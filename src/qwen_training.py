"""Response-only weighted SFT. Imports no Transformers until training starts."""
import copy
import json
import random
import re
from pathlib import Path

import torch

from .evidence import digest,file_digest
from .qwen_data import audit_source_groups,read_jsonl
from .qwen_protocol import messages_for


def response_encoding(tokenizer,target,supervision):
    """Offsets refer to the exact JSON taught to the model, including cites."""
    text=json.dumps(target,ensure_ascii=False,separators=(",",":"))
    encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    spans={}
    for field in ("claims","verdict","explanation"):
        prefix='"'+field+'":';start=text.index(prefix)
        later=[text.find(',"'+other+'":',start+len(prefix)) for other in ("claims","verdict","explanation") if other!=field]
        end=min([x for x in later if x!=-1] or [len(text)-1])
        spans[field]=(start,end)
    citations=[m.span() for m in re.finditer(r'"evidence_ids":\[[^\]]*\]',text)]
    roles,mask=[],[]
    for start,end in encoded["offset_mapping"]:
        overlap=[k for k,(a,b) in spans.items() if start<b and end>a]
        field=overlap[0] if overlap else "structure"
        # BPE tokens may straddle a field boundary. If ANY overlapping field
        # lacks a target, mask the whole token rather than leaking its value.
        known=all(supervision.get(k,False) for k in overlap) if overlap else True
        category="citation" if any(start<b and end>a for a,b in citations) else field
        roles.append(category);mask.append(bool(known))
    return text,encoded["input_ids"],roles,mask


def has_supervised_target(row):
    return any(bool(row["supervision"].get(k)) and row["target"].get(k) is not None for k in ("claims","verdict","explanation"))


def encode_training_row(processor,row,max_length=8192):
    prompt=processor.apply_chat_template(messages_for(row["input"]),tokenize=False,add_generation_prompt=True,enable_thinking=False)
    prompt_ids=processor.tokenizer.encode(prompt,add_special_tokens=False)
    supervision={k:bool(row["supervision"].get(k,False)) and row["target"].get(k) is not None for k in ("claims","verdict","explanation")}
    text,response,roles,mask=response_encoding(processor.tokenizer,row["target"],supervision)
    ids=prompt_ids+response+[processor.tokenizer.eos_token_id]
    if len(ids)>max_length:raise ValueError(f"{row['case_id']}: {len(ids)} tokens exceeds {max_length}; reduce evidence bins explicitly, do not truncate targets")
    labels=[-100]*len(prompt_ids)+[token if known else -100 for token,known in zip(response,mask)]
    # End-of-response supervision is meaningful only for a complete target.
    labels.append(ids[-1] if all(supervision.values()) else -100)
    return dict(input_ids=ids,labels=labels,roles=["input"]*len(prompt_ids)+roles+["structure"],
                prompt_tokens=len(prompt_ids),output_tokens=len(response)+1)


def collate_encoded(rows,pad_id):
    size=max(len(r["input_ids"]) for r in rows)
    return dict(input_ids=torch.tensor([r["input_ids"]+[pad_id]*(size-len(r["input_ids"])) for r in rows]),
        attention_mask=torch.tensor([[1]*len(r["input_ids"])+[0]*(size-len(r["input_ids"])) for r in rows]),
        labels=torch.tensor([r["labels"]+[-100]*(size-len(r["labels"])) for r in rows]))


def response_loss(model,encoded,device,field_weights,connector=None,soft_features=None):
    from torch.nn import functional as F
    batch=collate_encoded([encoded],model.config.text_config.pad_token_id or 0)
    batch={k:v.to(device) for k,v in batch.items()}
    labels=batch.pop("labels")
    roles=encoded["roles"]
    if connector is not None:
        from src.soft_tokens import prepend_soft_tokens
        batch,labels=prepend_soft_tokens(model,batch,labels,connector,soft_features)
        roles=["input"]*connector.token_count+roles
    positions=torch.where(labels[0]!=-100)[0]
    positions=positions[positions>0]
    if not len(positions):raise ValueError("No supervised output tokens")
    outputs=model(**batch,logits_to_keep=positions-1,use_cache=False)
    ce=F.cross_entropy(outputs.logits[0].float(),labels[0,positions],reduction="none")
    categories=[roles[int(i)] for i in positions]
    weights=torch.tensor([field_weights.get(c,1.) for c in categories],device=device)
    loss=(ce*weights).sum()/weights.sum()
    stats={}
    for category in sorted(set(categories)):
        mask=torch.tensor([c==category for c in categories],device=device)
        stats[category]=dict(tokens=int(mask.sum()),mean_ce=float(ce[mask].mean().detach()),
                             weighted_loss_sum=float((ce[mask]*weights[mask]).sum().detach()))
    return loss,stats


def train_qwen(dataset,config_path,out,device="cuda",resume=None,max_steps=None,soft_config=None):
    import sys,time
    import numpy as np
    import peft,transformers
    from peft import LoraConfig,get_peft_model
    sys.path.insert(0,str(Path("scripts").resolve()))
    from qwen_runtime import load_qwen
    started=time.perf_counter();config=json.loads(Path(config_path).read_text())
    if max_steps is not None:config["max_steps"]=max_steps
    rows=read_jsonl(dataset);audit=audit_source_groups(rows)
    allowed=set(config.get("review_statuses",["human_approved"]))
    selected=[r for r in rows if r["review"]["status"] in allowed]
    train=[r for r in selected if r["split"]=="train" and has_supervised_target(r)]
    validation=[r for r in selected if r["split"]=="validation" and has_supervised_target(r)]
    if not train:raise ValueError("No training rows in selected review statuses; review data or explicitly use smoke config")
    for row in train:
        if row["review"]["status"]=="human_approved" and not row["review"].get("reviewer"):raise ValueError("Approved data needs an identified reviewer")
        if row["supervision"].get("verdict") and row["target_sources"]["verdict"]=="rule_provisional" and not config.get("allow_weak_verdict_supervision",False):
            raise ValueError("Provisional verdict supervision needs explicit allow_weak_verdict_supervision; otherwise mask it")
    seed=config["seed"];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    model,processor,load_meta=load_qwen(device,resume,trainable=bool(resume))
    target_pattern=config["target_modules"]
    matched=[n for n,v in model.named_modules() if isinstance(v,torch.nn.Linear) and re.fullmatch(target_pattern,n)] if not resume else []
    if not resume:
        if not matched:raise ValueError("Configured LoRA modules matched none of the actual Qwen modules")
        model=get_peft_model(model,LoraConfig(r=config["rank"],lora_alpha=config["alpha"],lora_dropout=config["dropout"],
                                             target_modules=target_pattern,task_type="CAUSAL_LM"))
    model.config.use_cache=False
    if config.get("gradient_checkpointing",True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    model.train()
    connector=None
    soft_spec=json.loads(Path(soft_config).read_text()) if soft_config else None
    if soft_spec:
        from .soft_tokens import SoftConnector
        connector=SoftConnector(**soft_spec).to(device)
        if connector.output_dim!=model.get_input_embeddings().weight.shape[1]:raise ValueError("Soft output dimension differs from actual Qwen embeddings")
        if resume:connector.load_state_dict(torch.load(Path(resume)/"connector.pt",weights_only=True,map_location=device))
    trainable={n:p for n,p in model.named_parameters() if p.requires_grad}
    if any("lora_" not in n or "language_model" not in n for n in trainable):raise ValueError("Unexpected non-LoRA or vision parameter is trainable")
    params=list(trainable.values())+(list(connector.parameters()) if connector is not None else [])
    optimizer=torch.optim.AdamW(params,lr=config["learning_rate"],weight_decay=config.get("weight_decay",0.))
    initial={n:p.detach().clone() for n,p in trainable.items()}
    history=[];step=0;data_hash=file_digest(dataset)
    if resume:
        meta=json.loads((Path(resume)/"training_meta.json").read_text())
        if meta.get("soft_config")!=soft_spec:raise ValueError("Resume soft-connector configuration mismatch")
        if meta["dataset_sha256"]!=data_hash:raise ValueError("Resume data hash mismatch")
        for key in ("rank","alpha","target_modules","seed","review_statuses","field_weights"):
            if meta["config"][key]!=config[key]:raise ValueError(f"Resume config mismatch: {key}")
        state=torch.load(Path(resume)/"training_state.pt",map_location="cpu",weights_only=True)
        optimizer.load_state_dict(state["optimizer"]);step=state["step"];history=state["history"]
        torch.set_rng_state(state["torch_rng"])
        if device!="cpu" and state.get("cuda_rng") is not None:torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])
    start_step=step
    encoded_cache={}
    # Validate every eligible sequence before optimizing, so an irrelevant-
    # evidence augmentation cannot fail halfway through a training run.
    for row in train:
        encoded=encode_training_row(processor,row,config["max_length"])
        if len(encoded_cache)<128:encoded_cache[row["case_id"]]=encoded
    for current in range(step,config["max_steps"]):
        row=train[current%len(train)]
        if row["case_id"] not in encoded_cache:
            encoded_cache[row["case_id"]]=encode_training_row(processor,row,config["max_length"])
        encoded=encoded_cache[row["case_id"]]
        optimizer.zero_grad(set_to_none=True)
        features=None
        if connector is not None:
            from .soft_tokens import load_feature_cache
            if not row.get("soft_cache"):raise ValueError("Row has no frozen feature cache")
            features=load_feature_cache(row["soft_cache"],row["input"])[0].to(device)
        loss,stats=response_loss(model,encoded,device,config["field_weights"],connector,features)
        if not torch.isfinite(loss):raise FloatingPointError("Nonfinite SFT loss")
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(params,config.get("max_grad_norm",1.))
        if not torch.isfinite(norm):raise FloatingPointError("Nonfinite gradient")
        if connector is not None:
            connector_gradient=sum(float(p.grad.abs().sum()) for p in connector.parameters() if p.grad is not None)
            if len(features)>0 and connector_gradient<=0:raise RuntimeError("Soft connector gradient did not flow")
        else:connector_gradient=None
        optimizer.step();step=current+1
        record=dict(step=step,case_id=row["case_id"],loss=float(loss.detach()),gradient_norm=float(norm),
                    field_losses=stats,prompt_tokens=encoded["prompt_tokens"],output_tokens=encoded["output_tokens"],
                    connector_gradient_l1=connector_gradient)
        history.append(record);print(json.dumps(record),flush=True)
    changed=[n for n,p in trainable.items() if not torch.equal(initial[n],p.detach())]
    if step>start_step and not changed:
        raise RuntimeError("LoRA parameters did not change")
    out=Path(out);out.mkdir(parents=True,exist_ok=True);model.save_pretrained(out)
    if connector is not None:
        torch.save(connector.state_dict(),out/"connector.pt");(out/"soft_config.json").write_text(json.dumps(soft_spec,indent=2))
    ntrain,total=model.get_nb_trainable_parameters()
    meta=dict(base_model=json.loads(Path("configs/model_versions.json").read_text())["qwen"],dataset=str(dataset),dataset_sha256=data_hash,
        source_audit=audit,seed=seed,config=config,trainable_parameters=ntrain,total_parameters=total,
        connector_parameters=sum(p.numel() for p in connector.parameters()) if connector is not None else 0,
        matched_modules=matched or json.loads((Path(resume)/"training_meta.json").read_text())["matched_modules"],
        changed_lora_tensors=len(changed),train_source_groups=sorted({r["source_group_id"] for r in train}),
        train_original_ids=sorted({v for r in train for v in r["original_ids"]}),
        validation_source_groups=sorted({r["source_group_id"] for r in validation}),train_rows=len(train),validation_rows=len(validation),
        review_statuses=sorted(allowed),evaluation_scope="automatic_data_optimization_smoke" if allowed!={"human_approved"} else "reviewed_training_not_test_evaluation",
        supervised_rows={field:sum(bool(r["supervision"].get(field)) and r["target"].get(field) is not None for r in train) for field in ("claims","verdict","explanation")},
        step=step,steps_executed_this_run=step-start_step,used_case_ids=sorted({v["case_id"] for v in history}),
        resumed_from=resume,versions=dict(torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__),
        history=history,seconds=round(time.perf_counter()-started,3),soft_config=soft_spec)
    (out/"training_meta.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    torch.save(dict(optimizer=optimizer.state_dict(),step=step,history=history,torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if device!="cpu" else None,python_rng=random.getstate()),out/"training_state.pt")
    return meta
