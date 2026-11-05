"""Paired A/B/C(/D) evaluation on frozen pipeline predictions, no rule credit."""
import copy
import json
from pathlib import Path
import subprocess

from .evidence import digest,file_digest
from .qwen_data import audit_source_groups,read_jsonl,write_jsonl
from .qwen_metrics import validate_semantic_review


def evaluate_qwen(dataset,out,qwen_python,adapter=None,config="configs/qwen/inference.json",split="test",
                  conditions=("A","B","C"),base_cases=0,variants=None,device="cpu",semantic_reviews=None,
                  max_new_tokens=None,max_claims=None,soft_adapter=None):
    rows=read_jsonl(dataset);audit=audit_source_groups(rows)
    selected=[r for r in rows if r["split"]==split]
    # Include a pattern-reuse case in smoke, followed by hard negatives.
    bases=list(dict.fromkeys(r["base_case_id"] for r in sorted(selected,key=lambda r:r["evaluation_truth"]["kind"]!="insertion")))
    if base_cases:bases=bases[:base_cases]
    selected=[r for r in selected if r["base_case_id"] in bases and (not variants or r["variant"] in variants)]
    if not selected:raise ValueError("No evaluation cases")
    cfg=json.loads(Path(config).read_text()) if isinstance(config,(str,Path)) else copy.deepcopy(config)
    dec=cfg.setdefault("decoding",{})
    if max_new_tokens is not None:dec["max_new_tokens"]=max_new_tokens
    if max_claims is not None:dec["max_claims"]=max_claims
    # Training source groups and original IDs must be disjoint from evaluated
    # rows, even when an adapter is passed from another dataset/manifest.
    for path in [p for p in (adapter,soft_adapter) if p]:
        meta=json.loads((Path(path)/"training_meta.json").read_text())
        if set(meta["train_source_groups"]) & {r["source_group_id"] for r in selected}:
            raise ValueError("Adapter training/evaluation source group leakage")
        originals=set(meta.get("train_original_ids",[]))
        if originals & {x for r in selected for x in r["original_ids"]}:raise ValueError("Adapter original-ID leakage")
    all_results=[];condition_status={}
    review_lookup={}
    if semantic_reviews:
        for review in read_jsonl(semantic_reviews):review_lookup[(review["condition"],review["case_id"])]=review
    for condition in conditions:
        current=copy.deepcopy(cfg);current.pop("adapter",None);current.pop("soft_tokens",None)
        if condition=="A" and "legacy_decoding" in current:
            current["decoding"]=copy.deepcopy(current["legacy_decoding"])
            if max_new_tokens is not None:current["decoding"]["max_new_tokens"]=max_new_tokens
        if condition=="C":
            if not adapter:condition_status[condition]="unavailable_no_adapter";continue
            current["adapter"]=adapter
        elif condition=="D":
            if not soft_adapter:condition_status[condition]="unavailable_no_soft_adapter";continue
            current.update(adapter=soft_adapter,soft_tokens=True)
        elif condition not in ("A","B"):raise ValueError(condition)
        requests=[]
        for row in selected:
            requests.append(dict(case_id=row["case_id"],payload=row["input"],legacy_input=row["legacy_input"],
                                 mode="legacy" if condition=="A" else "claims",soft_cache=row.get("soft_cache")))
        if condition=="D":
            for row in [r for r in selected if r["variant"]=="original"]:
                for change in ("remove","swap"):
                    requests.append(dict(case_id=row["case_id"]+"_soft_"+change,payload=row["input"],mode="claims",
                                         soft_cache=row.get("soft_cache"),soft_intervention=change))
        try:
            process=subprocess.run([qwen_python,"scripts/qwen_claim_worker.py"],input=json.dumps(dict(device=device,config=current,requests=requests)),
                                   text=True,capture_output=True,timeout=max(1800,len(requests)*600),check=True)
            response=json.loads(process.stdout);condition_status[condition]="executed"
        except (subprocess.SubprocessError,OSError,ValueError) as error:
            condition_status[condition]=dict(status="worker_failed",error=(getattr(error,"stderr",None) or str(error))[-6000:]);continue
        byid={r["case_id"]:r for r in selected}
        for result in response["results"]:
            case_id=result["case_id"]
            base_id=case_id.split("_soft_")[0]
            row=byid[base_id]
            result.update(condition=condition,base_case_id=row["base_case_id"],variant=row["variant"] if "_soft_" not in case_id else "soft_"+case_id.split("_soft_")[1],
                input_sha256=digest(row["legacy_input"] if condition=="A" else row["input"]),decoding_sha256=digest(current["decoding"]),
                source_group_id=row["source_group_id"],review_status=row["review"]["status"],
                original_evidence_count=next((len(x["input"]["evidence"]) for x in selected if x["base_case_id"]==row["base_case_id"] and x["variant"]=="original"),None),
                intervention=row["intervention"],evaluation_truth=row["evaluation_truth"],
                fallback_used=False,fallback_required=not result["success"],
                fallback_policy="disabled_for_model_quality_comparison; production_fallback_is_reported_by_judge_pair")
            result["response_sha256"]=digest(result["attempts"][-1]["raw_response"])
            review=review_lookup.get((condition,case_id))
            if review and not validate_semantic_review(review,result):review=None
            result["human_semantic_review"]=review
            all_results.append(result)
        # Keep partial results even if a later condition fails.
        Path(out).parent.mkdir(parents=True,exist_ok=True)
        Path(str(out)+".partial.json").write_text(json.dumps(dict(results=all_results,condition_status=condition_status),ensure_ascii=False,indent=2))
    from .qwen_metrics import aggregate_metrics
    metrics=aggregate_metrics(all_results,conditions)
    bc={}
    bykey={(r["condition"],r["case_id"]):r for r in all_results}
    for row in selected:
        b,c=bykey.get(("B",row["case_id"])),bykey.get(("C",row["case_id"]))
        if b and c:
            bc[row["case_id"]]=dict(same_input=b["input_sha256"]==c["input_sha256"],
                same_rendered_prompt=b["metadata"].get("rendered_prompt_sha256")==c["metadata"].get("rendered_prompt_sha256"),
                same_implementation=b["metadata"].get("implementation_sha256")==c["metadata"].get("implementation_sha256"),
                same_decoding=b["decoding_sha256"]==c["decoding_sha256"])
    result=dict(schema_version=2,evaluation_mode="fixed_predicted_evidence",dataset_sha256=file_digest(dataset),source_audit=audit,
        selected_base_cases=len(bases),selected_rows=len(selected),conditions=condition_status,metrics=metrics,bc_controls=bc,
        human_semantic_gold_available=bool(semantic_reviews),results=all_results,
        limitations=["automatic_predicate_checks_do_not_measure_free_text_semantics","automatic_pilot_is_not_human_gold",
                     "legacy_A_has_no_claim_list_so_claim_metrics_are_unavailable","fixed_evidence_invariance_is_not_full_audio_pipeline_symmetry"])
    if adapter and soft_adapter:
        cm,dm=[json.loads((Path(p)/"training_meta.json").read_text()) for p in (adapter,soft_adapter)]
        def training_digest(meta):
            if file_digest(meta["dataset"])!=meta["dataset_sha256"]:raise ValueError("Training dataset changed after checkpoint save")
            samples=[r for r in read_jsonl(meta["dataset"]) if r["split"]=="train" and r["review"]["status"] in meta["review_statuses"]]
            return digest([{k:r[k] for k in ("case_id","input","target","supervision","source_group_id")} for r in samples])
        result["cd_training_controls"]=dict(same_base_model=cm["base_model"]==dm["base_model"],same_steps=cm["step"]==dm["step"],
            same_seed=cm["seed"]==dm["seed"],same_common_training_data=training_digest(cm)==training_digest(dm),
            same_source_groups=cm["train_source_groups"]==dm["train_source_groups"],
            C_step=cm["step"],D_step=dm["step"],D_connector_parameters=dm["connector_parameters"])
    Path(out).parent.mkdir(parents=True,exist_ok=True);Path(out).write_text(json.dumps(result,ensure_ascii=False,indent=2))
    write_jsonl(str(out)+".semantic_review.jsonl",[dict(condition=r["condition"],case_id=r["case_id"],response_sha256=r["response_sha256"],
        raw_response=r["attempts"][-1]["raw_response"],payload_case_id=r["case_id"],review_status="unreviewed",reviewer=None,supported_claim_ids=[],unsupported_claim_ids=[],statement_spans=[],
        contradiction=None,notes="Review against frozen dataset input; do not approve from the predicate checker alone") for r in all_results])
    return result
