"""Persistent within a batch: fixed evidence -> raw attempts -> validations."""
import contextlib
import copy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.claims import response_schema,validate_claim_response
from src.qwen_protocol import messages_for
from qwen_runtime import load_qwen,generation_stop_args
from qwen_decoding import build_tokenizer_data,build_schema_decoder,ExplanationNoRepeat


def run_requests(request):
    import torch
    cfg=request.get("config",{});dec=cfg.get("decoding",{})
    from src.evidence import file_digest
    implementation={k:file_digest(p) for k,p in dict(decoder="scripts/qwen_decoding.py",protocol="src/qwen_protocol.py",validator="src/claims.py").items()}
    model,processor,load_meta=load_qwen(request.get("device","cpu"),cfg.get("adapter"))
    connector=None
    if cfg.get("soft_tokens"):
        from src.soft_tokens import load_connector
        connector_started=time.perf_counter();connector=load_connector(cfg["adapter"],model.device)
        load_meta["connector_load_seconds"]=round(time.perf_counter()-connector_started,4)
        load_meta["model_load_seconds"]+=load_meta["connector_load_seconds"]
    model.eval();cache_started=time.perf_counter()
    data=build_tokenizer_data(processor.tokenizer,dec.get("prose_filter","korean"),dec.get("forbid_numeric_prose",False)) if dec.get("json_constraints",True) else None
    cache_seconds=time.perf_counter()-cache_started
    results=[]
    for item in request.get("requests",[request]):
        started=time.perf_counter();payload=item["payload"];mode=item.get("mode","claims")
        if mode=="legacy":
            from qwen_worker import PROMPT
            messages=[{"role":"system","content":[{"type":"text","text":PROMPT}]},
                      {"role":"user","content":[{"type":"text","text":json.dumps(dict(evidence=item["legacy_input"]),ensure_ascii=False)}]}]
        else:messages=messages_for(payload)
        from src.evidence import digest
        rendered_prompt_sha256=digest(messages)
        attempts=[];validated=None
        for attempt in range(dec.get("max_retries",1)+1):
            processing=time.perf_counter()
            inputs=processor.apply_chat_template(messages,tokenize=True,return_dict=True,return_tensors="pt",add_generation_prompt=True,enable_thinking=False).to(model.device)
            prefix=None;processors=[]
            if data is not None:
                if mode=="legacy":
                    ids=[e["evidence_id"] for e in item["legacy_input"]]
                    schema=dict(type="object",properties=dict(judgment=dict(type="string",enum=["high_similarity","review","low_similarity","insufficient_evidence"]),
                        evidence_ids=dict(type="array",items=dict(type="string",enum=ids or ["unused"]),maxItems=len(ids)),
                        explanation=dict(type="string",minLength=1,maxLength=400)),required=["judgment","evidence_ids","explanation"],additionalProperties=False)
                else:schema=response_schema(payload,dec.get("max_claims",4))
                prefix=build_schema_decoder(data,schema,dec.get("prose_filter","korean"))
                if dec.get("no_repeat_ngram",0):processors=[ExplanationNoRepeat(prefix,inputs["input_ids"].shape[-1],dec["no_repeat_ngram"])]
            process_seconds=time.perf_counter()-processing
            soft_processing=time.perf_counter()
            soft_info=None
            kwargs={}
            if cfg.get("soft_tokens"):
                from src.soft_tokens import load_feature_cache,prepend_soft_tokens
                features,soft_info=load_feature_cache(item["soft_cache"],payload)
                if item.get("soft_intervention")=="swap":
                    from src.soft_tokens import swap_feature_sides
                    features=swap_feature_sides(features)
                if item.get("soft_intervention")!="remove":
                    kwargs,_=prepend_soft_tokens(model,dict(inputs),None,connector,features.to(model.device))
                    # input_ids track generated text for JSON constraints;
                    # embeddings carry the actual prefix. No vocabulary resize.
                    dummy=torch.full((1,connector.token_count),processor.tokenizer.pad_token_id or 0,device=model.device,dtype=torch.long)
                    kwargs["input_ids"]=torch.cat([dummy,inputs["input_ids"]],-1)
            if not kwargs:kwargs=dict(inputs)
            soft_processing_seconds=time.perf_counter()-soft_processing
            input_length=kwargs["input_ids"].shape[-1]
            if prefix and dec.get("no_repeat_ngram",0):processors=[ExplanationNoRepeat(prefix,input_length,dec["no_repeat_ngram"])]
            generation=time.perf_counter()
            with torch.inference_mode():
                generated=model.generate(**kwargs,max_new_tokens=dec.get("max_new_tokens",1000),do_sample=False,
                    repetition_penalty=dec.get("repetition_penalty",1.05),prefix_allowed_tokens_fn=prefix,logits_processor=processors,
                    **generation_stop_args(processor))
            output=generated[0,input_length:]
            raw=processor.decode(output,skip_special_tokens=True).strip()
            generation_seconds=time.perf_counter()-generation
            try:
                value=json.loads(raw)
                if mode=="legacy":
                    from src.judge import validate_judgment
                    validate_judgment(value,item["legacy_input"])
                    validation=dict(valid=True,automatic_errors=[],claims=[],semantic_accuracy="not_measured_by_automatic_validation")
                else:validation=validate_claim_response(value,payload)
            except json.JSONDecodeError as error:
                value=None;validation=dict(valid=False,automatic_errors=[dict(code="json_parse_failure",detail=str(error))],claims=[])
            except (ValueError,TypeError,KeyError) as error:
                value=None;validation=dict(valid=False,automatic_errors=[dict(code="json_or_schema_failure",detail=str(error))],claims=[])
            attempts.append(dict(raw_response=raw,response=value,validation=validation,input_tokens=int(inputs["input_ids"].shape[-1]),
                output_tokens=len(output),input_processing_seconds=round(process_seconds+soft_processing_seconds,4),
                text_processing_seconds=round(process_seconds,4),soft_input_processing_seconds=round(soft_processing_seconds,4),
                soft_prefix_tokens=connector.token_count if cfg.get("soft_tokens") and item.get("soft_intervention")!="remove" else 0,
                input_embedding_positions=int(kwargs["attention_mask"].shape[-1]),active_input_positions=int(kwargs["attention_mask"].sum()),
                generation_seconds=round(generation_seconds,4)))
            if validation["valid"]:validated=value;break
            messages.extend([{"role":"assistant","content":[{"type":"text","text":raw}]},
                {"role":"user","content":[{"type":"text","text":"검증 오류를 원본 근거에 따라 고치세요: "+json.dumps(validation["automatic_errors"],ensure_ascii=False)}]}])
        results.append(dict(case_id=item.get("case_id"),result=validated,validation=attempts[-1]["validation"],attempts=attempts,
            success=validated is not None,metadata={**load_meta,"thinking":False,"device":str(model.device),"dtype":str(next(model.parameters()).dtype),
                "weights_frozen":all(not p.requires_grad for p in model.parameters()),"adapter_applied":bool(cfg.get("adapter")),"training_at_inference":False,"decoding":dec,
                "tokenizer_cache_seconds":round(cache_seconds,4),"case_seconds":round(time.perf_counter()-started,4),
                "input_tokens":attempts[0]["input_tokens"],"output_tokens":sum(a["output_tokens"] for a in attempts),"soft_cache":soft_info}))
        results[-1]["metadata"]["generation_stop"]=generation_stop_args(processor)
        results[-1]["metadata"]["rendered_prompt_sha256"]=rendered_prompt_sha256
        results[-1]["metadata"]["implementation_sha256"]=implementation
    return dict(results=results,load=load_meta,tokenizer_cache_seconds=cache_seconds)


if __name__=="__main__":
    request=json.load(sys.stdin)
    with contextlib.redirect_stdout(sys.stderr):response=run_requests(request)
    print(json.dumps(response,ensure_ascii=False))
