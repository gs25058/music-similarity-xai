"""Opt-in claims path; legacy report consumers still see selected.judgment."""
import json
from pathlib import Path
import subprocess

from .claims import draft_response,legacy_projection,validate_claim_response
from .evidence import compact_evidence


def judge_configured(comparison,backend,qwen_python,device,config,context=None):
    context=context or {};cfg=json.loads(Path(config).read_text()) if isinstance(config,(str,Path)) else dict(config)
    if cfg.get("evidence_format","compact")!="compact" or cfg.get("response_format","claims")!="claims":
        raise ValueError("Claims configuration requires compact/claims; omit --qwen-config to run the unchanged legacy CLI path")
    calibration=json.loads(Path(cfg["calibration"]).read_text()) if cfg.get("calibration") else None
    payload=compact_evidence(comparison,context.get("backend"),context.get("songs"),calibration,
                             cfg.get("evidence_limit",3),cfg.get("time_bins",6))
    from .judge import rule_judge
    rule=rule_judge(comparison)
    response=draft_response(payload);claim_rule=legacy_projection(response)
    candidate={**response,"verdict":rule["judgment"],"explanation":rule["explanation"]}
    legacy_validation=validate_claim_response(candidate,payload)
    fallback_source="legacy_rule" if legacy_validation["valid"] else "claim_rule"
    if fallback_source=="legacy_rule":response=candidate
    status=dict(backend="rule",status="executed",fallback_source=fallback_source)
    model_validation=None;trace=None
    if backend=="qwen":
        if not qwen_python:raise ValueError("--qwen-python is required")
        request=dict(device=device,config=cfg,payload=payload)
        if cfg.get("soft_tokens"):request["soft_cache"]=cfg["soft_cache"]
        try:
            process=subprocess.run([qwen_python,"scripts/qwen_claim_worker.py"],input=json.dumps(request),text=True,capture_output=True,timeout=1800,check=True)
            trace=json.loads(process.stdout)["results"][0];model_validation=trace["validation"]
            if trace["success"]:model_validation=validate_claim_response(trace["result"],payload)
            if trace["success"] and model_validation["valid"]:
                response=trace["result"];status=dict(backend="qwen",status="executed_validated",metadata=trace["metadata"])
            else:status=dict(backend="qwen",status="failed_rule_fallback",error="claim_validation_failed",fallback_source=fallback_source)
        except (OSError,subprocess.SubprocessError,ValueError,KeyError) as e:
            status=dict(backend="qwen",status="failed_rule_fallback",fallback_source=fallback_source,error=(getattr(e,"stderr",None) or str(e))[-6000:])
    selected=legacy_projection(response);byid={e["evidence_id"]:e for e in payload["evidence"]}
    status["fallback_used"]=backend=="qwen" and status["status"]=="failed_rule_fallback"
    selected["cited_intervals"]=[dict(evidence_id=eid,a=byid[eid]["a"],b=byid[eid]["b"]) for eid in selected["evidence_ids"]]
    return dict(selected=selected,rule_baseline=rule,claim_rule_baseline=claim_rule,legacy_rule_validation=legacy_validation,
                execution=status,claim_response=response,
                claim_validation=validate_claim_response(response,payload),model_validation=model_validation,
                generation_trace=trace,evidence_input=payload)
