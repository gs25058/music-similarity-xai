"""Attach human annotations and rescore immutable model outputs, without GPU."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.evidence import digest,file_digest
from src.qwen_data import read_jsonl
from src.qwen_metrics import aggregate_metrics,validate_semantic_review

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--report",required=True);p.add_argument("--out",required=True);p.add_argument("--reviews");p.add_argument("--dataset")
    args=p.parse_args();r=json.loads(Path(args.report).read_text())
    if r.get("evaluation_mode")=="fixed_predicted_evidence":
        for x in r["results"]:
            if x.get("fallback_policy","").startswith("rule_fallback_available"):
                x["old_fallback_required_flag"]=x.get("fallback_used")
                x.update(fallback_used=False,fallback_required=not x["success"],fallback_policy="disabled_for_model_quality_comparison")
    if args.dataset:
        rows=read_jsonl(args.dataset);counts={x["base_case_id"]:len(x["input"]["evidence"]) for x in rows if x["variant"]=="original"};byid={x["case_id"]:x for x in rows}
        for x in r["results"]:
            x["original_evidence_count"]=counts.get(x["base_case_id"])
            case_id=x["case_id"].split("_soft_")[0];row=byid[case_id]
            if digest(row["legacy_input"] if x["condition"]=="A" else row["input"])!=x["input_sha256"]:raise ValueError("Scoring dataset altered the frozen model input")
            x["evaluation_truth"]=row["evaluation_truth"]
        r["scoring_dataset_sha256"]=file_digest(args.dataset)
    if args.reviews:
        reviews={(x["condition"],x["case_id"]):x for x in read_jsonl(args.reviews)}
        for x in r["results"]:
            review=reviews.get((x["condition"],x["case_id"]))
            if review and validate_semantic_review(review,x):
                x["human_semantic_review"]=review
    r["metrics"]=aggregate_metrics(r["results"],r["conditions"])
    r["human_semantic_gold_available"]=any(x.get("human_semantic_review") for x in r["results"])
    r["scored_from"]=args.report
    Path(args.out).parent.mkdir(parents=True,exist_ok=True);Path(args.out).write_text(json.dumps(r,ensure_ascii=False,indent=2))
    print(json.dumps(r["metrics"],ensure_ascii=False))
