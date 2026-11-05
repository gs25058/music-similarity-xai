"""Merge separately executed conditions on identical frozen evaluation rows."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.qwen_metrics import aggregate_metrics

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--reports",nargs="+",required=True);p.add_argument("--out",required=True)
    args=p.parse_args();reports=[json.loads(Path(path).read_text()) for path in args.reports]
    first=reports[0];result={k:v for k,v in first.items() if k not in ("results","metrics","conditions","bc_controls")}
    results=[];conditions={};bc={};seen=set()
    for report in reports:
        if report["dataset_sha256"]!=first["dataset_sha256"] or report["selected_rows"]!=first["selected_rows"]:raise ValueError("Reports use different frozen datasets or case selections")
        for key,value in report["conditions"].items():
            if key in conditions:raise ValueError("Duplicate condition")
            conditions[key]=value
        bc.update(report.get("bc_controls",{}))
        for row in report["results"]:
            key=(row["condition"],row["case_id"])
            if key in seen:raise ValueError("Duplicate condition/case")
            seen.add(key);results.append(row)
    result.update(results=results,conditions=conditions,bc_controls=bc,metrics=aggregate_metrics(results,conditions),merged_from=args.reports)
    Path(args.out).parent.mkdir(parents=True,exist_ok=True);Path(args.out).write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:dict(successful=v["successful_cases"],requested=v["requested_cases"]) for k,v in result["metrics"].items()}))
