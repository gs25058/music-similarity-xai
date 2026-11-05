"""Prepare/review/audit pilot data in the core environment."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

if __name__=="__main__":
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest="command",required=True)
    prep=sub.add_parser("prepare");prep.add_argument("--manifest",required=True);prep.add_argument("--out",required=True)
    prep.add_argument("--backend",choices=["dsp","ours","merit"],default="dsp");prep.add_argument("--checkpoint");prep.add_argument("--device",default="cpu")
    prep.add_argument("--max-per-split",type=int,default=3);prep.add_argument("--evidence-limit",type=int,default=2);prep.add_argument("--time-bins",type=int,default=6)
    prep.add_argument("--reports",nargs="*")
    prep.add_argument("--supervise-rule-verdict",action="store_true",help="Explicit weak-label smoke only; default masks provisional verdict loss")
    export=sub.add_parser("review-export");export.add_argument("--dataset",required=True);export.add_argument("--out",required=True)
    apply=sub.add_parser("review-apply");apply.add_argument("--dataset",required=True);apply.add_argument("--patches",required=True);apply.add_argument("--out",required=True)
    audit=sub.add_parser("audit");audit.add_argument("--dataset",required=True)
    order=sub.add_parser("order-test");order.add_argument("--dataset",required=True);order.add_argument("--out",required=True)
    view=sub.add_parser("review-view");view.add_argument("--dataset",required=True);view.add_argument("--out",required=True)
    cal=sub.add_parser("calibrate");cal.add_argument("--manifest",required=True);cal.add_argument("--out",required=True)
    cal.add_argument("--backend",choices=["dsp","ours","merit"],default="dsp");cal.add_argument("--checkpoint");cal.add_argument("--device",default="cpu")
    args=vars(p.parse_args());command=args.pop("command")
    import torch
    torch.set_num_threads(2)
    from src.qwen_data import prepare_pilot,review_export,review_apply,audit_source_groups,read_jsonl,prepare_order_test,dataset_coverage,review_view
    if command in ("prepare","calibrate"):
        from src.backends import make_backend
        backend=make_backend(args.pop("backend"),args.pop("checkpoint"),args.pop("device"))
        args["manifest_path"]=args.pop("manifest")
        if command=="prepare":result=prepare_pilot(backend=backend,**args)
        else:
            from src.evidence import calibrate_scores
            result=calibrate_scores(backend=backend,**args)
        print(json.dumps(result,ensure_ascii=False))
    elif command=="review-export":review_export(**args)
    elif command=="review-apply":review_apply(**args)
    elif command=="order-test":print(json.dumps(prepare_order_test(**args),ensure_ascii=False))
    elif command=="review-view":print(json.dumps(review_view(**args),ensure_ascii=False))
    else:print(json.dumps(dataset_coverage(read_jsonl(args["dataset"])),ensure_ascii=False))
