"""Run from core environment, delegating model inference to isolated Python."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.qwen_faithfulness import evaluate_qwen

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--dataset",required=True);p.add_argument("--out",required=True)
    p.add_argument("--qwen-python",default=".venv-qwen/bin/python");p.add_argument("--adapter");p.add_argument("--soft-adapter")
    p.add_argument("--config",default="configs/qwen/inference.json");p.add_argument("--split",default="test",choices=["validation","test"])
    p.add_argument("--conditions",nargs="+",default=["A","B","C"]);p.add_argument("--base-cases",type=int,default=0)
    p.add_argument("--variants",nargs="*");p.add_argument("--device",default="cpu");p.add_argument("--semantic-reviews")
    p.add_argument("--max-new-tokens",type=int);p.add_argument("--max-claims",type=int)
    r=evaluate_qwen(**vars(p.parse_args()));print(json.dumps(dict(conditions=r["conditions"],metrics=r["metrics"]),ensure_ascii=False))
