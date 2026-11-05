"""Run in the isolated Qwen environment, from repository root."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.qwen_training import train_qwen

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",required=True);p.add_argument("--config",default="configs/qwen/lora.json")
    p.add_argument("--out",required=True);p.add_argument("--device",default="cuda")
    p.add_argument("--resume");p.add_argument("--max-steps",type=int);p.add_argument("--soft-config")
    args=p.parse_args();meta=train_qwen(args.dataset,args.config,args.out,args.device,args.resume,args.max_steps,args.soft_config)
    print(json.dumps({k:meta[k] for k in ("step","train_rows","changed_lora_tensors","trainable_parameters","evaluation_scope","seconds")},ensure_ascii=False))
