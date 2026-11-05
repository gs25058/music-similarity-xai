import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.soft_tokens import prepare_soft_caches

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--dataset",required=True);p.add_argument("--out",required=True)
    p.add_argument("--checkpoint",required=True);p.add_argument("--device",default="cpu");p.add_argument("--samples-per-window",type=int,default=16)
    args=vars(p.parse_args())
    import torch
    torch.set_num_threads(2)
    print(json.dumps(prepare_soft_caches(**args),ensure_ascii=False))
