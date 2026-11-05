import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.backends import make_backend
from src.qwen_audio_eval import evaluate_audio_intervention
from src.structure import SongFormerStructure

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--manifest",required=True);p.add_argument("--out",required=True)
    p.add_argument("--backend",choices=["dsp","ours","merit"],default="ours");p.add_argument("--checkpoint")
    p.add_argument("--config",required=True);p.add_argument("--qwen-python",default=".venv-qwen/bin/python")
    p.add_argument("--device",default="cpu");p.add_argument("--songformer-python");p.add_argument("--split",choices=["validation","test"],default="test")
    args=vars(p.parse_args())
    import torch
    torch.set_num_threads(2)
    backend=make_backend(args.pop("backend"),args.pop("checkpoint"),args["device"])
    python=args.pop("songformer_python");structure=SongFormerStructure(python,args["device"]) if python else None
    print(json.dumps(evaluate_audio_intervention(backend=backend,structure=structure,**args),ensure_ascii=False))
