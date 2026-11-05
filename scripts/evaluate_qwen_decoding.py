"""Change one decoder control at a time on the same held-out original case."""
import argparse
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.qwen_faithfulness import evaluate_qwen

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--dataset",required=True);p.add_argument("--adapter",required=True)
    p.add_argument("--out",required=True);p.add_argument("--config",default="configs/qwen/inference_smoke.json")
    p.add_argument("--qwen-python",default=".venv-qwen/bin/python");p.add_argument("--device",default="cpu")
    p.add_argument("--variants",nargs="+",choices=["baseline","no_json","unicode","repeat4"],default=["baseline","no_json","unicode","repeat4"])
    args=p.parse_args();cfg=json.loads(Path(args.config).read_text());out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    experiments={};controls={};reference=None
    for name in args.variants:
        current=copy.deepcopy(cfg)
        if name=="no_json":current["decoding"]["json_constraints"]=False
        elif name=="unicode":current["decoding"]["prose_filter"]="none"
        elif name=="repeat4":current["decoding"]["no_repeat_ngram"]=4
        result=evaluate_qwen(args.dataset,str(out/(name+".json")),args.qwen_python,adapter=args.adapter,config=current,
                             conditions=["C"],base_cases=1,variants=["original"],device=args.device)
        row=result["results"][0] if result["results"] else None
        if row:
            fingerprint=(row["input_sha256"],row["metadata"]["rendered_prompt_sha256"])
            if reference is None:reference=fingerprint
            controls[name]=dict(same_frozen_input=fingerprint==reference)
        experiments[name]=result["metrics"]["C"]
    (out/"summary.json").write_text(json.dumps(dict(experiments=experiments,controls=controls,
        note="Single-case decoder engineering comparison, no human semantic gold"),ensure_ascii=False,indent=2))
    print(json.dumps(experiments,ensure_ascii=False))
