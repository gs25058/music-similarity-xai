import argparse
import json
from pathlib import Path

import torch

from .backends import make_backend
from .compare import analyze_song, compare_songs, public_song
from .data import save_json
from .judge import judge_pair
from .structure import SongFormerStructure


def model_arguments(p):
    p.add_argument("--backend", choices=["dsp", "merit", "ours"], default="dsp")
    p.add_argument("--checkpoint")
    p.add_argument("--device", default="cpu")


def main():
    p = argparse.ArgumentParser(description="Full-song musical similarity and grounded Korean explanation")
    commands = p.add_subparsers(dest="command", required=True)
    synth = commands.add_parser("synth")
    synth.add_argument("--out", default="data/synthetic")
    synth.add_argument("--originals", type=int, default=12)
    synth.add_argument("--seed", type=int, default=42)
    compare = commands.add_parser("compare")
    compare.add_argument("a")
    compare.add_argument("b")
    compare.add_argument("--out", required=True)
    compare.add_argument("--window", type=float, default=10.)
    compare.add_argument("--hop", type=float, default=5.)
    compare.add_argument("--threshold", type=float, default=.8)
    compare.add_argument("--top-k", type=int, default=20)
    compare.add_argument("--weights", help='JSON factor weights, e.g. {"melody":1,"rhythm":1,"timbre":0}')
    compare.add_argument("--songformer-python")
    compare.add_argument("--judge", choices=["rule", "qwen"], default="rule")
    compare.add_argument("--qwen-python")
    compare.add_argument("--qwen-config",help="Opt-in evidence/claims/adapter settings JSON; default keeps legacy schema")
    model_arguments(compare)
    training = commands.add_parser("train")
    training.add_argument("--manifest", required=True)
    training.add_argument("--out", required=True)
    training.add_argument("--backbone", choices=["dsp", "mert95"], default="mert95")
    training.add_argument("--epochs", type=int, default=2)
    training.add_argument("--batch-size", type=int, default=4)
    training.add_argument("--lr", type=float, default=.001)
    training.add_argument("--prediction-weight", type=float, default=1.)
    training.add_argument("--unfreeze-last", type=int, choices=[0, 2], default=0)
    training.add_argument("--init-checkpoint")
    training.add_argument("--device", default="cpu")
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--max-pairs", type=int, default=0)
    training.add_argument("--max-windows-per-pair", type=int, default=0)
    training.add_argument("--smoke", action="store_true")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--manifest", required=True)
    evaluation.add_argument("--out", required=True)
    evaluation.add_argument("--split", choices=["validation", "test"], default="test")
    evaluation.add_argument("--threshold", type=float, default=.8)
    evaluation.add_argument("--max-pairs", type=int, default=0)
    evaluation.add_argument("--judge", choices=["rule", "qwen"], default="rule")
    evaluation.add_argument("--qwen-python")
    evaluation.add_argument("--judge-cases", type=int, default=2)
    model_arguments(evaluation)
    prepare = commands.add_parser("prepare-real")
    prepare.add_argument("--pop909")
    prepare.add_argument("--slakh")
    prepare.add_argument("--slakh-redux")
    prepare.add_argument("--structure-root")
    prepare.add_argument("--out", required=True)
    prepare.add_argument("--max-originals", type=int, default=0)
    prepare.add_argument("--max-segments", type=int, default=0)
    prepare.add_argument("--seed", type=int, default=42)
    args = vars(p.parse_args())
    command = args.pop("command")
    torch.set_num_threads(2)
    if command == "synth":
        from .synthetic import prepare_synthetic
        result = prepare_synthetic(**args)
        print(json.dumps(dict(pairs=len(result["pairs"]), insertion_cases=len(result["insertion_cases"]))))
    elif command == "train":
        from .train import train
        args["manifest_path"] = args.pop("manifest")
        train(**args)
    elif command == "prepare-real":
        from .prepare import prepare_real
        result = prepare_real(**args)
        print(json.dumps(dict(pairs=len(result["pairs"]))))
    elif command == "evaluate":
        from .evaluate import evaluate
        backend = make_backend(args.pop("backend"), args.pop("checkpoint"), args["device"])
        args["manifest_path"] = args.pop("manifest")
        result = evaluate(backend=backend, **args)
        print(json.dumps(result["factor_metrics"], ensure_ascii=False))
    else:
        backend = make_backend(args["backend"], args["checkpoint"], args["device"])
        structure = SongFormerStructure(args["songformer_python"], args["device"]) if args["songformer_python"] else None
        a, b = [analyze_song(args[side], backend, args["window"], args["hop"], structure) for side in ("a", "b")]
        comparison = compare_songs(a, b, backend, args["threshold"], args["top_k"],
                                   json.loads(args["weights"]) if args["weights"] else None)
        judgment = judge_pair(comparison, args["judge"], args["qwen_python"], args["device"],args["qwen_config"],
                              dict(backend=backend.metadata(),songs=dict(a=a,b=b)))
        report = dict(schema_version=1, task="musical_similarity_for_further_review", backend=backend.metadata(),
                      songs=dict(a=public_song(a), b=public_song(b)), comparison=comparison, judgment=judgment,
                      analysis=dict(window_seconds=args["window"], hop_seconds=args["hop"], sample_rate=24000,
                                    unsupported_elements=[factor for factor in ("pitch_intervals", "rhythm", "harmony", "accompaniment_bass", "phrase_structure") if factor not in backend.supported]))
        save_json(args["out"], report)
        print(json.dumps(dict(report=str(Path(args["out"]).resolve()), judgment=judgment["selected"]["judgment"],
                              candidates=len(comparison["evidence"])), ensure_ascii=False))


if __name__ == "__main__":
    main()
