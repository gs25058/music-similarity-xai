"""Isolated official SongFormer inference, JSON stdout, diagnostics stderr.

Adapted from HF ASLP-lab/SongFormer README at a75880ed... .
Code provenance/license details: THIRD_PARTY_NOTICES.md.
"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    specs = json.loads(Path("configs/model_versions.json").read_text())
    spec = specs["songformer"]
    os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))
    os.environ.setdefault("HF_MODULES_CACHE", str(Path(".cache/hf_modules_songformer").resolve()))
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
    os.environ["HF_HUB_OFFLINE"] = "1"
    model_dir = Path(spec["local_dir"]).resolve()
    if not (model_dir / "modeling_songformer.py").exists():
        raise FileNotFoundError("Run scripts/prepare_models.py songformer first")
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        from transformers import AutoModel, Wav2Vec2ConformerConfig
        torch.set_num_threads(2)
        sys.path.insert(0, str(model_dir))
        os.environ["SONGFORMER_LOCAL_DIR"] = str(model_dir)
        # MusicFM's official constructor looks up this config by a floating Hub
        # ID. Redirect only that config lookup to a pinned local config. No model
        # source, architecture, or weights are replaced.
        original_config_loader = Wav2Vec2ConformerConfig.from_pretrained
        auxiliary = specs["songformer_aux_config"]
        def pinned_config(model_id, *args, **kwargs):
            if model_id == auxiliary["repo_id"]:
                return original_config_loader(str(Path(auxiliary["local_dir"]).resolve()), local_files_only=True)
            return original_config_loader(model_id, *args, **kwargs)
        with patch.object(Wav2Vec2ConformerConfig, "from_pretrained", side_effect=pinned_config):
            model, loading = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True,
                                    local_files_only=True, low_cpu_mem_usage=False, output_loading_info=True)
        if loading["missing_keys"] or loading.get("mismatched_keys"):
            raise ValueError(f"SongFormer has missing/mismatched weights: {loading}")
        model.to(args.device).eval()
        with torch.inference_mode():
            result = model(str(Path(args.audio).resolve()))
    # Official forward returns start/end/label dictionaries.
    segments = [dict(start=float(s["start"]), end=float(s["end"]), label=str(s["label"])) for s in result]
    print(json.dumps(dict(segments=segments, metadata={**spec, "auxiliary_config": auxiliary,
                                                      "unexpected_checkpoint_keys": loading["unexpected_keys"]}), ensure_ascii=False))


if __name__ == "__main__":
    main()
