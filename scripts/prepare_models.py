"""Download pinned public artifacts. No data or checkpoints enter Git."""
import argparse
import json
import time
from pathlib import Path
from huggingface_hub import snapshot_download


def main():
    p = argparse.ArgumentParser()
    p.add_argument("components", nargs="+", choices=["mert95", "mert330", "merit_heads", "songformer", "qwen"])
    args = p.parse_args()
    specs = json.loads(Path("configs/model_versions.json").read_text())
    components = list(args.components)
    if "songformer" in components:
        components.append("songformer_aux_config")
    for key in components:
        spec = specs[key]
        if key.startswith("mert"):
            allow = ["*.py", "config.json", "preprocessor_config.json", "pytorch_model.bin", "README.md"]
        elif key == "merit_heads":
            allow = ["head_*/best_head.pt", "README.md"]
        elif key == "songformer_aux_config":
            allow = ["config.json"]
        elif key == "qwen":
            allow = ["*.json", "*.jinja", "*.txt", "*.safetensors", "README.md", "LICENSE"]
        else:
            allow = None
        for attempt in range(3):
            try:
                path = snapshot_download(spec["repo_id"], revision=spec["revision"],
                                         local_dir=spec["local_dir"], allow_patterns=allow,
                                         ignore_patterns=["*fairseq*", "SongFormer.pt", "SongFormer.safetensors"],
                                         max_workers=3)
                break
            except (OSError, RuntimeError) as error:
                if attempt == 2:
                    raise
                print(f"Retrying {key}: {error}", flush=True)
                time.sleep(1)
        Path(path, "artifact_version.json").write_text(json.dumps(spec, indent=2)+"\n")
        print(key, path, spec["revision"], flush=True)


if __name__ == "__main__":
    main()
