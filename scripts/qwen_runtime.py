"""Official Qwen loading and optional PEFT adapter shared by train/workers."""
import json
from pathlib import Path
import time
import os


def generation_stop_args(processor):
    # Pinned Qwen tokenizer ends an assistant turn with im_end (248046),
    # while GenerationConfig.from_model_config chooses endoftext (248044).
    # LMFE permits tokenizer.eos_token_id, so the IDs must agree explicitly.
    return dict(eos_token_id=processor.tokenizer.eos_token_id,pad_token_id=processor.tokenizer.pad_token_id)


def load_qwen(device="cpu",adapter=None,trainable=False):
    import torch
    from transformers import AutoProcessor,AutoModelForImageTextToText
    started=time.perf_counter();torch.set_num_threads(2)
    spec=json.loads(Path("configs/model_versions.json").read_text())["qwen"]
    path=Path(spec["local_dir"])
    actual=json.loads((path/"artifact_version.json").read_text())
    if any(actual[k]!=spec[k] for k in ("repo_id","revision")):raise ValueError("Qwen artifact revision mismatch")
    processor=AutoProcessor.from_pretrained(path,local_files_only=True)
    model=AutoModelForImageTextToText.from_pretrained(path,local_files_only=True,
        dtype=torch.float32 if device=="cpu" else torch.bfloat16).to(device)
    for p in model.parameters():p.requires_grad_(False)
    if adapter:
        from peft import PeftModel
        meta=json.loads((Path(adapter)/"training_meta.json").read_text())
        if meta["base_model"]!=spec:raise ValueError("LoRA base-model revision mismatch")
        model=PeftModel.from_pretrained(model,adapter,is_trainable=trainable)
    model.train(trainable)
    return model,processor,dict(base_model=spec,adapter=str(adapter) if adapter else None,
                              model_load_seconds=round(time.perf_counter()-started,4),
                              gpu_name=torch.cuda.get_device_name(device) if device!="cpu" else None,
                              cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
