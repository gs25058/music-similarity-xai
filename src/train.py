import hashlib
import json
from pathlib import Path
import random
from collections import OrderedDict
import soundfile as sf

import numpy as np
import torch

from .audio import Window
from .data import frame_labels, save_json
from .features import FACTORS, FPS, INPUT_DIM, frame_features
from .model import OUTPUTS, FactorModel, masked_prediction_loss, pair_loss


def audit_manifest(manifest):
    membership = dict(manifest.get("split_originals", {}))
    for item in manifest["pairs"]+manifest.get("insertion_cases", []):
        if item["split"] not in ("train", "validation", "test"):
            raise ValueError("Unknown data split")
        for sid in item["original_ids"]:
            if sid in membership and membership[sid] != item["split"]:
                raise ValueError(f"Original ID leakage: {sid}")
            membership[sid] = item["split"]
    return membership


def alignment_specs(pair, seconds=10., hop=5., max_windows=0):
    """Store only start times, not full waveforms, for large-corpus training."""
    durations = [sf.info(pair[side]["audio"]).duration for side in ("a", "b")]
    index = 0
    start = 0.
    while start < durations[0] and (not max_windows or index < max_windows):
        bstart = start/durations[0]*durations[1]
        if min(durations[0]-start, durations[1]-bstart) >= 800/24000:
            yield start, bstart
        if start+seconds >= durations[0]:
            break
        start += hop
        index += 1


def read_training_window(item, start, seconds=10.):
    # Prepared manifests use WAV; random access avoids retaining full songs.
    info = sf.info(item["audio"])
    if info.samplerate != 24000 or info.channels != 1:
        raise ValueError("Training manifests require prepared 24 kHz mono audio")
    begin = round(start*24000)
    data, _ = sf.read(item["audio"], start=begin, stop=begin+round(seconds*24000), dtype="float32")
    return Window(0, begin/24000, (begin+len(data))/24000,
                  np.pad(data, (0, round(seconds*24000)-len(data))), len(data))


def train(manifest_path, out, backbone="mert95", epochs=2, batch_size=4, lr=.001,
          prediction_weight=1., unfreeze_last=0, init_checkpoint=None, device="cpu", seed=42,
          max_pairs=0, max_windows_per_pair=0, smoke=False):
    if epochs < 1 or batch_size < 1 or lr <= 0 or prediction_weight < 0:
        raise ValueError("Require epochs, batch size, learning rate > 0 and prediction weight >= 0")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True, warn_only=True)
    raw_manifest = Path(manifest_path).read_bytes()
    manifest = json.loads(raw_manifest)
    memberships = audit_manifest(manifest)
    train_pairs = [p for p in manifest["pairs"] if p["split"] == "train"]
    val_pairs = [p for p in manifest["pairs"] if p["split"] == "validation"]
    if max_pairs:
        train_pairs, val_pairs = train_pairs[:max_pairs], val_pairs[:max_pairs]
    if not train_pairs or not val_pairs:
        raise ValueError("Nonempty train and validation pairs required")
    encoder = None
    if backbone == "mert95":
        from .backends import MertEncoder
        encoder = MertEncoder("95", device)
        encoder.unfreeze_last(unfreeze_last)
    elif backbone != "dsp":
        raise ValueError(backbone)
    elif unfreeze_last:
        raise ValueError("Unfreezing requires the MERT backbone")
    model = FactorModel(encoder.output_dim if encoder else INPUT_DIM).to(device)
    if init_checkpoint:
        initial = torch.load(init_checkpoint, map_location=device, weights_only=True)
        if initial["metadata"]["backbone"] != backbone:
            raise ValueError("Checkpoint backbone mismatch")
        model.load_state_dict(initial["state_dict"], strict=True)
        if encoder and "encoder_state_dict" in initial:
            encoder.model.load_state_dict(initial["encoder_state_dict"], strict=True)
    groups = [{"params": model.parameters(), "lr": lr}]
    if encoder and unfreeze_last:
        groups.append({"params": [p for p in encoder.parameters() if p.requires_grad], "lr": lr*.05})
    optimizer = torch.optim.AdamW(groups, foreach=False)
    cache, annotations, task_counts, trained_task_counts = OrderedDict(), {}, {}, {}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    signature = raw_manifest+backbone.encode()+Path("configs/model_versions.json").read_bytes()
    if init_checkpoint:
        digest = hashlib.sha256()
        with open(init_checkpoint, "rb") as stream:
            for block in iter(lambda: stream.read(1024*1024), b""):
                digest.update(block)
        signature += digest.digest()
    cache_signature = hashlib.sha256(signature).hexdigest()[:16]
    disk_cache = out.parent/("feature_cache_"+cache_signature)
    if not unfreeze_last:
        disk_cache.mkdir(exist_ok=True)
    accompaniment_sources = set()

    def prepare_item(item, chunk):
        key = (item["audio"], chunk.start, chunk.end)
        if key in cache:
            frames, valid, labels = cache[key]
            cache.move_to_end(key)
            return frames.to(device), valid.to(device), {k: v.to(device) for k, v in labels.items()}
        frames, valid_np, _ = frame_features(chunk.samples, chunk.valid_samples)
        if item["annotation"] not in annotations:
            annotations[item["annotation"]] = json.loads(Path(item["annotation"]).read_text())
        annotation = annotations[item["annotation"]]
        accompaniment_sources.add(annotation.get("accompaniment_source", "unavailable"))
        labels_np = frame_labels(annotation, chunk.start, len(frames), valid_np)
        valid = torch.tensor(valid_np, device=device)
        labels = {k: torch.tensor(v, device=device) for k, v in labels_np.items()}
        for task in (k for k in labels if k.endswith("_mask")):
            task_counts[task.removesuffix("_mask")] = task_counts.get(task.removesuffix("_mask"), 0)+int(labels[task].sum())
            if model.training:
                key_task = task.removesuffix("_mask")
                trained_task_counts[key_task] = trained_task_counts.get(key_task, 0)+int(labels[task].sum())
        stat = Path(item["audio"]).stat()
        feature_key = hashlib.sha256(repr((key, stat.st_size, stat.st_mtime_ns)).encode()).hexdigest()
        feature_path = disk_cache/(feature_key+".pt")
        if not unfreeze_last and feature_path.exists():
            x = torch.load(feature_path, map_location=device, weights_only=True)
        elif encoder:
            # Keep frozen encoders deterministic even while heads are training.
            encoder.eval() if not unfreeze_last else encoder.train(model.training)
            x = encoder(chunk.samples, chunk.valid_samples, len(frames))[0]
        else:
            x = torch.tensor(frames, device=device)
        if not unfreeze_last:
            if not feature_path.exists():
                torch.save(x.detach().cpu(), feature_path)
            cache[key] = (x.detach().cpu(), valid.cpu(), {k: v.cpu() for k, v in labels.items()})
            if len(cache) > 128:
                cache.popitem(last=False)
        return x, valid, labels

    # Window indices are fixed before epochs; validation never updates parameters.
    examples = {}
    for split, pairs in (("train", train_pairs), ("validation", val_pairs)):
        examples[split] = [(p, spec) for p in pairs for spec in alignment_specs(p, max_windows=max_windows_per_pair)]
        if not examples[split]:
            raise ValueError(f"No usable audio windows in {split}")
    history, best = [], float("inf")
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for epoch in range(epochs):
        row = {"epoch": epoch+1}
        for split in ("train", "validation"):
            model.train(split == "train")
            items = list(examples[split])
            if split == "train":
                random.shuffle(items)
            metrics = []
            for offset in range(0, len(items), batch_size):
                batch = items[offset:offset+batch_size]
                with torch.set_grad_enabled(split == "train"):
                    sides = []
                    for side in (0, 1):
                        items_side = []
                        for pair, spec in batch:
                            item = pair["a" if side == 0 else "b"]
                            items_side.append(prepare_item(item, read_training_window(item, spec[side])))
                        x = torch.stack([i[0] for i in items_side])
                        valid = torch.stack([i[1] for i in items_side])
                        labels = {k: torch.stack([i[2][k] for i in items_side]) for k in items_side[0][2]}
                        outputs = model(x, valid)
                        prediction = masked_prediction_loss(outputs, labels, valid) if prediction_weight else outputs[FACTORS[0]]["embedding"].sum()*0
                        sides.append((outputs, prediction))
                    targets = torch.tensor([[pair["targets"].get(f) or 0 for f in FACTORS] for pair, _ in batch], device=device).float()
                    masks = torch.tensor([[pair["targets"].get(f) is not None for f in FACTORS] for pair, _ in batch], device=device)
                    similarity = pair_loss(sides[0][0], sides[1][0], targets, masks)
                    prediction = (sides[0][1]+sides[1][1])/2
                    loss = prediction_weight*prediction+similarity
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite {split} loss at epoch {epoch+1}, batch {offset}")
                    if split == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], 5., error_if_nonfinite=True)
                        optimizer.step()
                    metrics.append([float(loss.detach()), float(prediction.detach()), float(similarity.detach())])
            means = np.mean(metrics, axis=0)
            row[split] = dict(zip(("loss", "prediction_loss", "pair_loss"), means.tolist()))
        history.append(row)
        print(json.dumps(row), flush=True)
        if row["validation"]["loss"] < best:
            best = row["validation"]["loss"]
            trained_tasks = sorted(k for k, count in trained_task_counts.items() if count > 0) if prediction_weight else []
            trained_factors = [factor for factor in FACTORS if
                               any(task in trained_tasks for task in OUTPUTS[factor]) or
                               any(pair["targets"].get(factor) is not None for pair, _ in examples["train"])]
            metadata = dict(backbone=backbone, unfreeze_last=unfreeze_last, prediction_weight=prediction_weight,
                            seed=seed, smoke=smoke, epochs_completed=epoch+1,
                            train_original_ids=sorted(s for s, value in memberships.items() if value == "train"),
                            validation_original_ids=sorted(s for s, value in memberships.items() if value == "validation"),
                            manifest_sha256=hashlib.sha256(raw_manifest).hexdigest(),
                            accompaniment_sources=sorted(accompaniment_sources), task_annotation_counts=task_counts,
                            trained_prediction_tasks=trained_tasks, trained_factors=trained_factors,
                            trainable_encoder_blocks=[10, 11] if unfreeze_last else [],
                            trainable_encoder_parameters=sum(p.numel() for p in encoder.parameters() if p.requires_grad) if encoder else 0,
                            torch_version=str(torch.__version__),
                            model_versions=json.loads(Path("configs/model_versions.json").read_text()) if encoder else {})
            checkpoint = dict(config=model.config, state_dict=model.state_dict(), metadata=metadata)
            if encoder and unfreeze_last:
                checkpoint["encoder_state_dict"] = encoder.model.state_dict()
            torch.save(checkpoint, out)
    changed = any(not torch.equal(value, before[key]) for key, value in model.state_dict().items())
    summary = dict(history=history, checkpoint=str(out), parameters_changed=changed,
                   train_examples=len(examples["train"]), validation_examples=len(examples["validation"]),
                   feature_cache=str(disk_cache) if not unfreeze_last else None,
                   note="Small-run optimization/checkpoint validation; not an estimate of real music performance.")
    save_json(out.with_suffix(".history.json"), summary)
    return summary
