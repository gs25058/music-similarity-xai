import numpy as np
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

from .features import FACTORS, INPUT_DIM, PITCH_BINS, frame_features

OUTPUTS = {
    "pitch_intervals": {"melody_pitch": PITCH_BINS, "melody_onset": 1},
    "rhythm": {"beat": 1, "onset": 1},
    "harmony": {"chroma": 12, "chord_root": 13, "chord_quality": 5},
    "accompaniment_bass": {"bass_pitch": PITCH_BINS, "bass_onset": 1},
}


class TemporalHead(nn.Module):
    def __init__(self, input_dim, outputs, hidden=64, embedding_dim=32):
        super().__init__()
        self.temporal = nn.Sequential(nn.Conv1d(input_dim, hidden, 5, padding=2), nn.GELU(),
                                      nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU())
        self.predictors = nn.ModuleDict({name: nn.Linear(hidden, size) for name, size in outputs.items()})
        self.projection = nn.Linear(hidden*2, embedding_dim)

    def forward(self, frames, valid):
        z = self.temporal((frames * valid.unsqueeze(-1)).transpose(1, 2)).transpose(1, 2)
        mask = valid.unsqueeze(-1).float()
        mean = (z*mask).sum(1) / mask.sum(1).clamp_min(1)
        # Temporal changes as well as mean; retain order information in the head.
        differences = F.pad(z[:, 1:]-z[:, :-1], (0, 0, 1, 0))
        changes = (differences.abs()*mask).sum(1)/mask.sum(1).clamp_min(1)
        embedding = F.normalize(self.projection(torch.cat([mean, changes], -1)), dim=-1)
        return dict(embedding=embedding, predictions={name: layer(z) for name, layer in self.predictors.items()})


class FactorModel(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden=64, embedding_dim=32):
        super().__init__()
        self.config = dict(input_dim=input_dim, hidden=hidden, embedding_dim=embedding_dim)
        self.heads = nn.ModuleDict({factor: TemporalHead(input_dim, outputs, hidden, embedding_dim)
                                    for factor, outputs in OUTPUTS.items()})

    def forward(self, frames, valid):
        return {factor: head(frames, valid) for factor, head in self.heads.items()}


def masked_prediction_loss(outputs, labels, valid):
    """Every task has its own annotation mask, intersected with audio validity."""
    losses = []
    for factor, out in outputs.items():
        for task, logits in out["predictions"].items():
            target = labels[task].to(logits.device)
            mask = labels[task + "_mask"].to(logits.device).bool() & valid.bool()
            if not mask.any():
                continue
            if task.endswith("pitch") or task in ("chord_root", "chord_quality"):
                loss = F.cross_entropy(logits[mask], target[mask].long())
            else:
                if target.ndim < logits.ndim:
                    target = target.unsqueeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits[mask], target[mask].float())
            losses.append(loss)
    # Return differentiable zero when no annotations are present.
    zero = next(iter(outputs.values()))["embedding"].sum() * 0
    return torch.stack(losses).mean() if losses else zero


def pair_loss(a, b, targets, masks):
    terms = []
    for i, factor in enumerate(FACTORS):
        if masks[:, i].any():
            cosine = (a[factor]["embedding"] * b[factor]["embedding"]).sum(-1)
            # Positive -> cosine 1; known dissimilar -> cosine <= .2.
            y = targets[:, i]
            loss = y*(1-cosine).square() + (1-y)*F.relu(cosine-.2).square()
            terms.append(loss[masks[:, i]].mean())
    zero = a[FACTORS[0]]["embedding"].sum()*0
    return torch.stack(terms).mean() if terms else zero


class LearnedBackend:
    name = "ours"
    supported = FACTORS

    def __init__(self, checkpoint, device="cpu"):
        self.device = device
        ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
        self.meta = ckpt["metadata"]
        self.supported = tuple(self.meta.get("trained_factors", FACTORS))
        self.model = FactorModel(**ckpt["config"]).to(device)
        self.model.load_state_dict(ckpt["state_dict"], strict=True)
        self.model.eval()
        self.encoder = None
        if self.meta["backbone"] == "mert95":
            expected = self.meta.get("model_versions", {}).get("mert95")
            current = json.loads(Path("configs/model_versions.json").read_text())["mert95"]
            if expected is not None and expected != current:
                raise ValueError("Checkpoint MERT revision differs from current model manifest")
            from .backends import MertEncoder
            self.encoder = MertEncoder("95", device)
            if "encoder_state_dict" in ckpt:
                self.encoder.model.load_state_dict(ckpt["encoder_state_dict"], strict=True)
            self.encoder.eval()
        self.checkpoint = str(checkpoint)

    @torch.no_grad()
    def encode(self, window):
        frames, valid, _ = frame_features(window.samples, window.valid_samples)
        if np.max(np.abs(window.samples)) < 1e-6 or window.valid_samples < 800:
            return {}, {"status": "unavailable", "reason": "silence_or_too_short"}
        x = (self.encoder(window.samples, window.valid_samples, len(frames)) if self.encoder else
             torch.tensor(frames[None], device=self.device))
        mask = torch.tensor(valid[None], device=self.device)
        outputs = self.model(x, mask)
        emb = {factor: out["embedding"][0].cpu().numpy() for factor, out in outputs.items() if factor in self.supported}
        info = dict(status="model_prediction_not_verified", source=self.meta["backbone"],
                    training_accompaniment_sources=self.meta.get("accompaniment_sources", []),
                    predictions={}, missing=["verified_main_melody", "verified_bass", "verified_chords"])
        for factor, out in outputs.items():
            if factor not in self.supported:
                info["predictions"][factor] = dict(status="unavailable", reason="no_training_supervision_for_factor")
                continue
            tasks = {}
            for task, value in out["predictions"].items():
                if task not in self.meta.get("trained_prediction_tasks", []):
                    tasks[task] = dict(status="unavailable", reason="no_prediction_supervision")
                    continue
                v = value[0][mask[0]]
                if task.endswith("pitch"):
                    pitch = v.argmax(-1).cpu().numpy()
                    voiced = pitch[pitch != 84] + 24
                    tasks[task] = dict(midi_median=float(np.median(voiced)) if len(voiced) else None,
                                       unvoiced_fraction=float(np.mean(pitch == 84)))
                elif task == "chroma":
                    tasks[task] = torch.sigmoid(v).mean(0).cpu().tolist()
                elif task in ("chord_root", "chord_quality"):
                    tasks[task] = dict(class_histogram=torch.bincount(v.argmax(-1), minlength=v.shape[-1]).cpu().tolist())
                else:
                    tasks[task] = dict(mean_probability=float(torch.sigmoid(v).mean()))
            info["predictions"][factor] = tasks
        return emb, info

    def metadata(self):
        return dict(backend=self.name, checkpoint=self.checkpoint, supported=list(self.supported),
                    status="trained_smoke_model" if self.meta.get("smoke") else "trained_model",
                    training=self.meta)
