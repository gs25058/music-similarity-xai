"""Public encoders are lazy, local-only, and never silently replaced by DSP."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .features import FPS, dsp_encode


class DSPBackend:
    name = "dsp"
    supported = ("pitch_intervals", "rhythm", "harmony", "accompaniment_bass")
    status = "mixed_audio_proxy"

    def encode(self, window):
        return dsp_encode(window)

    def metadata(self):
        return dict(backend=self.name, status=self.status, supported=list(self.supported),
                    limitations=["no_source_separation", "no_verified_chords", "no_phrase_labels"])


class MertEncoder(nn.Module):
    """Official MERT loading API; 95M for ours, 330M exclusively for MERIT.

    Frames are returned without pooling. Actual unpadded audio is passed to MERT,
    so padding cannot enter normalization, self-attention, or mean pooling.
    """
    def __init__(self, size="95", device="cpu", model_dir=None):
        super().__init__()
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
        self.size, self.device_name = size, device
        self.model_dir = str(Path(model_dir or f"models/mert{size}").resolve())
        if not Path(self.model_dir, "config.json").exists():
            raise FileNotFoundError(f"Prepare official MERT first: scripts/prepare_models.py mert{size}")
        expected_version = json.loads(Path("configs/model_versions.json").read_text())[f"mert{size}"]
        actual_version = json.loads(Path(self.model_dir, "artifact_version.json").read_text())
        if any(actual_version.get(key) != expected_version[key] for key in ("repo_id", "revision")):
            raise ValueError("Local MERT artifact does not match the pinned official revision")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_dir, local_files_only=True)
        # Official 2023 checkpoint uses legacy weight_norm names. Recent torch
        # exposes the same tensors under parametrizations; translate names only.
        state = torch.load(Path(self.model_dir, "pytorch_model.bin"), map_location="cpu", weights_only=True)
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            prefix = "encoder.pos_conv_embed.conv."
            for old, new in (("weight_g", "parametrizations.weight.original0"),
                             ("weight_v", "parametrizations.weight.original1")):
                if prefix+old in state:
                    state[prefix+new] = state.pop(prefix+old)
        self.model, loading = AutoModel.from_pretrained(self.model_dir, trust_remote_code=True,
                                local_files_only=True, state_dict=state, output_loading_info=True)
        if loading["missing_keys"] or loading["unexpected_keys"] or loading.get("mismatched_keys"):
            raise ValueError(f"MERT checkpoint was not loaded exactly: {loading}")
        self.model = self.model.to(device).eval()
        # Supervised downstream fine-tuning does not use MERT's pretraining
        # span masking; time-aligned labels must see the unmasked waveform.
        self.model.config.apply_spec_augment = False
        expected = 768 if size == "95" else 1024
        if self.model.config.hidden_size != expected:
            raise ValueError(f"Expected official MERT-v1-{size}M ({expected} dims)")
        self.output_dim = expected
        for param in self.model.parameters():
            param.requires_grad_(False)

    def unfreeze_last(self, count):
        if count not in (0, 2):
            raise ValueError("Only frozen or last two blocks are supported")
        for p in self.model.parameters():
            p.requires_grad_(False)
        if count:
            for layer in self.model.encoder.layers[-count:]:
                for p in layer.parameters():
                    p.requires_grad_(True)

    def hidden_states(self, samples):
        x = self.processor(np.asarray(samples), sampling_rate=24000, return_tensors="pt")
        x = {key: value.to(self.device_name) for key, value in x.items()}
        return self.model(**x, output_hidden_states=True).hidden_states

    def forward(self, samples, valid_samples=None, total_frames=None):
        valid_samples = len(samples) if valid_samples is None else valid_samples
        if valid_samples < 800:
            raise ValueError("MERT needs at least 800 valid samples")
        grad = any(p.requires_grad for p in self.model.parameters())
        with torch.set_grad_enabled(grad and self.training):
            h = self.hidden_states(samples[:valid_samples])[-1]
            valid_frames = max(1, int(np.ceil(valid_samples / 24000 * FPS)))
            h = F.interpolate(h.transpose(1, 2), size=valid_frames, mode="linear",
                              align_corners=False).transpose(1, 2)
            if total_frames and valid_frames < total_frames:
                h = F.pad(h, (0, 0, 0, total_frames-valid_frames))
            return h


# Adapted from AMAAI-Lab/MERIT README at aa0039b4..., MIT.
# Full attribution and original copyright: THIRD_PARTY_NOTICES.md.
class MeritProjectionHead(nn.Module):
    def __init__(self, in_dim=5120, hidden_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True),
                                 nn.Linear(hidden_dim, out_dim, bias=False))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class MeritBackend:
    name, status = "merit", "public_pretrained"
    supported = ("melody", "rhythm", "timbre")
    extract_layers = (3, 4, 5, 6, 23)

    def __init__(self, device="cpu"):
        self.encoder = MertEncoder("330", device)
        spec = json.loads(Path("configs/model_versions.json").read_text())["merit_heads"]
        local = json.loads(Path("models/merit_heads/artifact_version.json").read_text())
        if any(local.get(key) != spec[key] for key in ("repo_id", "revision")):
            raise ValueError("MERIT heads do not match the pinned revision")
        self.heads = {}
        for factor, folder in zip(self.supported, ("head_mel", "head_rhy", "head_tim")):
            path = Path("models/merit_heads", folder, "best_head.pt")
            ckpt = torch.load(path, map_location=device, weights_only=True)
            if ckpt["in_dim"] != 5120:
                raise ValueError("MERIT public heads require 330M, 5120 input dimensions")
            head = MeritProjectionHead(ckpt["in_dim"], ckpt["hidden_dim"], ckpt["out_dim"])
            head.load_state_dict(ckpt["state_dict"], strict=True)
            self.heads[factor] = head.to(device).eval()

    @torch.no_grad()
    def encode(self, window):
        if window.valid_samples < 800 or np.max(np.abs(window.samples)) < 1e-6:
            return {}, dict(status="unavailable", reason="silence_or_too_short")
        hs = self.encoder.hidden_states(window.samples[:window.valid_samples])
        backbone = torch.cat([hs[layer].mean(1) for layer in self.extract_layers], dim=-1)
        emb = {factor: head(backbone)[0].cpu().numpy() for factor, head in self.heads.items()}
        return emb, dict(status="embedding_only", missing=["music_information_predictions", "chords", "bass", "structure"],
                         semantics={"melody": "MERIT melody (not our interval head)",
                                    "rhythm": "MERIT rhythm", "timbre": "MERIT timbre (not accompaniment)"})

    def metadata(self):
        specs = json.loads(Path("configs/model_versions.json").read_text())
        return dict(backend=self.name, status=self.status, supported=list(self.supported),
                    backbone=specs["mert330"], heads=specs["merit_heads"],
                    extract_layers=list(self.extract_layers), padding_policy="encode_actual_audio_only")


def make_backend(name="dsp", checkpoint=None, device="cpu"):
    if name == "dsp":
        return DSPBackend()
    if name == "merit":
        return MeritBackend(device)
    if name == "ours":
        from .model import LearnedBackend
        if not checkpoint:
            raise ValueError("--checkpoint is required for ours")
        return LearnedBackend(checkpoint, device)
    raise ValueError(name)
