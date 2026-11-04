import json
import subprocess


class SongFormerStructure:
    def __init__(self, python, device="cpu"):
        self.python, self.device = python, device

    def __call__(self, audio):
        try:
            proc = subprocess.run([self.python, "scripts/songformer_worker.py", "--audio", audio.path,
                                   "--device", self.device], text=True, capture_output=True, check=True, timeout=1800)
            result = json.loads(proc.stdout)
            segments = result["segments"]
            previous_end = 0.
            for s in segments:
                if set(s) != {"start", "end", "label"} or not isinstance(s["label"], str):
                    raise ValueError("Invalid SongFormer segment schema")
                if not (0 <= s["start"] < s["end"] <= audio.duration+.25) or s["start"] < previous_end-.05:
                    raise ValueError("Invalid or overlapping SongFormer interval")
                s["end"] = min(s["end"], audio.duration)
                previous_end = s["end"]
            return dict(backend="songformer", status="executed", segments=segments, metadata=result.get("metadata"))
        except (OSError, subprocess.SubprocessError, ValueError, KeyError) as exc:
            return dict(backend="songformer", status="unavailable", segments=[],
                        reason=(getattr(exc, "stderr", None) or str(exc))[-5000:])
