"""Audio-first, evidence-grounded music similarity MVP."""
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / ".cache" / "huggingface"))
os.environ.setdefault("HF_MODULES_CACHE", str(Path(__file__).resolve().parents[1] / ".cache" / "hf_modules"))
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
__version__ = "0.1.0"
