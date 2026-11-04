#!/usr/bin/env bash
set -euo pipefail
# Run from repository root. No external weights or network required.
PYTHON=${PYTHON:-python}
"$PYTHON" -m src synth --out data/synthetic
"$PYTHON" -m src compare data/synthetic/insertion/test/song_a.wav data/synthetic/insertion/test/song_b.wav --out results/smoke/dsp_report.json
"$PYTHON" -m src train --manifest data/synthetic/manifest.json --backbone dsp --epochs 2 --smoke --out checkpoints/dsp_smoke.pt
"$PYTHON" -m src compare data/synthetic/insertion/test/song_a.wav data/synthetic/insertion/test/song_b.wav --backend ours --checkpoint checkpoints/dsp_smoke.pt --out results/smoke/learned_report.json
"$PYTHON" -m src evaluate --manifest data/synthetic/manifest.json --backend ours --checkpoint checkpoints/dsp_smoke.pt --out results/smoke/evaluation.json
