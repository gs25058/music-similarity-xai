#!/usr/bin/env bash
set -euo pipefail
# Same backbone, data, seed and epoch budget for heads and ablation.
MANIFEST=${1:?Usage: bash scripts/train_full.sh data/prepared/manifest.json}
PYTHON=${PYTHON:-python}
DEVICE=${DEVICE:-cuda}
EPOCHS=${EPOCHS:-30}
"$PYTHON" -m src train --manifest "$MANIFEST" --backbone mert95 --device "$DEVICE" --epochs "$EPOCHS" --batch-size 8 --out checkpoints/full/heads.pt
"$PYTHON" -m src train --manifest "$MANIFEST" --backbone mert95 --device "$DEVICE" --epochs "$EPOCHS" --batch-size 8 --prediction-weight 0 --out checkpoints/full/ablation.pt
"$PYTHON" -m src train --manifest "$MANIFEST" --backbone mert95 --device "$DEVICE" --epochs 10 --batch-size 4 --lr .0001 --unfreeze-last 2 --init-checkpoint checkpoints/full/heads.pt --out checkpoints/full/finetuned.pt
"$PYTHON" -m src evaluate --manifest "$MANIFEST" --backend ours --checkpoint checkpoints/full/heads.pt --device "$DEVICE" --out results/full/heads_eval.json
"$PYTHON" -m src evaluate --manifest "$MANIFEST" --backend ours --checkpoint checkpoints/full/ablation.pt --device "$DEVICE" --out results/full/ablation_eval.json
"$PYTHON" -m src evaluate --manifest "$MANIFEST" --backend ours --checkpoint checkpoints/full/finetuned.pt --device "$DEVICE" --out results/full/finetuned_eval.json
