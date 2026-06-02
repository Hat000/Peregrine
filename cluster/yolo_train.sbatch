#!/bin/bash
#SBATCH --job-name=peregrine-yolo-train
#SBATCH -N 1 -n 8
#SBATCH --gres=gpu:3g.20gb:1
#SBATCH --time=02:00:00
#SBATCH --output=/home/fl3689/logs/peregrine-%j.out
# The REAL detector run (v2): YOLO11s-pose on a mixed L1/L2/L3 curriculum, now with MULTI-GATE
# scenes + 2x volume to shrink the corner-error tail (one-corner-off from clipping/clutter +
# rare high-roll swaps; median was already ~2px). One idle MIG A100 slice, time-capped. [Peregrine]
set -e
module load anaconda3/2024.2
cd /scratch/network/fl3689/peregrine
echo "[job] node $(hostname)"; nvidia-smi -L

# 1. Mixed-difficulty, multi-gate dataset (geometry maxed; appearance clean->chaos per image).
if [ ! -f data/mix_v2/data.yaml ]; then
  echo "[job] generating mixed multi-gate dataset (6000 train / 800 val)..."
  conda run -n yolo python scripts/gen_synthetic_dataset.py data/mix_v2 --mix --n-train 6000 --n-val 800
fi
echo "[job] label files:"; find data/mix_v2 -name '*.txt' | wc -l
echo "[job] multi-gate frames (>1 row):"; find data/mix_v2/labels -name '*.txt' -exec sh -c 'wc -l < "$1"' _ {} \; | awk '$1>1' | wc -l

# 2. Train YOLO11s-pose. patience early-stops once val stops improving.
echo "[job] training yolo11s-pose (mixed multi-gate curriculum, up to 100 epochs)..."
conda run -n yolo yolo pose train \
  model=/scratch/network/fl3689/peregrine/yolo11s-pose.pt \
  data=data/mix_v2/data.yaml epochs=100 patience=20 imgsz=640 batch=32 workers=8 device=0 plots=False \
  project=/scratch/network/fl3689/peregrine/runs name=curriculum_11s_v2 exist_ok=True

echo "[job] weights:"; ls -la runs/curriculum_11s_v2/weights/
echo "JOB_DONE_OK"
