#!/bin/bash
#SBATCH --job-name=peregrine-yolo-train
#SBATCH -N 1 -n 8
#SBATCH --gres=gpu:3g.20gb:1
#SBATCH --time=01:30:00
#SBATCH --output=/home/fl3689/logs/peregrine-%j.out
# The REAL detector run: YOLO11s-pose on a mixed L1/L2/L3 curriculum, long enough for the
# keypoint head to converge. One idle MIG A100 slice, time-capped. [Peregrine]
set -e
module load anaconda3/2024.2
cd /scratch/network/fl3689/peregrine
echo "[job] node $(hostname)"; nvidia-smi -L

# 1. Mixed-difficulty dataset (geometry maxed; appearance clean->chaos sampled per image).
if [ ! -f data/mix/data.yaml ]; then
  echo "[job] generating mixed dataset (3000 train / 400 val)..."
  conda run -n yolo python scripts/gen_synthetic_dataset.py data/mix --mix --n-train 3000 --n-val 400
fi
echo "[job] label files:"; find data/mix -name '*.txt' | wc -l

# 2. Train YOLO11s-pose. patience early-stops once val stops improving.
echo "[job] training yolo11s-pose (mixed curriculum, up to 100 epochs)..."
conda run -n yolo yolo pose train \
  model=/scratch/network/fl3689/peregrine/yolo11s-pose.pt \
  data=data/mix/data.yaml epochs=100 patience=20 imgsz=640 batch=32 workers=8 device=0 plots=False \
  project=/scratch/network/fl3689/peregrine/runs name=curriculum_11s exist_ok=True

echo "[job] weights:"; ls -la runs/curriculum_11s/weights/
echo "JOB_DONE_OK"
