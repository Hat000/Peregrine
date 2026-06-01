#!/bin/bash
#SBATCH --job-name=peregrine-yolo-smoke
#SBATCH -N 1 -n 4
#SBATCH --gres=gpu:3g.20gb:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/fl3689/logs/peregrine-%j.out
# Smoke test: validate the whole pipeline (gen -> dataloader -> YOLO-pose train -> weights) on
# a SMALL dataset + few epochs, on one idle MIG A100 slice. Scale up once this is green.
set -e
module load anaconda3/2024.2
cd /scratch/network/fl3689/peregrine
echo "[job] node $(hostname)"; nvidia-smi -L

# 1. Generate a small L1/L2/L3 curriculum on the compute node (NOT the shared login node).
if [ ! -f data/l2/data.yaml ]; then
  echo "[job] generating curriculum dataset (L1/L2/L3, 300 train / 60 val each)..."
  conda run -n yolo python scripts/gen_synthetic_dataset.py data --all-levels --n-train 300 --n-val 60
fi
echo "[job] label files:"; find data -name '*.txt' | wc -l

# 2. Smoke-train YOLO11n-pose on L2 for a few epochs.
echo "[job] training yolo11n-pose (smoke: L2, 5 epochs)..."
conda run -n yolo yolo pose train \
  model=/scratch/network/fl3689/peregrine/yolo11n-pose.pt \
  data=data/l2/data.yaml epochs=5 imgsz=640 batch=16 workers=4 device=0 plots=False \
  project=/scratch/network/fl3689/peregrine/runs name=smoke_l2 exist_ok=True

echo "[job] weights produced:"; ls -la runs/smoke_l2/weights/ 2>/dev/null
echo "JOB_DONE_OK"
