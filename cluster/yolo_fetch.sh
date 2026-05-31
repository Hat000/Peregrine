#!/bin/bash
# Pre-fetch pretrained YOLO11-pose weights on the LOGIN node (compute nodes have no internet).
# Caches the .pt files into the workspace so SLURM jobs load them locally. [Peregrine]
set -e
module load anaconda3/2024.2
cd /scratch/network/fl3689/peregrine
for m in yolo11n-pose.pt yolo11s-pose.pt; do
  conda run -n yolo python -c "from ultralytics import YOLO; YOLO('$m')"
done
ls -la yolo11n-pose.pt yolo11s-pose.pt
echo "FETCH_DONE_OK"
