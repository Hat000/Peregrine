#!/bin/bash
#SBATCH --job-name=peregrine-vq2-pose
#SBATCH -N 1 -n 8
#SBATCH --gres=gpu:3g.20gb:1
#SBATCH --time=03:00:00
#SBATCH --output=/home/fl3689/logs/peregrine-vq2-%j.out
# VQ2 photoreal 8-keypoint YOLO-pose detector. On-the-fly augmentation: Ultralytics built-in
# geometric (pose-aware, flip_idx) + an AlbumentationsX photometric/sensor hook (sim-to-real).
# The CLEAN photoreal base set (rendered on ShadowPC) must be transferred here first (cluster/push_dir.py),
# e.g. to data/vq2_sets/{allhue,vq1red}/. [Peregrine]
set -e
module load anaconda3/2024.2
cd /scratch/network/fl3689/peregrine
echo "[job] node $(hostname)"; nvidia-smi -L

SETS=data/vq2_sets          # <- transferred photoreal base sets live here (allhue/ + vq1red/)

# 1. Merge the sets + carve a 90/10 val split -> data.yaml (8-kpt kpt_shape + flip_idx). Idempotent-ish.
if [ ! -f $SETS/data.yaml ]; then
  echo "[job] building merged data.yaml + train/val split..."
  conda run -n yolo python cluster/vq2_pose_dataset.py --root $SETS --val-frac 0.1 --seed 0
fi
echo "[job] train/val:"; wc -l < $SETS/train.txt; wc -l < $SETS/val.txt

# 2. AlbumentationsX provides the photometric hook (maintained fork; AGPL -- fine for research).
#    ultralytics>=8.3 is required for the `augmentations=` train kwarg.
conda run -n yolo python -m pip install --quiet "albumentationsx>=2.0" "ultralytics>=8.3"

# 3. Train yolo11s-pose with the VQ2 augmentation strategy (Python API -- CLI cannot pass custom aug).
echo "[job] training vq2 8-keypoint yolo11s-pose..."
conda run -n yolo python cluster/vq2_pose_train.py \
  --data $SETS/data.yaml --model /scratch/network/fl3689/peregrine/yolo11s-pose.pt \
  --epochs 100 --patience 20 --imgsz 640 --batch 32 --device 0 \
  --project /scratch/network/fl3689/peregrine/runs --name vq2_pose_8kp

echo "[job] weights:"; ls -la runs/vq2_pose_8kp/weights/ 2>/dev/null || true
echo "JOB_DONE_OK"
