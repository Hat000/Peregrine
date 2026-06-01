#!/bin/bash
# Create the YOLO-pose training conda env on Adroit. Run on the LOGIN node (it has internet;
# compute nodes do not). Pip cache is redirected to scratch so it doesn't fill the tight /home
# quota. Idempotent-ish: re-running recreates the env. [Peregrine / detector training]
set -e
module load anaconda3/2024.2
export PIP_CACHE_DIR=/scratch/network/fl3689/.pipcache
ENV=yolo

echo "[setup] creating conda env '$ENV' (python 3.11) ..."
conda create -y -n "$ENV" python=3.11

echo "[setup] installing torch + torchvision (CUDA 12.1 wheels) ..."
conda run -n "$ENV" pip install --no-input torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "[setup] installing ultralytics + albumentations + headless cv/numpy/scipy ..."
conda run -n "$ENV" pip install --no-input ultralytics albumentations opencv-python-headless numpy scipy

echo "[setup] installed versions:"
conda run -n "$ENV" python -c "import torch, ultralytics, albumentations, cv2, numpy, scipy; print('torch', torch.__version__, '| cuda_build', torch.version.cuda, '| ultralytics', ultralytics.__version__, '| cv2', cv2.__version__)"

echo "SETUP_DONE_OK"
