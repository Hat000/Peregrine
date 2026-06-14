#!/bin/bash
#SBATCH --job-name=peregrine_inc8_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/network/fl3689/peregrine_inc8_smoke.out
#
# inc8 GPU SMOKE (P2 INC8-RL Worker #3, Deliverable 2). 1 seed, arm A, short budget. Acceptance:
#   * no NaN (GuardedPPO + the obs finite-guard); the run completes.
#   * GPU SATURATED -- the 2048-env PPO must keep utilization.gpu high (the gpu_util.log sampler
#     below records it; Adroit kills zero-GPU-util jobs at 2h, so a saturated smoke also proves the
#     env is not accidentally CPU-bound on the numpy boundary).
#   * reward curve RISING + the camera-pointing metric IMPROVING: grep inc8_pointing_rate /
#     inc8_terminal_pointing / inc8_fix_rate / total_reward from the log -- pointing emerges under A.
# This is a LEARN-AT-ALL + POINTING-EMERGES smoke, NOT the ladder ({L0,L1,L2}x{A>=5,B3,C1} = next worker).
#
# ADROIT AUP (Princeton RC SUSPENDS accounts): compute on SLURM only (this batch); compute nodes have
# NO internet -- pre-download git pull / conda env on the LOGIN or vis node BEFORE submitting; output
# -> /scratch; accurate --mem; 1 GPU for GPU code; the 2048-env PPO saturates it (verified by the
# sampler). Run `checkquota` routinely. See reference_adroit_princeton.md.
set -x
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
source /etc/profile.d/modules.sh
module load anaconda3/2024.10
eval "$(conda shell.bash hook)"
conda activate diffaero
ln -sfn /scratch/network/fl3689/diffaero_repo /scratch/network/fl3689/diffaero
export PYTHONPATH=/scratch/network/fl3689:/scratch/network/fl3689/peregrine_repo/src:/scratch/network/fl3689/peregrine_repo/rl:$PYTHONPATH
cd /scratch/network/fl3689/diffaero

SEED=${SEED:-0}
NENVS=${NENVS:-2048}
NUPD=${NUPD:-300}
TRAIN=/scratch/network/fl3689/peregrine_repo/rl/peregrine_train_inc8.py
# inc7-class measured-plant DR (map-ON) + contact-true geometry (doctrine band) + BSR3 + arm A.
COMMON=(
  env=racing env.name=peregrine_racing_inc8 +env.inc8=true +env.r5_arm=A
  +env.standing_start_frac=0.3
  +env.spin_rate_abort=10.0 +env.spin_time_abort=3.0
  +env.body_radius_lo=0.28 +env.body_radius_hi=0.38 +env.frame_depth_m=0.30
  dynamics=quad dynamics.name=peregrine_plant dynamics.g=9.80665
  +dynamics.dr=true +dynamics.dr_aero=true +dynamics.dr_mixer=true
  +dynamics.dr_force_bias=true +dynamics.dr_latency_min_steps=1 +dynamics.dr_latency_max_steps=3
  dynamics.controller.max_normed_thrust=3.765
  algo=ppo headless=True device=0
)

# ---- GPU-util sampler (background): records utilization.gpu + mem every 20 s for the saturation check
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 20 \
  > /scratch/network/fl3689/peregrine_inc8_gpu_util.log 2>&1 &
SMI_PID=$!

# ---- STAGE A: FAST PRECHECK (3 updates) -- catch a wiring/NaN/dim bug before burning the smoke ----
echo "=== inc8 PRECHECK: construct + 3 updates (obs_dim 20, critic 36, no NaN) ==="
python ${TRAIN} "${COMMON[@]}" \
  n_envs=512 n_updates=3 save_freq=1000 log_freq=1 seed=${SEED} runname=inc8_precheck
echo "PRECHECK_RC=$?"

# ---- STAGE B: the SMOKE (1 seed, arm A, short budget; 2048 envs to saturate the GPU) ----
echo "=== inc8 SMOKE: arm A, ${NUPD} updates, ${NENVS} envs, seed ${SEED} ==="
python ${TRAIN} "${COMMON[@]}" \
  n_envs=${NENVS} n_updates=${NUPD} save_freq=50 log_freq=10 \
  seed=${SEED} runname=inc8_smoke_A_seed${SEED}
echo "SMOKE_RC=$?"

kill ${SMI_PID} 2>/dev/null
echo "=== GPU utilization summary (must be HIGH -- saturation check) ==="
awk -F, 'NR>1 && $2 ~ /%/ {gsub(/ %/,"",$2); s+=$2; n++; if($2+0>mx)mx=$2} END{if(n)printf "mean_util=%.1f%%  max_util=%.0f%%  samples=%d\n", s/n, mx, n}' \
  /scratch/network/fl3689/peregrine_inc8_gpu_util.log

echo "=== inc8 SIGNAL TRACE (pointing emerges + reward rising): pull from the run log ==="
RUN=$(ls -dt /scratch/network/fl3689/diffaero/outputs/train/*/inc8_smoke_A_seed${SEED}* 2>/dev/null | head -1)
echo "RUN=$RUN"
# the env logs these into loss_components every log_freq updates (stdout + tensorboard):
grep -hoE "inc8_(pointing_rate|terminal_pointing|fix_rate|c_inplane)[^ ]*|total_reward[^ ]*" \
  /scratch/network/fl3689/peregrine_inc8_smoke.out | tail -60
echo "INC8_SMOKE_DONE"
echo "=== checkpoint sidecar (S7: obs_dim must be 20, inc8=true, r5_arm=A) ==="
find "$RUN" -name actor.json -exec cat {} \; 2>/dev/null | head -3
