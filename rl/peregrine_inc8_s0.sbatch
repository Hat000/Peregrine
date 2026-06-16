#!/bin/bash
#SBATCH --job-name=peregrine_inc8_s0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/network/fl3689/peregrine_inc8_s0.out
#
# inc8 ARCHITECTURE PIVOT -- STAGE S0 (P2 INC8-RL, 2026-06-16). 1 seed, arm A, default entropy 0.01.
# After 4 weight-tuning NO-GOs (low-pass exploit / band-pass abandons / fix-driven no-gradient = an
# EXPLORATION wall: PPO can't discover pointing->fix from a non-pointing start at any weight), we STOP
# tuning reward weights and ENGINEER the pointing: an active-perception LOOK-AT PRIMITIVE composed onto
# the CTBR action (rl/inc8_reward.lookat_correction + the env). S0 = YAW-ONLY (the cheap heading DoF):
# null the gate AZIMUTH so the gate is held in the wide 90deg HFoV; elevation left to the racing line.
#   omega_corr = K*(zhat_cam x uhat) -> FRD via the frames-canonical R_body_from_camera (baked, PINNED to
#   numpy frames by tests/test_inc8_lookat.py) -> FLU action via diag(1,-1,-1) (== diffaero _FLIP). The
#   SIGN is analytically resolved (gate-right -> dyaw_flu<0 -> turn right); inc8_band_az_abs_deg confirms
#   it on-GPU in the first ~hundred steps. Range-gated to [lookat_r_lo, lookat_r_hi].
#
# WHAT'S DIFFERENT vs the band-pass/fix-driven runs: the failed fix-driven weights are DROPPED
# (rw_conf_shape back to default 0.05, rw_fix_bonus 0) -- S0 tests the PRIMITIVE on top of the EXISTING
# spine (R1' progress + GT estimerr + annealed conf + arm-A band-pass R5'). centering reward stays OFF
# (rw_centering=0); it is the S1 lever (the NEXT run, if S0 clears).
#
# READ THE TRACE IN THIS ORDER (do NOT skip step 1 -- it is the cheap sign gate):
#   1. SIGN / EFFICACY  -- inc8_band_az_abs_deg must be LOW and FALLING in the band (a few deg: the
#      primitive holds the gate near azimuth 0). If it is HIGH and RISING (toward ~90), the FLU/FRD sign
#      is inverted -> KILL, resubmit with +env.lookat_g_yaw=-3.0, no other change. (Analytically it should
#      be +3.0; this is the belt-and-suspenders confirm. Visible by the first logged points ~step 100.)
#   2. PRIMITIVE WORKS  -- with azimuth nulled: inc8_fix_rate climbs OFF the ~0.001 floor + inc8_lockband
#      _pointing[12,28] rises + inc8_estim_err_inplane_m FALLS (forced pointing -> fixes -> KF accuracy).
#   3. LINE SURVIVES    -- success_rate / finish_time_s / mean_speed not wrecked (yaw is near-free; a big
#      drop means the yaw primitive is too disruptive -> lower g_yaw or narrow the band).
#
# ADJUDICATION (commander): GO (-> S1: add +env.rw_centering) if fix_rate climbs AND the line survives.
# INFORMATIVE PARTIAL: band_az_abs FALLS but fix_rate does NOT climb => the gate is in the HORIZONTAL FoV
# but below the VERTICAL FoV (the known 20deg-mount elevation gap) => yaw alone is insufficient => go to
# S2 (add a fixed pitch component, +env.lookat_g_pitch). This is a real finding, not a failure.
#
# ADROIT AUP (accounts get SUSPENDED): compute on SLURM only (this batch); compute nodes have NO internet
# -- git pull / conda env on the LOGIN or vis node BEFORE submitting; output -> /scratch; accurate --mem;
# 1 GPU (2048-env PPO saturates it). Run `checkquota` routinely. See reference_adroit_princeton.md.
#
# RUNTAG passthrough (no-clobber): submit with --export=ALL,RUNTAG=<tag> (e.g. RUNTAG=s0_yaw3). EXTRA is
# appended to BOTH invocations for a submit-time override without editing the file.
RUNTAG=${RUNTAG:-}
RUN_SUFFIX="${RUNTAG:+_$RUNTAG}"
GPUUTIL=/scratch/network/fl3689/peregrine_inc8_s0${RUN_SUFFIX}_gpu_util.log
exec > /scratch/network/fl3689/peregrine_inc8_s0${RUN_SUFFIX}.out 2>&1
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
RUNNAME=inc8_s0_yawprim_seed${SEED}${RUN_SUFFIX}
NENVS=${NENVS:-2048}
NUPD=${NUPD:-4000}
TRAIN=/scratch/network/fl3689/peregrine_repo/rl/peregrine_train_inc8.py
TBTRACE=/scratch/network/fl3689/peregrine_repo/rl/inc8_tb_trace.py
# inc7-class measured-plant DR (map-ON) + contact-true geometry + BSR3 + arm A + the band-pass perception
# window, PLUS the S0 look-at primitive (yaw-only). conf_shape/fix_bonus/centering = DEFAULTS (the failed
# fix-driven weights are NOT set here).
COMMON=(
  env=racing env.name=peregrine_racing_inc8 +env.inc8=true +env.r5_arm=A
  +env.standing_start_frac=0.3
  +env.spin_rate_abort=10.0 +env.spin_time_abort=3.0
  +env.body_radius_lo=0.28 +env.body_radius_hi=0.38 +env.frame_depth_m=0.30
  +env.rw_perc_r_lo=12 +env.rw_perc_r_hi=28 +env.rw_perc_w_lo=1.5 +env.rw_perc_w_hi=1.5 +env.rw_perc_w0=0.05
  +env.lookat_g_yaw=3.0 +env.lookat_g_pitch=0.0 +env.lookat_r_lo=8.0 +env.lookat_r_hi=30.0
  dynamics=quad dynamics.name=peregrine_plant dynamics.g=9.80665
  +dynamics.dr=true +dynamics.dr_aero=true +dynamics.dr_mixer=true
  +dynamics.dr_force_bias=true +dynamics.dr_latency_min_steps=1 +dynamics.dr_latency_max_steps=3
  dynamics.controller.max_normed_thrust=3.765
  algo=ppo headless=True device=0
)

# ---- GPU-util sampler (background) ----
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 20 > ${GPUUTIL} 2>&1 &
SMI_PID=$!

# ---- STAGE A: FAST PRECHECK (3 updates) -- catch a wiring/NaN/dim bug (incl. the new look-at compose
# + the band_az/centering metrics) BEFORE the pilot. ----
echo "=== inc8 S0 PRECHECK: construct + 3 updates (obs_dim 20, critic 36, look-at compose, no NaN) ==="
python ${TRAIN} "${COMMON[@]}" \
  n_envs=512 n_updates=3 save_freq=1000 log_freq=1 seed=${SEED} runname=inc8_s0_precheck${RUN_SUFFIX} ${EXTRA:-}
echo "PRECHECK_RC=$?"

# ---- STAGE B: the S0 PILOT (1 seed, arm A, 4000 updates; 2048 envs) ----
echo "=== inc8 S0: yaw-only look-at primitive, ${NUPD} updates, ${NENVS} envs, seed ${SEED} ==="
python ${TRAIN} "${COMMON[@]}" \
  n_envs=${NENVS} n_updates=${NUPD} save_freq=50 log_freq=10 \
  seed=${SEED} runname=${RUNNAME} ${EXTRA:-}
echo "S0_RC=$?"

kill ${SMI_PID} 2>/dev/null
echo "=== GPU utilization summary ==="
awk -F, 'NR>1 && $2 ~ /%/ {gsub(/ %/,"",$2); s+=$2; n++; if($2+0>mx)mx=$2} END{if(n)printf "mean_util=%.1f%%  max_util=%.0f%%  samples=%d\n", s/n, mx, n}' \
  ${GPUUTIL}

echo "=== inc8 S0 SIGNAL TRACE (READ band_az_abs_deg FIRST, then fix_rate/lockband, then estim_err) ==="
TRAINOUT=/scratch/network/fl3689/diffaero/outputs/train
RUNDECOR=$(ls -dt ${TRAINOUT}/*/*/*${RUNNAME}__* 2>/dev/null | head -1)
if [ -n "$RUNDECOR" ]; then RUN=$(dirname "$RUNDECOR"); else RUN=$(ls -dt ${TRAINOUT}/*/*/ 2>/dev/null | head -1); RUN=${RUN%/}; fi
echo "RUN=$RUN"
python ${TBTRACE} "$RUN" --stride-updates 100
echo "INC8_S0_DONE"
echo "=== checkpoint sidecar (obs_dim 20, inc8=true, r5_arm=A) ==="
find "$RUN" -name actor.json -exec cat {} \; 2>/dev/null | head -3
