#!/bin/bash
#SBATCH --job-name=peregrine_inc8_repilot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/network/fl3689/peregrine_inc8_repilot.out
#
# inc8 FIX-DRIVEN REWARD RE-PILOT v3 (P2 INC8-RL, 2026-06-15). 1 seed, arm A, default entropy 0.01.
# PRIOR (band-pass run 3274167, v2): the BAND-PASS w_term KILLED the terminal exploit (terminal_pointing
# 0.035->0.005) but the converged policy then ABANDONED pointing (lockband[12,28]~=0, fix~=0) -- band
# pointing is achievable (step-200 lockband=0.011) but NOT WORTH the racing-line cost at R5'~5% of
# progress. SHAPE alone is insufficient; the reward must make ACTUAL FIXES worth the line.
#
# THE FIX (this run, fix-driven fallback; KEEPS the band-pass shape r_lo=12/r_hi=28/w0=0.05):
#   VERIFIED first (handoff check_c_inplane.py): the emulator KF c_inplane is HONEST -- ~0.05 on a cold
#   no-fix coast, transient ~0.24 after one fix then decays -- so NO #74 covariance-floor needed and the
#   conf_shape anneal is already always-on (warmup=0/ramp=1). The conf_shape reward was just too WEAK.
#   1. rw_conf_shape 0.05 -> 0.3 : strengthen the UN-GAMEABLE (c_inplane - age_norm) staleness/confidence
#      incentive (age_norm resets ONLY on a real accepted fix => earnable ONLY by band pointing).
#   2. rw_fix_bonus 0.0 -> 1.5 : NEW direct per-ACCEPTED-fix bonus, progress-gated (anti-loiter). The most
#      direct un-gameable pointing->fix incentive; cannot be earned by terminal pointing or a coasting KF.
#   NOT touched: rw_estimerr (coast-satisfied, don't lean on it); rw_perc (the weak proxy we bracketed).
#
# THIS IS A PILOT: judge the TREND, not a final number.
#   * GO: inc8_lockband_pointing[12,28] CLIMBING + inc8_fix_rate CLIMBING + entropy mid-range = the
#     CONVERGED policy points the FIXABLE band and gets fixes => bake the band-pass + fix-driven weights
#     into COMMON -> L0. inc8_terminal_pointing@<=5 m must NOT dominate lockband_pointing.
#   * STOP CONDITION (iteration 4): if lockband~=0 / fix~=0 AGAIN, the problem is reward ARCHITECTURE
#     (direct terminal-sigma_p0 centering reward, or a pointing curriculum / demo-init), NOT more weight
#     tuning -- STOP and flag for the commander; do NOT sweep weights further.
#   * DO NOT gate on pooled inc8_fix_rate >= 0.50: the terminal 6 m is a GEOMETRIC dead zone, so pooled
#     >=0.50 was never achievable; the target is dense centring/fixes in the 12-18 m band + coast.
#   * PASSTHROUGH: ${EXTRA:-} is appended to BOTH train invocations so an override can still be injected
#     at submit time WITHOUT editing this file:  sbatch --export=ALL,SEED=0,EXTRA="<key>=<value>" ...
#     For THIS band-pass run, submit with NO EXTRA (default entropy_weight=0.01).
#
# ADROIT AUP (Princeton RC SUSPENDS accounts): compute on SLURM only (this batch); compute nodes have
# NO internet -- pre-download git pull / conda env on the LOGIN or vis node BEFORE submitting; output
# -> /scratch; accurate --mem; 1 GPU for GPU code; the 2048-env PPO saturates it (verified by the
# sampler). Run `checkquota` routinely. See reference_adroit_princeton.md.
#
# RUNTAG passthrough: tag the .out + runname + gpu-util log + TB-resolution glob so a resubmit can NOT
# clobber a prior run's artifacts and the TB newest-glob stays unambiguous. Submit with
# `--export=ALL,RUNTAG=<tag>` (e.g. RUNTAG=ent0.01). Default empty => byte-identical to the untagged
# workflow. SBATCH --output can't expand env vars, so we self-redirect stdout/stderr here based on RUNTAG
# (when RUNTAG is empty the target == the #SBATCH --output default, so nothing changes).
RUNTAG=${RUNTAG:-}
RUN_SUFFIX="${RUNTAG:+_$RUNTAG}"
GPUUTIL=/scratch/network/fl3689/peregrine_inc8_repilot${RUN_SUFFIX}_gpu_util.log
exec > /scratch/network/fl3689/peregrine_inc8_repilot${RUN_SUFFIX}.out 2>&1
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
RUNNAME=inc8_repilot_A_seed${SEED}${RUN_SUFFIX}
NENVS=${NENVS:-2048}
# Re-pilot budget = 4000 updates (was 1000 in the smoke). Pilot #2 (6000 updates) plateaued early and
# its entropy went NEGATIVE at step ~2300; 4000 runs well past that so we can see (a) whether the
# realigned reward lets inc8_fix_rate climb and (b) whether exploration survived to discover the band.
# Empirical throughput (pilot #2: 6000 updates in ~59 min wall @ ~28% util) => 4000 updates ~40 min;
# --time=04:00:00 is ~5x margin (precheck + startup/export + queue/IO slack). Keep NENVS=2048 (real
# GPU-saturation reading), SEED=0, arm A, precheck.
NUPD=${NUPD:-4000}
TRAIN=/scratch/network/fl3689/peregrine_repo/rl/peregrine_train_inc8.py
TBTRACE=/scratch/network/fl3689/peregrine_repo/rl/inc8_tb_trace.py
# inc7-class measured-plant DR (map-ON) + contact-true geometry (doctrine band) + BSR3 + arm A,
# PLUS the realigned perception-reward window (the conversion-fix under test).
COMMON=(
  env=racing env.name=peregrine_racing_inc8 +env.inc8=true +env.r5_arm=A
  +env.standing_start_frac=0.3
  +env.spin_rate_abort=10.0 +env.spin_time_abort=3.0
  +env.body_radius_lo=0.28 +env.body_radius_hi=0.38 +env.frame_depth_m=0.30
  +env.rw_perc_r_lo=12 +env.rw_perc_r_hi=28 +env.rw_perc_w_lo=1.5 +env.rw_perc_w_hi=1.5 +env.rw_perc_w0=0.05
  +env.rw_conf_shape=0.3 +env.rw_fix_bonus=1.5
  dynamics=quad dynamics.name=peregrine_plant dynamics.g=9.80665
  +dynamics.dr=true +dynamics.dr_aero=true +dynamics.dr_mixer=true
  +dynamics.dr_force_bias=true +dynamics.dr_latency_min_steps=1 +dynamics.dr_latency_max_steps=3
  dynamics.controller.max_normed_thrust=3.765
  algo=ppo headless=True device=0
)

# ---- GPU-util sampler (background): records utilization.gpu + mem every 20 s for the saturation check
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 20 \
  > ${GPUUTIL} 2>&1 &
SMI_PID=$!

# ---- STAGE A: FAST PRECHECK (3 updates) -- catch a wiring/NaN/dim/EXTRA-typo bug before the pilot ----
echo "=== inc8 PRECHECK: construct + 3 updates (obs_dim 20, critic 36, no NaN; EXTRA=${EXTRA:-<none>}) ==="
python ${TRAIN} "${COMMON[@]}" \
  n_envs=512 n_updates=3 save_freq=1000 log_freq=1 seed=${SEED} runname=inc8_repilot_precheck${RUN_SUFFIX} ${EXTRA:-}
echo "PRECHECK_RC=$?"

# ---- STAGE B: the RE-PILOT (1 seed, arm A, 4000 updates; 2048 envs to saturate the GPU) ----
echo "=== inc8 RE-PILOT: arm A, ${NUPD} updates, ${NENVS} envs, seed ${SEED}, EXTRA=${EXTRA:-<none>} ==="
python ${TRAIN} "${COMMON[@]}" \
  n_envs=${NENVS} n_updates=${NUPD} save_freq=50 log_freq=10 \
  seed=${SEED} runname=${RUNNAME} ${EXTRA:-}
echo "REPILOT_RC=$?"

kill ${SMI_PID} 2>/dev/null
echo "=== GPU utilization summary (must be HIGH -- saturation check) ==="
awk -F, 'NR>1 && $2 ~ /%/ {gsub(/ %/,"",$2); s+=$2; n++; if($2+0>mx)mx=$2} END{if(n)printf "mean_util=%.1f%%  max_util=%.0f%%  samples=%d\n", s/n, mx, n}' \
  ${GPUUTIL}

echo "=== inc8 SIGNAL TRACE (fix_rate climbs + lockband pointing rises + reward rising): from TensorBoard ==="
# diffaero/hydra writes each run to outputs/train/<date>/<time>/ ; the runname appears only in the
# DECORATED TB logdir <date>/<time>/quad__<env>__<algo>__<model>__<runname>__<idx>/ (NOT a top-level
# dir). Resolve the run dir by the decorated logdir for THIS re-pilot; fall back to the newest run dir
# (the re-pilot follows the precheck, so newest == re-pilot). Either way the whole run-dir tree goes
# to the recursive TB/sidecar readers.
TRAINOUT=/scratch/network/fl3689/diffaero/outputs/train
RUNDECOR=$(ls -dt ${TRAINOUT}/*/*/*${RUNNAME}__* 2>/dev/null | head -1)
if [ -n "$RUNDECOR" ]; then
  RUN=$(dirname "$RUNDECOR")
else
  RUN=$(ls -dt ${TRAINOUT}/*/*/ 2>/dev/null | head -1); RUN=${RUN%/}
fi
echo "RUN=$RUN"
# inc8 signal metrics are TensorBoard-only (env_loss/inc8_*, env_loss/total_reward, agent_loss/*);
# they never reach stdout, so we parse the TB event file (event_accumulator is in the diffaero env --
# NO download needed on the compute node). Prints the TRAJECTORY every ~100 updates, not just endpoints.
python ${TBTRACE} "$RUN" --stride-updates 100
echo "INC8_REPILOT_DONE"
echo "=== checkpoint sidecar (S7: obs_dim must be 20, inc8=true, r5_arm=A) ==="
find "$RUN" -name actor.json -exec cat {} \; 2>/dev/null | head -3
