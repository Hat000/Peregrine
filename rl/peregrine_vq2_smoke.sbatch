#!/bin/bash
#SBATCH --job-name=peregrine_vq2_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#
# VQ2 RL controller SMOKE -- the GPU-spend-gate rung 1 (spec T3.5). Derived from
# peregrine_inc8_recenter.sbatch (the ONLY inc8 recipe that ever flew a lap). ONE seed, arm A, the
# multi_gate curriculum stage (full 6-gate vq2_like random course), short budget. GREEN =
# no-NaN + GPU-sat >=70% + sidecar dims correct + FLIGHTCHECK not-immediately-dead -> commander's GO to
# the portfolio.
#
# WHAT IS VQ2-IZED vs the recenter recipe:
#   * course_mode=random + track_difficulty=vq2_like: PER-ENV random VQ2 courses (upright gates,
#     23.7-38.5 m spacing, the >=+5 m HIGH-gate climb class) instead of the fixed VQ1 course. The inc8
#     env DROPS the global-line R1' arc-progress for random courses (Option B, commander-endorsed) --
#     through_centering (the only term with a flown-lap pedigree) carries course-completion.
#   * the estimator emulator runs PER-ENV gate geometry (each env's fix geometry from ITS OWN gates).
#   * the pose-age DR + terminal blackout are ON at the measured VQ2 regime (the ONE stochastic DR
#     channel per the determinism doctrine).
#
# 🚩 ASYMMETRIC CRITIC (algo=appo): the spec (T2.4) wants the privileged critic ON, and it IS wired on
# this branch (0392b64 applied: GuardedAPPO(AsymmetricPPO) + AGENT_ALIAS["appo"] in
# rl/peregrine_train_inc8.py; critic reads get_state()=37-dim, independent of the obs-20/23 actor arm).
# Default here stays algo=ppo for the byte-frozen control arm; pass --export=ALL,ALGO=appo to run the
# privileged critic (the recipe is otherwise unchanged).
#
# 🚩 PRECONDITION -- ADROIT BRIDGE: the adroit-connector (serve+x daemon) must be re-established on
# ShadowPC first (Fengyou: Duo + SSH passphrase + VPN). PRE-SYNC these edited files to
# /scratch/network/fl3689/peregrine_repo/rl/ (the Adroit repo is a FILE COPY, no git pull):
#   peregrine_course.py  peregrine_racing.py  peregrine_racing_inc8.py  inc8_estimator_emul.py
#   inc8_reward.py  vq2_curriculum.py  peregrine_vq2_smoke.sbatch  peregrine_vq2_curriculum.sbatch
#
# ADROIT AUP (binding): compute on SLURM only; compute nodes have NO internet (pre-download deps on the
# LOGIN/vis node BEFORE submit -- the diffaero conda env is pre-built); output -> /scratch; accurate
# --mem; 1 GPU (the 2048-env PPO saturates it); zero-GPU-util killed at 2 h; `checkquota` routinely.
#
# RUNTAG passthrough (no-clobber): submit with --export=ALL,RUNTAG=<tag>. REWARD selects the reward
# freeze arm. 🚩 REWARD FROZEN by Fengyou (2026-07-04) = a (recenter-faithful): the DEFAULT + the launch
# arm. b (+dense-centering) / c (look-at+learned-gain) stay as DORMANT portfolio toggles only. A_BODY=1
# flips the 23-dim felt-accel obs arm. ALGO switches ppo<->appo (both wired on this branch).
# NENVS/NUPD override the budget.
set -x
RUNTAG=${RUNTAG:-}
RUN_SUFFIX="${RUNTAG:+_$RUNTAG}"
GPUUTIL=/scratch/network/fl3689/peregrine_vq2_smoke${RUN_SUFFIX}_gpu_util.log
exec > /scratch/network/fl3689/peregrine_vq2_smoke${RUN_SUFFIX}.out 2>&1
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
source /etc/profile.d/modules.sh
module load anaconda3/2024.10
eval "$(conda shell.bash hook)"
conda activate diffaero
ln -sfn /scratch/network/fl3689/diffaero_repo /scratch/network/fl3689/diffaero
export PYTHONPATH=/scratch/network/fl3689:/scratch/network/fl3689/peregrine_repo/src:/scratch/network/fl3689/peregrine_repo/rl:$PYTHONPATH
cd /scratch/network/fl3689/diffaero

NENVS=${NENVS:-2048}
NUPD=${NUPD:-800}          # smoke budget: enough to read util + FLIGHTCHECK, not a full converge
REWARD=${REWARD:-a}
A_BODY=${A_BODY:-0}
ALGO=${ALGO:-ppo}         # appo = privileged asymmetric critic (wired on this branch, 0392b64)
STAGE=${STAGE:-multi_gate}
ENT_WEIGHT=${ENT_WEIGHT:-0.01}  # PPO/APPO entropy bonus coeff (diffaero default). See
                                # peregrine_vq2_curriculum.sbatch header for the 2026-07-05 log_std-
                                # saturation finding this knob is for.
TRAIN=/scratch/network/fl3689/peregrine_repo/rl/peregrine_train_inc8.py
TBTRACE=/scratch/network/fl3689/peregrine_repo/rl/inc8_tb_trace.py
CURR=/scratch/network/fl3689/peregrine_repo/rl/vq2_curriculum.py

# BASE == the recenter-convergent config (measured-plant DR map-ON + contact-true geometry + BSR3 + arm
# A + the band-pass perception window). The look-at signs + through-centering come from the curriculum
# STAGE overrides (rendered below), NOT hardcoded here, so a stage change is a single knob.
BASE=(
  env=racing env.name=peregrine_racing_inc8 +env.inc8=true +env.r5_arm=A
  +env.standing_start_frac=0.3
  +env.spin_rate_abort=10.0 +env.spin_time_abort=3.0
  +env.body_radius_lo=0.28 +env.body_radius_hi=0.38 +env.frame_depth_m=0.30
  +env.rw_perc_r_lo=12 +env.rw_perc_r_hi=28 +env.rw_perc_w_lo=1.5 +env.rw_perc_w_hi=1.5 +env.rw_perc_w0=0.05
  +env.lookat_r_lo=8.0 +env.lookat_r_hi=30.0
  dynamics=quad dynamics.name=peregrine_plant dynamics.g=9.80665
  +dynamics.dr=true +dynamics.dr_aero=true +dynamics.dr_mixer=true
  +dynamics.dr_force_bias=true +dynamics.dr_latency_min_steps=1 +dynamics.dr_latency_max_steps=3
  dynamics.controller.max_normed_thrust=3.765
  algo=${ALGO} algo.entropy_weight=${ENT_WEIGHT} headless=True device=0
)

# REWARD-FREEZE ARMS (Fengyou owns the freeze; presented as toggles -- spec T3.3):
#   a = recenter-faithful (progress + through-centering + R5'-A; rw_centering OFF) -- the proven-flying
#       config, DEFAULT candidate.
#   b = + dense terminal-sigma_p0 centering (the honest GT-anchored objective).
#   c = look-at primitive + learned gain (the S0 architecture pivot).
case "${REWARD}" in
  a) REWARD_OV=( ) ;;                                         # base already carries through-centering via STAGE
  b) REWARD_OV=( +env.rw_centering=1.0 ) ;;
  c) REWARD_OV=( +env.rw_centering=0.0 ) ;;                   # (learned-gain seam; look-at gains from STAGE)
  *) echo "unknown REWARD=${REWARD} (a|b|c)"; exit 2 ;;
esac

# a_body 23-dim felt-accel arm (A/B against the frozen obs-20 control).
if [ "${A_BODY}" = "1" ]; then ABODY_OV=( +env.obs_a_body=true ); else ABODY_OV=( ); fi

# CURRICULUM STAGE overrides (course_mode=random + vq2_like + blackout/pose-age DR + look-at signs +
# through-centering) rendered from rl/vq2_curriculum.py so the sbatch and the stager never drift.
STAGE_OV=( $(python ${CURR} ${STAGE}) )
echo "=== VQ2 STAGE=${STAGE} overrides: ${STAGE_OV[@]} ==="

# ---- GPU-util sampler (background): saturation check, every 20 s ----
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 20 > ${GPUUTIL} 2>&1 &
SMI_PID=$!

# ---- STAGE A: FAST PRECHECK (3 updates) -- catch a wiring/NaN/dim bug (per-env emulator gather, the
# a_body arm, the random-course R1'-drop) BEFORE the smoke run. ----
echo "=== VQ2 PRECHECK: construct + 3 updates (no NaN, sidecar dims) ==="
python ${TRAIN} "${BASE[@]}" "${STAGE_OV[@]}" "${REWARD_OV[@]}" "${ABODY_OV[@]}" \
  n_envs=512 n_updates=3 save_freq=1000 log_freq=1 seed=0 \
  runname=vq2_smoke_precheck${RUN_SUFFIX} ${EXTRA:-}
echo "PRECHECK_RC=$?"

# ---- STAGE B: the SMOKE (1 seed, arm A, short budget) ----
RUNNAME=vq2_smoke_seed0${RUN_SUFFIX}
RUNDIR=/scratch/network/fl3689/diffaero/outputs/train/${RUNNAME}
echo "=== VQ2 SMOKE seed 0: ${NUPD} updates, ${NENVS} envs, REWARD=${REWARD} A_BODY=${A_BODY} ALGO=${ALGO} ==="
python ${TRAIN} "${BASE[@]}" "${STAGE_OV[@]}" "${REWARD_OV[@]}" "${ABODY_OV[@]}" \
  hydra.run.dir=${RUNDIR} \
  n_envs=${NENVS} n_updates=${NUPD} save_freq=50 log_freq=10 \
  seed=0 runname=${RUNNAME} ${EXTRA:-}
echo "SMOKE_RC=$?"

echo "=== VQ2 SMOKE signal trace (obs_nonfinite==0, fix_rate/pointing rising, err bounded, FLIGHTCHECK) ==="
RUNDECOR=$(ls -dt ${RUNDIR}/*${RUNNAME}__* 2>/dev/null | head -1)
RUN=${RUNDECOR:-$RUNDIR}
echo "RUN=$RUN"
python ${TBTRACE} "$RUN" --stride-updates 100
echo "=== checkpoint sidecar (obs_dim 20 or 23, inc8 true, r5_arm A) ==="
find "$RUN" -name actor.json -exec cat {} \; 2>/dev/null | head -3

kill ${SMI_PID} 2>/dev/null
echo "=== GPU utilization summary (must be >=70% -- saturation gate) ==="
awk -F, 'NR>1 && $2 ~ /%/ {gsub(/ %/,"",$2); s+=$2; n++; if($2+0>mx)mx=$2} END{if(n)printf "mean_util=%.1f%%  max_util=%.0f%%  samples=%d\n", s/n, mx, n}' \
  ${GPUUTIL}
echo "VQ2_SMOKE_DONE"

# =====================================================================================================
# LAUNCH (from the Adroit LOGIN node, after PRE-SYNC + the adroit-connector serve daemon is up):
#   default (arm A, obs-20, symmetric, multi_gate stage):
#     sbatch --export=ALL,RUNTAG=vq2s0 /scratch/network/fl3689/peregrine_repo/rl/peregrine_vq2_smoke.sbatch
#   the 23-dim felt-accel A/B arm:
#     sbatch --export=ALL,RUNTAG=vq2s0_abody,A_BODY=1 .../peregrine_vq2_smoke.sbatch
#   an EARLY curriculum stage (hover / single_gate / blackout_pass):
#     sbatch --export=ALL,RUNTAG=vq2s0_hover,STAGE=hover .../peregrine_vq2_smoke.sbatch
#   GREEN -> the PORTFOLIO ladder (peregrine_vq2_curriculum.sbatch: reward arms x seeds, the winning
#   arm's speed rungs GATED on T1 above-hover thrust/drag validation).
# =====================================================================================================
