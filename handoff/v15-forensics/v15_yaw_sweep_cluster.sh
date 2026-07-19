#!/bin/bash
# =====================================================================================================
# v1 YAW SWEEP (++rollout_only harness) -- CLUSTER-ONLY REHEARSAL: RUN ON THE SMOKE CKPT FIRST.
#
# WHY THIS FILE EXISTS: launch_v1.sh's SELECT block (lines 197-201) says "run the ++rollout_only
# harness over the run's periodic + best_npg/ snapshots" but NO concrete command exists anywhere in
# the repo/handoff. rollout_only requires diffaero (peregrine_train_ego.py:67 imports it at module
# top; ModuleNotFoundError locally) -> it CANNOT be dry-run on the laptop. This is the exact
# cluster-side invocation, assembled key-for-key from peregrine_vq2_ego.sbatch (BASE lines 86-105,
# BOUNDARY_OV 123-124, env setup 49-55) + launch_v1.sh (BASE_VPEFFS0 104-106, VPEF8NC 110-114,
# ANTIDITHER 117-118, FAITHFUL_SHARED 123-124), with EXACTLY TWO deliberate edits:
#   EDIT 1: ++eval_det_steps=600 REMOVED from the arm EXTRA -> BOUNDARY's +eval_det_steps=1200 stands.
#           600 steps = 20 s censors 8-gate laps (seg 10-20 m x 8 at 3-5 m/s = 27-53 s); 1200 halves
#           the n_passed censoring. signflips_per_s is per-step and unaffected either way. RANK CKPTS
#           WITHIN THIS SWEEP ONLY (in-run YAW_EVAL/DET lines were at 600 -- not directly comparable).
#   EDIT 2: +init_from REMOVED -> ++rollout_ckpt is the explicit ckpt (rollout_ckpt takes precedence
#           anyway, peregrine_train_ego.py:702-704; dropping init_from removes the fallback ambiguity).
# Uniform census: seed=0 + ego_vision_cadence_seed=20260715 for ALL evals (identical courses +
# detector-miss realization across every ckpt -> a fair ranking; per-run training seeds NOT replicated
# on purpose).
#
# REHEARSAL (do this BEFORE the arms finish, on the completed smoke):
#   SWEEP_RUNS="ego_dual_gate_fullstack_floor_pef_seed0_v1smoke_s0" bash v15_yaw_sweep_cluster.sh
#   (the smoke has ONLY checkpoints/ -- UPD=100 < save_freq=500 means best_npg/ and periodic/ never
#    saved; promotion in the smoke log prints "no best_npg/ snapshot to promote" -- EXPECTED, not a bug.
#    PASS = one "[rollout-only] LOADED .../checkpoints" line + one parseable YAW_EVAL[..._sweep/...]
#    line with finite signflips_per_s that ~matches the smoke's own in-run final YAW_EVAL -- same ckpt,
#    same seed 0; the in-run line sampled 600 steps, the sweep 1200, but the metric is a per-second RATE,
#    so agreement to sampling noise is the pass criterion.)
# REAL SWEEP (after the arms):
#   SWEEP_RUNS="ego_dual_gate_fullstack_floor_pef_seed0_v1A_s0 ego_dual_gate_fullstack_floor_pef_seed1_v1A_s1 \
#               ego_dual_gate_fullstack_floor_pef_seed0_v1R_s0 ego_dual_gate_fullstack_floor_pef_seed0_v1C_s0" \
#   bash v15_yaw_sweep_cluster.sh
#
# READOUT: grep YAW_EVAL /scratch/network/fl3689/v15_yaw_sweep_<jobid>.out
#   HARD GATE: discard signflips_per_s > ~4.0 regardless of n_passed; rank survivors on n_passed
#   (this sweep's own lines); tiebreak lowest signflips then lowest roll_swing.
#   CKPT ASSOCIATION FOOTGUN: in the TRAINING logs, if BEST_CKPT_PROMOTED[<run>] printed, the shipped
#   checkpoints/ = best_npg -> its in-run yaw line is YAW_EVAL[<run>/best], NOT YAW_EVAL[<run>].
#   This sweep sidesteps that: it evals checkpoints/ (post-promotion contents) directly by path.
#
# AUP: SLURM only (sbatch below), output -> /scratch, ~16 evals x ~2 min << 2 h wall.
# =====================================================================================================
set -euo pipefail
SWEEP_RUNS=${SWEEP_RUNS:?set SWEEP_RUNS=\"<runname> [...]\"}

cat > /scratch/network/fl3689/v15_yaw_sweep.sbatch <<'SB'
#!/bin/bash
#SBATCH --job-name=v15_yaw_sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
exec > /scratch/network/fl3689/v15_yaw_sweep_${SLURM_JOB_ID}.out 2>&1
set -x
hostname
source /etc/profile.d/modules.sh
module load anaconda3/2024.10
eval "$(conda shell.bash hook)"
conda activate diffaero
ln -sfn /scratch/network/fl3689/diffaero_repo /scratch/network/fl3689/diffaero
export PYTHONPATH=/scratch/network/fl3689:/scratch/network/fl3689/peregrine_repo/src:/scratch/network/fl3689/peregrine_repo/rl:$PYTHONPATH
cd /scratch/network/fl3689/diffaero

TRAIN=/scratch/network/fl3689/peregrine_repo/rl/peregrine_train_ego.py
CURR=/scratch/network/fl3689/peregrine_repo/rl/vq2_ego_curriculum.py
STAGE=dual_gate_fullstack_floor_pef
STAGE_OV=( $(python ${CURR} ${STAGE}) )

# BASE -- VERBATIM peregrine_vq2_ego.sbatch lines 86-105 (ENT_WEIGHT default 0.01)
BASE=( env=racing env.name=peregrine_racing_ego
  +env.body_radius_lo=0.28 +env.body_radius_hi=0.38 +env.frame_depth_m=0.30
  dynamics=quad dynamics.name=peregrine_plant dynamics.g=9.80665
  +dynamics.dr=true +dynamics.dr_aero=true +dynamics.dr_mixer=true
  +dynamics.dr_force_bias=true +dynamics.dr_latency_min_steps=1 +dynamics.dr_latency_max_steps=3
  dynamics.controller.max_normed_thrust=3.765
  algo=appo algo.entropy_weight=0.01 headless=True device=0
  algo.critic_grad_norm=1.0 export.jit=false export.onnx=false
  +algo.noise_anneal=true +algo.noise_std_hold=0.30 +algo.noise_std_floor=0.30
  +algo.noise_hold_frac=1.0 +algo.noise_entropy_floor=0.0 )
# BOUNDARY -- VERBATIM sbatch lines 123-124 (eval_det_steps=1200 STANDS: EDIT 1, no 600 override below)
BOUNDARY_OV=( +warmstart_reset_logstd=true +warmstart_reset_logstd_std=0.18 +critic_warmup_updates=100
              +eval_det_steps=1200 )
# arm-A EXTRA -- VERBATIM launch_v1.sh BASE_VPEFFS0 + VPEF8NC + ANTIDITHER + FAITHFUL_SHARED, minus
# EDIT 1 (++eval_det_steps=600 dropped) + EDIT 2 (+init_from dropped). All keys inert pre-rollout
# (rollout_only returns before anneal/select wiring, peregrine_train_ego.py:845-846) but kept verbatim
# so the BUILT eval env is bit-identical to the arms' training env (clamp 0.7, 8 gates, cadence ON,
# faithful_rate ON, spin fence at the configured END values).
EXTRA_COMMON="++env.ego_blur_gate=false ++env.ego_yaw_cmd_clamp_rad_s=0.7 ++env.spin_abort_anneal=true \
++env.spin_abort_scale_start=2.6 ++env.spin_abort_hold_frac=0.25 ++env.ego_est_dt_ticks_hi=1 \
++warmstart_reset_logstd=true ++warmstart_reset_logstd_std=0.18 \
++env.course_n_gates=8 ++env.rw_v_cap=0 ++env.rw_cross_zero_m=0.75 ++env.rw_finish_time=0.25 \
++algo.noise_anneal=true ++algo.noise_std_hold=0.6 ++algo.noise_std_floor=0.05 ++algo.noise_hold_frac=0.5 \
++critic_grad_clip=10 ++ckpt_select_metric=n_passed_gates ++eval_yaw_log=true ++save_freq=500 \
++env.rw_yaw_dither=0.4 ++env.overspeed_abort_mps=12.0 ++env.rw_progress_vcap_mps=7.5 ++env.rw_yaw_duty=0.15 ++env.yaw_duty_free_band=0.25 ++env.rw_yaw_jerk=0.05 ++env.rw_roll_recover=0.5 ++env.recovery_anneal=true ++env.recovery_start=0.0 ++env.recovery_hold_frac=0.3 ++env.yaw_dither_anneal=true ++env.yaw_dither_start=1.0 ++env.yaw_dither_hold_frac=0.3 \
++dynamics.faithful_rate=true ++env.ego_vision_cadence=true \
++env.ego_vision_frame_hz=30.0 ++env.ego_vision_detect_p=0.35 ++env.ego_vision_cadence_seed=20260715"

OUTROOTS="/scratch/network/fl3689/diffaero/outputs/train /scratch/network/fl3689/diffaero_repo/outputs/train"
for RUN in ${SWEEP_RUNS}; do
  # snapshot inventory per run (peregrine_train_ego.py behavior): checkpoints/ (final OR promoted best),
  # best_npg/ + periodic/ + periodic_prev/ under the DECORATED logdir (may sit under either root -- the
  # Bug-2 postmortem saw best_npg under diffaero_repo/outputs/train/<date>/<time>/), emergency/ on crash.
  # periodic/ is OVERWRITTEN each boundary (only the last two survive: :1169-1175).
  CANDS=$(find ${OUTROOTS} -maxdepth 6 -type d \
            \( -name checkpoints -o -name best_npg -o -name periodic -o -name periodic_prev -o -name emergency \) \
            -path "*${RUN}*" 2>/dev/null | sort -u)
  if [ -z "${CANDS}" ]; then
    echo "SWEEP_WARN[${RUN}]: no ckpt dirs found under ${OUTROOTS} matching *${RUN}* -- if the run used a"
    echo "  date/time-decorated logdir WITHOUT the runname in the path, locate best_npg/periodic manually:"
    echo "  find ${OUTROOTS} -maxdepth 6 -type d -name best_npg -newermt 2026-07-15"
    continue
  fi
  for D in ${CANDS}; do
    if [ ! -f "${D}/actor.pth" ] || [ ! -f "${D}/critic.pth" ]; then
      echo "SWEEP_SKIP[${RUN}]: ${D} lacks actor.pth+critic.pth (agent.load needs BOTH: :706-710)"; continue
    fi
    TAG="$(basename $(dirname ${D}))_$(basename ${D})"
    python ${TRAIN} "${BASE[@]}" "${STAGE_OV[@]}" "${BOUNDARY_OV[@]}" \
      hydra.run.dir=/scratch/network/fl3689/diffaero/outputs/rollout/${RUN}_$(basename ${D}) \
      n_envs=2048 n_updates=1 save_freq=100000 log_freq=10 seed=0 runname=${RUN}_sweep \
      ${EXTRA_COMMON} \
      ++rollout_only=true ++rollout_ckpt=${D} ++rollout_tag=${TAG}
    echo "SWEEP_RC[${RUN}/${TAG}]=$?"
  done
done
echo "=== SWEEP SUMMARY (grep-ready) ==="
grep -h "YAW_EVAL\|rollout-only\|SWEEP_" /scratch/network/fl3689/v15_yaw_sweep_${SLURM_JOB_ID}.out || true
SB

# GENERIC gres (header gpu:1 stands): typed nvidia_a100 misses free MIG slices -- the arms-launch lesson.
# 2h walltime -> QOS auto-maps gpu-short = does NOT touch the gpu-medium 2-cap where R/C queue.
# Exclude adroit-h11g3: V100 sm_70 unvalidated (#57) -- don't burn the rehearsal on a fail-loud probe.
SBATCH_EXCLUDE=adroit-h11g3 sbatch --export=ALL,SWEEP_RUNS="${SWEEP_RUNS}" /scratch/network/fl3689/v15_yaw_sweep.sbatch
