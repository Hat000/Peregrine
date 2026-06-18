#!/bin/bash
#SBATCH --job-name=peregrine_inc8_recenter
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/network/fl3689/peregrine_inc8_recenter.out
#
# inc8 RE-TRAIN -- RESTORE THROUGH-APPROACH CENTERING (P2 INC8-RL, 2026-06-17). 3 FRESH seeds, arm A.
#
# WHY: the inc8 lineage NEVER flew a lap (in-training success_rate ≡ 0 across ALL runs incl. the S2-seed2
# parent + warm-start; dies at gate-0 ~1.5 s in). ROOT (confound-free, Adroit-confirmed): the inc8 reward
# redesign DROPPED inc7's through-approach centering -- R1-to-centre (peregrine_racing.py:364, the
# Euclidean distance-to-gate-CENTRE delta) was ZEROED (peregrine_racing_inc8.py: w_frozen progress=0.0)
# and replaced by arc-Γ progress (R1', ALONG-track only) + a NEAR-gate centering that penalises ESTIMATOR
# error (σ_p0), not physical offset. Arc-Γ rewards advancing along the racing line but has NO restoring
# force when the drone drifts OFF the gate-centre line -> it under-centres through the approach -> misses
# gate-0. inc7 (which flies 3/3 in the same harness) KEEPS R1-to-centre. The whole ladder optimised a
# PROXY (camera pointing) and never gated the OBJECTIVE (course-completion), so this was invisible.
#
# THE FIX (rl/inc8_reward.through_centering_reward + the env): re-add the LATERAL pull of R1-to-centre as
# a NEW gated reward term, +env.rw_through_centering=<w>. It is the cross-track (gate-frame in-plane
# hypot(y,z)) distance-to-centre-line delta -- R1-to-centre's centring component, DECOUPLED from forward
# progress so it composes with arc-Γ (ORTHOGONAL: arc-Γ = along-track, this = cross-track), the existing
# near-gate σ_p0 centering, and the look-at primitive (camera, not flight path). Default 0 -> byte-id inc8.
# We deliberately change ONE thing (add through-centering) for an interpretable GO/NO-GO; rw_centering
# (the σ_p0 near-gate term, the S3 cliff driver) stays OFF -- it is only meaningful once laps complete and
# fresh-init would risk re-triggering the S3 reward cliff. Layer it in a follow-up once flight is recovered.
#
# 🚩 GO CRITERION (the lesson -- COURSE-COMPLETION is PRIMARY, NOT pointing):
#   1. FLIGHT RECOVERED (the headline): metrics/n_passed_gates climbs OFF 0 (the drone passes gate-0+) and
#      metrics/success_rate climbs OFF 0 (full laps). The TB trace surfaces BOTH from step 0; the trace's
#      FLIGHTCHECK line summarises their max. EARLY-STOP: if success_rate stays ~0 AND n_passed_gates stays
#      ~0 by ~step 800-1000, the restore did NOT recover flight -> STOP (scancel) + report; do NOT burn the
#      full run. This script AUTO-gates the 2nd/3rd seeds on the 1st seed's flight-check (below).
#   2. CENTERING FIRES: inc8_through_centering goes non-zero (the lateral restoring reward is active) and
#      inc8_r1p (arc-Γ) stays healthy (the two compose, they do not fight).
#   3. POINTING RETAINED: inc8_lockband_pointing / inc8_terminal_pointing stay > 0 (the look-at primitive
#      still points while the new centring keeps the drone on the line).
#   4. ONLY IF laps complete: σ_p0 (terminal centring miss) via rl/contact_true_eval.py becomes meaningful
#      (judge on the GT-anchored PHYSICAL miss, NOT the TB inc8_estim_err_inplane_m, which is estimator-
#      floored ~0.115-0.13 regardless of policy). A pointing-but-non-flying policy yields ZERO laps, so
#      σ_p0 is UN-MEASURABLE until flight is recovered -- that is exactly the trap the lineage fell into.
#
# ADROIT AUP (Princeton RC SUSPENDS accounts): compute on SLURM only (this batch); compute nodes have NO
# internet -- git pull / conda env on the LOGIN or vis node BEFORE submitting; output -> /scratch; accurate
# --mem; 1 GPU (the 2048-env PPO saturates it). Run `checkquota` routinely. See reference_adroit_princeton.md.
#
# 🚩 PRE-SYNC (the Adroit peregrine_repo is a FILE COPY, NOT a git clone -- there is NO `git pull` there):
# BEFORE submitting, SYNC the edited files to the LOGIN node (compute nodes have no internet):
#     /scratch/network/fl3689/peregrine_repo/rl/inc8_reward.py             (the new term)
#     /scratch/network/fl3689/peregrine_repo/rl/peregrine_racing_inc8.py   (the env wiring)
#     /scratch/network/fl3689/peregrine_repo/rl/inc8_tb_trace.py           (success_rate/n_passed cols)
#     /scratch/network/fl3689/peregrine_repo/rl/peregrine_inc8_recenter.sbatch
# (e.g. via the adroit-connector upload, or scp from the laptop). The diffaero conda env is pre-built.
#
# RUNTAG passthrough (no-clobber): submit with --export=ALL,RUNTAG=<tag> (e.g. RUNTAG=rc1). EXTRA is
# appended to BOTH the precheck and every train invocation for a submit-time override without editing this
# file. SEEDS overrides the seed list (default "0 1 2"; the 1st is the auto flight-check gate). RW_TC sets
# the through-centering weight (default 10.0 == inc7's rw_progress / the arc-Γ weight, applied to the
# cross-track delta only -> gentler than inc7's full-Euclidean 10.0). LOOKAT_WARMUP ramps the look-at gain
# (fresh seeds: 200, to survive the fresh-policy value_loss spike that killed 2/3 S3 cold seeds @ step100).
RUNTAG=${RUNTAG:-}
RUN_SUFFIX="${RUNTAG:+_$RUNTAG}"
GPUUTIL=/scratch/network/fl3689/peregrine_inc8_recenter${RUN_SUFFIX}_gpu_util.log
exec > /scratch/network/fl3689/peregrine_inc8_recenter${RUN_SUFFIX}.out 2>&1
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

SEEDS=${SEEDS:-0 1 2}
NENVS=${NENVS:-2048}
NUPD=${NUPD:-4000}
RW_TC=${RW_TC:-10.0}                 # through-centering weight (the new lever; the ONLY change vs S0)
LOOKAT_WARMUP=${LOOKAT_WARMUP:-200}  # look-at gain-warmup updates (fresh seeds need the ramp)
TRAIN=/scratch/network/fl3689/peregrine_repo/rl/peregrine_train_inc8.py
TBTRACE=/scratch/network/fl3689/peregrine_repo/rl/inc8_tb_trace.py

# COMMON == the EXACT S0/S2-convergent config (rl/peregrine_inc8_s0.sbatch lines 71-83, VERBATIM):
# inc7-class measured-plant DR (map-ON) + contact-true geometry + BSR3 + arm A + the band-pass perception
# window + the look-at primitive. lookat_g_yaw/g_pitch are SET here (so the per-run sign fix below uses ++
# DOUBLE-plus to OVERRIDE them); rw_through_centering / lookat_warmup_updates are NOT here (+ single-plus).
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

# RECENTER overrides applied on top of COMMON, for BOTH precheck and the seed runs:
#   ++env.lookat_g_yaw=-3.0        empirical S0 yaw sign (COMMON default +3.0 is INVERTED) -- DOUBLE-plus
#   ++env.lookat_g_pitch=3.0       empirical S2 pitch sign (OPPOSITE yaw) -- DOUBLE-plus (both in COMMON)
#   +env.lookat_warmup_updates=N   fresh-seed gain-warmup ramp (NOT in COMMON) -- single-plus
#   +env.rw_through_centering=W    THE RESTORE: arc-Γ-compatible lateral centring pull (NEW) -- single-plus
RC_OVERRIDES=(
  ++env.lookat_g_yaw=-3.0 ++env.lookat_g_pitch=3.0
  +env.lookat_warmup_updates=${LOOKAT_WARMUP}
  +env.rw_through_centering=${RW_TC}
)

# ---- GPU-util sampler (background): saturation check, every 20 s ----
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 20 > ${GPUUTIL} 2>&1 &
SMI_PID=$!

# ---- STAGE A: FAST PRECHECK (3 updates, seed 0) -- catch a wiring/NaN/dim bug (incl. the new through-
# centering term + the look-at compose) BEFORE the seed runs. obs_dim 20, symmetric critic, no NaN. ----
echo "=== inc8 RECENTER PRECHECK: construct + 3 updates (obs_dim 20, through-centering ON, no NaN) ==="
python ${TRAIN} "${COMMON[@]}" "${RC_OVERRIDES[@]}" \
  n_envs=512 n_updates=3 save_freq=1000 log_freq=1 seed=0 \
  runname=inc8_recenter_precheck${RUN_SUFFIX} ${EXTRA:-}
echo "PRECHECK_RC=$?"

# ---- STAGE B: the RECENTER FINE-TUNE (3 FRESH seeds, arm A, 4000 updates, 2048 envs) ----
# Per-seed EXPLICIT hydra.run.dir (the S1@3.0 collision fix): each seed writes to its own dir so two
# seeds can never share a timestamped dir and come out bit-identical. The FIRST seed is the FLIGHT-CHECK
# gate: after it trains, we grep its trace FLIGHTCHECK line; if the restore did NOT recover flight
# (success_rate ~0 AND no gates passed), we STOP and do not burn the remaining seeds.
FIRST_SEED=$(echo ${SEEDS} | awk '{print $1}')
for SEED in ${SEEDS}; do
  RUNNAME=inc8_recenter_seed${SEED}${RUN_SUFFIX}
  RUNDIR=/scratch/network/fl3689/diffaero/outputs/train/${RUNNAME}
  echo "=== inc8 RECENTER seed ${SEED}: ${NUPD} updates, ${NENVS} envs, rw_through_centering=${RW_TC} ==="
  python ${TRAIN} "${COMMON[@]}" "${RC_OVERRIDES[@]}" \
    hydra.run.dir=${RUNDIR} \
    n_envs=${NENVS} n_updates=${NUPD} save_freq=50 log_freq=10 \
    seed=${SEED} runname=${RUNNAME} ${EXTRA:-}
  echo "RECENTER_SEED${SEED}_RC=$?"

  echo "=== inc8 RECENTER seed ${SEED} SIGNAL TRACE (READ success_rate / n_passed_gates FIRST -- the "
  echo "    PRIMARY GO; then through_centering fires + r1p healthy; then pointing retained) ==="
  RUNDECOR=$(ls -dt ${RUNDIR}/*${RUNNAME}__* 2>/dev/null | head -1)
  RUN=${RUNDECOR:-$RUNDIR}
  echo "RUN=$RUN"
  TRACE_LOG=/scratch/network/fl3689/peregrine_inc8_recenter${RUN_SUFFIX}_seed${SEED}_trace.txt
  python ${TBTRACE} "$RUN" --stride-updates 100 | tee ${TRACE_LOG}
  echo "=== seed ${SEED} checkpoint sidecar (obs_dim 20, inc8 true, r5_arm A) ==="
  find "$RUN" -name actor.json -exec cat {} \; 2>/dev/null | head -3

  # ---- FLIGHT-CHECK GATE (first seed only): STOP before burning the rest if flight was not recovered ----
  if [ "${SEED}" = "${FIRST_SEED}" ]; then
    VERDICT=$(awk '/FLIGHTCHECK/ {
        for (i=1;i<=NF;i++) {
          if ($i ~ /^success_rate_max=/)   { split($i,a,"="); sr=a[2]+0 }
          if ($i ~ /^n_passed_gates_max=/) { split($i,b,"="); npg=b[2]+0 }
        }
        if (sr < 0.005 && npg < 0.1) print "FAILED"; else print "FLEW";
      }' ${TRACE_LOG})
    echo "FLIGHTCHECK_VERDICT=${VERDICT:-UNKNOWN} (seed ${SEED})"
    if [ "${VERDICT}" = "FAILED" ]; then
      echo "🚩 FLIGHTCHECK FAILED: success_rate ~0 AND no gates passed -- the through-centering restore did"
      echo "   NOT recover flight on the first seed. STOPPING (not burning the remaining seeds). REPORT this"
      echo "   + the trace; consider RW_TC sweep / reconcile arc-Γ-vs-centring before a re-run."
      break
    fi
    echo "FLIGHTCHECK PASSED (flight recovered on seed ${SEED}) -- continuing to the remaining seeds."
  fi
done

kill ${SMI_PID} 2>/dev/null
echo "=== GPU utilization summary (must be HIGH -- saturation check) ==="
awk -F, 'NR>1 && $2 ~ /%/ {gsub(/ %/,"",$2); s+=$2; n++; if($2+0>mx)mx=$2} END{if(n)printf "mean_util=%.1f%%  max_util=%.0f%%  samples=%d\n", s/n, mx, n}' \
  ${GPUUTIL}
echo "INC8_RECENTER_DONE"

# =====================================================================================================
# LAUNCH (from the Adroit LOGIN node, after the PRE-SYNC above; the connector submits to SLURM):
#   one job, 3 fresh seeds (auto flight-check gate on the 1st):
#     sbatch --export=ALL,RUNTAG=rc1 /scratch/network/fl3689/peregrine_repo/rl/peregrine_inc8_recenter.sbatch
#   OR 3 SEPARATE jobs (more robust -- one seed's crash can't kill the others; NO auto-gate across jobs,
#   so launch seed 0 FIRST, read its FLIGHTCHECK, then launch 1,2 only if it flew):
#     sbatch --export=ALL,SEEDS=0,RUNTAG=rc1_s0 /scratch/network/fl3689/peregrine_repo/rl/peregrine_inc8_recenter.sbatch
#     # if seed 0 flew:
#     for s in 1 2; do sbatch --export=ALL,SEEDS=$s,RUNTAG=rc1_s$s \
#       /scratch/network/fl3689/peregrine_repo/rl/peregrine_inc8_recenter.sbatch; done
#   weight sweep (if RW_TC=10.0 over/under-centres): add RW_TC=<w> to --export (e.g. RW_TC=6.0).
# =====================================================================================================
