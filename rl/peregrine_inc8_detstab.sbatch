#!/bin/bash
#SBATCH --job-name=peregrine_inc8_detstab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/network/fl3689/peregrine_inc8_detstab.out
#
# inc8 DETERMINISTIC-STABILITY RE-TRAIN (inc8-deterministic-retrain, 2026-06-19). Produce a policy whose
# DETERMINISTIC mean (test=True == the deployed/eval policy) flies the 2-axis camera-pointing course, so
# the binding gate-4 terminal-centering sigma_p0 becomes MEASURABLE (today it is NO-DATA: rc1 flies only
# STOCHASTICALLY; its deterministic mean crashes ~1 s after spawn -- sigma_p0 job 3278151).
#
# ROOT CAUSE (measured, handoff probe_logstd.py): rc1 seed0's actor_logstd SATURATED two action channels
# to std=exp(2)=7.39 (the LOG_STD_MAX ceiling) -- after the tanh squash those channels are near-random
# bang-bang in the rollout. The policy learned to fly IN EXPECTATION OVER that chaos (feedback masks it),
# so its MEAN is an unvisited, unstable operating point. The entropy bonus (0.01, no schedule) with no std
# ceiling drove the saturation. Seeds 1/2 sit at a tamer ~0.25 but their means also crash.
#
# THE FIX = NOISE-ANNEAL (rl/inc8_noise_anneal.py, the LOAD-BEARING lever): clamp actor_logstd to a
# decaying std ceiling (loose hold std_hold from update 0 to kill the 7.39 saturation; geometric decay
# std_hold->std_floor over the back 1-hold_frac of training to collapse the rollout onto the mean) + anneal
# entropy_weight->0 over the same window. The policy is then TRAINED on near-deterministic actions, so its
# mean flies. Paired with DENSE SNAPSHOTS (rl/inc8_snapshots.py) for DETERMINISTIC post-hoc selection
# (rl/inc8_select_ckpt.sbatch) -- the runner's stochastic high-water "best" did NOT transfer to the mean.
#
# SINGLE-VAR DISCIPLINE: this holds rc1's WORKING reward (rw_through_centering=10.0, look-at warmup 200,
# g_yaw=-3) and changes ONLY (a) noise-anneal ON and (b) g_pitch (the elevation-pointing gain we are trying
# to push HIGHER than rc1's deterministic ceiling of 0.5). rw_centering (the S3 cliff driver) stays OFF.
#
# 🚩 GO CRITERION = DETERMINISTIC reach, NOT stochastic success. A config that trains to 50% STOCHASTIC
#   success but crashes on the mean is a FAILURE here (that is exactly rc1). The in-sbatch FLIGHTCHECK is
#   a STOCHASTIC necessary-not-sufficient early kill (catches total non-flyers ~step 800-1000); the REAL
#   criterion is measured post-hoc/mid-run by evaling a snapshot DETERMINISTICALLY with
#   rl/inc8_sigmap0_torch_eval.py (test=True). MID-RUN PROBE (early kill at the anneal tail, ~update 3000+):
#     python rl/inc8_sigmap0_torch_eval.py --ckpt <RUNDIR>/snapshots/upd03000 --lookat auto --n-envs 128 --horizons 1.0
#   reach_rate>0 there = the mean is starting to fly; still 0 with std already small => scancel.
#
# ADROIT AUP (account SUSPENSION risk): SLURM only (this batch); compute nodes have NO internet (pre-sync
# on login BEFORE submit); output -> /scratch; accurate --mem; 1 GPU (2048-env PPO saturates it); checkquota.
#
# 🚩 PRE-SYNC (the Adroit peregrine_repo is a FILE COPY -- NO git pull). Sync these to the login node first:
#     /scratch/network/fl3689/peregrine_repo/rl/inc8_noise_anneal.py          (NEW -- the lever)
#     /scratch/network/fl3689/peregrine_repo/rl/inc8_snapshots.py             (NEW -- dense retention)
#     /scratch/network/fl3689/peregrine_repo/rl/peregrine_train_inc8.py       (wires the two gated calls)
#     /scratch/network/fl3689/peregrine_repo/rl/peregrine_inc8_detstab.sbatch (this file)
# (via the adroit-connector base64 write or upload.) The diffaero conda env is pre-built.
#
# SUBMIT KNOBS (--export=ALL,KEY=VAL): GPITCH (look-at pitch gain to sweep, default 1.0), SEEDS (default 0),
#   NOISE_STD_HOLD/NOISE_STD_FLOOR/NOISE_HOLD_FRAC (anneal shape), SNAP_EVERY/SNAP_FROM (snapshot ladder),
#   NUPD (default 4000), NENVS (2048), RW_TC (10.0), LOOKAT_WARMUP (200), RUNTAG, EXTRA (appended to train).
RUNTAG=${RUNTAG:-}
RUN_SUFFIX="${RUNTAG:+_$RUNTAG}"
exec > /scratch/network/fl3689/peregrine_inc8_detstab${RUN_SUFFIX}.out 2>&1
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

SEEDS=${SEEDS:-0}
NENVS=${NENVS:-2048}
NUPD=${NUPD:-4000}
GPITCH=${GPITCH:-1.0}                       # look-at PITCH gain (the elevation-pointing knob we SWEEP)
RW_TC=${RW_TC:-10.0}                        # through-centering weight (held at rc1's working value)
LOOKAT_WARMUP=${LOOKAT_WARMUP:-200}        # look-at gain-warmup updates (held at rc1's value)
NOISE_STD_HOLD=${NOISE_STD_HOLD:-0.6}      # std ceiling during the hold (kills the 7.39 saturation)
NOISE_STD_FLOOR=${NOISE_STD_FLOOR:-0.03}   # final std ceiling (near-deterministic rollout)
NOISE_HOLD_FRAC=${NOISE_HOLD_FRAC:-0.5}    # fraction of training to hold before annealing
SNAP_EVERY=${SNAP_EVERY:-100}              # dense snapshot cadence (updates)
SNAP_FROM=${SNAP_FROM:-0.4}                # snapshot from this fraction of training (skip stochastic era)
TRAIN=/scratch/network/fl3689/peregrine_repo/rl/peregrine_train_inc8.py
TBTRACE=/scratch/network/fl3689/peregrine_repo/rl/inc8_tb_trace.py
GPUUTIL=/scratch/network/fl3689/peregrine_inc8_detstab${RUN_SUFFIX}_gpu_util.log

# COMMON == the EXACT rc1 (recenter) convergent config (peregrine_inc8_recenter.sbatch L91-103, VERBATIM).
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

# DETSTAB overrides on top of COMMON (both precheck + seed runs):
#   ++env.lookat_g_yaw=-3.0     empirical S0 yaw sign (COMMON default +3.0 is INVERTED) -- DOUBLE-plus
#   ++env.lookat_g_pitch=GPITCH the SWEEP knob (COMMON default 0.0) -- DOUBLE-plus
#   +env.lookat_warmup_updates  fresh-seed gain-warmup ramp -- single-plus
#   +env.rw_through_centering   rc1's lateral centring restore (held) -- single-plus
#   +algo.noise_anneal=true     THE LEVER + its shape (std ceiling decay + entropy anneal) -- single-plus
#   +snapshot_every/from_frac   dense retention for deterministic selection -- single-plus (root-level)
DS_OVERRIDES=(
  ++env.lookat_g_yaw=-3.0 ++env.lookat_g_pitch=${GPITCH}
  +env.lookat_warmup_updates=${LOOKAT_WARMUP}
  +env.rw_through_centering=${RW_TC}
  +algo.noise_anneal=true
  +algo.noise_std_hold=${NOISE_STD_HOLD}
  +algo.noise_std_floor=${NOISE_STD_FLOOR}
  +algo.noise_hold_frac=${NOISE_HOLD_FRAC}
  +algo.noise_entropy_floor=0.0
  +snapshot_every=${SNAP_EVERY}
  +snapshot_from_frac=${SNAP_FROM}
)

# ---- GPU-util sampler (background): saturation check, every 20 s ----
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 20 > ${GPUUTIL} 2>&1 &
SMI_PID=$!

# ---- STAGE A: FAST PRECHECK (3 updates, seed 0) -- catch a wiring/NaN/dim bug in the NEW noise-anneal +
# snapshot path BEFORE the 8 h burn. snapshot_every=1 here so it also exercises the snapshot save. ----
echo "=== inc8 DETSTAB PRECHECK: construct + 3 updates (noise-anneal ON, snapshot ON, no NaN) ==="
python ${TRAIN} "${COMMON[@]}" "${DS_OVERRIDES[@]}" \
  n_envs=512 n_updates=3 save_freq=1000 log_freq=1 seed=0 ++snapshot_every=1 \
  runname=inc8_detstab_precheck${RUN_SUFFIX} ${EXTRA:-}
PRECHECK_RC=$?
echo "PRECHECK_RC=${PRECHECK_RC}"
if [ "${PRECHECK_RC}" != "0" ]; then
  echo "🚩 PRECHECK FAILED (RC=${PRECHECK_RC}) -- a wiring/NaN/dim bug in the noise-anneal/snapshot path."
  echo "   NOT burning the 8 h seed run. Fix + re-sync + re-submit."
  kill ${SMI_PID} 2>/dev/null
  exit 1
fi
echo "PRECHECK PASSED -- proceeding to the seed run(s)."

# ---- STAGE B: the DETSTAB TRAIN (seed(s), arm A, NUPD updates, NENVS envs, noise-anneal ON) ----
FIRST_SEED=$(echo ${SEEDS} | awk '{print $1}')
for SEED in ${SEEDS}; do
  RUNNAME=inc8_detstab_seed${SEED}${RUN_SUFFIX}
  RUNDIR=/scratch/network/fl3689/diffaero/outputs/train/${RUNNAME}
  echo "=== inc8 DETSTAB seed ${SEED}: ${NUPD} updates, ${NENVS} envs, g_pitch=${GPITCH}, "
  echo "    noise-anneal std ${NOISE_STD_HOLD}->${NOISE_STD_FLOOR} hold_frac=${NOISE_HOLD_FRAC} ==="
  python ${TRAIN} "${COMMON[@]}" "${DS_OVERRIDES[@]}" \
    hydra.run.dir=${RUNDIR} \
    n_envs=${NENVS} n_updates=${NUPD} save_freq=50 log_freq=10 \
    seed=${SEED} runname=${RUNNAME} ${EXTRA:-}
  echo "DETSTAB_SEED${SEED}_RC=$?"

  echo "=== seed ${SEED} SIGNAL TRACE (STOCHASTIC flightcheck -- necessary-not-sufficient; the GO is the "
  echo "    DETERMINISTIC snapshot probe, see header) ==="
  RUNDECOR=$(ls -dt ${RUNDIR}/*${RUNNAME}__* 2>/dev/null | head -1)
  RUN=${RUNDECOR:-$RUNDIR}
  echo "RUN=$RUN"
  TRACE_LOG=/scratch/network/fl3689/peregrine_inc8_detstab${RUN_SUFFIX}_seed${SEED}_trace.txt
  python ${TBTRACE} "$RUN" --stride-updates 100 | tee ${TRACE_LOG}
  echo "=== seed ${SEED} snapshots retained (for deterministic selection) ==="
  ls -d ${RUNDIR}/snapshots/upd* 2>/dev/null | tail -8
  echo "=== seed ${SEED} checkpoint sidecar (obs_dim 20, inc8 true, r5_arm A) ==="
  find "$RUN" -name actor.json -exec cat {} \; 2>/dev/null | head -3

  # ---- FLIGHTCHECK GATE (first seed only): STOP before burning the rest if flight not even STOCHASTIC ----
  if [ "${SEED}" = "${FIRST_SEED}" ]; then
    VERDICT=$(awk '/FLIGHTCHECK/ {
        for (i=1;i<=NF;i++) {
          if ($i ~ /^success_rate_max=/)   { split($i,a,"="); sr=a[2]+0 }
          if ($i ~ /^n_passed_gates_max=/) { split($i,b,"="); npg=b[2]+0 }
        }
        if (sr < 0.005 && npg < 0.1) print "FAILED"; else print "FLEW";
      }' ${TRACE_LOG})
    echo "FLIGHTCHECK_VERDICT=${VERDICT:-UNKNOWN} (seed ${SEED}; STOCHASTIC -- the deterministic GO is the snapshot probe)"
    if [ "${VERDICT}" = "FAILED" ]; then
      echo "🚩 FLIGHTCHECK FAILED: not even stochastic flight -- noise-anneal cap likely TOO TIGHT for this"
      echo "   g_pitch (raise NOISE_STD_HOLD) or g_pitch too high. STOPPING (not burning remaining seeds)."
      break
    fi
    echo "FLIGHTCHECK PASSED (stochastic flight) -- continuing; CONFIRM deterministic flight via the snapshot probe."
  fi
done

kill ${SMI_PID} 2>/dev/null
echo "=== GPU utilization summary (must be HIGH -- saturation check) ==="
awk -F, 'NR>1 && $2 ~ /%/ {gsub(/ %/,"",$2); s+=$2; n++; if($2+0>mx)mx=$2} END{if(n)printf "mean_util=%.1f%%  max_util=%.0f%%  samples=%d\n", s/n, mx, n}' \
  ${GPUUTIL}
echo "INC8_DETSTAB_DONE"

# =====================================================================================================
# LAUNCH (Adroit login node, after PRE-SYNC). BATCH 1 -- the g_pitch x cap-strength bracket (seed0, parallel):
#   sbatch --export=ALL,GPITCH=1.0,NOISE_STD_HOLD=0.6,NOISE_STD_FLOOR=0.03,RUNTAG=ds_gp1p0_h06 rl/peregrine_inc8_detstab.sbatch
#   sbatch --export=ALL,GPITCH=1.0,NOISE_STD_HOLD=1.5,NOISE_STD_FLOOR=0.05,RUNTAG=ds_gp1p0_h15 rl/peregrine_inc8_detstab.sbatch
#   sbatch --export=ALL,GPITCH=1.5,NOISE_STD_HOLD=0.6,NOISE_STD_FLOOR=0.03,RUNTAG=ds_gp1p5_h06 rl/peregrine_inc8_detstab.sbatch
# BATCH 2 (informed): take the best DETERMINISTIC config, add seeds 1,2 for the >=3-seed GO + push g_pitch.
#   for s in 1 2; do sbatch --export=ALL,GPITCH=<best>,NOISE_STD_HOLD=<best>,SEEDS=$s,RUNTAG=ds_<best>_s$s rl/peregrine_inc8_detstab.sbatch; done
# Then SELECT + MEASURE: rl/inc8_select_ckpt.sbatch on each winning run dir.
# =====================================================================================================
