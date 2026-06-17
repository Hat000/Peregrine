#!/bin/bash
#SBATCH --job-name=peregrine_inc8_warmstart
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/network/fl3689/peregrine_inc8_warmstart.out
#
# inc8 WARM-START PROBE (P2 INC8-RL, 2026-06-17). Fine-tune the CENTERING reward from the CONVERGED S2
# seed-2 checkpoint instead of FRESH init -- the cheap pre-appo probe for the S3 NOT-GO.
#
# WHY: S3 (rw_centering=1.0, FRESH seeds) was NOT-GO from two COLD-START artifacts:
#   - 2/3 seeds died in the look-at gain-warmup (value_loss 432/572 @ step100), AND
#   - the 1 seed that broke out hit a REWARD CLIFF at centering-onset (value_loss 650 @ step1200) and
#     collapsed (the KF suddenly gets fixes -> err_ip jumps from KF-drift-to-zero to the honest ~0.12 ->
#     the symmetric critic can't anticipate the new reward term -> PPO destabilizes).
# A warm-start from S2 seed-2 (already PAST warmup, already pointing at FULL look-at gain, KF already
# getting fixes from step 0) removes the regime change that drives BOTH: the policy starts INSIDE the
# pointing basin -> no warmup ramp left to survive, and no fix-onset discontinuity for centering to slam
# into. The look-at gain-warmup is FORCED to 0 (the loaded policy already points -- re-ramping would
# disrupt it); the trainer enforces this AND we pass +env.lookat_warmup_updates=0 belt-and-braces.
#
# ===========================  ${CKPT}: the S2 seed-2 converged checkpoint  ===========================
# CKPT = the S2 seed-2 CONVERGED checkpoint DIR (jobs 3275300-02; seed 2 was THE converged seed -- band_el
# 24-31, band_az 22-37, fix_rate 0.11-0.16, terminal_pointing 0.03-0.13, stable steps 600-3990).
# LOCATE on the Adroit LOGIN node (this is a FILE READ -- AUP-safe, NOT compute):
#   ls -dt /scratch/network/fl3689/diffaero/outputs/train/*/*/*inc8_s0_yawprim_seed2_s2full_s2__*/periodic | head -1
#   cat <that dir>/actor.json     # MUST print  "obs_dim": 20,  "inc8": true,  "r5_arm": "A"
# The dir MUST hold BOTH actor.pth and critic.pth (the trainer lifeline periodic-save writes both + the
# sidecar). A deploy-only actor.pth alone is rejected -- the critic must resume too. Then submit with
# CKPT=<that dir> (see the LAUNCH block at the very bottom).
#
# ===========================  MEASUREMENT / GO  (read in THIS order)  ================================
#   1. NO CLIFF (the headline): inc8 value_loss must NOT spike at the step S3 cliffed (~step 1200 in the
#      FRESH run). A successful warm-start holds value_loss in the converged band (S2 seed-2: 3-25; the
#      transient ~112 at the entropy-zero crossing ~step1100 is NORMAL PPO, not a cliff). A 400-650 spike
#      at centering-onset = the cliff SURVIVED the warm-start => warm-start alone does NOT solve it =>
#      escalate to appo (asymmetric critic). This is the single most important readout.
#   2. CENTERING TAKES + POINTING RETAINED: inc8_centering reward goes non-zero WHILE terminal_pointing /
#      lockband_pointing STAY > 0 (the loaded pointing is kept, centering is added ON TOP). If pointing
#      collapses when centering switches on, the policy is trading the line away again.
#   3. 🚩 PHYSICAL GATE (the REAL verdict -- do NOT crown on estim_err): judge terminal centering on the
#      GT-anchored PHYSICAL miss sigma_p0, via rl/contact_true_eval.py on the warm-started checkpoint
#      (read the TRUE in-plane miss / true err_ip / contact_true geometry), NOT the TB
#      inc8_estim_err_inplane_m. estim_err is FLOORED at the estimator's gate-relative RMS ~0.115-0.13
#      REGARDLESS of policy (S2 seed-2 plateaued there WITH perfect pointing), so "estim_err <= 0.08" is
#      geometrically unachievable and is NOT the gate. The honest objective is the physical centering
#      miss sigma_p0 <~ 0.08 m (gate-4 closure, r=0.30). The TB inc8_centering trace shows the reward is
#      FIRING; contact_true_eval shows whether the drone is actually CENTERED. Crown on the latter.
#
# ADROIT AUP (Princeton RC SUSPENDS accounts): compute on SLURM only (this batch); compute nodes have
# NO internet -- pre-download git pull / conda env on the LOGIN or vis node BEFORE submitting; output
# -> /scratch; accurate --mem; 1 GPU for GPU code; the 2048-env PPO saturates it (verified by the
# sampler). Run `checkquota` routinely. See reference_adroit_princeton.md.
#
# 🚩 WARM-START PRE-SYNC (the Adroit peregrine_repo is a FILE COPY, NOT a git clone -- there is NO
# `git pull` there): BEFORE submitting, SYNC the two new/edited files to the LOGIN node so the compute
# node sees them (compute nodes have no internet):
#     /scratch/network/fl3689/peregrine_repo/rl/inc8_warmstart.py        (NEW)
#     /scratch/network/fl3689/peregrine_repo/rl/peregrine_train_inc8.py  (the +init_from wiring)
# (e.g. via the adroit-connector upload, or scp from the laptop). The diffaero conda env is pre-built.
#
# RUNTAG passthrough (no-clobber): submit with --export=ALL,CKPT=<dir>,RUNTAG=<tag> (e.g. RUNTAG=ws1).
# EXTRA is appended to BOTH the precheck and every train invocation for a submit-time override without
# editing this file. SEEDS overrides the seed list (default "0 1 2"); each seed gets its OWN explicit
# hydra.run.dir (the S1@3.0 Hydra dir-COLLISION footgun: a shared timestamped dir made two seeds
# bit-identical). SBATCH --output can't expand env vars, so we self-redirect below on RUNTAG.
RUNTAG=${RUNTAG:-}
RUN_SUFFIX="${RUNTAG:+_$RUNTAG}"
GPUUTIL=/scratch/network/fl3689/peregrine_inc8_warmstart${RUN_SUFFIX}_gpu_util.log
exec > /scratch/network/fl3689/peregrine_inc8_warmstart${RUN_SUFFIX}.out 2>&1
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

# ---- ${CKPT} is REQUIRED (the S2 seed-2 converged checkpoint dir; see the locator above) ----
CKPT=${CKPT:?set CKPT=<S2 seed-2 checkpoint dir with actor.pth+critic.pth>  (see header for the locator)}
if [ ! -f "${CKPT}/actor.pth" ] || [ ! -f "${CKPT}/critic.pth" ]; then
  echo "FATAL: CKPT=${CKPT} must contain BOTH actor.pth and critic.pth (a full agent.save dir)." >&2
  exit 2
fi
echo "=== warm-start source checkpoint sidecar (MUST be obs_dim 20, inc8 true) ==="
cat "${CKPT}/actor.json" 2>/dev/null | head -3

SEEDS=${SEEDS:-0 1 2}
NENVS=${NENVS:-2048}
# Fine-tune budget = 4000 updates (== S0/S2/S3). The S3 cliff hit ~step1200; 4000 runs well past it so a
# clean (no-cliff) warm-start is unambiguous and the centering plateau is visible. Throughput ~4000
# updates/~40 min @ 2048 envs; 3 seeds serial ~2-2.5 h compute; --time=08:00:00 covers startup/export/IO
# per seed + queue slack with margin.
NUPD=${NUPD:-4000}
TRAIN=/scratch/network/fl3689/peregrine_repo/rl/peregrine_train_inc8.py
TBTRACE=/scratch/network/fl3689/peregrine_repo/rl/inc8_tb_trace.py

# COMMON == the EXACT S0/S2-convergent config (rl/peregrine_inc8_s0.sbatch lines 71-83, VERBATIM):
# inc7-class measured-plant DR (map-ON) + contact-true geometry + BSR3 + arm A + band-pass perception
# window + the look-at primitive. lookat_g_yaw/g_pitch are SET here (so the per-run sign fix below uses
# ++ DOUBLE-plus to OVERRIDE them); rw_centering / lookat_warmup_updates / init_from are NOT here (so
# they are + SINGLE-plus new keys per run).
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

# WARM-START overrides applied on top of COMMON, for BOTH precheck and the seed runs:
#   +init_from=${CKPT}             ROOT-level new key (training-loop concern, like seed/n_updates/
#                                  save_freq -- diffaero has NO `train` group; single-plus).
#   ++env.lookat_g_yaw=-3.0        empirical S0 yaw sign (sbatch default +3.0 is INVERTED) -- DOUBLE-plus
#   ++env.lookat_g_pitch=3.0       empirical S2 pitch sign (OPPOSITE yaw) -- DOUBLE-plus (both in COMMON)
#   +env.lookat_warmup_updates=0   warm-start holds full look-at gain from update 0 (no re-ramp) -- single
#   +env.rw_centering=1.0          the centering reward under test (S3's lever; NOT in COMMON) -- single
WS_OVERRIDES=(
  +init_from=${CKPT}
  ++env.lookat_g_yaw=-3.0 ++env.lookat_g_pitch=3.0
  +env.lookat_warmup_updates=0
  +env.rw_centering=1.0
)

# ---- GPU-util sampler (background): saturation check, every 20 s ----
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 20 > ${GPUUTIL} 2>&1 &
SMI_PID=$!

# ---- STAGE A: FAST PRECHECK (3 updates, seed 0) -- catch a bad CKPT / load-wiring / dim / EXTRA bug
# BEFORE the long seed runs. obs_dim 20, SYMMETRIC critic (obs-input 20; NOT the get_state 36 -- the
# active GuardedPPO critic sees the 20-dim obs), look-at compose, warm-start LOAD, no NaN. ----
echo "=== inc8 WARM-START PRECHECK: load ${CKPT} + 3 updates (obs_dim 20, symmetric critic, no NaN) ==="
python ${TRAIN} "${COMMON[@]}" "${WS_OVERRIDES[@]}" \
  hydra.run.dir=/scratch/network/fl3689/diffaero/outputs/train/warmstart_precheck${RUN_SUFFIX} \
  n_envs=512 n_updates=3 save_freq=1000 log_freq=1 seed=0 \
  runname=inc8_warmstart_precheck${RUN_SUFFIX} ${EXTRA:-}
echo "PRECHECK_RC=$?"

# ---- STAGE B: the WARM-START FINE-TUNE (3 seeds, arm A, 4000 updates, 2048 envs) ----
# Per-seed EXPLICIT hydra.run.dir (the S1@3.0 collision fix): each seed writes to its own dir so two
# seeds can never share a timestamped dir and come out bit-identical.
for SEED in ${SEEDS}; do
  RUNNAME=inc8_warmstart_seed${SEED}${RUN_SUFFIX}
  RUNDIR=/scratch/network/fl3689/diffaero/outputs/train/${RUNNAME}
  echo "=== inc8 WARM-START seed ${SEED}: ${NUPD} updates, ${NENVS} envs, init_from=${CKPT} ==="
  python ${TRAIN} "${COMMON[@]}" "${WS_OVERRIDES[@]}" \
    hydra.run.dir=${RUNDIR} \
    n_envs=${NENVS} n_updates=${NUPD} save_freq=50 log_freq=10 \
    seed=${SEED} runname=${RUNNAME} ${EXTRA:-}
  echo "WARMSTART_SEED${SEED}_RC=$?"

  echo "=== inc8 WARM-START seed ${SEED} SIGNAL TRACE (READ value_loss @~step1200 FIRST -- no cliff; "
  echo "    then centering rises + terminal_pointing held; estim_err is FLOORED, judge physical sigma_p0) ==="
  RUNDECOR=$(ls -dt ${RUNDIR}/*${RUNNAME}__* 2>/dev/null | head -1)
  RUN=${RUNDECOR:-$RUNDIR}
  echo "RUN=$RUN"
  python ${TBTRACE} "$RUN" --stride-updates 100
  echo "=== seed ${SEED} checkpoint sidecar (obs_dim 20, inc8 true, r5_arm A) ==="
  find "$RUN" -name actor.json -exec cat {} \; 2>/dev/null | head -3
done

kill ${SMI_PID} 2>/dev/null
echo "=== GPU utilization summary (must be HIGH -- saturation check) ==="
awk -F, 'NR>1 && $2 ~ /%/ {gsub(/ %/,"",$2); s+=$2; n++; if($2+0>mx)mx=$2} END{if(n)printf "mean_util=%.1f%%  max_util=%.0f%%  samples=%d\n", s/n, mx, n}' \
  ${GPUUTIL}
echo "INC8_WARMSTART_DONE"

# =====================================================================================================
# LAUNCH (from the Adroit LOGIN node, after the PRE-SYNC above; the connector submits to SLURM):
#   1. locate the checkpoint:
#        CKPT=$(ls -dt /scratch/network/fl3689/diffaero/outputs/train/*/*/*inc8_s0_yawprim_seed2_s2full_s2__*/periodic | head -1)
#        cat "$CKPT/actor.json"          # confirm obs_dim 20, inc8 true
#   2. submit (3 seeds, one job):
#        sbatch --export=ALL,CKPT="$CKPT",RUNTAG=ws1 /scratch/network/fl3689/peregrine_repo/rl/peregrine_inc8_warmstart.sbatch
#      OR, for 3 SEPARATE jobs (more robust -- one seed's crash can't kill the others):
#        for s in 0 1 2; do sbatch --export=ALL,CKPT="$CKPT",SEEDS="$s",RUNTAG=ws1_s$s \
#          /scratch/network/fl3689/peregrine_repo/rl/peregrine_inc8_warmstart.sbatch; done
# =====================================================================================================
