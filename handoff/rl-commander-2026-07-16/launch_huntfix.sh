#!/bin/bash
# =====================================================================================================
# huntfix = vpef8nc + THE MISSING ANTI-DITHER. Extend the champion 2-gate smooth-yaw policy vpeffs0 to 8
# gates WITHOUT the yaw limit-cycle by re-adding the ANTI-DITHER yaw-jerk penalty that the base `_pef`
# stage OMITS. vpef8nc (= launch_vpefcap.sh at RWVCAP=0: ffs0-warm, 8-gate, NO velocity cap, all
# BASE_VPEFFS0) is the BEST 8-gate flyer to date but rides the +-clamp yaw rail (the intrinsic aw0/trackA
# oscillation is UNMASKED once the drone is on an 8-gate course). aw1 proved the CURE: the annealed
# anti-dither yaw-jerk penalty rw_yaw_dither, CALIBRATED for the yaw clamp, kills the rail-flip while
# preserving the yaw->gate-in-view->altitude coupling (it penalises yaw JITTER only, not gate-tracking yaw).
#
# WHY THIS IS MINIMAL (reuse, don't reinvent):
#   * SAME stage as vpef8nc: `dual_gate_fullstack_floor_pef` (NOT the trackA stage). The _pef stage has NO
#     att-cap and NO yaw_dither -> roll/speed stay FREE (Fengyou: fence only the axis the course doesn't
#     use; roll/speed/velocity-cap = NO). We add ONLY the yaw_dither arm, nothing that touches roll/pitch.
#   * SAME BASE_VPEFFS0 tokens as vpefcap (yaw clamp 0.7, ticks_hi=1, blur off, spin-abort anneal, logstd
#     reset). SAME 8-gate/reward/infra block as vpef8nc, MINUS the velocity cap (RWVCAP hard-0) and MINUS
#     the recovery-reward experiment (roll_recover / cross_level default 0 = OFF -> we simply omit those
#     tokens; that also sidesteps the recovery_anneal "both-zero" guard that vpefcap's block would trip).
#   * The ONLY additions vs vpef8nc = aw1's yaw_dither discipline (verbatim), appended as `++env.*` in EXTRA.
#
# THE YAW-DITHER + CLAMP CALIBRATION (the flip-tax footgun -- a MISMATCHED clamp makes it 4x off):
#   penalty for a full +-clamp RAIL FLIP = rw_yaw_dither * (2*clamp)^2. aw1 holds this INVARIANT while
#   lifting the clamp 0.35 -> 0.7: it scales yaw_dither 0.5 -> 0.125 in LOCKSTEP (4x down as clamp 2x up):
#       clamp 0.35, yd 0.5   -> 0.5   * (0.70)^2 = 0.245
#       clamp 0.70, yd 0.125 -> 0.125 * (1.40)^2 = 0.245   (IDENTICAL flip-tax)
#   vpeffs0 / vpef8nc ALREADY run at clamp 0.7 (BASE_VPEFFS0). So the MATCHED anti-dither target = 0.125,
#   ANNEALED in from 0 (yaw_dither_start=0.0, hold_frac=0.3) exactly as aw1. (The penalty is on the APPLIED
#   post-clamp yaw command, so this calibration is exact.)
#
# ARMS (each a SEPARATE SLURM job so one seed's crash can't kill the others):
#   A  (primary), seeds 0,1 : vpef8nc + aw1 yaw_dither (0.125, annealed) at clamp 0.7. TAG=huntfixA_s<seed>
#   C  (hardened), seed 0   : arm A + ++dynamics.dr_latency_max_steps=5 (extend control-latency DR to cover
#                             deploy's ~100 ms / 5-step tail; dr_latency_min stays 1). TAG=huntfixC_s0
#
# YAW METRIC OVERNIGHT (see BLOCK at the bottom): ++eval_yaw_log=true makes the end-of-stage DET evals
#   (final + promoted-best) ALSO print YAW_EVAL[...] signflips_per_s + roll_swing. For a during-run TREND,
#   point the existing ++rollout_only harness at the run's best_npg/ (and periodic/) snapshots post-hoc.
#
# SMOKE:  UPD=100 SAVEFREQ=25 SEEDS=0 ARMS=A bash launch_huntfix.sh   (wiring + ffs0 warm + yaw-dither ramp)
# LAUNCH: bash launch_huntfix.sh                                       (arm A seeds 0,1 + arm C seed 0)
# =====================================================================================================
set -euo pipefail
STAGE=dual_gate_fullstack_floor_pef
UPD=${UPD:-12000}
SAVEFREQ=${SAVEFREQ:-500}
NGATES=${NGATES:-8}
# --- anti-dither (aw1 discipline; MATCHED to the 0.7 clamp in BASE_VPEFFS0 -> target 0.125, annealed) ---
YAWDITHER=${YAWDITHER:-0.125}          # matched anti-dither target for clamp 0.7 (aw1's ++ value)
YDSTART=${YDSTART:-0.0}                # ramp-in start scale (INERT at birth, == aw1)
YDHOLDFRAC=${YDHOLDFRAC:-0.3}          # END-HOLD at full for the last 30% (== aw1)
# --- arm C control-latency DR tail (deploy ~100 ms / 5-step); min stays 1 (BASE default) ---
DRLATMAX=${DRLATMAX:-5}
# --- which arms + seeds (arm C is a single-seed hardened variant of arm A) ---
ARMS=${ARMS:-"A C"}
SEEDS_A=${SEEDS_A:-"0 1"}
SEEDS_C=${SEEDS_C:-"0"}
WARM=${WARM:-/scratch/network/fl3689/diffaero/outputs/train/ego_dual_gate_fullstack_floor_pef_seed0_vpeffs0/checkpoints}
SBATCH_FILE="$(cd "$(dirname "$0")" && pwd)/peregrine_vq2_ego.sbatch"

# ---- vpeffs0 base recipe (its proven EXTRA: yaw clamp 0.7, ticks_hi=1, blur off, spin-abort anneal,
#      logstd reset) -- VERBATIM from launch_vpefcap.sh's BASE_VPEFFS0 ----
BASE_VPEFFS0="++env.ego_blur_gate=false ++env.ego_yaw_cmd_clamp_rad_s=0.7 ++env.spin_abort_anneal=true \
++env.spin_abort_scale_start=2.6 ++env.spin_abort_hold_frac=0.25 ++env.ego_est_dt_ticks_hi=1 \
++warmstart_reset_logstd=true ++warmstart_reset_logstd_std=0.18"

# ---- vpef8nc 8-gate/reward/infra block = launch_vpefcap.sh's VPEFCAP with rw_v_cap HARD-0 (no velocity
#      cap) and the recovery tokens DROPPED (roll_recover/cross_level default 0 = OFF). Plus ++eval_yaw_log
#      =true so the end-of-stage DET evals emit YAW_EVAL. ----
VPEF8NC="++env.course_n_gates=${NGATES} ++env.rw_v_cap=0 \
++env.rw_cross_zero_m=0.75 ++env.rw_finish_time=0.25 \
++algo.noise_anneal=true ++algo.noise_std_hold=0.6 ++algo.noise_std_floor=0.05 ++algo.noise_hold_frac=0.5 \
++critic_grad_clip=10 ++ckpt_select_metric=n_passed_gates ++eval_det_steps=600 ++eval_yaw_log=true \
++save_freq=${SAVEFREQ}"

# ---- THE MISSING ANTI-DITHER (aw1 verbatim; the _pef stage renders NONE of these keys, so all four are
#      new `++env.*` adds calibrated for the 0.7 clamp above) ----
ANTIDITHER="++env.rw_yaw_dither=${YAWDITHER} ++env.yaw_dither_anneal=true \
++env.yaw_dither_start=${YDSTART} ++env.yaw_dither_hold_frac=${YDHOLDFRAC}"

export STAGES="${STAGE}"
export PRECHECK=1
export "UPD_${STAGE}=${UPD}"

launch_seed () {   # $1=ARM label  $2=SEED  $3=extra-per-arm overrides
  local ARM="$1" SEED="$2" ARMEXTRA="$3"
  export SEED
  export RUNTAG="huntfix${ARM}_s${SEED}"
  export EXTRA="${BASE_VPEFFS0} ${VPEF8NC} ${ANTIDITHER} ${ARMEXTRA} +init_from=${WARM}"
  echo "=== launching huntfix${ARM} SEED=${SEED} STAGE=${STAGE} UPD=${UPD} ngates=${NGATES} \
save_freq=${SAVEFREQ} yaw_dither=${YAWDITHER}(clamp0.7,annealed) warm=vpeffs0 armextra='${ARMEXTRA}' ==="
  sbatch --gres=gpu:nvidia_a100:1 --export=ALL "${SBATCH_FILE}"
}

for ARM in ${ARMS}; do
  case "${ARM}" in
    A) for s in ${SEEDS_A}; do launch_seed A "${s}" ""; done ;;
    C) for s in ${SEEDS_C}; do launch_seed C "${s}" "++dynamics.dr_latency_max_steps=${DRLATMAX}"; done ;;
    *) echo "!! unknown ARM '${ARM}' (expected A and/or C) -- skipping" ;;
  esac
done

echo "=== huntfix submitted [arms='${ARMS}' A-seeds='${SEEDS_A}' C-seed='${SEEDS_C}']. SELECT the deploy"
echo "    seed on highest DETERMINISTIC n_passed_gates (DET_EVAL lines), tiebreak LOWEST yaw oscillation"
echo "    (YAW_EVAL signflips_per_s) + lowest roll_swing -- NEVER on value. GOAL: >= vpef8nc's DET n_passed"
echo "    5.33 with signflips_per_s driven toward ~0 (rail-flip gone)."
