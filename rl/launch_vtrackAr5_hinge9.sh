#!/bin/bash
# =====================================================================================================
# vtrackAr5_hinge9 -- the VELOCITY-HINGE BRACKET (1 seed, seed 0). IDENTICAL to vtrackAr5 seed0 EXCEPT
# the TRUE-speed hinge is left at the OLD wide regime: rw_v_cap_soft=9.0 rw_v_cap_hard=12.0 (vs the
# vtrackAr5 5 m/s regime soft=4.0 / hard=5.5). Everything else -- the rs0 reversions, noise-anneal, and
# the Stage-1 infra fixes (ckpt-select on n_passed_gates, critic-grad-clip, DET n_passed_gates) -- is
# HELD, so any progress delta vs vtrackAr5 seed0 is attributable to the SPEED REGIME alone (adversarial-
# review MUST_FIX #3/BUNDLE_CRITIQUE: bracket the hinge so it doesn't confound the infra fixes).
#
# Read ALL the PRE-LAUNCH FLAGS in launch_vtrackAr5.sh (trackA scaffolding not in git; rw_cross_zero_m
# precedence; finish_time key; return_norm deferred; deploy governor separate) -- they apply here too.
# =====================================================================================================
set -euo pipefail

STAGE=dual_gate_fullstack_floor_pef_trackA
UPD=${UPD:-12000}
SEED=${SEED:-0}
WARM=${WARM:-/scratch/network/fl3689/diffaero/outputs/train/ego_dual_gate_fullstack_floor_pef_trackA_seed0_vtrackAw0/checkpoints}
SBATCH_FILE="$(cd "$(dirname "$0")" && pwd)/peregrine_vq2_ego.sbatch"

# ---- m8b BASE tokens (kept UNCHANGED) ----
BASE_M8B="++env.ego_blur_gate=false ++env.spin_abort_anneal=true ++env.spin_abort_scale_start=2.6 \
++env.spin_abort_hold_frac=0.25 ++env.ego_est_dt_ticks_hi=1 ++warmstart_reset_logstd=true \
++warmstart_reset_logstd_std=0.18 ++env.att_cap_anneal=false ++env.yaw_dither_anneal=false \
++env.course_n_gates=8 ++env.course_lateral_offset_lo=2.0 ++env.course_lateral_offset_hi=6.0 \
++env.ego_coarse_map_live=true ++env.ego_yaw_cmd_clamp_rad_s=0.7"

# ---- vtrackAr5 changes, but with the WIDE hinge (soft=9.0 hard=12.0) -- the ONLY delta vs vtrackAr5 ----
VTRACKAR5_HINGE9="++env.rw_v_cap=3.0 ++env.rw_v_cap_soft=9.0 ++env.rw_v_cap_hard=12.0 \
++env.rw_cross_zero_m=0.75 ++env.rw_exit_align=0.0 ++env.rw_perception=0.02 ++env.rw_perception_next=0.0 \
++env.rw_centering=0.0 ++env.rw_yaw_dither=0.125 ++env.rw_finish_time=0.25 \
++algo.noise_anneal=true ++algo.noise_std_hold=0.6 ++algo.noise_std_floor=0.05 ++algo.noise_hold_frac=0.5 \
++critic_grad_clip=10 ++ckpt_select_metric=n_passed_gates ++eval_det_steps=600"

export EXTRA="${BASE_M8B} ${VTRACKAR5_HINGE9} +init_from=${WARM}"
export STAGES="${STAGE}"
export PRECHECK=1
export SEED
export RUNTAG="vtrackAr5_hinge9"
export "UPD_${STAGE}=${UPD}"

echo "=== launching vtrackAr5_hinge9 (WIDE hinge soft=9/hard=12) SEED=${SEED} STAGE=${STAGE} UPD=${UPD} ==="
sbatch --gres=gpu:nvidia_a100:1 --export=ALL "${SBATCH_FILE}"
echo "=== hinge9 submitted. Compare DET n_passed_gates vs vtrackAr5 seed0 to attribute the speed regime. ==="
