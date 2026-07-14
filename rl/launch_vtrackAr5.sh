#!/bin/bash
# =====================================================================================================
# vtrackAr5 -- clean multi-seed ~5 m/s Track-A base (Stage 1). Launch SEEDS 0,1,2 IDENTICAL as separate
# SLURM jobs (one seed's crash can't kill the others). Warm from the vtrackAw0 lineage; UPD=12000;
# PRECHECK=1 (the sbatch runs a 3-update 512-env wiring smoke first and aborts nonzero on any crash).
#
# WHAT THIS ARM CHANGES vs m8b (the base tokens are kept verbatim; see EXTRA below): the layered ~5 m/s
# design (TRUE-speed hinge rw_v_cap 3 / soft 4 / hard 5.5 + rw_finish_time 0.25), the rs0 reward
# REVERSIONS (exit_align/perception_next->0, centering->0, perception 0.02, yaw_dither 0.125), the
# noise-anneal ceiling ON, and the Stage-1 INFRA fixes wired in peregrine_train_ego.py:
#   * ++ckpt_select_metric=n_passed_gates  -> rolling-best-on-n_passed_gates ckpt selection + a
#     post-train DET promotion of the PEAK over checkpoints/ (BEST_CKPT / BEST_CKPT_PROMOTED lines).
#   * ++critic_grad_clip=10                 -> critic-only grad-norm clip at the optim.step wrap.
#   * ++eval_det_steps=600                  -> DET_EVAL now aggregates n_passed_gates over enough
#                                              8-gate episodes for a low-variance read.
#
# ============================ PRE-LAUNCH FLAGS (READ BEFORE sbatch) ============================
# 1. TRACK-A SCAFFOLDING IS *NOT* IN THE GIT REPO (chaum). Confirmed absent from every branch:
#      - the curriculum stage `dual_gate_fullstack_floor_pef_trackA`, and
#      - the reward knobs `rw_v_cap` / `rw_v_cap_soft` / `rw_v_cap_hard` / `rw_perception_next` /
#        `rw_yaw_dither`.
#    They must already exist in the Adroit /scratch/network/fl3689/peregrine_repo checkout (the m8b/rs0
#    runs used them). If they do NOT, this launch FAILS ("unknown stage") or silently NO-OPs the velocity
#    hinge + those reward tokens (Hydra ++ adds the cfg key but EgoRewardWeights.from_cfg only reads keys
#    that map to a dataclass field). VERIFY the Adroit checkout carries the trackA scaffolding first.
#    🚩 SYNC HAZARD: when syncing chaum's rl/*.py to Adroit, MERGE ego_reward.py + vq2_ego_curriculum.py
#    (do NOT blind-overwrite) or you will CLOBBER the Adroit-only trackA scaffolding. peregrine_train_ego.py
#    (the Stage-1 infra) is safe to sync.
# 2. cross_zero_m: the effective 4.0 comes from the curriculum stage dict's "rw_cross_zero_m": 4.0.
#    EgoRewardWeights.from_cfg reads `rw_cross_zero_m` with PRECEDENCE over `cross_zero_m`, so the override
#    below is `++env.rw_cross_zero_m=0.75` (a bare `++env.cross_zero_m=0.75` would be INERT). Confirm the
#    _trackA stage has NO cross_zero_anneal (if it does, also pass ++env.cross_zero_anneal=false, else the
#    per-update anneal re-mutates cross_zero_m 4.0->0.75 and overrides this static value).
# 3. finish_time: consumed by the T4 finish term; the correct key is `++env.rw_finish_time=0.25` (rw_
#    prefix), included below. Verify it maps to weights.finish_time on the PRECHECK.
# 4. return_norm (PopArt) is DEFERRED -- NOT set here (Stage 1 relies on the critic-only grad-clip). Do
#    not add ++return_norm.
# 5. The deploy-side speed governor (fly_rl.py) is OUT of this arm; it ships separately on the deploy
#    branch and needs Fengyou's vel-bias log (k) to size v_gov_soft=5*k before it can hold TRUE ~5 m/s.
#
# 🚩 PRECHECK ASSERTIONS (the 3-update sbatch smoke is a WIRING check only; run a ~50-100 update smoke
#    with save_freq~=25 to also see BEST_CKPT): the log must show
#      [ckpt-select] ON ... ; BEST_CKPT[...] lines (a save boundary was hit + buffer populated);
#      [critic-grad-clip] ON ; entropy/noise-anneal live; DET_EVAL[...] n_passed_gates=... prints.
# =====================================================================================================
set -euo pipefail

STAGE=dual_gate_fullstack_floor_pef_trackA
UPD=${UPD:-12000}
SEEDS=${SEEDS:-"0 1 2"}
WARM=${WARM:-/scratch/network/fl3689/diffaero/outputs/train/ego_dual_gate_fullstack_floor_pef_trackA_seed0_vtrackAw0/checkpoints}
SBATCH_FILE="$(cd "$(dirname "$0")" && pwd)/peregrine_vq2_ego.sbatch"

# ---- m8b BASE tokens (kept UNCHANGED from the m8b recipe) ----
BASE_M8B="++env.ego_blur_gate=false ++env.spin_abort_anneal=true ++env.spin_abort_scale_start=2.6 \
++env.spin_abort_hold_frac=0.25 ++env.ego_est_dt_ticks_hi=1 ++warmstart_reset_logstd=true \
++warmstart_reset_logstd_std=0.18 ++env.att_cap_anneal=false ++env.yaw_dither_anneal=false \
++env.course_n_gates=8 ++env.course_lateral_offset_lo=2.0 ++env.course_lateral_offset_hi=6.0 \
++env.ego_coarse_map_live=true ++env.ego_yaw_cmd_clamp_rad_s=0.7"

# ---- vtrackAr5 CHANGES (the ~5 m/s hinge + rs0 reversions + noise-anneal + Stage-1 infra flags) ----
# NOTE: rw_v_cap_soft=4.0 / hard=5.5 is the ~5 m/s regime; the hinge9 bracket flips ONLY these two.
VTRACKAR5="++env.rw_v_cap=3.0 ++env.rw_v_cap_soft=4.0 ++env.rw_v_cap_hard=5.5 \
++env.rw_cross_zero_m=0.75 ++env.rw_exit_align=0.0 ++env.rw_perception=0.02 ++env.rw_perception_next=0.0 \
++env.rw_centering=0.0 ++env.rw_yaw_dither=0.125 ++env.rw_finish_time=0.25 \
++algo.noise_anneal=true ++algo.noise_std_hold=0.6 ++algo.noise_std_floor=0.05 ++algo.noise_hold_frac=0.5 \
++critic_grad_clip=10 ++ckpt_select_metric=n_passed_gates ++eval_det_steps=600"

export EXTRA="${BASE_M8B} ${VTRACKAR5} +init_from=${WARM}"
export STAGES="${STAGE}"
export PRECHECK=1
export "UPD_${STAGE}=${UPD}"

for SEED in ${SEEDS}; do
  export SEED
  export RUNTAG="vtrackAr5_s${SEED}"         # per-seed suffix -> distinct .out / rundir / runname
  echo "=== launching vtrackAr5 SEED=${SEED} STAGE=${STAGE} UPD=${UPD} warm=${WARM} ==="
  sbatch --gres=gpu:nvidia_a100:1 --export=ALL "${SBATCH_FILE}"
done

echo "=== vtrackAr5 seeds [${SEEDS}] submitted. SELECT the deploy seed on highest DETERMINISTIC"
echo "    n_passed_gates (DET_EVAL lines), tiebreak lowest exit_frame / lowest hazard -- NEVER on value."
