"""Tests for rl/vq2_ego_curriculum.py -- the VQ2 EGOCENTRIC (inc9) curriculum ladder (DESIGN.md §D/§E).

Config is pure DATA + a renderer, so this unit-tests on the laptop with no torch/diffaero. Coverage:
  (1) the ladder is well-formed: 4 stages, correct STAGE_ORDER, blackout_pass DELETED, ego/appo/gamma
      wired on EVERY stage.
  (2) the B2b CATASTROPHIC-FORGETTING scoping: the HARD-stage reward knobs (rw_passage bump, exit_align
      ON) are NOT in _COMMON and do NOT hit single_gate or handoff_drill; the easy stages stay on the
      clean champion reward.
  (3) render_overrides emits valid hydra tokens (bools lowercase; _raw verbatim; unknown stage raises).

Run from repo ROOT:
    .venv\\Scripts\\python.exe -m pytest tests/test_vq2_ego_curriculum.py -q
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rl"))

import vq2_ego_curriculum as C                                             # noqa: E402


# ================================================================================================
# (1) the ladder is well-formed.
# ================================================================================================
def test_stage_order_is_the_four_ego_stages_no_blackout():
    assert C.STAGE_ORDER == ("single_gate", "handoff_drill", "dual_gate_full", "multi_gate")
    assert len(C.STAGE_ORDER) == 4
    # blackout_pass is DELETED (emergent from the keypoint model, DESIGN.md §5.B/§D).
    assert "blackout_pass" not in C.STAGES
    assert "blackout_pass" not in C.STAGE_ORDER
    # every ordered stage is a real STAGES entry.
    for s in C.STAGE_ORDER:
        assert s in C.STAGES


def test_every_stage_wires_ego_appo_path_and_gamma_9975():
    for s in C.STAGE_ORDER:
        d = C.STAGES[s]
        # the egocentric env + standing start (trivial dash removed) on EVERY stage.
        assert d["ego"] is True, s
        assert d["standing_start_frac"] == 1.0, s
        assert d["course_mode"] == "random", s
        # gamma=0.9975 (LOAD-BEARING) on every stage, in the _raw verbatim block (algo.gamma).
        assert d["_raw"]["algo.gamma"] == 0.9975, s
    # NO stage carries any inc8 blackout / estimator-emul knob (deleted for the ego generation).
    for s in C.STAGE_ORDER:
        for k in C.STAGES[s]:
            assert not k.startswith("emul_"), (s, k)
            assert "blackout" not in k, (s, k)


def test_gate_counts_and_spacing_match_design():
    # single_gate = 1 gate; the rest = 2/2/N with spacing 10-20 m (DESIGN.md §D).
    assert C.STAGES["single_gate"]["course_n_gates"] == 1
    assert C.STAGES["handoff_drill"]["course_n_gates"] == 2
    assert C.STAGES["dual_gate_full"]["course_n_gates"] == 2
    assert C.STAGES["multi_gate"]["course_n_gates"] >= 2
    for s in ("handoff_drill", "dual_gate_full", "multi_gate"):
        assert C.STAGES[s]["course_seg_len_lo"] == 10.0, s
        assert C.STAGES[s]["course_seg_len_hi"] == 20.0, s


# ================================================================================================
# (2) THE B2b SCOPING DISCIPLINE: hard-stage knobs NOT in _COMMON / not on easy stages.
# ================================================================================================
def test_common_is_easy_safe_no_hard_stage_pressure():
    """_COMMON (applied to single_gate) must carry NO HARD-STAGE pressure: exit_align OFF (the B2b
    invariant -- a hard-stage knob in _COMMON regressed single_gate 0.82->0.035). The 2026-07-07 reward
    redesign (Fengyou) lives here because it is DISCOVERY-FRIENDLY, not hard-stage: moderate progress
    (2.0), a strong centered-passage (5.0 + per-gate increment), the distance-gated area coupling, a
    closer first gate (10-20 m), forgiving MISS, catastrophic CONTACT (200 -> sprint-and-clip defeated)."""
    assert C._COMMON["rw_exit_align"] == 0.0         # exit-line OFF on easy/discovery stages (hard only)
    assert C._COMMON["rw_progress"] == 2.0           # moderate dense pull (was 6.0 -- over-rewarded rushing)
    assert C._COMMON["rw_passage"] == 5.0            # centered passage BASE: passing >> approaching
    assert C._COMMON["rw_passage_increment"] == 1.0  # per-gate: gate g pays (5 + g) -> deeper = better
    assert C._COMMON["rw_area_dist_ref_m"] == 0.0    # area coupling OFF (reward-audit: taxed the homing)
    # PROGRESS-TO-CENTRE root-cause fix (2026-07-07): dense 3D homing to the gate centre (inc7/Swift),
    # replacing segment-only along-track progress that starved lateral/vertical homing (single_gate 0%).
    assert C._COMMON["rw_progress_to_center"] is True
    # the magnitude-penalty centering is now OFF (its give-up-and-OOB back-fire is superseded by the
    # constructive progress-to-centre homing); kept as a 0-knob, not deleted.
    assert C._COMMON["rw_centering"] == 0.0          # OFF (penalty form back-fired; homing now via progress)
    assert C._COMMON["rw_centering_max_m"] == 2.0    # perpendicular-offset clamp (unused while centering==0)
    assert C._COMMON["course_spawn_dist_lo"] == 10.0 and C._COMMON["course_spawn_dist_hi"] == 20.0
    # TERMINAL EQUALIZATION (Fengyou 2026-07-08): miss raised 8->100 to MATCH contact (100) so bailing wide
    # is no longer a cheap escape that teaches gate-avoidance; the residual gap is only the contact-only
    # banked-progress forfeit (the sprint-and-clip defence).
    assert C._COMMON["rw_terminal_miss"] == 100.0    # MISS == CONTACT base (no cheap bail; anti-gate-avoidance)
    assert C._COMMON["rw_terminal_oob"] == 200.0     # OOB (leaving arena) STAYS discouraged (96% OOB @ 30)
    assert C._COMMON["rw_terminal_progress_scaled"] is True
    assert C._COMMON["rw_terminal_base"] == 100.0    # CONTACT lowered 200->100 (still DQ-scale; miss-parity)


def test_single_gate_varied_gvf_stage():
    """The vector-field (GVF) branch: use_racing_line ON, contouring ON, same varied geometry as the
    aniso/mpcc branches, and the flag renders as a hydra +env token."""
    s = C.STAGES["single_gate_varied_gvf"]
    assert s["use_racing_line"] is True
    assert s["rw_corridor"] == 4.0                    # cross-track contouring pull onto the line
    assert s["rw_progress_to_center"] is False        # s comes from the line arc length (flag ignored)
    assert s["course_n_gates"] == 1
    assert s["course_spawn_dist_lo"] == 8.0 and s["course_spawn_dist_hi"] == 15.0
    assert s["course_spawn_below_g0_lo"] == -6.0 and s["course_spawn_below_g0_hi"] == 6.0
    toks = C.render_overrides("single_gate_varied_gvf")
    assert "+env.use_racing_line=true" in toks        # boolean renders hydra-lowercase
    # terminal equalization inherited from _COMMON
    assert "+env.rw_terminal_miss=100.0" in toks and "+env.rw_terminal_base=100.0" in toks


def test_hard_knobs_absent_from_easy_stages_present_on_hard_stages():
    """single_gate + handoff_drill stay on the clean champion reward (exit_align==0.0, no stage-specific
    reward override); dual_gate_full + multi_gate carry the STAGE-SPECIFIC turning pressure (exit>0). The
    passage weight is now UNIFORM across stages (the _COMMON base-5 + per-gate increment) -- the old flat
    x3 hard-stage passage bump is superseded (it would have UNDERCUT the new base 5)."""
    for s in ("single_gate", "handoff_drill"):
        assert C.STAGES[s]["rw_exit_align"] == 0.0, s   # exit-line OFF on the easy/discovery stages
    for s in ("dual_gate_full", "multi_gate"):
        assert C.STAGES[s]["rw_exit_align"] > 0.0, s    # next-gate exit-line ON (gate-gated -> safe)
    # passage is UNIFORM (base 5 + increment); NO stage carries a flat per-stage passage override.
    for s in C.STAGE_ORDER:
        assert C.STAGES[s]["rw_passage"] == 5.0, s


def test_single_gate_is_pure_common_no_extra_reward_knobs():
    """single_gate must be _COMMON + only course/gamma/max_time -- NO stage-specific reward override,
    so it trains on exactly the reward it validates at ~0.8 before promotion (validate-before-_COMMON)."""
    extra = set(C.STAGES["single_gate"]) - set(C._COMMON) - {"course_n_gates", "_raw"}
    assert extra == set(), extra


def test_single_gate_static_is_a_fixed_offladder_diagnostic():
    """The STATIC diagnostic stage (Fengyou 2026-07-07): a fixed 15 m, level, dead-ahead gate that does NOT
    move between episodes -- removes all course variance to isolate the gross control failure. It is
    OFF-LADDER (not in STAGE_ORDER); run standalone via STAGES=single_gate_static."""
    assert "single_gate_static" not in C.STAGE_ORDER                # off-ladder, standalone only
    s = C.STAGES["single_gate_static"]
    assert s["course_n_gates"] == 1
    assert s["course_spawn_dist_lo"] == s["course_spawn_dist_hi"] == 15.0   # gate does NOT move (fixed 15 m)
    assert s["course_drop_lo"] == s["course_drop_hi"] == 0.0                # level (no climb) -- isolate
    assert s["ego"] is True and s["_raw"]["algo.gamma"] == C._GAMMA
    # it renders valid tokens (fixed spawn-distance emitted as a +env. override).
    toks = C.render_overrides("single_gate_static")
    assert "+env.course_spawn_dist_lo=15.0" in toks and "+env.course_spawn_dist_hi=15.0" in toks
    assert "+env.course_drop_lo=0.0" in toks


# ================================================================================================
# (3) render_overrides emits valid hydra tokens.
# ================================================================================================
def test_render_overrides_tokens_and_bool_formatting():
    toks = C.render_overrides("dual_gate_full")
    # ego bool renders lowercase (hydra-true), prefixed +env.
    assert "+env.ego=true" in toks
    assert "+env.rw_terminal_progress_scaled=true" in toks
    # course + reward knobs are +env. prefixed.
    assert "+env.course_n_gates=2" in toks
    assert "+env.rw_passage=5.0" in toks          # uniform base-5 passage (hard-stage x3 bump removed)
    assert "+env.rw_exit_align=0.1" in toks
    assert "+env.course_spawn_dist_lo=10.0" in toks   # closer first gate (10-20 m) on every stage
    # _raw entries are VERBATIM (no +): algo.gamma / env.max_time.
    assert "algo.gamma=0.9975" in toks
    assert any(t.startswith("env.max_time=") for t in toks)
    # NO token double-prefixes the raw keys.
    assert not any(t.startswith("+algo.gamma") for t in toks)


def test_render_overrides_unknown_stage_raises():
    with pytest.raises(ValueError):
        C.render_overrides("does_not_exist")


def test_course_sampler_keys_documented():
    # the sampler-key contract (matches the inc8 curriculum convention; + the spawn-distance pair, the
    # fixed spawn-heading scalar, the gate-0 HEIGHT band added 2026-07-08 for the varied-gate stage, and
    # the gates-above-spawn floor clearance added 2026-07-10 for the floor_at_spawn stages).
    assert C.COURSE_SAMPLER_KEYS == ("course_n_gates", "course_seg_len_lo", "course_seg_len_hi",
                                     "course_drop_lo", "course_drop_hi",
                                     "course_spawn_dist_lo", "course_spawn_dist_hi",
                                     "course_spawn_below_g0_lo", "course_spawn_below_g0_hi",
                                     "course_spawn_heading", "course_spawn_yaw_jitter",
                                     "course_gates_above_spawn",
                                     "course_g1_out_of_fov_lo", "course_g1_out_of_fov_hi",
                                     "course_min_pair_dist_m")


def test_single_gate_varied_varies_position_and_height_offladder():
    """The varied-gate stage: OFF-LADDER, ONE gate at VARIABLE distance (10-20 m) AND VARIABLE height
    (+-6 m via spawn_below_g0) so it lands at different FOV positions each episode -- forces input-use +
    un-buries the vertical geometrically. Isotropic reward inherited (a clean geometry-only test)."""
    assert "single_gate_varied" not in C.STAGE_ORDER              # off-ladder, standalone only
    s = C.STAGES["single_gate_varied"]
    assert s["course_n_gates"] == 1
    assert s["course_spawn_dist_lo"] == 8.0 and s["course_spawn_dist_hi"] == 15.0    # varied range (8-15 m)
    assert s["course_spawn_below_g0_lo"] == -6.0 and s["course_spawn_below_g0_hi"] == 6.0  # +-6 m height
    assert s["rw_progress_vert_weight"] == 1.0        # ISOTROPIC (inherited) -> geometry-only test
    toks = C.render_overrides("single_gate_varied")
    assert "+env.course_spawn_below_g0_lo=-6.0" in toks and "+env.course_spawn_below_g0_hi=6.0" in toks


def test_varied_combo_stages_race_coupled_vs_decoupled():
    """The two varied-gate reward-form contenders (adversarial 2026-07-08): COUPLED aniso vs DECOUPLED
    MPCC, both OFF-LADDER, both on the SAME varied geometry (dist 8-15 m, height +-6 m)."""
    for s in ("single_gate_varied_aniso", "single_gate_varied_mpcc"):
        assert s not in C.STAGE_ORDER
        d = C.STAGES[s]
        assert d["course_spawn_dist_lo"] == 8.0 and d["course_spawn_dist_hi"] == 15.0
        assert d["course_spawn_below_g0_lo"] == -6.0 and d["course_spawn_below_g0_hi"] == 6.0
    # COUPLED aniso: isotropic-base progress ON + moderate vertical weight.
    a = C.STAGES["single_gate_varied_aniso"]
    assert a["rw_progress_to_center"] is True and a["rw_progress_vert_weight"] == 6.0
    # DECOUPLED MPCC: along-track LAG (progress_to_center False) + separate corridor k raised from 2 -> 4.
    m = C.STAGES["single_gate_varied_mpcc"]
    assert m["rw_progress_to_center"] is False and m["rw_corridor"] == 4.0
    assert "+env.rw_corridor=4.0" in C.render_overrides("single_gate_varied_mpcc")


def test_common_pins_fixed_spawn_heading():
    # Fengyou 2026-07-07: the egocentric obs is heading-invariant, so segment-0 heading is pinned (0.0)
    # to de-circle the world-frame layout without changing the egocentric training distribution.
    assert C._COMMON["course_spawn_heading"] == 0.0


# ================================================================================================
# HOVER-HOLD probe stage (Fengyou greenlight 2026-07-08; the H1-vs-H2 disambiguator).
# ================================================================================================
def test_common_altitude_hold_default_off():
    # the hover-hold bonus is OFF on every real stage (turned ON only by the hover_hold probe stage).
    assert C._COMMON["rw_altitude_hold"] == 0.0
    # every ordered ladder stage inherits _COMMON's OFF (no stage accidentally enables the probe bonus).
    for s in C.STAGE_ORDER:
        assert C.STAGES[s]["rw_altitude_hold"] == 0.0, s


def test_hover_hold_is_a_fixed_offladder_altitude_probe():
    """The hover_hold probe: OFF-LADDER, a fixed 15 m LEVEL gate, and the reward is the altitude-hold
    bonus ONLY -- every gate-homing term (progress/passage/increment/area/centering/exit) is zeroed, so
    holding spawn altitude is the unique optimum with no competing forward objective."""
    assert "hover_hold" not in C.STAGE_ORDER                       # off-ladder, standalone only
    s = C.STAGES["hover_hold"]
    assert s["course_n_gates"] == 1
    assert s["course_spawn_dist_lo"] == s["course_spawn_dist_hi"] == 15.0   # fixed 15 m
    assert s["course_drop_lo"] == s["course_drop_hi"] == 0.0               # level
    # the probe bonus is ON ...
    assert s["rw_altitude_hold"] == 1.0
    assert s["rw_altitude_hold_band_m"] == 8.0
    # ... and EVERY gate-homing term is OFF (pure altitude-hold, no forward pull).
    for k in ("rw_progress", "rw_passage", "rw_passage_increment", "rw_area_dist_ref_m",
              "rw_centering", "rw_exit_align"):
        assert s[k] == 0.0, k
    assert s["ego"] is True and s["_raw"]["algo.gamma"] == C._GAMMA


def test_hover_hold_renders_valid_tokens():
    toks = C.render_overrides("hover_hold")
    assert "+env.rw_altitude_hold=1.0" in toks
    assert "+env.rw_progress=0.0" in toks              # forward homing zeroed
    assert "+env.rw_passage=0.0" in toks
    assert "+env.course_spawn_dist_lo=15.0" in toks
    assert "algo.gamma=0.9975" in toks                # _raw verbatim (no +)


# ================================================================================================
# MPCC-clean contouring lever (Fengyou greenlight 2026-07-08; hover-hold-confirmed).
# ================================================================================================
def test_common_corridor_default_off():
    # MPCC contouring OFF on every real stage (turned ON only by the single_gate_static_mpcc lever stage).
    assert C._COMMON["rw_corridor"] == 0.0
    for s in C.STAGE_ORDER:
        assert C.STAGES[s]["rw_corridor"] == 0.0, s
    # the ladder stays on the isotropic 3D homing (progress_to_center) -- the MPCC lag/contouring split is
    # the diagnostic lever only, not (yet) baked into the ladder.
    assert C._COMMON["rw_progress_to_center"] is True


def test_single_gate_static_mpcc_is_lag_plus_contouring_offladder():
    """The MPCC lever: OFF-LADDER, SAME fixed 15 m level gate as single_gate_static (apples-to-apples box-
    exit vs the 100%-floor baseline), progress switched to ALONG-TRACK LAG (progress_to_center False) with
    the PBRS CONTOURING term ON to supply the perpendicular homing."""
    assert "single_gate_static_mpcc" not in C.STAGE_ORDER              # off-ladder, standalone only
    s = C.STAGES["single_gate_static_mpcc"]
    assert s["course_n_gates"] == 1
    assert s["course_spawn_dist_lo"] == s["course_spawn_dist_hi"] == 15.0   # == the sgs baseline geometry
    assert s["course_drop_lo"] == s["course_drop_hi"] == 0.0
    assert s["rw_progress_to_center"] is False        # ALONG-TRACK LAG (segment-arc), NOT the isotropic norm
    assert s["rw_corridor"] == 2.0                    # PBRS contouring ON
    assert s["ego"] is True and s["_raw"]["algo.gamma"] == C._GAMMA


def test_single_gate_static_mpcc_renders_valid_tokens():
    toks = C.render_overrides("single_gate_static_mpcc")
    assert "+env.rw_corridor=2.0" in toks
    assert "+env.rw_progress_to_center=false" in toks   # lag mode (bool lowercase)
    assert "+env.course_drop_lo=0.0" in toks
    assert "algo.gamma=0.9975" in toks


# ================================================================================================
# Anisotropic vertical-weight lever (Fengyou greenlight 2026-07-08; supersedes MPCC contouring).
# ================================================================================================
def test_common_vert_weight_default_isotropic():
    # isotropic (1.0) on every real stage -> byte-compatible with the plain Euclidean norm.
    assert C._COMMON["rw_progress_vert_weight"] == 1.0
    for s in C.STAGE_ORDER:
        assert C.STAGES[s]["rw_progress_vert_weight"] == 1.0, s


def test_single_gate_static_aniso_cranks_vertical_weight_offladder():
    """The anisotropic lever: OFF-LADDER, SAME fixed 15 m level gate as the sgs baseline, isotropic-base
    progress ON (progress_to_center True) with the VERTICAL weight cranked up to un-bury the altitude
    signal (the floor-dive fix)."""
    assert "single_gate_static_aniso" not in C.STAGE_ORDER            # off-ladder, standalone only
    s = C.STAGES["single_gate_static_aniso"]
    assert s["course_n_gates"] == 1
    assert s["course_spawn_dist_lo"] == s["course_spawn_dist_hi"] == 15.0   # == the sgs baseline geometry
    assert s["course_drop_lo"] == s["course_drop_hi"] == 0.0
    assert s["rw_progress_to_center"] is True         # isotropic-base homing ON (the un-buried driver)
    assert s["rw_progress_vert_weight"] == 25.0       # vertical weight cranked (>> 1)
    assert s["ego"] is True and s["_raw"]["algo.gamma"] == C._GAMMA


def test_single_gate_static_aniso_renders_valid_tokens():
    toks = C.render_overrides("single_gate_static_aniso")
    assert "+env.rw_progress_vert_weight=25.0" in toks
    assert "+env.rw_progress_to_center=true" in toks
    assert "+env.course_spawn_dist_lo=15.0" in toks
    assert "algo.gamma=0.9975" in toks


# ================================================================================================
# PERCEPTION-HONESTY / HARD NO-SPIN chain (2026-07-10, DESIGN.md §P): the _percept stage pair.
# ================================================================================================
# The exact contents of the LIVE floor-chain stages (consumed by running SLURM jobs -- vdflr0/vdff1
# lineage). SNAPSHOT-PINNED literally: the perception-honesty package must NEVER mutate them; new
# behavior lives only in the *_percept variants. If a legitimate future edit changes these stages,
# update this pin CONSCIOUSLY (and check no live run consumes them).
_DUAL_GATE_BOOT_FLOOR_PIN = {
    **C._COMMON,
    "course_n_gates": 2,
    "course_spawn_dist_lo": 8.0, "course_spawn_dist_hi": 15.0,
    "course_spawn_below_g0_lo": 0.5, "course_spawn_below_g0_hi": 6.0,
    "course_spawn_yaw_jitter": 0.25,
    "course_seg_len_lo": 10.0, "course_seg_len_hi": 20.0,
    "course_gates_above_spawn": 0.5,
    "floor_at_spawn": True,
    "use_racing_line": True,
    "rw_progress_to_center": False,
    "rw_corridor": 4.0,
    "rw_centering": 0.4, "rw_centering_max_m": 6.0,
    "rw_parabola_crossing": True,
    "rw_cross_center": 20.0, "rw_cross_zero_m": 4.0, "rw_cross_neg_cap": 100.0,
    "ego_noise_scale": 0.0,
    "_raw": {"env.max_time": 60, "algo.gamma": 0.9975,
             "++algo.noise_std_hold": 0.30, "++algo.noise_std_floor": 0.03,
             "++algo.noise_hold_frac": 0.5},
}
_DUAL_GATE_FULLSTACK_FLOOR_PIN = {
    **C._COMMON,
    "course_n_gates": 2,
    "course_spawn_dist_lo": 8.0, "course_spawn_dist_hi": 15.0,
    "course_spawn_below_g0_lo": 0.5, "course_spawn_below_g0_hi": 6.0,
    "course_spawn_yaw_jitter": 0.25,
    "course_seg_len_lo": 10.0, "course_seg_len_hi": 20.0,
    "course_gates_above_spawn": 0.5,
    "floor_at_spawn": True,
    "use_racing_line": True,
    "rw_progress_to_center": False,
    "rw_corridor": 4.0,
    "rw_centering": 0.4, "rw_centering_max_m": 6.0,
    "rw_parabola_crossing": True,
    "rw_cross_center": 20.0, "rw_cross_zero_m": 4.0, "rw_cross_neg_cap": 100.0,
    "_raw": {"env.max_time": 60, "algo.gamma": 0.9975,
             "++algo.noise_std_hold": 0.12, "++algo.noise_std_floor": 0.03,
             "++algo.noise_hold_frac": 0.5},
}

# The package knobs a _percept variant adds on top of its base stage -- EXACTLY these, nothing else.
_PERCEPT_KEYS = {
    "ego_blur_gate": True,
    "ego_blur_rate_lo_rad_s": 2.0,       # PLACEHOLDER pending the A2 curve
    "ego_blur_rate_hi_rad_s": 4.0,       # PLACEHOLDER pending the A2 curve
    "ego_spin_rate_abort": 3.5,
    "ego_spin_time_abort": 0.4,
    "ego_spin_rev_abort": 1.5,
    "ego_spin_rev_window_s": 4.0,
    "ego_yaw_cmd_clamp_rad_s": 0.35,
    "rw_perception": 0.02,
    "rw_perception_exponent": 4.0,
}

# Keys the sbatch itself plus-appends (BASE + BOUNDARY_OV, rl/peregrine_vq2_ego.sbatch): a stage _raw
# entry re-appending any of these is a hydra append-collision that kills the launch (curriculum
# footgun :530-535). The dead inc8 spin keys were REMOVED from BASE 2026-07-10.
_SBATCH_APPENDED_KEYS = (
    "+env.body_radius_lo", "+env.body_radius_hi", "+env.frame_depth_m",
    "+dynamics.dr", "+dynamics.dr_aero", "+dynamics.dr_mixer", "+dynamics.dr_force_bias",
    "+dynamics.dr_latency_min_steps", "+dynamics.dr_latency_max_steps",
    "+algo.noise_anneal", "+algo.noise_std_hold", "+algo.noise_std_floor",
    "+algo.noise_hold_frac", "+algo.noise_entropy_floor",
    "+warmstart_reset_logstd", "+warmstart_reset_logstd_std", "+critic_warmup_updates",
    "+eval_det_steps",
    "+init_from",                       # the ladder auto-appends this on chained stage 2
)

_PERCEPT_STAGES = ("dual_gate_boot_floor_percept", "dual_gate_fullstack_floor_percept")

# The exact contents of the LIVE _percept chain stages (consumed by the running vperc0 job 3305662,
# deployed snapshot @ 6615458). SNAPSHOT-PINNED literally -- previously these stages had only the
# RELATIVE superset test below (which silently passes if base+_PERCEPT_KEYS mutate together); the
# estimator-faithful package (2026-07-11) must land STRICTLY additively on top, so the live configs
# get their own byte-literal guard FIRST. If a legitimate future edit changes them, update this pin
# CONSCIOUSLY (and check no live run consumes them).
_DUAL_GATE_BOOT_FLOOR_PERCEPT_PIN = {
    **_DUAL_GATE_BOOT_FLOOR_PIN,
    "ego_blur_gate": True,
    "ego_blur_rate_lo_rad_s": 2.0,
    "ego_blur_rate_hi_rad_s": 4.0,
    "ego_spin_rate_abort": 3.5,
    "ego_spin_time_abort": 0.4,
    "ego_spin_rev_abort": 1.5,
    "ego_spin_rev_window_s": 4.0,
    "ego_yaw_cmd_clamp_rad_s": 0.35,
    "rw_perception": 0.02,
    "rw_perception_exponent": 4.0,
}
_DUAL_GATE_FULLSTACK_FLOOR_PERCEPT_PIN = {
    **_DUAL_GATE_FULLSTACK_FLOOR_PIN,
    "ego_blur_gate": True,
    "ego_blur_rate_lo_rad_s": 2.0,
    "ego_blur_rate_hi_rad_s": 4.0,
    "ego_spin_rate_abort": 3.5,
    "ego_spin_time_abort": 0.4,
    "ego_spin_rev_abort": 1.5,
    "ego_spin_rev_window_s": 4.0,
    "ego_yaw_cmd_clamp_rad_s": 0.35,
    "rw_perception": 0.02,
    "rw_perception_exponent": 4.0,
}


def test_live_floor_chain_stages_snapshot_pinned():
    """LIVE-RUN GUARD: dual_gate_boot_floor / dual_gate_fullstack_floor byte-identical to the pinned
    literals -- the perception-honesty package lands STRICTLY additively."""
    assert C.STAGES["dual_gate_boot_floor"] == _DUAL_GATE_BOOT_FLOOR_PIN
    assert C.STAGES["dual_gate_fullstack_floor"] == _DUAL_GATE_FULLSTACK_FLOOR_PIN


def test_live_percept_chain_stages_snapshot_pinned():
    """LIVE-RUN GUARD (vperc0, job 3305662 @ 6615458): the two _percept stages byte-identical to the
    pinned literals -- the estimator-faithful (_pef) package lands STRICTLY additively."""
    assert C.STAGES["dual_gate_boot_floor_percept"] == _DUAL_GATE_BOOT_FLOOR_PERCEPT_PIN
    assert C.STAGES["dual_gate_fullstack_floor_percept"] == _DUAL_GATE_FULLSTACK_FLOOR_PERCEPT_PIN


def test_percept_stages_are_strict_supersets_of_their_bases():
    """Each _percept variant = its base stage VERBATIM + exactly the package knobs (same values in
    both stages). The boot keeps ego_noise_scale=0.0 AND blur ON (blur independent of noise_scale --
    a blur-free boot re-learns spin-scan); the fullstack keeps real noise (no ego_noise_scale key)."""
    for name, base in (("dual_gate_boot_floor_percept", "dual_gate_boot_floor"),
                       ("dual_gate_fullstack_floor_percept", "dual_gate_fullstack_floor")):
        p, b = C.STAGES[name], C.STAGES[base]
        for k, v in b.items():
            assert p[k] == v, (name, k)                    # base keys present, values equal
        extra = {k: v for k, v in p.items() if k not in b}
        assert extra == _PERCEPT_KEYS, (name, extra)
    # boot: noise-0 calibration + blur ON (the load-bearing independence).
    boot = C.STAGES["dual_gate_boot_floor_percept"]
    assert boot["ego_noise_scale"] == 0.0 and boot["ego_blur_gate"] is True
    # fullstack: REAL noise (no override), blur ON.
    full = C.STAGES["dual_gate_fullstack_floor_percept"]
    assert "ego_noise_scale" not in full and full["ego_blur_gate"] is True


def test_percept_fullstack_has_no_init_from_and_no_sbatch_collision():
    """(a) stage 2 of the chain must NOT carry +init_from (the sbatch ladder appends it -- a
    hardcoded path hydra-collides); (b) no _percept token re-appends a key the sbatch BASE /
    BOUNDARY_OV already plus-appends."""
    assert not any("init_from" in k for k in C.STAGES["dual_gate_fullstack_floor_percept"]["_raw"])
    for s in _PERCEPT_STAGES:
        assert s not in C.STAGE_ORDER                      # off-ladder, standalone/chained only
        for tok in C.render_overrides(s):
            key = tok.split("=", 1)[0]
            if key.startswith("++"):
                continue                                   # ++ force-override is the sanctioned form
            assert key not in _SBATCH_APPENDED_KEYS, (s, tok)


def test_percept_threshold_ordering_blind_policy_defense():
    """THE ORDERING INVARIANT (owner directive design consequence): clamped REALIZED yaw (~3.5x the
    command) < blur-free band lo < the rate abort -- a full-authority pointing sweep is never
    blur-punished and never fatal, so a non-spinning policy can still SEE the gate by pointing at
    it. A future A2 recalibration that breaks this ordering strands the clamped policy blind: this
    test fails first."""
    REALIZED_YAW_GAIN = 3.5                                # empirical (A2); verify from the first trace
    for s in _PERCEPT_STAGES:
        d = C.STAGES[s]
        realized_yaw = d["ego_yaw_cmd_clamp_rad_s"] * REALIZED_YAW_GAIN
        assert realized_yaw <= 1.5, (s, realized_yaw)      # inside the owner 1-1.5 rad/s target
        assert realized_yaw < d["ego_blur_rate_lo_rad_s"], (s, realized_yaw)   # looking is blur-free
        assert d["ego_blur_rate_lo_rad_s"] < d["ego_spin_rate_abort"], s      # blur bites before fatal
        # hi ABOVE the rate abort is deliberate (NOT because [abort, hi) is "fatal anyway" -- blur
        # cuts on instantaneous LOS-perp rate, the abort on SUSTAINED all-axis ||omega||, different
        # abscissas; hi=4.0 keeps the A2 calibration structure) -- but hi must never sit below lo.
        assert d["ego_blur_rate_hi_rad_s"] >= d["ego_blur_rate_lo_rad_s"], s


def test_percept_rperc_farm_neutrality_bound():
    """FARM-NEUTRALITY construction rule (2026-07-10): in every stage that sets rw_perception,
    rw_perception <= rw_time (effective; default 0.02) so hover-and-stare nets <= 0 per tick --
    staring is never a positive-return strategy on its own. ONE historical exemption:
    single_gate_varied_gvf_lpara_perc (0.05) is the DOCUMENTED DETONATION precedent (farmable
    fly-away on a warm start) -- kept verbatim per the existing-stages-untouched invariant; it is
    exactly the failure this bound exists to prevent, so it stays exempt-and-frozen, never copied."""
    # LIVE default, not a mirrored constant: if EgoRewardWeights.time ever changes, the bound checked
    # here moves with it (a hardcoded 0.02 would let real violations pass silently).
    from ego_reward import EgoRewardWeights
    DEFAULT_RW_TIME = EgoRewardWeights().time
    LEGACY_DETONATION_EXEMPT = ("single_gate_varied_gvf_lpara_perc",)
    for s, d in C.STAGES.items():
        if s in LEGACY_DETONATION_EXEMPT:
            assert d["rw_perception"] == 0.05, s           # frozen historical value
            continue
        if d.get("rw_perception", 0.0) > 0.0:
            eff_time = d.get("rw_time", DEFAULT_RW_TIME)
            assert d["rw_perception"] <= eff_time, (s, d["rw_perception"], eff_time)
    # and the _percept pair actually sets it (the framing gradient is the constructive-path guide).
    for s in _PERCEPT_STAGES:
        assert C.STAGES[s]["rw_perception"] == 0.02, s


# ================================================================================================
# ESTIMATOR-FAITHFUL chain (_pef, 2026-07-11): _percept VERBATIM + exactly the package knobs.
# ================================================================================================
_PEF_STAGES = ("dual_gate_boot_floor_pef", "dual_gate_fullstack_floor_pef")
# the env-key additions (rendered as +env.*). ego_est_dt_ticks_hi=4 = the CHOKED-LOOP dt
# emulation over the MEASURED wire tick-gap band (reviewer-caught 2026-07-11: the wire never ran
# at the 30 Hz training tick; see ego_ins_emul.MEASURED_TICK_GAP_PMF + the stage block comment).
_PEF_ENV_KEYS = {"ego_faithful": True, "ego_est_dt_ticks_hi": 4}
# the _raw additions (++ add-or-override form -- deliberately NOT '+', so they can never collide
# with an existing config key NOR appear in _SBATCH_APPENDED_KEYS)
_PEF_RAW_KEYS = {"++dynamics.n_substeps": 5, "++dynamics.capture_specific_force": True}


def test_pef_stages_are_strict_supersets_of_percept():
    """Each _pef variant = its _percept base VERBATIM + exactly the estimator-faithful knobs
    (same values in both stages): +env.ego_faithful=true and the two ++dynamics plant knobs in
    _raw. The boot keeps ego_noise_scale=0.0 -- and the leveler/KF are DELIBERATELY
    noise_scale-independent (blur precedent), so the boot already flies the lying attitude."""
    for name, base in (("dual_gate_boot_floor_pef", "dual_gate_boot_floor_percept"),
                       ("dual_gate_fullstack_floor_pef", "dual_gate_fullstack_floor_percept")):
        p, b = C.STAGES[name], C.STAGES[base]
        for k, v in b.items():
            if k == "_raw":
                for rk, rv in v.items():
                    assert p["_raw"][rk] == rv, (name, rk)     # base _raw verbatim
                extra_raw = {rk: rv for rk, rv in p["_raw"].items() if rk not in v}
                assert extra_raw == _PEF_RAW_KEYS, (name, extra_raw)
            else:
                assert p[k] == v, (name, k)
        extra = {k: v for k, v in p.items() if k not in b}
        assert extra == _PEF_ENV_KEYS, (name, extra)
    boot = C.STAGES["dual_gate_boot_floor_pef"]
    assert boot["ego_noise_scale"] == 0.0 and boot["ego_faithful"] is True
    full = C.STAGES["dual_gate_fullstack_floor_pef"]
    assert "ego_noise_scale" not in full and full["ego_faithful"] is True


def test_pef_no_init_from_and_no_sbatch_collision():
    """(a) chain stage 2 must NOT carry +init_from (the sbatch ladder appends it); (b) no _pef
    token re-appends a key the sbatch BASE/BOUNDARY_OV plus-appends (the ++dynamics knobs are the
    sanctioned force-override form and are skipped like the ++algo ones)."""
    assert not any("init_from" in k for k in C.STAGES["dual_gate_fullstack_floor_pef"]["_raw"])
    for s in _PEF_STAGES:
        assert s not in C.STAGE_ORDER                      # off-ladder, chained only
        for tok in C.render_overrides(s):
            key = tok.split("=", 1)[0]
            if key.startswith("++"):
                continue
            assert key not in _SBATCH_APPENDED_KEYS, (s, tok)


def test_pef_renders_valid_tokens():
    toks = C.render_overrides("dual_gate_boot_floor_pef")
    assert "+env.ego_faithful=true" in toks
    assert "+env.ego_est_dt_ticks_hi=4" in toks            # measured choked-loop dt band
    assert "++dynamics.n_substeps=5" in toks               # the aliasing channel (load-bearing)
    assert "++dynamics.capture_specific_force=true" in toks
    assert "+env.ego_noise_scale=0.0" in toks              # boot keeps the calibration regime
    assert "+env.ego_blur_gate=true" in toks               # the _percept package rides along
    assert "algo.gamma=0.9975" in toks
    full_toks = C.render_overrides("dual_gate_fullstack_floor_pef")
    assert "+env.ego_faithful=true" in full_toks
    assert "++dynamics.n_substeps=5" in full_toks
    assert not any(t.startswith("+env.ego_noise_scale") for t in full_toks)   # real noise


def test_pef_does_not_set_per_channel_ablation_knobs():
    """The _pef stages arm ONLY the master knob: ego_att_model/ego_vel_model/ego_rate_model stay
    unset (they default from ego_faithful in the env) so an ablation run must set them explicitly."""
    for s in _PEF_STAGES:
        for k in ("ego_att_model", "ego_vel_model", "ego_rate_model"):
            assert k not in C.STAGES[s], (s, k)


def test_percept_renders_valid_tokens():
    toks = C.render_overrides("dual_gate_boot_floor_percept")
    assert "+env.ego_blur_gate=true" in toks
    assert "+env.ego_spin_rate_abort=3.5" in toks
    assert "+env.ego_yaw_cmd_clamp_rad_s=0.35" in toks
    assert "+env.rw_perception=0.02" in toks
    assert "+env.ego_noise_scale=0.0" in toks              # boot keeps the calibration regime
    assert "algo.gamma=0.9975" in toks                     # _raw verbatim
    full_toks = C.render_overrides("dual_gate_fullstack_floor_percept")
    assert "+env.ego_blur_gate=true" in full_toks
    assert not any(t.startswith("+env.ego_noise_scale") for t in full_toks)   # real noise


# ================================================================================================
# BEHAVIORAL-CAP chain (_pefcap, 2026-07-12): _pef VERBATIM + the default-OFF behavioral package.
# ================================================================================================
_PEFCAP_STAGES = ("dual_gate_boot_floor_pefcap", "dual_gate_fullstack_floor_pefcap")
_PEFCAP_NEW_KEYS = {"rw_att_pitch", "att_pitch_limit_rad", "rw_att_roll", "att_roll_limit_rad",
                    "course_g1_out_of_fov_lo", "course_g1_out_of_fov_hi",
                    "ego_obs_slot_range_cap_m", "rw_perception_next"}


def test_pefcap_stages_exist_offladder_and_arm_the_package():
    for s in _PEFCAP_STAGES:
        assert s in C.STAGES
        assert s not in C.STAGE_ORDER                              # off-ladder, chained only
        d = C.STAGES[s]
        # (A) soft attitude caps ARMED, pitch limit ABOVE the 17.8deg (0.31 rad) nose-down rest tilt.
        assert d["rw_att_pitch"] == 0.5 and d["rw_att_roll"] == 0.3
        assert d["att_pitch_limit_rad"] > 0.31 and d["att_pitch_limit_rad"] == pytest.approx(0.5235988)
        assert d["att_roll_limit_rad"] == pytest.approx(0.6981317)
        # (B) gate-1 out-of-fov target bearing past the ~45deg (0.785 rad) camera half-HFOV edge.
        assert d["course_g1_out_of_fov_lo"] > 0.785
        assert d["course_g1_out_of_fov_hi"] > d["course_g1_out_of_fov_lo"]
        # (B') far-gate slot-fill cap 30 m (BOTH slots; train/deploy parity + empty slot1 between gates).
        assert d["ego_obs_slot_range_cap_m"] == 30.0
        # (D) perception SPLIT current+next STRICTLY below rw_time (0.02) -> farm-neutral hover-stare.
        assert d["rw_perception"] == 0.014 and d["rw_perception_next"] == 0.004
        assert d["rw_perception"] + d["rw_perception_next"] < 0.02


def test_pefcap_seg_len_ramp_and_sharper_fullstack_band():
    boot = C.STAGES["dual_gate_boot_floor_pefcap"]
    full = C.STAGES["dual_gate_fullstack_floor_pefcap"]
    # (C) seg-len ramp: boot GENTLE (10-20 m, protect the warm boot); fullstack TIGHTER min (8-20 m).
    assert boot["course_seg_len_lo"] == 10.0 and boot["course_seg_len_hi"] == 20.0
    assert full["course_seg_len_lo"] == 8.0 and full["course_seg_len_hi"] == 20.0
    # the fullstack lowers the sampler rejection floor so 8 m spacing is ACTUALLY realized (default 10 m
    # would redraw every <10 m course and silently truncate seg_len_lo=8 back to 10); boot keeps the default.
    assert full["course_min_pair_dist_m"] == 7.0
    assert "course_min_pair_dist_m" not in boot
    # the out-of-fov band is SHARPER in the fullstack (gate 1 hidden earlier on the approach).
    assert full["course_g1_out_of_fov_lo"] > boot["course_g1_out_of_fov_lo"]


def test_pefcap_inherits_pef_verbatim_except_documented_deviations():
    """_pefcap = its _pef base VERBATIM + exactly the ARMED behavioral knobs, with two DOCUMENTED
    deviations: the perception re-split (rw_perception 0.02 -> 0.014) and the fullstack seg-len ramp
    (course_seg_len_lo 10 -> 8). Everything else -- incl. the whole estimator-faithful + no-spin _raw
    -- is byte-identical to _pef."""
    for cap, pef in (("dual_gate_boot_floor_pefcap", "dual_gate_boot_floor_pef"),
                     ("dual_gate_fullstack_floor_pefcap", "dual_gate_fullstack_floor_pef")):
        c, p = C.STAGES[cap], C.STAGES[pef]
        is_full = cap.endswith("fullstack_floor_pefcap")
        for k, v in p.items():
            if k == "rw_perception":
                assert c[k] == 0.014                               # DEVIATION 1: re-split
            elif k == "course_seg_len_lo" and is_full:
                assert c[k] == 8.0                                 # DEVIATION 2: fullstack seg ramp
            elif k == "_raw":
                assert c["_raw"] == v                              # estimator-faithful plant knobs byte-identical
            else:
                assert c[k] == v, (cap, k)
        extra = {k for k in c if k not in p}
        expected = set(_PEFCAP_NEW_KEYS)
        if is_full:
            expected |= {"course_min_pair_dist_m"}                # DEVIATION 3: fullstack lowers reject floor
        assert extra == expected, (cap, extra)


def test_pefcap_farm_neutrality_current_plus_next_below_time():
    from ego_reward import EgoRewardWeights
    t = EgoRewardWeights().time
    for s in _PEFCAP_STAGES:
        d = C.STAGES[s]
        assert d["rw_perception"] + d["rw_perception_next"] < t + 1e-12
        # constructing the weights must NOT trip the farm-neutrality guard.
        EgoRewardWeights(perception=d["rw_perception"], perception_next=d["rw_perception_next"])


def test_pefcap_renders_valid_tokens():
    for s in _PEFCAP_STAGES:
        toks = C.render_overrides(s)
        assert "+env.rw_att_pitch=0.5" in toks and "+env.rw_att_roll=0.3" in toks
        assert "+env.ego_obs_slot_range_cap_m=30.0" in toks
        assert "+env.rw_perception_next=0.004" in toks
        assert "+env.rw_perception=0.014" in toks              # the re-split current lever
        assert any(t.startswith("+env.course_g1_out_of_fov_lo=") for t in toks)
        # the _pef estimator-faithful + hard-no-spin package rides along VERBATIM.
        assert "+env.ego_faithful=true" in toks and "+env.ego_blur_gate=true" in toks
        assert "++dynamics.n_substeps=5" in toks
        assert "algo.gamma=0.9975" in toks


def test_pefcap_no_sbatch_collision_and_no_init_from():
    assert not any("init_from" in k for k in C.STAGES["dual_gate_fullstack_floor_pefcap"]["_raw"])
    for s in _PEFCAP_STAGES:
        for tok in C.render_overrides(s):
            key = tok.split("=", 1)[0]
            if key.startswith("++"):
                continue
            assert key not in _SBATCH_APPENDED_KEYS, (s, tok)


# ================================================================================================
# NODITHER fine-tune (2026-07-12): dual_gate_fullstack_floor_pef VERBATIM + exactly rw_yaw_dither armed.
# ================================================================================================
_NODITHER_STAGE = "dual_gate_fullstack_floor_pef_nodither"
_NODITHER_NEW_KEYS = {"rw_yaw_dither"}


def test_nodither_is_pef_fullstack_verbatim_plus_only_yaw_dither():
    """The fine-tune stage = dual_gate_fullstack_floor_pef VERBATIM (course + reward + the no-spin
    package + the estimator-faithful _raw all byte-identical) PLUS exactly ONE new armed knob:
    rw_yaw_dither. Nothing else changes -> the flight skill + the HARD no-spin guarantee are preserved."""
    nd, pef = C.STAGES[_NODITHER_STAGE], C.STAGES["dual_gate_fullstack_floor_pef"]
    for k, v in pef.items():
        if k == "_raw":
            assert nd["_raw"] == v                                  # _raw byte-identical (noise + plant knobs)
        else:
            assert nd[k] == v, k                                    # every _pef key verbatim
    extra = {k for k in nd if k not in pef}
    assert extra == _NODITHER_NEW_KEYS, extra
    assert nd["rw_yaw_dither"] > 0.0                                # ARMED (the whole point of the fine-tune)


def test_nodither_keeps_hard_no_spin_and_yaw_clamp_untouched():
    """The no-spin guarantee stays BY CONSTRUCTION: the fatal spin abort (incl. the accumulated-rotation
    trigger that closes the constant-drift/slow-spin escape) + the realized-yaw clamp are EXACTLY the _pef
    values -- the anti-dither jerk penalty rides ALONGSIDE them, it does not touch or weaken them."""
    nd = C.STAGES[_NODITHER_STAGE]
    assert nd["ego_spin_rate_abort"] == 3.5 and nd["ego_spin_time_abort"] == 0.4
    assert nd["ego_spin_rev_abort"] == 1.5 and nd["ego_spin_rev_window_s"] == 4.0   # the drift closer
    assert nd["ego_yaw_cmd_clamp_rad_s"] == 0.35


def test_nodither_offladder_no_init_from_no_sbatch_collision_renders():
    nd = C.STAGES[_NODITHER_STAGE]
    assert _NODITHER_STAGE not in C.STAGE_ORDER                     # off-ladder (standalone warm-start)
    assert not any("init_from" in k for k in nd["_raw"])           # launcher/EXTRA wires vpeffs0 at run time
    toks = C.render_overrides(_NODITHER_STAGE)
    assert "+env.rw_yaw_dither=0.5" in toks                         # the ONE armed anti-dither knob
    assert "+env.ego_faithful=true" in toks                         # the _pef estimator-faithful rides along
    assert "+env.ego_yaw_cmd_clamp_rad_s=0.35" in toks             # the no-spin clamp preserved
    assert "+env.ego_spin_rev_abort=1.5" in toks                    # the constant-drift spin closer preserved
    assert "algo.gamma=0.9975" in toks
    assert not any(t.startswith("+env.ego_noise_scale") for t in toks)   # real-noise fullstack regime
    for tok in toks:                                               # no _pef/BASE-appended key collision
        key = tok.split("=", 1)[0]
        if key.startswith("++"):
            continue
        assert key not in _SBATCH_APPENDED_KEYS, tok
