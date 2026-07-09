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
    # fixed spawn-heading scalar, and the gate-0 HEIGHT band added 2026-07-08 for the varied-gate stage).
    assert C.COURSE_SAMPLER_KEYS == ("course_n_gates", "course_seg_len_lo", "course_seg_len_hi",
                                     "course_drop_lo", "course_drop_hi",
                                     "course_spawn_dist_lo", "course_spawn_dist_hi",
                                     "course_spawn_below_g0_lo", "course_spawn_below_g0_hi",
                                     "course_spawn_heading", "course_spawn_yaw_jitter")


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
