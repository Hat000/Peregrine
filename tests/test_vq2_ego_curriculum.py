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
    """_COMMON (applied to single_gate) must carry ONLY the clean champion reward: base rw_passage=1.0
    and exit_align OFF. The B2b lesson: a hard-stage knob in _COMMON regressed single_gate 0.82->0.035."""
    assert C._COMMON["rw_passage"] == 1.0
    assert C._COMMON["rw_exit_align"] == 0.0
    assert C._COMMON["rw_progress"] == 1.0
    assert C._COMMON["rw_terminal_progress_scaled"] is True
    assert C._COMMON["rw_terminal_base"] == 200.0


def test_hard_knobs_absent_from_easy_stages_present_on_hard_stages():
    """single_gate + handoff_drill stay on the clean champion reward (rw_passage==1.0, exit_align==0.0);
    dual_gate_full + multi_gate carry the STAGE-SPECIFIC turning pressure (rw_passage bumped >1, exit>0)."""
    for s in ("single_gate", "handoff_drill"):
        d = C.STAGES[s]
        assert d["rw_passage"] == 1.0, s        # NO passage bump on the easy/discovery stages
        assert d["rw_exit_align"] == 0.0, s     # exit-line OFF on the easy/discovery stages
    for s in ("dual_gate_full", "multi_gate"):
        d = C.STAGES[s]
        assert d["rw_passage"] > 1.0, s         # wider centering basin for the sharp turn
        assert 2.0 <= d["rw_passage"] <= 4.0, s  # DESIGN.md: 2-4x
        assert d["rw_exit_align"] > 0.0, s      # next-gate exit-line ON (gate-gated -> safe)


def test_single_gate_is_pure_common_no_extra_reward_knobs():
    """single_gate must be _COMMON + only course/gamma/max_time -- NO stage-specific reward override,
    so it trains on exactly the reward it validates at ~0.8 before promotion (validate-before-_COMMON)."""
    extra = set(C.STAGES["single_gate"]) - set(C._COMMON) - {"course_n_gates", "_raw"}
    assert extra == set(), extra


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
    assert "+env.rw_passage=3.0" in toks
    assert "+env.rw_exit_align=0.1" in toks
    # _raw entries are VERBATIM (no +): algo.gamma / env.max_time.
    assert "algo.gamma=0.9975" in toks
    assert any(t.startswith("env.max_time=") for t in toks)
    # NO token double-prefixes the raw keys.
    assert not any(t.startswith("+algo.gamma") for t in toks)


def test_render_overrides_unknown_stage_raises():
    with pytest.raises(ValueError):
        C.render_overrides("does_not_exist")


def test_course_sampler_keys_documented():
    # the sampler-key contract (matches the inc8 curriculum convention).
    assert C.COURSE_SAMPLER_KEYS == ("course_n_gates", "course_seg_len_lo", "course_seg_len_hi",
                                     "course_drop_lo", "course_drop_hi")
