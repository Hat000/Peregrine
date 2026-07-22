"""The gate-emission port must actually reach the detector that FLIES (2026-07-22).

The deployed M model is 8-keypoint (4 inner + 4 outer). Before this port the flight path sliced the
outer 4 off at the model boundary and discarded them, and dropped every detection with <3 confident
inner corners -- exactly the cropped close-range frames just before a gate pass. Two seams decide
whether the fix is live in the air, and both are easy to get silently wrong:

  1. ``GateDetector`` must DEFAULT to fusion + rescue ON, because ``fly_rl`` constructs it with no
     kwargs. A default flip here is invisible at the call site.
  2. BOTH loader sites must pass the CLI kill-switches. ``fly_rl`` pre-warms a detector at startup
     and the seeker REUSES that instance -- so if only ``_build_casec_seeker`` threaded the flags,
     ``--seeker-no-outer`` would be a no-op on every real flight (the pre-warm always wins) while
     still "working" in any test that skipped the pre-warm.

These are wiring pins, not geometry: the geometry lives in test_partial_rescue / test_outer_fusion.

Run: python -m pytest tests/test_seeker_gate_emission_wiring.py -q
"""
import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))

from racer.vision.detector import (  # noqa: E402
    KPT_CONF_THRESH_DEFAULT,
    EnsembleGateDetector,
    GateDetector,
)


def test_detector_defaults_are_fusion_and_rescue_on():
    """fly_rl calls GateDetector.load(weights) with NO kwargs, so the defaults ARE the flight config."""
    det = GateDetector(model=None)
    assert det.use_outer is True, "outer fusion must default ON -- fly_rl passes no kwargs"
    assert det.partial_rescue is True, "partial rescue must default ON -- fly_rl passes no kwargs"
    assert det.kpt_conf_thresh == KPT_CONF_THRESH_DEFAULT == 0.2

    ens = EnsembleGateDetector(models=[])
    assert ens.use_outer is True and ens.partial_rescue is True
    assert ens.kpt_conf_thresh == KPT_CONF_THRESH_DEFAULT


def test_kill_switch_kwargs_map_flags_to_detector():
    import fly_rl

    on = argparse.Namespace(seeker_no_outer=False, seeker_no_rescue=False)
    assert fly_rl._seeker_detector_kwargs(on) == {"use_outer": True, "partial_rescue": True}

    off = argparse.Namespace(seeker_no_outer=True, seeker_no_rescue=True)
    assert fly_rl._seeker_detector_kwargs(off) == {"use_outer": False, "partial_rescue": False}

    # and they must be REAL constructor kwargs, not silently swallowed by **kwargs somewhere
    det = GateDetector(model=None, **fly_rl._seeker_detector_kwargs(off))
    assert det.use_outer is False and det.partial_rescue is False


def test_both_loader_sites_thread_the_kwargs():
    """The pre-warmed instance is the one that flies -- a flag threaded into only one site is a
    no-op in the air. Asserted on the AST so it survives reformatting but not a dropped call."""
    tree = ast.parse((ROOT / "rl" / "fly_rl.py").read_text(encoding="utf-8"))
    loads = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "load"
             and isinstance(n.func.value, ast.Name) and n.func.value.id == "GateDetector"]
    assert len(loads) == 2, f"expected the pre-warm + seeker load sites, found {len(loads)}"
    for call in loads:
        starred = [k for k in call.keywords if k.arg is None]      # a **kwargs expansion
        assert starred, f"GateDetector.load at line {call.lineno} does not thread the kill-switches"
        assert any(isinstance(k.value, ast.Call)
                   and getattr(k.value.func, "id", None) == "_seeker_detector_kwargs"
                   for k in starred), \
            f"GateDetector.load at line {call.lineno} expands the wrong kwargs source"


def test_cli_exposes_both_switches_defaulting_off():
    import fly_rl

    p = fly_rl.build_parser() if hasattr(fly_rl, "build_parser") else None
    if p is None:                                   # parser built inline in main(); fall back to AST
        src = (ROOT / "rl" / "fly_rl.py").read_text(encoding="utf-8")
        assert '"--seeker-no-outer"' in src and '"--seeker-no-rescue"' in src
        return
    ns = p.parse_args([])
    assert ns.seeker_no_outer is False and ns.seeker_no_rescue is False
