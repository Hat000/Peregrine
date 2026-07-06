"""B2/M2 measurement stack -- laptop tests (torch-only, no diffaero, no tensorboard).

M2a: rl/inc8_spawn_metrics.py -- the spawn-class partition (0=standing, 1=near non-last,
2=near LAST gate = the RC1 trivial-success class) + the decayed class accumulators the inc8 env
logs as inc8_success_standing / inc8_success_near / inc8_success_nearlast (+ mass fracs + npass).
M2b: rl/inc8_tb_trace.py FLIGHTCHECK window -- verdict max over updates >= 100 (excludes the
step-0 warm-start restore artifact: the diagnosis' dual_gate read 0.64 at step 0 vs sustained
0.343) and gates on inc8_success_standing when the run logs it.
M2c: the trace table carries the B1/B2 KPIs so stage reads need no manual TB pulls.

Run from repo ROOT: .venv\\Scripts\\python.exe -m pytest tests/test_b2_spawn_class_metrics.py -q
"""
import math
import re
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rl"))

import inc8_spawn_metrics as SM                                        # noqa: E402
import inc8_tb_trace as TB                                             # noqa: E402


# ============================================================ M2a: the spawn-class partition
def test_spawn_class_partition():
    standing = torch.tensor([True, False, False, True, False])
    tg = torch.tensor([0, 1, 0, 0, 1], dtype=torch.int32)
    cls = SM.spawn_class_of(standing, tg, n_gates=2)
    assert cls.dtype == torch.long
    assert cls.tolist() == [SM.CLS_STANDING, SM.CLS_NEARLAST, SM.CLS_NEAR,
                            SM.CLS_STANDING, SM.CLS_NEARLAST]
    # G=1: every near-spawn is class 2 by definition (the only gate IS the last gate)
    cls1 = SM.spawn_class_of(torch.tensor([False, True]),
                             torch.tensor([0, 0], dtype=torch.int32), n_gates=1)
    assert cls1.tolist() == [SM.CLS_NEARLAST, SM.CLS_STANDING]


def test_class_accumulators_exact_and_empty_safe():
    ep = torch.zeros(SM.N_CLASSES)
    su = torch.zeros(SM.N_CLASSES)
    np_ = torch.zeros(SM.N_CLASSES)
    # 0/0 guard: zero accumulators -> all-finite all-zero, never NaN
    r_s, r_n, r_f = SM.class_rates(ep, su, np_)
    for t in (r_s, r_n, r_f):
        assert torch.isfinite(t).all() and (t == 0.0).all()
    # one exact update (decay=1.0): cls=[0,0,1,2,1], reset=[T,T,T,T,F], succ=[T,F,T,F,T], np=[2,0,1,0,1]
    cls = torch.tensor([0, 0, 1, 2, 1])
    reset = torch.tensor([True, True, True, True, False])
    succ = torch.tensor([True, False, True, False, True])
    npass = torch.tensor([2.0, 0.0, 1.0, 0.0, 1.0])
    SM.update_class_accumulators(ep, su, np_, cls, reset, succ, npass, decay=1.0)
    r_s, r_n, r_f = SM.class_rates(ep, su, np_)
    assert r_s.tolist() == pytest.approx([0.5, 1.0, 0.0])
    assert r_n.tolist() == pytest.approx([1.0, 1.0, 0.0])
    assert r_f.tolist() == pytest.approx([0.5, 0.25, 0.25])   # env 4 (mid-episode) never counted


def test_class_accumulators_decay_ratio_invariant():
    """The EMA decays numerator and denominator TOGETHER: a class's success RATE is invariant under
    empty updates while its episode MASS decays -- pins the window semantics."""
    ep = torch.zeros(SM.N_CLASSES)
    su = torch.zeros(SM.N_CLASSES)
    np_ = torch.zeros(SM.N_CLASSES)
    cls = torch.tensor([0])
    SM.update_class_accumulators(ep, su, np_, cls, torch.tensor([True]),
                                 torch.tensor([True]), torch.tensor([1.0]), decay=1.0)
    none = torch.tensor([False])
    for _ in range(2000):
        SM.update_class_accumulators(ep, su, np_, cls, none,
                                     torch.tensor([False]), torch.tensor([0.0]), decay=0.999)
    r_s, _, _ = SM.class_rates(ep, su, np_)
    assert r_s[0].item() > 0.99                       # ratio preserved
    assert ep[0].item() < 0.14                        # mass decayed (~0.999^2000 = 0.135)


# ============================================================ M2b: the FLIGHTCHECK window
def test_flightcheck_window_excludes_step0():
    assert TB.FLIGHTCHECK_MIN_STEP == 100
    tag = "env_loss/success_rate"
    # the diagnosis' exact dual_gate numbers: 0.64 step-0 restore artifact, sustained max 0.343
    series = {tag: {0: 0.64, 50: 0.2, 100: 0.31, 200: 0.343, 300: 0.30}}
    assert TB._series_max_from(series, tag, 100) == (0.343, True)
    # short run: no step clears the window -> full-series fallback, flagged False
    short = {tag: {0: 0.64, 50: 0.1}}
    assert TB._series_max_from(short, tag, 100) == (0.64, False)
    # missing tag -> (nan, False)
    v, ok = TB._series_max_from(series, None, 100)
    assert math.isnan(v) and ok is False
    v, ok = TB._series_max_from(series, "env_loss/absent", 100)
    assert math.isnan(v) and ok is False


# ============================================================ M2c: the trace table KPIs
def test_trace_columns_carry_b2_kpis():
    cols = dict(TB.COLUMNS)
    all_cands = [c for cands in cols.values() for c in cands]
    for tag in ("inc8_tcam_front_frac", "inc8_gate_progress", "inc8_success_standing",
                "inc8_success_near", "inc8_success_nearlast", "inc8_frac_nearlast",
                "inc8_through_centering"):
        assert tag in all_cands, tag
    # A2 coordination: the renamed near-centering row reads the NEW tag first, old tag as fallback
    assert cols["near_centering"][0] == "inc8_near_centering"
    assert "inc8_centering" in cols["near_centering"]


def test_awk_field_anchor_safety():
    """The sbatch FLIGHTCHECK awk anchors fields with ^success_rate_max= -- the new
    raw_success_rate_max= field must never false-match it (field-start anchored)."""
    assert re.match(r"^success_rate_max=", "raw_success_rate_max=0.6") is None
    assert re.match(r"^success_rate_max=", "success_rate_max=0.3") is not None
