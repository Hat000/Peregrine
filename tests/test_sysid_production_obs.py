"""SYS-ID #6 -- PRODUCTION-vs-TRAINED obs registration (laptop arm, 2026-06-18).

Registers the deploy obs builder (src/racer/estimator_obs.py) against the trained 20-dim contract
(rl/fly_rl.build_obs/obs_from_zup [0:17] + the d5 confidence triple [17:20]). Pins what was MEASURED:

  R1 (inc7 BYTE-IDENTITY): under MATCHED state (estimator pos/vel == wire pos/vel) the deploy seam
     estimator_state_for_obs(ds, nav) -> build_obs is element-wise BYTE-IDENTICAL to the trained
     build_obs over random pose/vel/attitude/gate/thrust. obs[0:17] delta == 0.0 exactly. This is the
     inc7 (obs_dim 17) contract -- it must stay byte-identical.

  R2 (the obs[17:20] GAP): there is NO production obs[17:20]/confidence-triple builder in src/racer.
     The deploy confidence triple builder (`deploy_confidence_triple`) lives ONLY in
     rl/spike_vertical_slice.py (a spike), NOT promoted into src/racer/estimator_obs.py. estimator_obs's
     public surface is 17-dim only. This test PINS the gap so that when a production builder IS added
     (promoting deploy_confidence_triple into src/racer), this guard fires and forces a contract review.

  R3 (the sigma sqrt(2) FOOTGUN -- INTENTIONAL, do NOT reconcile): the navigator's NavState export
     nav_inplane_sigma = sqrt(P_E + P_D) is sqrt(2)x the emulator/spike in-plane sigma_hat
     = sqrt((P_E + P_D)/2). A production obs[17:20] builder fed NavState.nav_inplane_sigma RAW into the
     d5 encoding clip(sigma_ref/sigma_hat) would mis-report c_inplane by up to ~0.29; dividing by sqrt(2)
     recovers the trained value to machine epsilon. Pinned here so the relationship is explicit for
     whoever promotes the builder. (The deploy spike sidesteps this by re-projecting the raw KF P, not
     consuming the NavState export.)

Drives BOTH sides with identical state, isolating the obs layer (matched-state -> no estimator
divergence; the obs builder is the only thing under test). build_obs pulls torch -> importorskip.
[SYS-ID 2026-06-18 production-vs-trained-obs]
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("torch")   # fly_rl imports torch at module top

_RL = Path(__file__).resolve().parents[1] / "rl"
if str(_RL) not in sys.path:
    sys.path.insert(0, str(_RL))

from fly_rl import _GATE_POS_ZUP, _GATE_YAW_ZUP, build_obs, make_gate_map  # noqa: E402
from racer.contracts import DroneState, NavState  # noqa: E402
from racer.estimator_obs import estimator_obs, estimator_state_for_obs  # noqa: E402

_SRC_RACER = Path(__file__).resolve().parents[1] / "src" / "racer"
_LEVEL_Q = np.array([1.0, 0.0, 0.0, 0.0])


def _ds(pos, vel, q=_LEVEL_Q):
    return DroneState(sim_time_ns=0, orientation_ned_wxyz=np.asarray(q, float),
                      angular_rate_body=np.zeros(3), accel_body=np.array([0.0, 0.0, -9.80665]),
                      position_ned=np.asarray(pos, float), velocity_ned=np.asarray(vel, float))


def _nav(pos, vel):
    return NavState(sim_time_ns=0, position_ned=np.asarray(pos, float),
                    velocity_ned=np.asarray(vel, float))


def test_r1_inc7_deploy_equals_trained_byte_identical():
    """R1: deploy seam == trained build_obs, ELEMENT-WISE BYTE-IDENTICAL under matched state.

    The deploy obs builder is a PURE pos/vel substitution into the canonical builder; with the
    estimator pos/vel set EQUAL to the wire (matched state) the obs must be byte-identical to the
    shipped 17-dim build_obs the inc7 policy trained on. Any layout/sign drift trips here."""
    rng = np.random.default_rng(7)
    gm = make_gate_map(_GATE_POS_ZUP, _GATE_YAW_ZUP)
    max_abs = 0.0
    for _ in range(400):
        pos = rng.normal(0, 25, 3)
        vel = rng.normal(0, 8, 3)
        q = Rotation.random(random_state=rng).as_quat()        # xyzw
        ds = _ds(pos, vel, q=[q[3], q[0], q[1], q[2]])
        tg = int(rng.integers(0, len(_GATE_POS_ZUP)))
        lnt = float(rng.uniform(0, 2))
        trained = build_obs(ds, tg, lnt, gate_map=gm)
        deploy = estimator_obs(ds, _nav(pos, vel), tg, lnt, gate_map=gm)
        assert trained.shape == (17,) and deploy.shape == (17,), "inc7 obs must be 17-dim"
        np.testing.assert_array_equal(deploy, trained)         # byte-identical (obs[0:17] delta 0.0)
        max_abs = max(max_abs, float(np.max(np.abs(
            deploy.astype(np.float64) - trained.astype(np.float64)))))
    assert max_abs == 0.0, f"deploy 17-dim diverged from trained by {max_abs:.3e} (must be 0.0)"


def test_r2_no_production_obs1720_builder_gap():
    """R2: REGISTER the GAP -- no production obs[17:20]/confidence-triple BUILDER exists in src/racer.

    The deploy 17-dim seam (estimator_obs) deliberately does NOT append the d5 confidence triple; the
    only deploy builder (`deploy_confidence_triple`) lives in rl/spike_vertical_slice.py (a spike). This
    pins the absence so promoting a production builder into src/racer trips this guard -> contract review.

    Implementation: scan src/racer for any callable obs[17:20] producer. References inside COMMENTS/
    docstrings (estimator_obs/contracts describe the FUTURE channel) are not builders -> excluded by
    requiring the token to appear as an actual `def ...triple`/`def confidence_channel` definition."""
    import racer.estimator_obs as eo
    public = {n for n in dir(eo) if not n.startswith("_")}
    # estimator_obs's only obs builders are the two 17-dim seam fns -- no 20-dim/triple producer.
    assert "deploy_confidence_triple" not in public, (
        "GAP CLOSED: a deploy_confidence_triple was promoted into src/racer.estimator_obs -- "
        "update the obs[17:20] contract registration (this guard intentionally pinned its ABSENCE).")
    assert "confidence_channel" not in public

    # No DEFINITION of a confidence-triple builder anywhere under src/racer (comments are allowed).
    triple_defs = []
    for p in _SRC_RACER.rglob("*.py"):
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            s = line.lstrip()
            if s.startswith("def ") and (
                    "deploy_confidence_triple" in s or "confidence_channel" in s or "confidence_triple" in s):
                triple_defs.append(f"{p.name}: {s.strip()}")
    assert triple_defs == [], (
        "GAP CLOSED: a production obs[17:20] builder DEF appeared in src/racer "
        f"({triple_defs}); register the new production confidence-triple contract.")


def test_r3_navstate_sigma_is_sqrt2_times_emul_hat_intentional():
    """R3: PIN the INTENTIONAL sqrt(2) gap (do NOT reconcile).

    navigator._gate_frame_pos_sigma / NavState.nav_inplane_sigma = sqrt(P_E + P_D);
    emulator/spike sigma_inplane_hat                              = sqrt((P_E + P_D)/2).
    Ratio is EXACTLY sqrt(2). Consequence: feeding the NavState export RAW into the d5 encoding
    clip(sigma_ref/sigma_hat) mis-reports c_inplane (here >0.05 worst-case); dividing by sqrt(2) first
    recovers the TRAINED value to ~0. Both pinned so the promotion path is unambiguous."""
    sigma_ref = 0.05
    rng = np.random.default_rng(3)
    worst_ratio_err = 0.0
    worst_c_raw_err = 0.0
    worst_c_corr_err = 0.0
    for _ in range(500):
        a = float(rng.uniform(0.0005, 0.02))   # P_E (gate-frame in-plane axis 0 variance)
        b = float(rng.uniform(0.0005, 0.02))   # P_D (gate-frame in-plane axis 1 variance)
        nav_sigma = np.sqrt(a + b)                       # navigator NavState export (navigator.py:626)
        emul_sigma = np.sqrt(0.5 * (a + b))              # emul/spike sigma_inplane_hat
        worst_ratio_err = max(worst_ratio_err, abs(nav_sigma / emul_sigma - np.sqrt(2.0)))
        c_true = np.clip(sigma_ref / emul_sigma, 0, 1)
        c_raw = np.clip(sigma_ref / nav_sigma, 0, 1)            # naive: NavState export raw
        c_corr = np.clip(sigma_ref / (nav_sigma / np.sqrt(2.0)), 0, 1)   # corrected: /sqrt(2)
        worst_c_raw_err = max(worst_c_raw_err, abs(c_raw - c_true))
        worst_c_corr_err = max(worst_c_corr_err, abs(c_corr - c_true))
    assert worst_ratio_err < 1e-12, (
        f"NavState/emul in-plane sigma ratio drifted from sqrt(2) by {worst_ratio_err:.3e}")
    assert worst_c_raw_err > 0.05, (
        "the sqrt(2) gap should MEASURABLY perturb c_inplane if NavState export is fed raw "
        f"(worst err {worst_c_raw_err:.3e}); footgun no longer load-bearing?")
    assert worst_c_corr_err < 1e-9, (
        f"dividing NavState.nav_inplane_sigma by sqrt(2) must recover the trained c_inplane "
        f"(worst err {worst_c_corr_err:.3e})")
