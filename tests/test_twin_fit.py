"""Task B -- fitting a sim-faithful CtbrPlantConfig from a ShadowPC sysid extract (twin_fit).

Two layers: (1) SYNTHETIC round-trips -- generate a response from a KNOWN plant and confirm the
fitter recovers its parameters (correctness, independent of the real data); (2) the REAL extract --
confirm the fit is plausible and the faithful config reproduces the recorded gate0_course1 flight.
"""
from pathlib import Path

import numpy as np
import pytest

from racer.contracts import ControlCommand, ControlMode
from racer.twin import CtbrPlant, CtbrPlantConfig
from racer.twin_fit import (Run, faithful_config, fit_plant, fit_rate, fit_thrust, load_run, validate)

_EXTRACT = Path(__file__).resolve().parent.parent / "handoff/shadowpc-followups-2026-06-05/sysid"
_FIT_RUNS = ("rate1", "rate2", "hsweep1", "hsweep2", "hsweep3")


def _sim_run(cfg: CtbrPlantConfig, schedule, *, dt: float = 0.02, label: str = "syn",
             mode: str = "rate") -> Run:
    """Drive a CtbrPlant(cfg) through ``schedule`` = [(rate(3), thrust, kind, axis, dur_s, phase)]
    and capture a :class:`Run` shaped like a real extract (records the plant's emitted state)."""
    plant = CtbrPlant(cfg)
    cols = {k: [] for k in ("t", "q", "cr", "ct", "orr", "v", "p", "ph", "kd", "ax")}
    for rate, thrust, kind, axis, dur, phase in schedule:
        for _ in range(int(round(dur / dt))):
            st = plant.state()
            cols["t"].append(plant.t_ns); cols["q"].append(st.orientation_ned_wxyz.tolist())
            cols["cr"].append(list(rate)); cols["ct"].append(thrust)
            cols["orr"].append(st.angular_rate_body.tolist()); cols["v"].append(st.velocity_ned.tolist())
            cols["p"].append(st.position_ned.tolist()); cols["ph"].append(phase)
            cols["kd"].append(kind); cols["ax"].append(axis)
            plant.step(ControlCommand(mode=ControlMode.BODY_RATE, body_rate=np.array(rate, float),
                                      thrust=thrust), dt)
    return Run(label=label, t=np.array(cols["t"], float) * 1e-9, q=np.array(cols["q"], float),
               cmd_rate=np.array(cols["cr"], float), cmd_thrust=np.array(cols["ct"], float),
               odo_rate=np.array(cols["orr"], float), vel_world=np.array(cols["v"], float),
               pos=np.array(cols["p"], float), phase=cols["ph"], kind=cols["kd"], axis=cols["ax"],
               meta={"mode": mode})


def test_fit_rate_recovers_a_known_plant():
    # A plant whose rate loop INVERTS roll+yaw (not pitch) at ~2.6x with tau 0.04 s -- the fitter
    # must recover the signed gain (-> magnitude + sign) and tau from a doublet response.
    cfg = CtbrPlantConfig(rate_tau_s=0.04, rate_gain=np.array([2.70, 2.60, 2.40]),
                          rate_sign=np.array([-1.0, 1.0, -1.0]))
    sched = []
    for ax, name in ((1, "pitch"), (0, "roll"), (2, "yaw")):
        sched.append((np.zeros(3), 0.26, "hold", -1, 0.4, "lvl"))
        rp = np.zeros(3); rp[ax] = 0.25
        rm = np.zeros(3); rm[ax] = -0.25
        sched.append((rp, 0.26, "step", ax, 0.5, f"{name}+"))
        sched.append((rm, 0.26, "step", ax, 0.5, f"{name}-"))
    rf = fit_rate([_sim_run(cfg, sched)])
    np.testing.assert_allclose(rf["gain"], [2.70, 2.60, 2.40], rtol=0.05)
    np.testing.assert_array_equal(rf["sign"], [-1.0, 1.0, -1.0])
    assert 0.02 < rf["tau"] < 0.08            # recovers ~0.04 s (within discretisation)


def test_fit_thrust_recovers_a_known_hover():
    # Level holds at several collectives -> the fitter recovers the hover (zero net vertical accel).
    cfg = CtbrPlantConfig(hover_thrust=0.27)
    sched = [(np.zeros(3), th, "hold", -1, 0.6, f"thr{th}") for th in (0.24, 0.26, 0.28, 0.30)]
    tf = fit_thrust([_sim_run(cfg, sched)])
    assert tf["n"] >= 3
    assert tf["hover"] == pytest.approx(0.27, abs=0.01)


def test_fit_recovers_plant_from_the_real_extract():
    # The real ShadowPC doublets: roll & yaw command-inverted (not pitch), |gain| ~2.2-2.6, hover
    # ~0.26, fast inner-loop tau. (No drag fit here -> faster; drag is exercised below.)
    runs = [load_run(_EXTRACT / f"{n}.json") for n in _FIT_RUNS]
    cfg, rep = fit_plant(runs)
    np.testing.assert_array_equal(cfg.rate_sign, [-1.0, 1.0, -1.0])
    assert np.all((cfg.rate_gain > 2.0) & (cfg.rate_gain < 3.0))
    assert 0.24 < cfg.hover_thrust < 0.28
    assert 0.005 < cfg.rate_tau_s < 0.05
    assert rep["rate"]["per_axis"]["pitch"]["r2"] > 0.99      # clean linear rate fit


def test_faithful_config_reproduces_course1():
    # The committed faithful config (twin_fit.faithful_config) reproduces the recorded closed-loop
    # gate0_course1 attitude + velocity when driven by its commands. One-step-ahead is the
    # drag-independent fidelity check; open-loop confirms the drag closes the velocity drift.
    course = load_run(_EXTRACT / "gate0_course1.json")
    v = validate(faithful_config(), course)
    assert v["n"] > 200
    assert np.all(v["step_att_rms_deg"] < 0.5)               # per-step attitude < 0.5 deg
    assert np.all(v["step_vel_rms_mps"] < 0.05)              # per-step velocity < 5 cm/s
    assert np.all(v["ol_att_rms_deg"] < 1.0)                 # open-loop attitude < 1 deg over 6.6 s
    assert v["ol_speed_rms_mps"] < 0.6                       # open-loop speed drift bounded by drag
