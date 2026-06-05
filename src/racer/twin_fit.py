"""twin_fit -- fit a SIM-FAITHFUL :class:`~racer.twin.CtbrPlantConfig` from a ShadowPC sysid extract
(Task B / roadmap B). Consumes the downsampled per-run JSON in
``handoff/shadowpc-followups-2026-06-05/sysid/`` (schema in that dir's ``manifest.json``) and fits:

  * **rate loop** -- ``rate_gain`` (|steady realized/commanded| per axis, ~2.5-2.7), ``rate_sign``
    (the sim's command-sign inversions: roll & yaw inverted, pitch not), ``rate_tau_s`` (inner-loop
    first-order time constant, from the step rise). REALIZED rate is the SIGN-CORRECT quaternion
    finite-difference (``frames.body_rate_from_quats``), NOT the raw ODOMETRY ``angular_rate`` (which
    is sign-inverted on pitch -- a measurement quirk the controller's ``odo_rate_sign`` undoes, not a
    plant property).
  * **thrust** -- ``hover_thrust`` (collective at zero net vertical accel) + the slope d(up-accel)/
    d(thrust), POOLED across the discrete held levels {0.25, 0.30, 0.48} (the hsweeps aborted on the
    altitude limit, so there is no single clean sweep) plus the near-hover rate/course holds.

All fits go through :mod:`racer.sysid` (``fit_rate_gain`` / ``fit_hover_thrust`` /
``step_response_metrics``). The output ``CtbrPlantConfig`` is then VALIDATED open-loop against the
clean closed-loop ``gate0_course1`` run (drive the twin with its recorded command stream, compare the
reproduced attitude/velocity to the recording). The validation -- not the fit alone -- is what earns
transfer-confidence; a poor reproduction means the first-order model structure is wrong (say so).

NB: this fits the CTBR rotor PLANT only. The sim's broken vertical *velocity-setpoint* auto-thrust
(the "altitude balloon", memo Section 7) is a SEPARATE, still-uncharacterised path that the twin does
NOT model -- do not assume a good fit here covers it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from racer.contracts import ControlCommand, ControlMode
from racer.frames import body_rate_from_quats, euler_from_quat_wxyz
from racer.sysid import fit_hover_thrust, fit_rate_gain, step_response_metrics
from racer.twin import CtbrPlant, CtbrPlantConfig

_G = 9.80665
_AXES = ("roll", "pitch", "yaw")


# --------------------------------------------------------------------------- loading
@dataclass
class Run:
    """One parsed sysid run: per-sample arrays in a uniform shape (NaN where a field is absent)."""

    label: str
    t: np.ndarray            # (N,) seconds (relative)
    q: np.ndarray            # (N,4) ODOMETRY quaternion wxyz (body->world)
    cmd_rate: np.ndarray     # (N,3) OUR commanded body rate (rad/s); NaN before the first command
    cmd_thrust: np.ndarray   # (N,) OUR commanded collective 0..1; NaN before the first command
    odo_rate: np.ndarray     # (N,3) RAW ODOMETRY angular rate (pitch sign-inverted)
    vel_world: np.ndarray    # (N,3) odo_vel_world (the c3b5a8e-corrected world velocity)
    pos: np.ndarray          # (N,3) odo_pos (world NED)
    phase: list              # (N,) ctx phase label
    kind: list               # (N,) ctx kind ('hold'/'step'/None)
    axis: list               # (N,) ctx axis (0/1/2 for a step, else -1/None)
    meta: dict


def _v(x, n):
    """Coerce ``x`` to a length-``n`` float vector, or NaNs if it is missing/malformed."""
    if isinstance(x, list) and len(x) == n:
        return np.asarray(x, dtype=np.float64)
    return np.full(n, np.nan)


def load_run(path: str | Path) -> Run:
    """Parse one extract JSON (see manifest ``per_sample_fields``) into a :class:`Run`."""
    d = json.loads(Path(path).read_text())
    S = d["samples"]
    ctx = [s.get("ctx") or {} for s in S]
    return Run(
        label=d.get("meta", {}).get("label", Path(path).stem),
        t=np.asarray([s["t_ns"] for s in S], dtype=np.float64) * 1e-9,
        q=np.asarray([s["odo_q_wxyz"] for s in S], dtype=np.float64),
        cmd_rate=np.asarray([_v(s.get("cmd_body_rate"), 3) for s in S]),
        cmd_thrust=np.asarray([s.get("cmd_thrust") if isinstance(s.get("cmd_thrust"), (int, float))
                               else np.nan for s in S], dtype=np.float64),
        odo_rate=np.asarray([_v(s.get("odo_angular_rate"), 3) for s in S]),
        vel_world=np.asarray([_v(s.get("odo_vel_world"), 3) for s in S]),
        pos=np.asarray([_v(s.get("odo_pos"), 3) for s in S]),
        phase=[c.get("phase") for c in ctx],
        kind=[c.get("kind") for c in ctx],
        axis=[c.get("axis") for c in ctx],
        meta=d.get("meta", {}),
    )


def realized_rate(run: Run) -> np.ndarray:
    """Sign-correct body rate (N,3) by quaternion finite-difference (sidesteps the inverted raw
    ODOMETRY pitch rate). Row 0 is zero (no previous sample)."""
    w = np.zeros((len(run.t), 3))
    for i in range(1, len(run.t)):
        w[i] = body_rate_from_quats(run.q[i - 1], run.q[i], run.t[i] - run.t[i - 1])
    return w


def _segments(run: Run):
    """Yield ``(i, j, phase, kind, axis)`` for each contiguous same-phase span."""
    i = 0
    n = len(run.phase)
    while i < n:
        j = i
        while j < n and run.phase[j] == run.phase[i]:
            j += 1
        yield i, j, run.phase[i], run.kind[i], run.axis[i]
        i = j


# --------------------------------------------------------------------------- rate fit
def fit_rate(runs: list[Run], *, settle_frac: float = 0.35) -> dict:
    """Fit ``rate_gain`` (|.|), ``rate_sign``, ``rate_tau_s`` per axis from the ``*_rate*`` doublets.

    Per step segment: commanded = the held cmd on its axis; STEADY realized = mean of the settled
    tail (last ``settle_frac``) of the quaternion-finite-diff rate; tau from ``step_response_metrics``
    (the segment plus a few pre-step baseline samples). ``fit_rate_gain`` then regresses steady-vs-
    commanded per axis (pooling +/- magnitudes through 0), giving the SIGNED gain -> magnitude +
    sign. Returns ``{gain(3), sign(3), tau, per_axis, n_segments}``."""
    pairs: dict[int, list[tuple[float, float]]] = {0: [], 1: [], 2: []}
    taus: list[float] = []
    for run in runs:
        w = realized_rate(run)
        for i, j, _phase, kind, axis in _segments(run):
            if kind != "step" or axis is None or axis < 0:
                continue
            cmd = float(np.nanmedian(run.cmd_rate[i:j, axis]))
            if not np.isfinite(cmd) or abs(cmd) < 1e-6:
                continue
            seg = w[i:j, axis]
            k = max(1, int(len(seg) * settle_frac))
            steady = float(np.mean(seg[-k:]))
            pairs[axis].append((cmd, steady))
            # tau: include a short pre-step baseline so the rise is measured from ~0
            p0 = max(0, i - 5)
            m = step_response_metrics(run.t[p0:j], w[p0:j, axis], t_step=run.t[i])
            if np.isfinite(m["tau_s"]) and m["tau_s"] > 0:
                taus.append(m["tau_s"])
    gain = np.ones(3)
    sign = np.ones(3)
    per_axis = {}
    for axis in (0, 1, 2):
        if not pairs[axis]:
            per_axis[_AXES[axis]] = {"gain": float("nan"), "n": 0}
            continue
        c = np.asarray([p[0] for p in pairs[axis]])
        m = np.asarray([p[1] for p in pairs[axis]])
        # anchor the line through the origin (a zero command -> zero rate) by appending (0, 0)
        fit = fit_rate_gain(np.append(c, 0.0), np.append(m, 0.0))
        g = fit["gain"]
        gain[axis] = abs(g)
        sign[axis] = -1.0 if g < 0 else 1.0
        per_axis[_AXES[axis]] = {"gain": float(g), "r2": fit["r2"], "n": int(fit["n"])}
    tau = float(np.median(taus)) if taus else float("nan")
    return {"gain": gain, "sign": sign, "tau": tau, "per_axis": per_axis,
            "n_segments": sum(len(v) for v in pairs.values()), "tau_n": len(taus)}


# --------------------------------------------------------------------------- thrust fit
def fit_thrust(runs: list[Run], *, level_cos: float = 0.98, min_seg: int = 10,
               skip_modes: tuple = ("hover",)) -> dict:
    """Fit ``hover_thrust`` + slope by pooling near-LEVEL, ~constant-collective HOLD windows.

    Per window: ``thrust`` = the held collective; net WORLD-UP accel = -d(vz)/dt (NED z+ down), from a
    linear fit of vz over the window's settled (latter 60%) part. ``fit_hover_thrust`` then regresses
    up-accel vs thrust -> (hover, slope); hover = thrust at zero net vertical accel.

    The hsweep (mode='hover') runs are SKIPPED: per the extract manifest their odo_vel_world is
    unreliable (inflated 0.04-0.06 m/s by the 75/97 Hz LPN/ODO stagger during their fast vertical
    aborts) -- they read physically-impossible net-up (e.g. +7 m/s^2 at a 0.25 sinking collective). So
    hover comes from the RELIABLE near-hover level holds in the rate runs (slow vertical motion, RMS
    ~0.011), which bracket it; the twin's thrust model is one-parameter (``a_up = g*thrust/hover``), so
    a tight hover matters more than a wide-range slope. Level + hold only so body thrust ~= vertical
    thrust. Returns ``{hover, slope, points, n}``."""
    thr: list[float] = []
    acc: list[float] = []
    for run in runs:
        if run.meta.get("mode") in skip_modes:
            continue
        cos_tilt = np.array([float(np.cos(euler_from_quat_wxyz(q)[0]) * np.cos(euler_from_quat_wxyz(q)[1]))
                             for q in run.q])
        for i, j, _phase, kind, _axis in _segments(run):
            if kind != "hold" or j - i < min_seg:                    # dedicated level holds only
                continue
            th = run.cmd_thrust[i:j]
            if not np.all(np.isfinite(th)) or np.ptp(th) > 0.005:     # ~constant collective only
                continue
            tail = slice(i + int((j - i) * 0.4), j)                   # let the vertical accel settle
            if float(np.mean(cos_tilt[tail])) < level_cos:            # near-level only
                continue
            t, vz = run.t[tail], run.vel_world[tail, 2]
            if not np.all(np.isfinite(vz)) or np.ptp(t) < 0.1:
                continue
            az = float(np.polyfit(t - t[0], vz, 1)[0])                # d(vz)/dt (NED down+)
            thr.append(float(np.mean(th)))
            acc.append(-az)                                           # net world-up accel
    hover, slope = fit_hover_thrust(thr, acc)
    return {"hover": hover, "slope": slope, "points": sorted(zip(thr, acc)), "n": len(thr)}


# --------------------------------------------------------------------------- assemble + validate
def fit_drag(config: CtbrPlantConfig, course: Run, *, lo: float = 0.0, hi: float = 0.6,
             iters: int = 3, n: int = 13) -> float:
    """Estimate world-frame linear drag (1/s) by minimising the twin's OPEN-LOOP speed RMS vs the
    recorded ``course`` run (coarse-to-fine 1-D scan, deterministic).

    Drag is the real sim's terminal velocity (~4 m/s in level-ish forward flight) that the IDEAL-rotor
    twin lacks -- visible ONLY in the sustained-forward ``course1`` (the rate/hsweep runs are too short
    and erratic). So unlike rate_gain/hover (fit from the open-loop probes, independent of course1),
    drag is estimated FROM the validation run; the attitude + one-step-ahead velocity RMS remain the
    independent fidelity checks (drag is a translational force, it does not touch the attitude)."""
    from dataclasses import replace

    def speed_rms(d: float) -> float:
        v = validate(replace(config, linear_drag=float(d)), course)
        return v["ol_speed_rms_mps"] if v["n"] else 1e9

    best = lo
    for _ in range(iters):
        grid = np.linspace(lo, hi, n)
        vals = [speed_rms(d) for d in grid]
        b = int(np.argmin(vals))
        best = float(grid[b])
        step = (hi - lo) / (n - 1)
        lo, hi = max(0.0, best - step), best + step
    return best


def fit_plant(runs: list[Run], *, drag_run: Run | None = None) -> tuple[CtbrPlantConfig, dict]:
    """Fit a full faithful :class:`CtbrPlantConfig` from the runs. Returns ``(config, report)``.

    ``drag_run`` (course1): if given, also estimate ``linear_drag`` from it (see :func:`fit_drag`);
    else drag stays 0 (an ideal-rotor twin that over-runs in sustained forward flight)."""
    from dataclasses import replace

    rate_runs = [r for r in runs if r.meta.get("mode") == "rate"]
    rf = fit_rate(rate_runs or runs)
    tf = fit_thrust(runs)
    hover = tf["hover"] if np.isfinite(tf["hover"]) else 0.26
    cfg = CtbrPlantConfig(hover_thrust=float(hover), rate_tau_s=float(rf["tau"]),
                          rate_gain=rf["gain"].copy(), rate_sign=rf["sign"].copy())
    drag = 0.0
    if drag_run is not None:
        drag = fit_drag(cfg, drag_run)
        cfg = replace(cfg, linear_drag=float(drag))
    return cfg, {"rate": rf, "thrust": tf, "drag": drag}


def validate(config: CtbrPlantConfig, course: Run) -> dict:
    """Drive a :class:`CtbrPlant` (``config``) with ``course``'s recorded command stream from its
    initial recorded state and compare the reproduced attitude + velocity to the recording.

    Reports BOTH:
      * open-loop trajectory RMS (integrate the whole run; accumulates any per-step bias) and
      * one-step-ahead RMS (re-init each step from the RECORDED state, predict one dt) -- which
        isolates per-step model fidelity from integration drift.
    Attitude RMS in degrees (roll/pitch/yaw), velocity RMS in m/s. ``n`` usable steps."""
    n = len(course.t)
    valid = [k for k in range(n) if np.all(np.isfinite(course.cmd_rate[k]))
             and np.isfinite(course.cmd_thrust[k])]
    if len(valid) < 3:
        return {"n": 0}
    k0 = valid[0]

    def _cmd(k):
        return ControlCommand(mode=ControlMode.BODY_RATE, body_rate=course.cmd_rate[k],
                              thrust=float(course.cmd_thrust[k]))

    # -- open-loop: one rollout from the first commanded sample --
    plant = CtbrPlant(config, position_ned=course.pos[k0], velocity_ned=course.vel_world[k0],
                      q_wxyz=course.q[k0])
    att_err, vel_err = [], []
    for k in range(k0, n - 1):
        if not (np.all(np.isfinite(course.cmd_rate[k])) and np.isfinite(course.cmd_thrust[k])):
            break
        plant.step(_cmd(k), course.t[k + 1] - course.t[k])
        st = plant.state()
        rec = np.array(euler_from_quat_wxyz(course.q[k + 1]))
        att_err.append(_wrap(np.array([st.roll, st.pitch, st.yaw]) - rec))
        vel_err.append(st.velocity_ned - course.vel_world[k + 1])

    # -- one-step-ahead: re-init from the recorded state each step --
    att1, vel1 = [], []
    for k in valid:
        if k + 1 >= n or not np.all(np.isfinite(course.cmd_rate[k])):
            continue
        p = CtbrPlant(config, position_ned=course.pos[k], velocity_ned=course.vel_world[k],
                      q_wxyz=course.q[k])
        p.step(_cmd(k), course.t[k + 1] - course.t[k])
        st = p.state()
        rec = np.array(euler_from_quat_wxyz(course.q[k + 1]))
        att1.append(_wrap(np.array([st.roll, st.pitch, st.yaw]) - rec))
        vel1.append(st.velocity_ned - course.vel_world[k + 1])

    def _rms(e):
        e = np.asarray(e)
        return np.sqrt(np.mean(e ** 2, axis=0)) if len(e) else np.full(3, np.nan)

    return {
        "n": len(att_err),
        "ol_att_rms_deg": np.degrees(_rms(att_err)),
        "ol_vel_rms_mps": _rms(vel_err),
        "ol_speed_rms_mps": float(np.sqrt(np.mean(np.sum(np.square(vel_err), axis=1)))) if vel_err else float("nan"),
        "step_att_rms_deg": np.degrees(_rms(att1)),
        "step_vel_rms_mps": _rms(vel1),
    }


def _wrap(a: np.ndarray) -> np.ndarray:
    """Wrap an angle error to (-pi, pi] so a yaw near +-pi doesn't read as a ~2pi error."""
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---- FAITHFUL CONFIG (from scripts/fit_twin.py on the 2026-06-05 ShadowPC extract; deterministic) --
# rate_gain/rate_sign/rate_tau + hover_thrust are fit from the OPEN-LOOP rate/hsweep probes
# (independent of course1); linear_drag is estimated from course1's sustained forward flight (the only
# run that reveals the sim's terminal velocity). VALIDATED open-loop vs gate0_course1: attitude RMS
# 0.17/0.44/0.36 deg (roll/pitch/yaw), speed RMS 0.285 m/s; one-step-ahead velocity RMS < 0.006 m/s.
# rate_sign = the sim's COMMAND inversions (roll & yaw inverted -> controller body_rate_sign=[-1,1,-1]
# undoes it; ff_gain ~= the |gain| ~2.5 undoes the amplification). The ODOMETRY pitch-rate measurement
# inversion (raw angular_rate vs the quat finite-diff) is a SEPARATE reporting quirk -- modelled by
# odo_rate_report_sign for the closed-loop Task-C re-tune, not needed for this open-loop validation.
def faithful_config():
    """The sim-faithful :class:`CtbrPlantConfig` fitted from the ShadowPC sysid extract (Task B)."""
    return CtbrPlantConfig(
        hover_thrust=0.2656,
        rate_tau_s=0.0190,
        rate_gain=np.array([2.501, 2.504, 2.231]),
        rate_sign=np.array([-1.0, 1.0, -1.0]),
        linear_drag=0.2111,
    )
