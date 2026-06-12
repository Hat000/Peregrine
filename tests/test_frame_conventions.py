"""Golden-file handedness tests (LAPTOP-FRAME-AUDIT 2026-06-12).

Pin every per-axis frame convention in the RL deploy path against RECORDED LIVE
data (tests/data/frame_audit_golden.json: banked windows from the 2026-06-12
inc6_rollfix_f1 + inc6_bridge_f1 ShadowPC flights), so no future refactor can
silently mirror an axis again. The external invariant throughout is the
finite-difference of the pristine vel_ned/pos_ned -- NEVER the telemetry's own
internal consistency (a proper frame conjugation passes every internal check;
that is precisely how two mirrors survived previous validations).

History being guarded against (three strikes):
  1. pre-bcc93f9: roll-negated quat + [-1,-1,+1] rates (level-attitude alias).
  2. bcc93f9:     quat as-is + [+1,-1,+1] rates + wire [-1,-1,-1] -- a SECOND
                  self-consistent mirror; East thrust projection flipped.
  3. FRAME-AUDIT: true quat = raw*[1,-1,1,-1] (R_y(pi) telemetry conjugation),
                  true rate = -w_raw, live command sign [+1,+1,+1],
                  wire = rate_flu*[+1,-1,+1].
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rl"))

import fly_rl
import offline_rollout as OR
from racer import frames
from racer.rl_plant import (ALPHA_MAX_RPS2_MEASURED, COLL_MAP_ACCEL_MEASURED,
                            COLL_MAP_THR_MEASURED, MIXER_IDLE_MEASURED,
                            MIXER_KAPPA_ERR_MEASURED, MIXER_KAPPA_HOLD_MEASURED,
                            MIXER_ZETA_YAW_MEASURED, PlantParams, PlantState,
                            QUAD_DRAG_C2_MEASURED, SUPER_RATE_S_MEASURED,
                            step as plant_step)

_G = 9.80665
GOLDEN = json.loads((Path(__file__).parent / "data" / "frame_audit_golden.json")
                    .read_text(encoding="utf-8"))


def _arrays(win):
    rows = GOLDEN[win]["rows"]
    return dict(
        t=np.array([r["t_mono"] for r in rows]),
        pos=np.array([r["pos_ned"] for r in rows]),
        vel=np.array([r["vel_ned"] for r in rows]),
        q=np.array([r["q_raw_wxyz"] for r in rows]),
        w=np.array([r["w_raw"] for r in rows]),
        coll=np.array([r["collective"] for r in rows]),
        wire=np.array([r["rate_frd"] for r in rows]),
    )


def _R(q_wxyz):
    return Rotation.from_quat(np.asarray(q_wxyz)[..., [1, 2, 3, 0]]).as_matrix()


def _model_accel(Rk, coll_delayed, v):
    K = np.interp(coll_delayed, COLL_MAP_THR_MEASURED, COLL_MAP_ACCEL_MEASURED)
    vb = Rk.T @ v
    cc = np.where(vb >= 0, QUAD_DRAG_C2_MEASURED[:, 0], QUAD_DRAG_C2_MEASURED[:, 1])
    return K * (Rk @ [0.0, 0.0, -1.0]) + Rk @ (-cc * np.abs(vb) * vb) + np.array([0, 0, _G])


# ---------------------------------------------------------------- constants pinned
def test_deploy_constants_pinned():
    assert np.array_equal(fly_rl._ODO_QUAT_TRUE_CONJ, [1.0, -1.0, 1.0, -1.0])
    assert np.array_equal(fly_rl._ODO_RATE_SIGN, [-1.0, -1.0, -1.0])
    assert np.array_equal(fly_rl._ACT_FLU_TO_FRD, [1.0, -1.0, 1.0])
    assert np.array_equal(OR._RATE_SIGN_LIVE, [1.0, 1.0, 1.0])
    assert np.array_equal(frames.ODO_QUAT_TRUE_CONJ_WXYZ, [1.0, -1.0, 1.0, -1.0])
    # the closed-loop composition equals the TRAINED semantics: the training
    # adapter ([1,-1,-1] FLU->FRD) into the training plant ([+1,+1,-1]) gives
    # realized = g*[+1,-1,+1]*rate_flu; deploy must compose to the same.
    trained = np.array([1.0, -1.0, -1.0]) * np.array([1.0, 1.0, -1.0])
    deployed = fly_rl._ACT_FLU_TO_FRD * OR._RATE_SIGN_LIVE
    assert np.array_equal(deployed, trained)


def test_frames_helpers_involutory():
    rng = np.random.default_rng(7)
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    q2 = frames.true_attitude_from_odo_quat_wxyz(frames.true_attitude_from_odo_quat_wxyz(q))
    assert np.allclose(q2, q)
    w = rng.normal(size=3)
    assert np.allclose(frames.true_rate_from_odo_angular_rate(
        frames.true_rate_from_odo_angular_rate(w)), w)


# ---------------------------------------------------------------- force projection
@pytest.mark.parametrize("win", ["rollfix_f1", "bridge_f1"])
def test_force_projection_handedness(win):
    """TRUE attitude (conjugated) force model must track the measured world accel
    on ALL axes at bank; the as-is reading must FAIL on East (anti-correlate).
    Measured accel = central FD of pristine vel_ned (the external invariant)."""
    d = _arrays(win)
    meas, mdl_true, mdl_raw = [], [], []
    for k in range(2, len(d["t"]) - 2):
        dt = d["t"][k + 1] - d["t"][k - 1]
        if not (0.05 < dt < 0.09):
            continue
        a = (d["vel"][k + 1] - d["vel"][k - 1]) / dt
        meas.append(a)
        c = d["coll"][k - 2]
        mdl_true.append(_model_accel(_R(d["q"][k] * fly_rl._ODO_QUAT_TRUE_CONJ), c, d["vel"][k]))
        mdl_raw.append(_model_accel(_R(d["q"][k]), c, d["vel"][k]))
    meas, mdl_true, mdl_raw = np.array(meas), np.array(mdl_true), np.array(mdl_raw)
    assert len(meas) >= 20
    for ax in range(3):
        cor = np.corrcoef(mdl_true[:, ax], meas[:, ax])[0, 1]
        assert cor > 0.8, f"TRUE attitude axis {ax}: corr {cor:.2f}"
        assert np.median(np.abs(mdl_true[:, ax] - meas[:, ax])) < 5.0
    # the mirror canary: as-is East must anti-correlate in these banked windows
    cor_e_raw = np.corrcoef(mdl_raw[:, 1], meas[:, 1])[0, 1]
    assert cor_e_raw < 0.0, f"as-is East corr {cor_e_raw:.2f} -- telemetry convention changed?"


# ---------------------------------------------------------------- rate channel
@pytest.mark.parametrize("win", ["rollfix_f1", "bridge_f1"])
def test_rate_channel_is_negated_true_rate(win):
    """quat-FD of the TRUE (conjugated) attitude == -w_raw, gain ~1, per axis."""
    d = _arrays(win)
    Rm = _R(d["q"] * fly_rl._ODO_QUAT_TRUE_CONJ[None, :])
    fd, wneg = [], []
    for k in range(len(d["t"]) - 1):
        dtk = d["t"][k + 1] - d["t"][k]
        if not (0.02 < dtk < 0.06):
            continue
        fd.append(Rotation.from_matrix(Rm[k].T @ Rm[k + 1]).as_rotvec() / dtk)
        wneg.append(-d["w"][k])
    fd, wneg = np.array(fd), np.array(wneg)
    assert len(fd) >= 20
    for ax in range(3):
        cor = np.corrcoef(fd[:, ax], wneg[:, ax])[0, 1]
        gain = np.polyfit(wneg[:, ax], fd[:, ax], 1)[0]
        assert cor > 0.8, f"axis {ax}: corr {cor:.2f}"
        assert 0.85 < gain < 1.15, f"axis {ax}: gain {gain:.2f}"


# ---------------------------------------------------------------- open-loop replay
def _mixer_params(rate_sign):
    return PlantParams(
        transport_delay_steps=0, rate_sign=np.array(rate_sign, float),
        super_rate_s=SUPER_RATE_S_MEASURED, alpha_max_rps2=ALPHA_MAX_RPS2_MEASURED.copy(),
        linear_drag=0.0, quad_drag_c2=QUAD_DRAG_C2_MEASURED.copy(),
        coll_map_thr=COLL_MAP_THR_MEASURED.copy(), coll_map_accel=COLL_MAP_ACCEL_MEASURED.copy(),
        mixer_idle=MIXER_IDLE_MEASURED, mixer_kappa_err=MIXER_KAPPA_ERR_MEASURED,
        mixer_kappa_hold=MIXER_KAPPA_HOLD_MEASURED, mixer_zeta_yaw=MIXER_ZETA_YAW_MEASURED)


def _replay(d, i0, nt, true_frame):
    if true_frame:
        q0 = d["q"][i0] * fly_rl._ODO_QUAT_TRUE_CONJ
        w0 = -d["w"][i0]
        P = _mixer_params([1, 1, 1])
    else:   # the bcc93f9 mirror reading
        q0 = d["q"][i0].copy()
        w0 = d["w"][i0] * np.array([1.0, -1.0, 1.0])
        P = _mixer_params([-1, 1, -1])
    st = PlantState(pos=d["pos"][i0].copy(), vel=d["vel"][i0].copy(),
                    quat=q0 / np.linalg.norm(q0), omega=w0,
                    thrust=np.float64(d["coll"][i0]))
    cp = None; clp = 0.0
    for i in range(i0, i0 + nt):
        if cp is not None:
            st = plant_step(st, np.concatenate([cp, [clp]]), 1.0 / 30.0, P)
        cp = d["wire"][i]; clp = d["coll"][i]
    return st


def test_openloop_replay_east_velocity_sign():
    """Seed the twin at the live pre-divergence state and drive it with the exact
    recorded wire commands through the hard-banked gate-0 flare: the TRUE-frame
    reading must reproduce the live East velocity correctly SIGNED; the bcc93f9
    mirror reading must produce the opposite-signed East velocity (S18 smoking
    gun, now pinned as a regression guard)."""
    d = _arrays("rollfix_f1")
    i0 = 36 - GOLDEN["rollfix_f1"]["k0"]          # live tick 36
    nt = 25
    live_vE = d["vel"][i0 + nt - 1][1]
    assert live_vE > 8.0                          # the window is the +E turn
    st_true = _replay(d, i0, nt, True)
    st_mirr = _replay(d, i0, nt, False)
    assert abs(st_true.vel[1] - live_vE) < 2.5, \
        f"true-frame replay vE {st_true.vel[1]:+.1f} vs live {live_vE:+.1f}"
    assert st_mirr.vel[1] < -8.0, \
        f"mirror replay should flip East: got {st_mirr.vel[1]:+.1f}"
    # attitude must also track (true roll/pitch within ~8 deg at the end tick)
    q_live = d["q"][i0 + nt - 1] * fly_rl._ODO_QUAT_TRUE_CONJ
    e_live = Rotation.from_quat(q_live[[1, 2, 3, 0]]).as_euler("ZYX")
    e_twin = Rotation.from_quat(np.asarray(st_true.quat)[[1, 2, 3, 0]]).as_euler("ZYX")
    assert np.max(np.abs(np.degrees(e_live - e_twin))) < 10.0


# ---------------------------------------------------------------- obs pipeline
def test_build_obs_telemetry_roundtrip():
    """fly_rl.build_obs must exactly undo offline_rollout.telemetry_from_truth
    (the emulated wire artifacts) at random tilted states."""
    worst = OR.check_build_obs(np.random.default_rng(0))
    assert worst < 1e-9


def test_build_obs_golden_banked_tick():
    """Pin build_obs attitude/rate dims on a recorded hard-banked live tick.
    Values computed from the FRAME-AUDIT convention (true roll -57 deg regime);
    any silent re-mirroring of the deploy path breaks these numbers."""
    d = _arrays("rollfix_f1")
    i = 51 - GOLDEN["rollfix_f1"]["k0"]           # live tick 51: roll -59, +E turn
    st = type("S", (), dict(
        position_ned=d["pos"][i], velocity_ned=d["vel"][i],
        orientation_ned_wxyz=d["q"][i], angular_rate_body=d["w"][i]))()
    obs = fly_rl.build_obs(st, 0, 0.0, virtual_flip=True)
    q_true = d["q"][i] * fly_rl._ODO_QUAT_TRUE_CONJ
    roll_true = Rotation.from_quat(q_true[[1, 2, 3, 0]]).as_euler("ZYX")[2]
    # obs[6:9] = euler of (R_w2g @ R_b2w_zup @ Rz(pi)); independently recompute
    R_ned = Rotation.from_quat(q_true[[1, 2, 3, 0]]).as_matrix()
    R_zup = (fly_rl._FLIP[:, None] * R_ned) * fly_rl._FLIP[None, :]
    rpy_g = fly_rl._euler_zyx(fly_rl._R_W2G @ (R_zup @ fly_rl._RZ_PI_BODY))
    assert np.allclose(obs[6:9], rpy_g.astype(np.float32), atol=1e-6)
    # the roll the policy sees corresponds to the TRUE physical roll, not the
    # reported one: live tick 51 banks right-side-down ~ -59 deg true
    assert -1.2 < roll_true < -0.85
    # rates: w_flu = (-w_raw FRD->true) -> FLU -> virtual flip
    w_true_frd = d["w"][i] * fly_rl._ODO_RATE_SIGN
    w_expect = fly_rl._RZ_PI_BODY @ (w_true_frd * fly_rl._FLIP)
    assert np.allclose(obs[9:12], w_expect.astype(np.float32), atol=1e-6)
