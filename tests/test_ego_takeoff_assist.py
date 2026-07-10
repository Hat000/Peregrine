"""EGO autonomous takeoff assist (A2 ground-unstick) — state machine + closed-loop punch.

Pins the fly_ego takeoff-assist contract [ego-deploy A2, 2026-07-10]:

  * THE FREEZE: the trained ego opener is a low-thrust (~0.22-0.25 g) airborne attitude re-orient;
    on the VQ2 pad its ~6 % collective never unloads the drone, so the body can't rotate, the obs
    (attitude + bearing + measured rates) is frozen, and the opener idles forever (A1: both flights
    pinned at ~0.25 g, never lifted). The assist floors the EMITTED collective just above hover to
    unload the pad, passes the rate commands through untouched, and disarms PERMANENTLY once the
    drone is airborne (measured rate OR climb OR a hard time cap).

  * The state machine (EgoTakeoffAssist) is unit-tested as a pure object: the g-units floor AND its
    [0,1] collective conversion, the obs[8] feedback reflecting the ASSISTED value, each of the
    three handover triggers, rate-is-primary ordering, no re-activation after handover, GO-relative
    timing, non-finite-signal safety, and the loud ACTIVE/HANDOVER logs.

  * HANDOVER KEYS OFF THE MEASURED gyro, NOT the commanded rate: the opener rails yaw at ±3.14 from
    tick 0, so a commanded-rate trigger would fire before the drone ever moved (pinned indirectly by
    the closed-loop rollout, whose commanded rate is railed while the measured rate ramps).

  * Closed-loop punch (real actor, skipped when the gitignored .pth is absent): feed the recorded
    A1 frozen tick-0 obs -> the policy is trapped below hover; apply the assist -> emitted 1.10 g /
    0.292 collective. Then a faithful re-orient rollout (integrate the policy's OWN commanded rates
    into the attitude, rebuild the 21-dim obs via the real EgoObsBuilder, drive the assist with the
    MEASURED rate) shows the policy's own thrust punch past hover to >2 g once airborne — the freeze
    is escapable. The sequence is PRINTED (run with `pytest -s`).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_RL = Path(__file__).resolve().parents[1] / "rl"
if str(_RL) not in sys.path:
    sys.path.insert(0, str(_RL))

import fly_rl  # noqa: E402

_HOVER = fly_rl._HOVER_THRUST                       # 0.2656 g-units -> [0,1] collective
# primary flight ckpt first, then the noise-0 sibling (both gitignored release artifacts)
_CKPT = next((p for p in (_RL / "checkpoints" / "vn16_final_actor.pth",
                          _RL / "checkpoints" / "vczext_final_actor.pth") if p.exists()), None)

# Recorded A1 frozen tick-0 obs (data/runs/20260710_061821_ego_vn16_a1_f1/ego_obs.jsonl k=0):
# body rates obs[5:8]=0, attitude/bearing frozen -> the trapped opener. vn16 replays normed 0.2543.
_A1_FROZEN_OBS0 = np.array(
    [0.0, 0.0, 0.0, -0.00025, -0.3107, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     -10.25107, 0.09354, 4.29817, 1.0, 0.94706, -0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _mk(enabled=True, assist_g=1.10, max_s=1.5):
    return fly_rl.EgoTakeoffAssist(assist_g=assist_g, hover_collective=_HOVER,
                                   max_s=max_s, enabled=enabled)


# --------------------------------------------------------------------------- floor / obs[8] math
def test_floor_below_hover_lifts_to_assist_g():
    """A trapped opener (policy 0.25 g) is floored to the assist value in g-units; the wire
    collective is the g-units floor * hover, clipped to [0,1]."""
    a = _mk(assist_g=1.10)
    emitN, coll, on = a.apply(0.25, now=0.0, gyro_frd=np.zeros(3), nav_z=0.0)
    assert on is True
    assert emitN == pytest.approx(1.10)                      # g-units floor (hover = 1.0 g)
    assert coll == pytest.approx(1.10 * _HOVER)              # 0.29216 collective
    assert coll == pytest.approx(fly_rl._clip01(1.10 * _HOVER))


def test_policy_above_floor_passes_through():
    """When the policy already commands above the floor, the assist must NOT lower it."""
    a = _mk(assist_g=1.10)
    emitN, coll, on = a.apply(2.5, now=0.0, gyro_frd=np.zeros(3), nav_z=0.0)
    assert on is True                                        # still active (no handover yet)
    assert emitN == pytest.approx(2.5)                       # policy's own value passes through
    assert coll == pytest.approx(fly_rl._clip01(2.5 * _HOVER))


def test_obs8_feedback_is_the_assisted_value():
    """The FIRST return (fed back as obs[8]) is the ACTUALLY emitted g-units value, not the raw
    policy output — feeding the assisted value is what training's punch phase (thrust_prev rising)
    looks like, and avoids the thrust_prev self-feed that helps pin the freeze."""
    a = _mk(assist_g=1.10)
    emitN, _, _ = a.apply(0.22, now=0.0, gyro_frd=np.zeros(3), nav_z=0.0)
    assert emitN == pytest.approx(1.10)                      # NOT 0.22


def test_collective_floor_never_exceeds_one():
    """A large assist_g must still clip the collective to the wire's [0,1]."""
    a = _mk(assist_g=5.0)
    _, coll, _ = a.apply(0.1, now=0.0, gyro_frd=np.zeros(3), nav_z=0.0)
    assert coll == pytest.approx(1.0)


# --------------------------------------------------------------------------- disabled = no-op
def test_disabled_is_bit_identical_passthrough():
    """--no-ego-takeoff-assist: pass-through, assist_on False, collective bit-identical to what
    policy_step computes for the same normed thrust, and NO log emitted."""
    a = _mk(enabled=False)
    logs = []
    for polN in (0.05, 0.25, 1.88, 3.765):
        emitN, coll, on = a.apply(polN, now=0.0, gyro_frd=np.zeros(3), nav_z=0.0,
                                  log=logs.append)
        assert on is False
        assert emitN == pytest.approx(polN)
        assert coll == pytest.approx(float(np.clip(polN * fly_rl._HOVER_THRUST, 0.0, 1.0)))
    assert a.active is False and logs == []                  # disabled never activates or logs


# --------------------------------------------------------------------------- handover triggers
def test_handover_on_measured_rates():
    a = _mk()
    a.apply(0.25, now=0.0, gyro_frd=np.zeros(3), nav_z=0.0)          # GO (rest)
    emitN, coll, on = a.apply(0.25, now=0.03, gyro_frd=[0.0, 0.0, 1.2], nav_z=0.0)  # any axis
    assert a.last_trigger == "rates" and a.active is False
    assert on is False and emitN == pytest.approx(0.25)             # this tick already hands back


def test_handover_on_climb():
    """NED z is down-positive; climbing 0.6 m makes z0 - z = 0.6 > 0.5."""
    a = _mk()
    a.apply(0.25, now=0.0, gyro_frd=np.zeros(3), nav_z=0.0)          # GO, z0 = 0
    a.apply(0.25, now=0.03, gyro_frd=np.zeros(3), nav_z=-0.6)        # 0.6 m up
    assert a.last_trigger == "climb" and a.active is False


def test_handover_on_timeout_measured_from_go():
    """The hard cap is measured from GO (first active tick), NOT construction — a pre-GO delay
    must not eat the budget."""
    a = _mk(max_s=1.5)
    a.apply(0.25, now=100.0, gyro_frd=np.zeros(3), nav_z=0.0)        # GO at t=100
    _, _, on = a.apply(0.25, now=101.4, gyro_frd=np.zeros(3), nav_z=0.0)   # +1.4 s: still active
    assert on is True and a.active is True
    a.apply(0.25, now=101.6, gyro_frd=np.zeros(3), nav_z=0.0)        # +1.6 s: over cap
    assert a.last_trigger == "timeout" and a.active is False


def test_rates_is_primary_over_climb_and_timeout():
    """When every criterion is satisfied at once, the reported trigger is the rate (spec: rate is
    primary; climb/nav-z is the unreliable secondary)."""
    a = _mk(max_s=1.0)
    a.apply(0.25, now=0.0, gyro_frd=np.zeros(3), nav_z=0.0)
    a.apply(0.25, now=5.0, gyro_frd=[2.0, 0.0, 0.0], nav_z=-3.0)     # rates & climb & timeout
    assert a.last_trigger == "rates"


# --------------------------------------------------------------------------- latch / safety
def test_no_reactivation_after_handover():
    """Once handed over, the assist stays disarmed even if the drone returns to rest (a settle-back
    must NEVER re-floor thrust — that is the permanent-latch guarantee)."""
    a = _mk()
    a.apply(0.25, now=0.0, gyro_frd=np.zeros(3), nav_z=0.0)
    a.apply(0.25, now=0.03, gyro_frd=[0.0, 0.0, 2.0], nav_z=0.0)     # -> rates handover
    assert a.active is False
    for k in range(5):                                              # back to rest, well within cap
        emitN, _, on = a.apply(0.10, now=0.1 + 0.03 * k, gyro_frd=np.zeros(3), nav_z=0.0)
        assert on is False and emitN == pytest.approx(0.10)         # policy owns thrust forever
    assert a.active is False


def test_non_finite_signals_never_false_handover():
    """A transient NaN gyro / NaN nav-z must not be read as 'airborne' — no false handover."""
    a = _mk(max_s=1.5)
    a.apply(0.25, now=0.0, gyro_frd=np.zeros(3), nav_z=0.0)
    _, _, on = a.apply(0.25, now=0.5, gyro_frd=[np.nan, np.nan, np.nan], nav_z=np.nan)
    assert on is True and a.active is True and a.last_trigger is None


def test_missing_signals_fall_back_to_timeout():
    """gyro None and nav_z None (no fix at GO): rate + climb are simply not-yet-triggering, and the
    hard timeout remains the backstop."""
    a = _mk(max_s=0.5)
    a.apply(0.25, now=0.0, gyro_frd=None, nav_z=None)               # GO, no signals
    _, _, on = a.apply(0.25, now=0.2, gyro_frd=None, nav_z=None)
    assert on is True                                              # still floored
    a.apply(0.25, now=0.6, gyro_frd=None, nav_z=None)              # over cap
    assert a.last_trigger == "timeout" and a.active is False


def test_active_and_handover_logs_are_loud():
    """Exactly one ACTIVE line at GO and one HANDOVER line (with trigger + GO-relative time)."""
    a = _mk()
    logs = []
    a.apply(0.25, now=10.0, gyro_frd=np.zeros(3), nav_z=0.0, log=logs.append)
    a.apply(0.25, now=10.06, gyro_frd=[0.0, 0.0, 1.5], nav_z=0.0, log=logs.append)
    assert len(logs) == 2
    assert logs[0] == "[ego-assist] ACTIVE (thrust floor 1.100 g)"
    assert logs[1].startswith("[ego-assist] HANDOVER at t=0.060s trigger=rates")


def test_single_instantiation_only_on_ego_path():
    """The assist is wired into _fly_ego only: the RL/gate-seeker loops must never construct it
    (source guard against accidental plumbing into a non-ego path)."""
    src = (_RL / "fly_rl.py").read_text(encoding="utf-8")
    assert src.count("EgoTakeoffAssist(") == 1                     # constructed exactly once
    ego_start = src.index("def _fly_ego(")
    ego_end = src.index("def _fly_armed(")
    assert ego_start < src.index("EgoTakeoffAssist(") < ego_end    # ...inside _fly_ego


# --------------------------------------------------------------------------- closed-loop punch
@pytest.fixture()
def ego_bounds():
    amin, amax = fly_rl._ACT_MIN.copy(), fly_rl._ACT_MAX.copy()
    fly_rl._apply_ego_action_bounds()
    yield
    fly_rl._ACT_MIN[:] = amin
    fly_rl._ACT_MAX[:] = amax


@pytest.mark.skipif(_CKPT is None, reason="no ego .pth pulled (gitignored release artifact; see "
                    "handoff/ego-deploy-2026-07-09/REPORT.md for the ckpt logistics)")
def test_frozen_tick0_is_trapped_and_assist_lifts_it(ego_bounds):
    """The recorded A1 frozen tick-0 obs replays the trapped opener (below hover); the assist
    floors the emission to 1.10 g / 0.292 collective and the obs[8] feedback becomes 1.10."""
    actor = fly_rl.load_ego_actor(str(_CKPT))
    _, coll, polN = fly_rl.policy_step(actor, _A1_FROZEN_OBS0, virtual_flip=True)
    print(f"\n[frozen tick-0] policy normed={polN:.4f} g  collective={coll:.4f}  (< hover 1.0 g)")
    assert polN < 0.5                                              # trapped opener, well below hover
    a = _mk(assist_g=1.10)
    emitN, ecoll, on = a.apply(polN, now=0.0, gyro_frd=np.zeros(3), nav_z=0.0)
    assert on is True and emitN == pytest.approx(1.10) and ecoll == pytest.approx(1.10 * _HOVER)
    print(f"[assist]       emitted normed={emitN:.4f} g  collective={ecoll:.4f}  (obs[8] feedback)")


@pytest.mark.skipif(_CKPT is None, reason="no ego .pth pulled")
def test_closed_loop_reorient_engages_the_punch(ego_bounds):
    """Faithful mini closed-loop: integrate the policy's OWN commanded rates into the attitude
    (realized rate lags the command), rebuild the 21-dim obs via the REAL EgoObsBuilder, and drive
    the assist with the MEASURED (realized) rate. Once the drone unloads the pad and can rotate, the
    re-orient completes and the policy's own thrust PUNCHES past hover — the freeze is escapable."""
    from scipy.spatial.transform import Rotation as Rot

    from racer import frames
    from racer.contracts import GatePose
    from racer.ego_obs import EgoObsBuilder, EgoObsBuilderConfig

    actor = fly_rl.load_ego_actor(str(_CKPT))
    assist = _mk(assist_g=1.10, max_s=1.5)
    G, HOVER, DT, TAU = 9.81, 1.0, 1.0 / 30.0, 0.06

    gate_ned = np.array([10.25, 0.093, -4.30])      # ~ the A1 frozen slot0 geometry
    drone = np.zeros(3)                             # NED position; z down-positive
    vz = 0.0                                        # NED down-velocity
    R = np.eye(3)                                   # FRD->NED attitude, level at GO
    w_real = np.zeros(3)                            # MEASURED body rate (frozen at rest -> ramps)
    builder = EgoObsBuilder(EgoObsBuilderConfig())
    last_normed = 0.0

    trace = []
    for k in range(30):
        t = k * DT
        t_ns = int(t * 1e9)
        rel_frd = R.T @ (gate_ned - drone)
        pose = GatePose(frame_id=k + 1, sim_time_ns=t_ns, R_cam_gate=np.eye(3),
                        t_cam_gate=frames.R_camera_from_body() @ rel_frd, reproj_error_px=0.5)
        obs = builder.update(sim_time_ns=t_ns, gate_index=0, R_frd2ned=R,
                             vel_ned=np.array([0.0, 0.0, vz]), gyro_frd=w_real,
                             pose=pose, last_normed_thrust=last_normed)
        rate_frd, _, polN = fly_rl.policy_step(actor, obs, virtual_flip=True)
        emitN, ecoll, on = assist.apply(polN, now=t, gyro_frd=w_real, nav_z=float(drone[2]))
        last_normed = emitN                        # obs[8] next tick = emitted (assisted) value
        trace.append((k, on, polN, emitN, ecoll, float(np.max(np.abs(w_real))), float(-drone[2])))
        # physics: realized rate lags the (railed) command; attitude integrates the realized rate;
        # vertical acceleration = (emitted_g - 1) * g.
        w_real = w_real + (rate_frd - w_real) * (DT / TAU)
        R = R @ Rot.from_rotvec(w_real * DT).as_matrix()
        vz = vz - (emitN - HOVER) * G * DT
        drone = drone + np.array([0.0, 0.0, vz]) * DT

    print("\n[closed-loop reorient]  k active polN  emitN  coll |w_meas| climb")
    for (k, on, polN, emitN, ecoll, wm, climb) in trace:
        if k < 8 or k % 3 == 0:
            print(f"  {k:>3} {str(on)[0]:>6} {polN:5.3f} {emitN:5.3f} {ecoll:5.3f} "
                  f"{wm:7.2f} {climb:6.2f}  trig={assist.last_trigger}")

    polNs = [row[2] for row in trace]
    actives = [row[1] for row in trace]
    # 1) starts trapped below hover
    assert polNs[0] < 1.0
    # 2) the policy's OWN thrust punches well past hover once airborne
    assert max(polNs) > 2.0
    assert any(p > 1.0 for p in polNs)
    # 3) the assist engaged at GO and handed over exactly once (never re-armed)
    assert actives[0] is True
    assert assist.active is False and assist.last_trigger in {"rates", "climb", "timeout"}
    # once deactivated it stays deactivated (monotone latch)
    first_off = actives.index(False)
    assert all(v is False for v in actives[first_off:])
