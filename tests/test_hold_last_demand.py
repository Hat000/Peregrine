"""Hold-last-demand bridge — continuous per-tick command through pose-gaps (A13).

THE TRAP (live VQ2 A13, 2026-06-30): on the slow VQ2 cruise the detection->pose stage drops a
USABLE pose on ~75% of ticks (the track-continuity flap rejects a candidate whose PnP range/bearing
jumps vs the EMA track; plus the _valid_poses reproj/behind filters). Each such tick routes into
``command_visual``'s regime 2 -> ``_hold_command``, which ZEROES all body rates and re-levels (a
silent zero-coast). The result is POLYGONAL motion: a real pursuit command on ~25% of ticks with
0.5-1.2 s zero/hold coasts between, and no mechanism to hold the last pursuit demand across a gap.

THE FIX: ``hold_last_demand_s`` bridges short pose gaps -- for that long after the last USABLE pose
the seeker RE-ISSUES the last pursuit demand (cached forward lean on the frozen heading) instead of
regime-2's all-axes-zero hold, so control is CONTINUOUS per tick. Default 0.0 == OFF == legacy
(byte-identical: regime-2 falls straight through to ``_hold_command``). ``hold_last_demand_decay``
scales the held forward accel full->0 over the window. vq2_case_c wires it ON (0.6 s) via the
``DeployProfile.seeker_overrides`` seam; the track-continuity gate widening is DEFERRED to A14.

[VQ2 slow-is-smooth, A13, 2026-06-30]
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from racer.contracts import ControlMode, NavState  # noqa: E402
from racer.deploy_profile import get_profile, vq1_case_a, vq2_case_c  # noqa: E402
from racer.gate_seeker import GateSeeker, GateSeekerConfig  # noqa: E402


def _nav(sim_time_ns=0, *, pitch_deg=-5.0, anchored_fix=True):
    """A NavState past the cold-start: a finite ``time_since_vision_update_s`` auto-releases the
    launch anchor (the legacy map-fix latch), so ``command_visual`` reaches regimes 2/3."""
    return NavState(
        sim_time_ns=sim_time_ns,
        position_ned=np.array([0.0, 0.0, -2.5]),
        velocity_ned=np.zeros(3),
        roll=0.0,
        pitch=np.deg2rad(pitch_deg),
        yaw=0.0,
        time_since_vision_update_s=(0.05 if anchored_fix else float("inf")),
    )


def _anchored_seeker(cfg, *, t0_ns=0):
    """Build a seeker already settled + anchored (past regimes 0/1/1.5), so a pose-None tick lands in
    regime 2 (the hold/bridge fork). We DISABLE the launch phases on the cfg (settle_s=0 +
    use_spawn_egress=False) so ``command_visual`` reaches regime 2 immediately, and mark it anchored."""
    cfg.settle_s = 0.0          # skip the post-arm settle hold (regime 0)
    cfg.use_spawn_egress = False  # skip the spawn-gate egress (regime 1.5)
    s = GateSeeker(config=cfg, detector=None)
    s._t0_sim_ns = t0_ns
    s._anchored = True
    s._release_t_ns = t0_ns
    s._last_yaw = 0.0
    return s


# A cached pursuit demand: forward heading +x (world), small yaw, a real forward accel.
_DEMAND_LOS = np.array([1.0, 0.0, 0.0])
_DEMAND_YAW = 0.0
_DEMAND_ACCEL = 1.2


def _seed_demand(seeker, t_ns=0):
    """Cache a last-good pursuit demand on the seeker (as a fresh pursuit tick would)."""
    seeker._record_last_demand(_DEMAND_LOS, _DEMAND_YAW, _DEMAND_ACCEL, t_ns)


# ===========================================================================
# 1. DEFAULT: the bridge ships OFF (field exists, 0.0) -> byte-identical legacy hold
# ===========================================================================
def test_hold_last_demand_defaults_off():
    """The bridge is opt-in: a bare GateSeekerConfig has hold_last_demand_s == 0.0 (the field exists),
    so the legacy regime-2 zero-coast (and VQ1) is byte-identical."""
    cfg = GateSeekerConfig()
    assert cfg.hold_last_demand_s == 0.0
    assert hasattr(cfg, "hold_last_demand_s")


def test_off_pose_none_returns_legacy_hold():
    """With hold_last_demand_s == 0.0 a pose-None tick returns EXACTLY the legacy ``_hold_command``
    (zeroed horizontal lean), even with a cached demand present -> byte-identical legacy path."""
    cfg = GateSeekerConfig(launch_ramp_s=0.0, hold_last_demand_s=0.0)
    seeker = _anchored_seeker(cfg)
    _seed_demand(seeker, 0)
    # _maybe_hold_last_demand must decline (None) when OFF, so the call site falls through to the hold.
    assert seeker._maybe_hold_last_demand(_nav(1_000_000)) is None
    # and the full command_visual (frame=None -> pose=None) equals the bare legacy hold.
    bridged = seeker.command_visual(_nav(1_000_000), None, 0)
    ref = _anchored_seeker(cfg)
    legacy = ref._hold_command(_nav(1_000_000),
                               yaw_rate_cap=cfg.reacquire_yaw_rate_rps, attitude_safe=True)
    assert bridged.mode is ControlMode.BODY_RATE
    np.testing.assert_array_equal(bridged.body_rate, legacy.body_rate)
    assert bridged.thrust == pytest.approx(legacy.thrust)
    # the legacy hold zeroes the horizontal lean (no forward feedforward) -> accel_ned is None/zero.
    assert bridged.accel_ned is None or np.allclose(bridged.accel_ned, 0.0)


# ===========================================================================
# 2. ON + within window: a pose-None tick re-issues a NON-zero pursuit-like command
# ===========================================================================
def test_on_within_window_bridges_with_forward_lean():
    """With hold_last_demand_s=0.6 and a cached demand within the window, a pose-None tick returns a
    pursuit-like command whose forward lean (a NON-zero pitch body-rate) is present -- NOT the zeroed
    re-level hold -- and it matches the re-issued ``_feedforward_command`` on the cached heading.

    The forward feedforward shows up as a TILT in the CTBR ``body_rate`` (pitch), not as ``accel_ned``
    on the returned command (the controller consumes the setpoint accel and emits body rates)."""
    cfg = GateSeekerConfig(launch_ramp_s=0.0, hold_last_demand_s=0.6, hold_last_demand_decay=False)
    seeker = _anchored_seeker(cfg)
    _seed_demand(seeker, 0)
    t = 100_000_000   # 0.1 s after the demand (well inside the 0.6 s window)
    nav = _nav(t, pitch_deg=0.0)   # level: the forward feedforward then demands a clear nose-down tilt
    bridged = seeker._maybe_hold_last_demand(nav)
    assert bridged is not None, "within the window the bridge must fire"
    assert bridged.mode is ControlMode.BODY_RATE and bridged.body_rate is not None
    # forward lean present (a non-zero pitch body-rate), NOT the zeroed-attitude re-level hold.
    assert float(np.linalg.norm(bridged.body_rate)) > 1e-6
    assert abs(float(bridged.body_rate[1])) > 1e-6, "the bridge must carry the forward (pitch) lean"
    # it equals the direct re-issue of _feedforward_command on the cached demand (no decay).
    ref = _anchored_seeker(cfg)
    direct = ref._feedforward_command(nav, _DEMAND_LOS, _DEMAND_YAW, None,
                                      _DEMAND_ACCEL, 1.0, vz_cmd=0.0)
    np.testing.assert_allclose(bridged.body_rate, direct.body_rate)
    assert bridged.thrust == pytest.approx(direct.thrust)
    # and it is DIFFERENT from the legacy zeroed hold (which re-levels: zero roll/pitch rate).
    legacy = _anchored_seeker(cfg)._hold_command(
        nav, yaw_rate_cap=cfg.reacquire_yaw_rate_rps, attitude_safe=True)
    assert float(legacy.body_rate[1]) == 0.0       # the hold zeroes pitch (re-level)
    assert abs(float(bridged.body_rate[1])) > abs(float(legacy.body_rate[1]))


# ===========================================================================
# 3. DECAY: the held forward accel scales full->0 over the window
# ===========================================================================
def _bridge_accel_scale(seeker, t_ns, monkeypatch):
    """Capture the EFFECTIVE forward accel the bridge passes into _feedforward_command at time t_ns,
    by intercepting the call. Returns the accel argument (a*decay) the bridge re-issued."""
    captured = {}
    orig = seeker._feedforward_command

    def spy(nav, los, yaw, launch_ramp, accel_mps2, demand_ramp, vz_cmd=0.0):
        captured["accel"] = float(accel_mps2)
        return orig(nav, los, yaw, launch_ramp, accel_mps2, demand_ramp, vz_cmd=vz_cmd)

    monkeypatch.setattr(seeker, "_feedforward_command", spy)
    seeker._maybe_hold_last_demand(_nav(t_ns))
    return captured["accel"]


def test_decay_halves_accel_at_half_window(monkeypatch):
    """With decay ON the held forward accel is ~half at ~half the window; with decay OFF it is full.
    Probed at the bridge's re-issue boundary (the accel it hands _feedforward_command)."""
    win = 0.6
    half_ns = int(0.5 * win * 1e9)
    # decay ON: at half the window the re-issued accel is ~half the cached demand.
    on = _anchored_seeker(GateSeekerConfig(launch_ramp_s=0.0, hold_last_demand_s=win,
                                           hold_last_demand_decay=True))
    _seed_demand(on, 0)
    a_half = _bridge_accel_scale(on, half_ns, monkeypatch)
    assert a_half == pytest.approx(0.5 * _DEMAND_ACCEL, rel=1e-6)
    # decay OFF: full demand at the same elapsed time.
    off = _anchored_seeker(GateSeekerConfig(launch_ramp_s=0.0, hold_last_demand_s=win,
                                            hold_last_demand_decay=False))
    _seed_demand(off, 0)
    a_full = _bridge_accel_scale(off, half_ns, monkeypatch)
    assert a_full == pytest.approx(_DEMAND_ACCEL, rel=1e-6)
    assert a_half < a_full


# ===========================================================================
# 4. WINDOW EXPIRY: past hold_last_demand_s a pose-None tick falls back to the zeroed hold
# ===========================================================================
def test_after_window_falls_through_to_legacy_hold():
    """Once elapsed exceeds hold_last_demand_s the bridge declines (None) -> the pose-None tick falls
    through to the legacy zeroed hold again (a genuinely-lost gate is not rammed forever)."""
    cfg = GateSeekerConfig(launch_ramp_s=0.0, hold_last_demand_s=0.6)
    seeker = _anchored_seeker(cfg)
    _seed_demand(seeker, 0)
    # just inside the window -> bridges; just past it -> declines.
    assert seeker._maybe_hold_last_demand(_nav(int(0.59 * 1e9))) is not None
    assert seeker._maybe_hold_last_demand(_nav(int(0.61 * 1e9))) is None
    # and the full command_visual past the window is the legacy zeroed hold.
    cmd = seeker.command_visual(_nav(int(0.61 * 1e9)), None, 0)
    assert cmd.accel_ned is None or np.allclose(cmd.accel_ned, 0.0)


# ===========================================================================
# 5. CONTINUOUS-COMMAND property: every pose-None tick within the window is bridged (no zero-coast)
# ===========================================================================
def test_continuous_command_across_consecutive_none_ticks():
    """Across N consecutive pose-None ticks within the window EVERY tick issues a NON-zero (bridged)
    command -- no silent zero-coast -- whereas the legacy (OFF) path zeroes them all. This is the core
    polygonal-motion -> continuous-command fix, driven through command_visual end-to-end."""
    dt = 1.0 / 12.0   # ~12 Hz cruise loop
    n = 6             # 6 ticks * 0.083 s = 0.5 s, inside the 0.6 s window
    # ON: seed a demand, then run N pose-None ticks (frame=None) -> all bridged + non-zero.
    # use a LEVEL nav (pitch=0): the forward feedforward then demands a clear nose-down tilt -> a
    # non-zero pitch body-rate on every bridged tick, vs the re-leveled hold's exactly-zero pitch.
    on = _anchored_seeker(GateSeekerConfig(launch_ramp_s=0.0, hold_last_demand_s=0.6,
                                           hold_last_demand_decay=True))
    _seed_demand(on, 0)
    for i in range(1, n + 1):
        t = int(i * dt * 1e9)
        cmd = on.command_visual(_nav(t, pitch_deg=0.0), None, 0)
        # the forward lean (a non-zero pitch body-rate) is present on EVERY gap tick (continuous).
        assert abs(float(cmd.body_rate[1])) > 1e-6, \
            f"tick {i}: the bridge must issue a non-zero forward (pitch) command (no zero-coast)"
    # every one of these was counted as bridged (instrumentation), none held-legacy.
    assert on.diag_counts["bridged"] == n
    assert on.diag_counts["held_legacy"] == 0
    assert on.diag_counts["none_total"] == n
    # OFF: the same N ticks re-level (zero the pitch lean) every time -- the legacy zero-coast.
    off = _anchored_seeker(GateSeekerConfig(launch_ramp_s=0.0, hold_last_demand_s=0.0))
    _seed_demand(off, 0)
    for i in range(1, n + 1):
        t = int(i * dt * 1e9)
        cmd = off.command_visual(_nav(t, pitch_deg=0.0), None, 0)
        assert float(cmd.body_rate[1]) == 0.0, \
            f"tick {i}: legacy path must zero the pitch lean (the re-level zero-coast)"
    assert off.diag_counts["bridged"] == 0
    assert off.diag_counts["held_legacy"] == n


# ===========================================================================
# 6. PROFILE wiring: vq2_case_c carries the bridge (0.6) + still the freeze; vq1_case_a None
# ===========================================================================
def test_profile_wiring():
    """vq2_case_c carries hold_last_demand_s == 0.6 via seeker_overrides AND still egress_freeze_attitude
    True; vq1_case_a leaves seeker_overrides None (VQ1 / case-A byte-identical)."""
    ov = get_profile("vq2_case_c").seeker_overrides
    assert ov["hold_last_demand_s"] == 0.6
    assert ov["egress_freeze_attitude"] is True
    assert get_profile("vq1_case_a").seeker_overrides is None
    # constructors agree with the named lookup.
    assert vq2_case_c().seeker_overrides["hold_last_demand_s"] == 0.6
    assert vq1_case_a().seeker_overrides is None


def test_make_seeker_threads_bridge_for_vq2_case_c_only():
    """A seeker built from the vq2_case_c overrides has the bridge ON (0.6); one from vq1_case_a (no
    overrides) has it OFF (0.0) -- VQ1 / case-A byte-identical. Mirrors rl.fly_rl.make_seeker's splat."""
    def _seeker(name):
        return GateSeeker(config=GateSeekerConfig(
            cruise_speed=3.0, **(get_profile(name).seeker_overrides or {})))
    assert _seeker("vq2_case_c").config.hold_last_demand_s == 0.6
    assert _seeker("vq1_case_a").config.hold_last_demand_s == 0.0


# ===========================================================================
# 7. CACHE PERSISTENCE: a pose-None tick must NOT clear the demand cache (it drives the bridge)
# ===========================================================================
def test_cache_persists_across_pose_none_ticks():
    """The _last_demand_* cache is WRITTEN only on a fresh pursuit tick and NEVER cleared by a pose-None
    tick, so it keeps driving the bridge across the whole gap. Seed it, run several pose-None ticks, and
    assert the cache is unchanged + still bridges."""
    cfg = GateSeekerConfig(launch_ramp_s=0.0, hold_last_demand_s=0.6, hold_last_demand_decay=False)
    seeker = _anchored_seeker(cfg)
    _seed_demand(seeker, 0)
    los0 = seeker._last_demand_los.copy()
    accel0, t0 = seeker._last_demand_accel, seeker._last_demand_t_ns
    dt = 1.0 / 12.0
    for i in range(1, 5):
        seeker.command_visual(_nav(int(i * dt * 1e9)), None, 0)
        # the cache is untouched by the pose-None ticks.
        np.testing.assert_array_equal(seeker._last_demand_los, los0)
        assert seeker._last_demand_accel == accel0
        assert seeker._last_demand_t_ns == t0
    # and it still drives a non-zero bridge afterward (within the window).
    cmd = seeker._maybe_hold_last_demand(_nav(int(0.4 * 1e9)))
    assert cmd is not None and abs(float(cmd.body_rate[1])) > 1e-6
