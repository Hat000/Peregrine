"""Deploy profiles — named, opt-in bundles of the estimator + control settings for a
flight mode. ONE place that assembles the scattered ``NavigatorConfig`` flags + the
uplink ``cmd_rate_scale`` into a single, auditable preset, so a ShadowPC flight is one
named choice rather than a dozen hand-set flags that can drift out of sync.

Two profiles ship today:

- ``vq1_case_a()`` — the LEGACY VQ1 / case-A path: pristine given position/velocity on
  the wire, ODOMETRY attitude, no AHRS, no gate-relative chain. This is EXACTLY the
  default ``NavigatorConfig`` (every flag at its dataclass default) + ``cmd_rate_scale=1.0``
  (identity uplink). Provided so the legacy path is a named, tested baseline — NOT a new
  behaviour. The RL/VQ1 deploy path that does not ask for a profile is byte-identical to
  this (it constructs a bare ``NavigatorConfig``); this function just makes that explicit.

- ``vq2_case_c()`` — the VQ2 SELF-LOCALIZING (case-C) profile (GAP #5 of
  ``docs/reactivation-2026-06-27/case-c-integration-scope.md``). VQ2 §9.3 blocks
  ATTITUDE / LOCAL_POSITION_NED / ODOMETRY / GATE_INFO and the wire carries NO
  magnetometer / NO barometer, so the deployed stack must SELF-LOCALIZE: attitude from an
  IMU AHRS (``use_ahrs``), yaw + z pinned by VISION (``use_vp_yaw`` / ``use_gate_bearing_yaw``
  / ``use_floor_height``), and the gate-relative +L chain (``use_gate_relative`` +
  ``use_rewind_kf`` + ``use_range_channel``) doing the position fix — all with NO given
  position. The body-rate uplink compensates the VQ2 ~2.5x realization gain via
  ``cmd_rate_scale = 1/2.5 = 0.4`` (the live-confirmed control handshake, 2026-06-29).

These are ADDITIVE + OPT-IN. They construct config objects; they do NOT change any default.
Nothing imports this at module load on the VQ1 path, so a stack that never calls a profile
is unaffected. The case-C flags themselves are all gated OFF-by-default in ``NavigatorConfig``
(byte-identical when off); this profile is simply the curated ON bundle.
"""
from __future__ import annotations

from dataclasses import dataclass

from racer.navigator import NavigatorConfig


# The VQ2 command->realized body-rate gain (live-confirmed sim build 1.0.3379, 2026-06-29):
# a commanded body rate realizes at ~2.5x on the wire, so the uplink pre-scales by 1/2.5.
VQ2_CMD_RATE_SCALE = 0.4   # = 1 / 2.5 ; multiplies the FRD body rates at MavlinkClient.send_command

# LIVE-WIRE gyro-sign correction (live-confirmed sim build 1.0.3379, 2026-06-30): the VQ2
# HIGHRES_IMU gyro is FULLY SIGN-NEGATED on ALL THREE rate axes vs the code's FRD assumption.
# Single-axis probe (estimator-independent, cmd-vs-raw-gyro): commanded +1.0 rad/s realized raw
# gyro ~-2.1 on roll, pitch AND yaw; clean diagonal response (each cmd axis drives only its own
# gyro axis, negated; no coupling, no axis swap). So it is a GLOBAL handedness/convention mismatch
# (gyro_reported ~= -|gain|*omega), NOT y-only and NOT FRD<->FLU (which leaves roll un-inverted).
# Magnitude (the ~2.1x) is the known command->realized rate gain, handled separately by cmd_rate_scale;
# only the SIGN is corrected here. Applied at the wire (MavlinkClient.gyro_sign) before the AHRS.
VQ2_GYRO_SIGN = (-1.0, -1.0, -1.0)   # full angular-rate sign negation (roll, pitch, yaw all flipped)


@dataclass(frozen=True)
class DeployProfile:
    """A named flight preset: the estimator config + the uplink rate-scale + a label.

    ``nav_config`` -> ``Navigator(config=...)``; ``cmd_rate_scale`` -> ``MavlinkClient(cmd_rate_scale=...)``;
    ``gyro_sign`` -> ``MavlinkClient(gyro_sign=...)`` (live-wire per-axis gyro convention correction).
    ``self_localizing`` is True when the profile carries NO given position (case-C) — the deploy
    entry uses it to decide whether to thread a ground-truth seed (it must NOT in case-C).
    """

    name: str
    nav_config: NavigatorConfig
    cmd_rate_scale: float
    self_localizing: bool
    # Per-axis LIVE-WIRE gyro-sign correction -> MavlinkClient(gyro_sign=...). (1,1,1) == identity
    # (no change; VQ1 + every offline path byte-identical). vq2_case_c flips PITCH (the live-confirmed
    # HIGHRES_IMU convention mismatch). Default keeps existing callers / pickles forward-compatible.
    gyro_sign: tuple[float, float, float] = (1.0, 1.0, 1.0)


def vq1_case_a() -> DeployProfile:
    """LEGACY VQ1 / case-A deploy preset: default ``NavigatorConfig`` (given pos/vel, ODOMETRY
    attitude, no AHRS / no gate-relative chain) + identity uplink. Byte-identical to the bare
    default the VQ1 / inc7 deploy path already constructs — provided as a named, tested baseline.
    """
    return DeployProfile(
        name="vq1_case_a",
        nav_config=NavigatorConfig(),   # every flag at its dataclass default == today's VQ1 path
        cmd_rate_scale=1.0,             # identity uplink (no rate scaling)
        self_localizing=False,
        gyro_sign=(1.0, 1.0, 1.0),     # identity gyro (no live-wire correction)
    )


def vq2_case_c() -> DeployProfile:
    """VQ2 SELF-LOCALIZING (case-C) deploy preset — the curated ON bundle (GAP #5).

    Turns ON together the full self-localizing estimator chain, with NO given position:
      * ``use_given_position=False`` / ``use_given_velocity=False`` — VQ2 blocks the wire pose,
        so the KF seeds at the origin (pos_std=5.0) and self-localizes; a "case C" run is
        genuinely vision-only (P0-a).
      * ``use_ahrs=True`` — own an ESKF AHRS; source R_wb + Euler + body-rates from raw
        HIGHRES_IMU (accel + gyro), since ODOMETRY/ATTITUDE are blocked (GAP #1/#2/#3).
      * ``use_vp_yaw`` + ``use_gate_bearing_yaw`` + ``use_floor_height`` — pin yaw + z from
        VISION (no mag / no baro): vanishing-point Manhattan heading, gate-bearing yaw lock to
        the known active gate, and the floor-plane height channel.
      * ``use_gate_relative=True`` + ``use_rewind_kf=True`` + ``use_range_channel=True`` — the
        +L gate-relative in-plane fix (per-track map bias cancels), capture-time OOSM rewind,
        and the attitude-independent along-track span range (valuable while the AHRS is cold).
      * ``use_vision=True``, ``use_inplane_pos_floor=True`` (the latter is already a default;
        active only under the gate-relative/rewind chain).

    Sigmas / gates / quality thresholds stay at their (measured / bench-validated) defaults — one
    source of truth each; this profile flips the FLAGS, not the calibration constants. Uplink
    ``cmd_rate_scale = VQ2_CMD_RATE_SCALE = 0.4`` compensates the live 2.5x rate gain (the body-rate
    controller therefore uses ff_gain=1.0 so the compensation is applied EXACTLY once, at the wire).
    """
    cfg = NavigatorConfig(
        # --- no given pose (case-C foundation) ---
        use_given_position=False,
        use_given_velocity=False,
        # --- vision in the loop ---
        use_vision=True,
        # --- self-estimated attitude from IMU (ODOMETRY blocked) ---
        use_ahrs=True,
        ahrs_accel_motion_reject=True,   # A8 fix: reject accel-leveling under sustained linear accel
                                         # (|a|~=g but tilted) — the nose-up-and-retreat divergence
        # --- map-free vision yaw + z (no mag / no baro) ---
        use_vp_yaw=True,
        use_gate_bearing_yaw=True,
        use_floor_height=True,
        # --- gate-relative +L position chain ---
        use_gate_relative=True,
        use_rewind_kf=True,
        use_range_channel=True,
        use_inplane_pos_floor=True,
    )
    return DeployProfile(
        name="vq2_case_c",
        nav_config=cfg,
        cmd_rate_scale=VQ2_CMD_RATE_SCALE,
        self_localizing=True,
        gyro_sign=VQ2_GYRO_SIGN,   # live-confirmed HIGHRES_IMU PITCH-axis flip (x/z unconfirmed -> +1)
    )


# Named lookup for a one-flag CLI (`--deploy-profile vq2_case_c`). Keep keys == DeployProfile.name.
PROFILES = {
    "vq1_case_a": vq1_case_a,
    "vq2_case_c": vq2_case_c,
}


def get_profile(name: str) -> DeployProfile:
    """Resolve a profile by name (the CLI seam). Raises ``KeyError`` with the valid names listed."""
    try:
        return PROFILES[name]()
    except KeyError:
        raise KeyError(
            f"unknown deploy profile {name!r}; valid: {sorted(PROFILES)}"
        ) from None
