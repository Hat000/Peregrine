"""Gate-seeker — a TRANSPARENT slow pursuit guidance law for the VQ2 self-localizing lap.

Per the "slow is smooth" curriculum (2026-06-29): the FIRST closed-loop VQ2 target is a
slow, zero-contact, fully self-localized lap — NOT a fast time and NOT the RL racing
policy. The RL policy is a black box trained off-distribution (it never saw a self-
localized estimate); to ISOLATE the self-localization we want a controller whose every
command we can explain. This module is that controller: a point-and-go pursuit law that
flies the drone slowly through the ACTIVE gate's centre.

Contract
--------
``GateSeeker.command(nav, gate, active_gate_index, *, is_final_gate=False, dt=None)
   -> ControlCommand`` (BODY_RATE / CTBR — body rates FRD + normalized collective [0,1]).

Inputs:
  * ``nav`` : :class:`NavState` — the SELF pose (position + velocity in world NED, attitude
    Euler, body rates) from the case-C Navigator. We consume only ``position_ned`` /
    ``velocity_ned`` / attitude / rates — the same fields the proven decoupled CTBR controller
    reads. We do NOT read ``time_since_vision_update_s`` here (the caller owns the coast policy).
  * ``gate`` : :class:`Gate` — the ACTIVE target gate's world pose (centre + frame). This is
    the gate the navigator localizes against; on the live wire it is selected by
    ``RACE_STATUS.active_gate_index`` (``DroneState.active_gate_index``).
  * ``active_gate_index`` : int — the current target index (drives the advance bookkeeping).

Output: a CTBR :class:`ControlCommand` ready for ``MavlinkClient.send_command`` (which applies
the deploy ``cmd_rate_scale``).

Guidance law (transparent, bounded)
-----------------------------------
1. **Steer the velocity at the gate CENTRE.** Build a carrot a short distance BEYOND the gate
   along its through-direction (so the line-of-sight passes through the opening, not the front
   bar), and ask for a desired velocity of magnitude ``cruise_speed`` pointing at that carrot.
   The decoupled CTBR controller turns a (capped) velocity error into a tilt and the matching
   collective; the cross-track component is corrected at full gain (centering) while the
   along-track speed is capped — exactly the proven ReactivePlanner split.
2. **Keep the gate in the +20deg camera view + hold altitude.** Yaw faces the gate so the
   forward-and-up-tilted camera sees the opening (the estimator needs the gate in frame to fix
   yaw/z). Altitude is held by the controller's velocity-damped alt-hold around hover thrust
   (z from the estimator) — we never command a vertical velocity, so blur-free slow flight
   stays level.
3. **Slow + bounded.** ``cruise_speed`` is small (default 3 m/s); the controller caps the
   commanded body rate and tilt. A launch ramp eases the takeoff transient from the start-gate
   spawn so the attitude target never STEPS to a cruise lean (the build-1.0.3364 tumble guard).
4. **Advance.** The caller advances ``active_gate_index`` from ``RACE_STATUS`` (the authoritative
   gate-ordering signal). This module additionally exposes :meth:`should_advance` as a
   RANGE-based backstop: advance when the drone has passed the gate plane (along-track sign flips
   to the exit side) within a capture radius, OR — for the FINAL gate, which has no "next" gate to
   re-aim at — when the drone is within ``final_blowout_m`` of the gate centre, dead-reckoning
   straight through (the recon "final blow-out" past the last station).

This is an ALTERNATIVE control source to the RL policy: same uplink (BODY_RATE), different brain.
It reuses the flight-proven decoupled :class:`Controller` geometry + sim-sign compensation, so no
new CTBR math is introduced — only the slow, gate-pointing SETPOINT on top of it.

MAP-FREE VISUAL SERVO (the VQ2 live path — 2026-06-29 slow-lap fix)
------------------------------------------------------------------
The map-based :meth:`plan` / :meth:`command` above steer to an ABSOLUTE world gate position
(``gate.position_ned``) using the estimator's self-position. On the live VQ2 wire that is FATAL
at launch: VQ2 broadcasts NO gate map (fly_rl fell back to a stale VQ1 map whose gate 0 sat at
world ``(-23.3, -0.4, 0)``) and the estimator SEEDS at the origin with NO vision fix yet, so the
first tick demanded a ~180deg yaw U-turn — spinning the start gate (which the camera ALREADY saw
at spawn) out of frame before the estimator could anchor. See ``handoff/vq2-slowlap-2026-06-29``.

:meth:`command_visual` is the fix: it chases the gate the CAMERA SEES, never an absolute map
position. It runs the injected detector + PnP on the live frame to recover the active gate's
RELATIVE lever ``t_cam_gate`` (gate centre in the camera optical frame), rotates that bearing into
the body/world frame with the estimator's gravity-known attitude, and steers heading + velocity to
CENTER and fly THROUGH the seen opening at the slow cap. NO absolute gate map and NO absolute
self-position enter the steering. Two safety behaviours guard the blind-launch failure:

  * **Post-arm settle + launch anchor.** At arm the seeker HOLDS a conservative LEVEL attitude with
    bounded hover thrust and ALL rates clamped (roll/pitch/yaw) for a short settle window, so the
    cold mag-free AHRS gravity-aligns and the gyro bias converges before any lean (the 2026-06-29
    attempt-2 cold-AHRS tumble fix). It then stays in the launch anchor until it has SEEN the gate
    for N consecutive quality-gated detections of its OWN detector (``anchor_release_detections``)
    -- the MAP-FREE release signal the seeker owns, since on the live VQ2 wire the navigator runs
    map-free and ``nav.time_since_vision_update_s`` never goes finite (a finite tsv, when a real
    map IS present, latches the anchor too). It never maneuvers blind and never slews the visible
    start gate out of frame; normal pursuit begins only after release.
  * **No detection.** When the detector returns nothing this tick (between gates / momentarily
    lost) the seeker HOLDS heading and coasts level — never a blind large slew. It re-acquires
    when the gate re-enters frame.

The map-based path is retained UNCHANGED for the offline dry-run's kinematic check and the unit
tests (which use a consistent synthetic map); only the LIVE VQ2 deploy uses ``command_visual``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from racer.contracts import (
    ControlCommand,
    ControlMode,
    Frame,
    Gate,
    GateObservation,
    GatePose,
    NavState,
    Setpoint,
)
from racer.controller import Controller
from racer.frames import R_camera_from_body, R_world_from_body
from racer.vision.gate_pose import estimate_gate_pose


def _unit(v: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    """Unit vector; return ``fallback`` (or a zero vector) when ``v`` is ~0."""
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.zeros(3) if fallback is None else np.asarray(fallback, dtype=np.float64)
    return np.asarray(v, dtype=np.float64) / n


# Flight-proven decoupled-CTBR sim-sign compensation (MEASURED on ShadowPC; see
# scripts/twin_fly_course._FAITHFUL_SIGNS). The ODOMETRY roll-quat + roll/pitch-rate reporting
# inversions and the roll/yaw COMMAND inversion the live sim needs. ff_gain is 1.0 here (NOT 2.5):
# the VQ2 deploy applies the ~2.5x rate compensation at the UPLINK (cmd_rate_scale=0.4), so undoing
# it again in the controller would double-compensate. Apply it EXACTLY once, at the wire.
SEEKER_SIGNS = dict(
    body_rate_sign=np.array([1.0, 1.0, -1.0]),     # roll/yaw command inversion (flight-correct set)
    odo_att_sign=np.array([-1.0, 1.0, 1.0]),       # ODOMETRY-quat roll inverted
    odo_rate_sign=np.array([-1.0, -1.0, 1.0]),     # ODOMETRY rate: roll + pitch inverted
)


def make_seeker_controller(**overrides) -> Controller:
    """Build the decoupled CTBR controller the gate-seeker drives — the flight-proven gain set
    (twin_fly_course._CANONICAL_GAINS) + the live sim-sign compensation, with ``ff_gain=1.0``
    (the uplink cmd_rate_scale owns the 2.5x compensation). ``overrides`` set Controller fields."""
    params = dict(
        mode=ControlMode.BODY_RATE, decoupled=True,
        hover_thrust=0.2656,                       # fitted plant hover (plant-matched)
        kp_pos=1.2, kd_vel=3.0, max_speed=3.0, max_accel_mps2=8.0,
        kp_att=10.0, kd_att=0.30, max_body_rate_rps=4.0, ff_gain=1.0,
        kp_alt=2.0, kd_alt=3.0, alt_thrust_lo=0.05, alt_thrust_hi=0.6, tilt_comp=True,
    )
    params.update(SEEKER_SIGNS)
    params.update(overrides)
    return Controller(**params)


@dataclass
class GateSeekerConfig:
    """Tunables for the slow gate-seeker guidance law (all conservative defaults)."""

    cruise_speed: float = 3.0          # m/s, capped forward speed toward the gate (SLOW first)
    lookahead_m: float = 2.0           # carrot distance BEYOND the gate centre along its normal
    capture_radius_m: float = 0.6      # along-track capture: count the gate passed within this of its plane
    final_blowout_m: float = 4.0       # final gate: dead-reckon straight through within this range
    launch_ramp_s: float = 0.6         # ease the takeoff->cruise tilt over this long (anti-tumble)
    yaw_mode: str = "carrot"           # "carrot" faces the line-of-sight; "course" faces the gate normal

    # --- MAP-FREE visual servo (the live VQ2 path; command_visual) ---
    # Before the estimator's FIRST accepted vision fix the seeker is in the LAUNCH-ANCHOR phase:
    # it holds attitude and clamps the yaw rate to this small value so it never slews the visible
    # start gate out of frame (the 2026-06-29 blind-launch failure). 0.0 => hold yaw exactly.
    anchor_yaw_rate_rps: float = 0.0
    # When NO gate is detected this tick (between gates / momentarily lost) the seeker coasts level
    # and holds heading. The yaw rate is clamped to this (small) value so a re-acquire slew is gentle
    # and never spins the next gate out of frame.
    reacquire_yaw_rate_rps: float = 0.0
    # Minimum detection score + maximum PnP reproj error for a detection to drive the visual servo
    # (reject weak / mis-localised gates -> treat as "no detection" -> hold).
    min_detect_score: float = 0.0
    max_reproj_px: float = 12.0
    # Cap the per-tick yaw-rate command in the visual-servo pursuit phase so a large bearing error
    # (gate at the edge of frame) is turned toward smoothly, never a saturated slew that would spin
    # the gate out of frame faster than the controller can track it.
    visual_yaw_rate_cap_rps: float = 1.5

    # --- ANCHOR RELEASE on the seeker's OWN detections (the 2026-06-29 attempt-2 BUG A fix) ---
    # On the LIVE VQ2 wire the navigator is MAP-FREE (gates=[]), so its map-associated fix path
    # never fires and ``nav.time_since_vision_update_s`` stays inf FOREVER -- the old anchor-release
    # signal is structurally unreachable. The seeker owns its OWN detector, so it releases the
    # launch-hold on its OWN quality-gated detections instead: after this many CONSECUTIVE ticks
    # with a usable ``detect_gate_lever`` pose, the estimator is provably seeing the gate -> release.
    # (``nav.time_since_vision_update_s`` finite STILL latches the anchor too, when the map path is
    # live -- own-detection is an ADDITIONAL, map-free release path, never a regression.)
    anchor_release_detections: int = 3

    # --- ATTITUDE-SAFE COLD START (the 2026-06-29 attempt-2 BUG B fix) ---
    # POST-ARM SETTLE: after the first command, hold a conservative LEVEL attitude + bounded hover
    # thrust + zero horizontal lean for this long, letting the ESKF/AHRS gravity-align + the gyro
    # bias converge BEFORE any estimator-driven leaning or pursuit. 0.0 => no settle (legacy).
    settle_s: float = 0.75
    # In the LAUNCH-HOLD / SETTLE the roll/pitch body-rate command is clamped to this (the cold-AHRS
    # tumble guard): a bad cold estimate can demand a saturated pitch-over, so we bound EVERY axis,
    # not just yaw. Conservative-level small correction only -- never a saturated lean.
    hold_rp_rate_cap_rps: float = 0.6
    # In the LAUNCH-HOLD / SETTLE the alt-hold collective is bounded to [hover*lo, hover*hi] so a
    # weak/uncorrected cold z-estimate can't saturate the thrust into a climb into the gate.
    hold_thrust_lo_frac: float = 0.6
    hold_thrust_hi_frac: float = 1.4


@dataclass
class GateSeeker:
    """Slow pursuit guidance -> CTBR command. Stateless guidance + a small advance bookkeeper.

    Holds a :class:`Controller` (the decoupled CTBR inner loop) and a launch clock so the takeoff
    transient is ramped. ``plan(nav, gate, ...)`` returns the SETPOINT (for inspection / tests);
    ``command(nav, gate, ...)`` returns the CTBR :class:`ControlCommand`.
    """

    config: GateSeekerConfig = field(default_factory=GateSeekerConfig)
    controller: Controller = field(default_factory=make_seeker_controller)
    # Injected gate detector for the MAP-FREE visual servo (anything with
    # ``.detect(frame) -> [GateObservation]``; the live VQ2 path passes RedGlowGateDetector). None
    # => command_visual cannot see and always holds (the map-based plan/command path is unaffected).
    detector: object | None = field(default=None, repr=False)
    _t0_sim_ns: int | None = field(default=None, repr=False)   # first-command sim time (launch clock)
    _last_index: int | None = field(default=None, repr=False)  # last seen active_gate_index
    _anchored: bool = field(default=False, repr=False)         # first accepted vision fix seen?
    _last_yaw: float | None = field(default=None, repr=False)  # last commanded heading (no-detection hold)
    _last_frame_id: int | None = field(default=None, repr=False)  # detector idempotence across re-feeds
    _last_pose: GatePose | None = field(default=None, repr=False)  # cached detected lever for re-fed frames
    _consec_detections: int = field(default=0, repr=False)     # consecutive own-detection ticks (anchor release)

    # -- guidance: NavState + active gate -> Setpoint -----------------------
    def plan(self, nav: NavState, gate: Gate, *, is_final_gate: bool = False) -> Setpoint:
        """Slow pursuit setpoint: a carrot just beyond the gate centre, a capped desired velocity
        pointing at it, a yaw that keeps the gate in the +20deg camera view, and a launch ramp.

        The gate normal sign is arbitrary in the map; orient it to point the way we are TRAVELLING
        (downrange, away from us) so the carrot lands on the EXIT side and the drone is pulled
        THROUGH the opening rather than to a point in front of the front bar. Disambiguate by the
        velocity direction when moving fast enough to trust it, else by the position-to-gate vector
        (takeoff / hover) — exactly the ReactivePlanner discipline (a momentary speed dip must not
        flip the carrot 180deg and U-turn)."""
        if self._t0_sim_ns is None:           # arm the launch clock on the first plan/command
            self._t0_sim_ns = int(nav.sim_time_ns)
        position = np.asarray(nav.position_ned, dtype=np.float64)
        velocity = np.asarray(nav.velocity_ned, dtype=np.float64)
        gate_pos = np.asarray(gate.position_ned, dtype=np.float64)
        to_gate = gate_pos - position

        heading_ref = velocity if float(np.linalg.norm(velocity)) > 0.5 else to_gate
        travel = np.asarray(gate.normal_ned, dtype=np.float64)
        if float(travel @ heading_ref) < 0.0:
            travel = -travel
        travel = _unit(travel, fallback=_unit(to_gate))

        carrot = gate_pos + self.config.lookahead_m * travel
        los_dir = _unit(carrot - position, fallback=travel)
        heading = travel if self.config.yaw_mode == "course" else los_dir
        yaw = float(np.arctan2(heading[1], heading[0]))

        ramp = self._launch_ramp(int(nav.sim_time_ns))
        return Setpoint(
            sim_time_ns=int(nav.sim_time_ns),
            position_ned=carrot,
            velocity_ned=self.config.cruise_speed * los_dir,
            yaw=yaw,
            launch_ramp=ramp,
        )

    # -- guidance -> CTBR ControlCommand ------------------------------------
    def command(self, nav: NavState, gate: Gate, active_gate_index: int, *,
                is_final_gate: bool = False) -> ControlCommand:
        """Full guidance: plan the slow pursuit setpoint, then run the decoupled CTBR controller.

        Returns a BODY_RATE :class:`ControlCommand` (body rates FRD + normalized collective). Tracks
        ``active_gate_index`` so :meth:`advanced_since_last` can report a wire-driven advance."""
        if self._t0_sim_ns is None:
            self._t0_sim_ns = int(nav.sim_time_ns)
        self._last_index = int(active_gate_index)
        sp = self.plan(nav, gate, is_final_gate=is_final_gate)
        return self.controller.command(nav, sp)

    # =======================================================================
    # MAP-FREE VISUAL SERVO  (the live VQ2 path — chase the gate the camera SEES)
    # =======================================================================
    def detect_gate_lever(self, frame: Frame | None) -> GatePose | None:
        """Run the injected detector + PnP on ``frame`` and return the camera-relative pose of the
        gate to chase (``GatePose.t_cam_gate`` = gate centre in the camera optical frame), or ``None``
        when nothing usable is seen.

        MAP-FREE: no association to any map gate, no self-position — just "which opening is in front
        of me, and where is it relative to the camera". When several gates are detected we pick the
        CLOSEST (largest apparent span / smallest PnP range): the active gate is the one we are
        flying at, so it dominates the frame. Quality-gated by detection score + PnP reproj error."""
        if self.detector is None or frame is None or getattr(frame, "image_bgr", None) is None:
            return None
        observations: list[GateObservation] = list(self.detector.detect(frame))
        best: GatePose | None = None
        for obs in observations:
            if float(getattr(obs, "score", 1.0)) < self.config.min_detect_score:
                continue
            pose = estimate_gate_pose(obs, compute_covariance=False)
            if pose is None or not np.isfinite(pose.t_cam_gate).all():
                continue
            if float(pose.reproj_error_px) > self.config.max_reproj_px:
                continue
            if pose.t_cam_gate[2] <= 0.05:      # gate behind / on the image plane -> unusable bearing
                continue
            if best is None or pose.range_m < best.range_m:
                best = pose
        return best

    def command_visual(self, nav: NavState, frame: Frame | None, active_gate_index: int, *,
                       is_final_gate: bool = False) -> ControlCommand:
        """MAP-FREE visual-servo command: steer to CENTER + fly THROUGH the gate the camera SEES.

        The live VQ2 entry point (replaces the absolute-map :meth:`command` on the wire). Never reads
        an absolute gate map or absolute self-position for steering; every command is derived from the
        DETECTED gate's relative bearing plus the estimator's gravity-known attitude. Four regimes:

          0. **Post-arm settle** (first ``settle_s`` after the launch clock arms): HOLD a conservative
             LEVEL attitude + bounded hover thrust + zero rates, letting the cold AHRS gravity-align
             and the gyro bias converge BEFORE any estimator-driven leaning (the 2026-06-29 attempt-2
             cold-AHRS tumble fix). roll/pitch rates are clamped, not just yaw.
          1. **Launch anchor** (settled, but not yet anchored): HOLD attitude, clamp ALL rates small.
             We never maneuver blind, so the visible start gate stays in frame until the estimator
             anchors on it. The anchor RELEASES on the seeker's OWN consecutive quality-gated
             detections (``anchor_release_detections``) -- the map-free release signal the seeker
             owns, since on VQ2 ``nav.time_since_vision_update_s`` never goes finite (BUG A fix). A
             finite tsv (live map path) latches the anchor too.
          2. **No detection** (between gates / momentarily lost): coast level, hold the last heading,
             gentle re-acquire — never a blind large slew.
          3. **Pursuit** (anchored + gate seen): build a desired velocity toward the seen opening at
             the slow cap and a yaw that centers its bearing, capped so the turn is smooth.
        """
        if self._t0_sim_ns is None:
            self._t0_sim_ns = int(nav.sim_time_ns)
        self._last_index = int(active_gate_index)
        if self._last_yaw is None:
            self._last_yaw = float(nav.yaw)

        # Detect the gate to chase (idempotent across re-feeds of the same frame_id; a re-fed frame
        # keeps the cached bearing decision rather than re-running the detector).
        if frame is not None and frame.frame_id != self._last_frame_id:
            self._last_frame_id = int(frame.frame_id)
            self._last_pose = self.detect_gate_lever(frame)
            # Track CONSECUTIVE own-detection ticks for the map-free anchor release: a usable pose
            # increments, a miss resets (we want a STREAK of clean sightings, not one lucky frame).
            if self._last_pose is not None:
                self._consec_detections += 1
            else:
                self._consec_detections = 0
        pose = self._last_pose

        # ANCHOR RELEASE -- two independent latches (whichever fires first; never un-latches):
        #   (a) MAP-FREE (VQ2 live): N consecutive own quality-gated detections -> the seeker is
        #       provably seeing the gate, so it is safe to leave the hold and pursue. This is the
        #       BUG A fix: the seeker owns this signal, so it works even though the navigator's
        #       map-fix path (and thus tsv) never fires on the empty VQ2 map.
        #   (b) LIVE MAP: the navigator accepted a map-associated fix (tsv finite) -- the legacy
        #       signal, retained for the case where a real map IS present.
        if (self._consec_detections >= max(1, int(self.config.anchor_release_detections))
                or np.isfinite(nav.time_since_vision_update_s)):
            self._anchored = True

        # --- regime 0: POST-ARM SETTLE -> conservative level hold, all rates clamped, thrust bounded ---
        # Hold for settle_s from the launch clock so the cold AHRS gravity-aligns before we lean. We
        # stay here EVEN IF a detection arrives early (the estimate is not yet trustworthy to lean on).
        if self._in_settle(int(nav.sim_time_ns)):
            return self._hold_command(nav, yaw_rate_cap=self.config.anchor_yaw_rate_rps,
                                      attitude_safe=True)

        # --- regime 1: LAUNCH ANCHOR (settled, no release yet) -> hold attitude, clamp ALL rates ---
        if not self._anchored:
            return self._hold_command(nav, yaw_rate_cap=self.config.anchor_yaw_rate_rps,
                                      attitude_safe=True)

        # --- regime 2: NO DETECTION -> coast level on the last heading, gentle re-acquire ---
        if pose is None:
            return self._hold_command(nav, yaw_rate_cap=self.config.reacquire_yaw_rate_rps,
                                      attitude_safe=True)

        # --- regime 3: PURSUIT -> velocity toward the SEEN opening + centering yaw, smoothly capped ---
        return self._visual_pursuit_command(nav, pose)

    def _in_settle(self, sim_time_ns: int) -> bool:
        """True while within ``settle_s`` of the launch clock arming (the post-arm cold-AHRS settle)."""
        if self.config.settle_s <= 0.0 or self._t0_sim_ns is None:
            return False
        return (int(sim_time_ns) - self._t0_sim_ns) / 1e9 < self.config.settle_s

    def _gate_dir_world(self, nav: NavState, pose: GatePose) -> np.ndarray:
        """Unit world-NED direction from the drone to the DETECTED gate centre, from the relative lever.

        ``pose.t_cam_gate`` is the gate centre in the camera optical frame. Rotate it into the body
        frame (the fixed camera mount), then into world NED with the estimator's gravity-known
        attitude (roll/pitch from the AHRS accel-levelled tilt, yaw from the vision-pinned heading).
        NO absolute self-position enters — only the DIRECTION to the seen gate."""
        d_cam = _unit(np.asarray(pose.t_cam_gate, dtype=np.float64))
        d_body = R_camera_from_body().T @ d_cam
        R_wb = R_world_from_body(float(nav.roll), float(nav.pitch), float(nav.yaw))
        return _unit(R_wb @ d_body, fallback=np.array([np.cos(nav.yaw), np.sin(nav.yaw), 0.0]))

    def _visual_pursuit_command(self, nav: NavState, pose: GatePose) -> ControlCommand:
        """Build the slow pursuit CTBR from the SEEN gate's relative bearing (no map, no abs position).

        Desired velocity = cruise_speed along the world bearing to the gate, but HOLD ALTITUDE (zero
        the vertical component) so slow flight stays level + blur-free (z is the estimator's weakest
        axis on VQ2). Yaw centers the gate bearing's horizontal heading; the controller's body-rate is
        then clamped to the smooth visual cap so a wide bearing turns in steadily, never a slew."""
        gdir = self._gate_dir_world(nav, pose)
        horiz = np.array([gdir[0], gdir[1], 0.0])
        los = _unit(horiz, fallback=np.array([np.cos(nav.yaw), np.sin(nav.yaw), 0.0]))
        yaw = float(np.arctan2(los[1], los[0]))
        self._last_yaw = yaw
        sp = Setpoint(
            sim_time_ns=int(nav.sim_time_ns),
            velocity_ned=self.config.cruise_speed * los,    # level pursuit (altitude held by alt-hold)
            yaw=yaw,
            launch_ramp=self._launch_ramp(int(nav.sim_time_ns)),
        )
        cmd = self.controller.command(nav, sp)
        return self._cap_yaw_rate(cmd, self.config.visual_yaw_rate_cap_rps)

    def _hold_command(self, nav: NavState, *, yaw_rate_cap: float,
                      attitude_safe: bool = False) -> ControlCommand:
        """A SAFE, vision-preserving hold: level attitude (no horizontal lean), hover collective, and a
        yaw rate clamped to ``yaw_rate_cap`` (0 => hold yaw exactly). Used for the launch anchor and the
        no-detection coast so we NEVER slew the visible gate out of frame. Holds the LAST heading.

        When ``attitude_safe`` (the cold-start settle / launch / no-detection holds), the ROLL/PITCH
        body-rate command is ALSO clamped (``hold_rp_rate_cap_rps``) and the alt-hold collective is
        BOUNDED to [hover*lo, hover*hi]. This is the 2026-06-29 attempt-2 BUG B fix: on the cold,
        mag-free, gravity-aligned-but-still-converging AHRS the attitude/altitude estimate can demand
        a saturated pitch-over + thrust climb; bounding EVERY axis (not just yaw) keeps the hold from
        tumbling the drone into the gate while the estimator settles."""
        hold_yaw = self._last_yaw if self._last_yaw is not None else float(nav.yaw)
        sp = Setpoint(
            sim_time_ns=int(nav.sim_time_ns),
            velocity_ned=np.zeros(3),       # no horizontal lean -> level hover
            yaw=hold_yaw,
            launch_ramp=0.0,                # full anti-lean: keep the attitude level while holding
        )
        cmd = self.controller.command(nav, sp)
        cmd = self._cap_yaw_rate(cmd, yaw_rate_cap)
        if attitude_safe:
            cmd = self._cap_rp_rate(cmd, self.config.hold_rp_rate_cap_rps)
            cmd = self._bound_hold_thrust(cmd)
        return cmd

    def _cap_rp_rate(self, cmd: ControlCommand, cap_rps: float) -> ControlCommand:
        """Clamp the ROLL (FRD body-rate X) and PITCH (FRD body-rate Y) command to +/-``cap_rps`` --
        the cold-AHRS hold's per-axis tumble guard (a bad cold estimate can't demand a saturated lean)."""
        if cmd.body_rate is None:
            return cmd
        import dataclasses
        br = np.asarray(cmd.body_rate, dtype=np.float64).copy()
        c = abs(float(cap_rps))
        br[0] = float(np.clip(br[0], -c, c))
        br[1] = float(np.clip(br[1], -c, c))
        return dataclasses.replace(cmd, body_rate=br)

    def _bound_hold_thrust(self, cmd: ControlCommand) -> ControlCommand:
        """Bound the hold's collective to [hover*lo, hover*hi] so a weak cold z-estimate can't
        saturate the alt-hold into a climb into the gate (the BUG B thrust-ramp guard)."""
        if cmd.thrust is None:
            return cmd
        import dataclasses
        hover = float(self.controller.hover_thrust)
        lo = hover * float(self.config.hold_thrust_lo_frac)
        hi = hover * float(self.config.hold_thrust_hi_frac)
        return dataclasses.replace(cmd, thrust=float(np.clip(cmd.thrust, lo, hi)))

    @staticmethod
    def _cap_yaw_rate(cmd: ControlCommand, cap_rps: float) -> ControlCommand:
        """Clamp the yaw (FRD body-rate Z) component of a CTBR command to +/-``cap_rps`` (the launch /
        re-acquire / smooth-pursuit yaw clamp). Returns a new command; non-BODY_RATE pass through."""
        if cmd.body_rate is None:
            return cmd
        import dataclasses
        br = np.asarray(cmd.body_rate, dtype=np.float64).copy()
        br[2] = float(np.clip(br[2], -abs(cap_rps), abs(cap_rps)))
        return dataclasses.replace(cmd, body_rate=br)

    # -- advance logic ------------------------------------------------------
    def should_advance(self, nav: NavState, gate: Gate, *, is_final_gate: bool = False) -> bool:
        """RANGE/geometry backstop for advancing to the next gate (the wire's
        ``active_gate_index`` increment is authoritative; this is the fallback when it lags).

        Non-final gate: advance once the drone has crossed the gate PLANE onto the exit side
        (the along-track signed distance flips sign) within ``capture_radius_m`` of the centre.
        Final gate: no next gate to re-aim at, so dead-reckon — advance (treat as cleared) once
        within ``final_blowout_m`` of the centre, letting the drone fly straight through."""
        position = np.asarray(nav.position_ned, dtype=np.float64)
        gate_pos = np.asarray(gate.position_ned, dtype=np.float64)
        velocity = np.asarray(nav.velocity_ned, dtype=np.float64)
        to_gate = gate_pos - position
        rng = float(np.linalg.norm(to_gate))

        if is_final_gate:
            return rng <= self.config.final_blowout_m

        # orient the normal downrange (same disambiguation as plan), then the signed along-track
        # distance of the drone behind the gate plane is (gate - pos) . travel ; <= 0 means we have
        # crossed to the exit side.
        heading_ref = velocity if float(np.linalg.norm(velocity)) > 0.5 else to_gate
        travel = np.asarray(gate.normal_ned, dtype=np.float64)
        if float(travel @ heading_ref) < 0.0:
            travel = -travel
        travel = _unit(travel, fallback=_unit(to_gate))
        along = float(to_gate @ travel)         # > 0 : gate ahead ; <= 0 : crossed the plane
        # lateral miss at the crossing (reject a far side-pass that never threaded the opening)
        lateral = float(np.linalg.norm(to_gate - along * travel))
        return along <= 0.0 and lateral <= max(self.config.capture_radius_m, 0.75)

    def advanced_since_last(self, active_gate_index: int) -> bool:
        """True iff ``active_gate_index`` is greater than the index of the previous command (the
        wire-driven advance). ``False`` before the first command (no baseline)."""
        return self._last_index is not None and int(active_gate_index) > self._last_index

    def reset(self) -> None:
        """Drop the launch clock + advance baseline + visual-servo state (e.g. on a sim epoch restart).

        After a reset the seeker is back in the LAUNCH-ANCHOR regime (``_anchored`` False), so it
        re-holds until the estimator re-anchors on the visible gate — exactly the boot behaviour."""
        self._t0_sim_ns = None
        self._last_index = None
        self._anchored = False
        self._last_yaw = None
        self._last_frame_id = None
        self._last_pose = None
        self._consec_detections = 0

    # -- internals ----------------------------------------------------------
    def _launch_ramp(self, sim_time_ns: int) -> float | None:
        """Launch ramp [0,1] over ``launch_ramp_s`` from the first command, on the sim master clock.
        ``None`` once the ramp completes (full authority) or when disabled (ramp_s <= 0)."""
        if self.config.launch_ramp_s <= 0.0 or self._t0_sim_ns is None:
            return None
        elapsed_s = (int(sim_time_ns) - self._t0_sim_ns) / 1e9
        if elapsed_s >= self.config.launch_ramp_s:
            return None
        return float(np.clip(elapsed_s / self.config.launch_ramp_s, 0.0, 1.0))
