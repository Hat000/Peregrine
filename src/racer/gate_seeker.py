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
# Shared per-frame_id detection cache (A15/A17 double-detect fix): the seeker + the navigator hold the
# SAME detector instance. Routing the seeker's detect through detect_cached reuses the navigator's
# per-frame_id detections (computed first, same tick) instead of running YOLO a SECOND time.
from racer.vision.detector import detect_cached
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

    # --- TEMPORAL GATE TRACKING (the 2026-06-29 attempt-3 LAYER-2a fix) ---
    # A3 (3.0 m/s): with several red gates visible down the lit course the "pick the closest each
    # frame" selection FLAPPED -- the chosen gate's PnP range jumped 10<->26<->9.6<->30 m tick-to-
    # tick (a DIFFERENT gate each frame / unstable depth), so the steering bearing swung and the
    # roll command SATURATED (+3.43 rps) into a roll-over. The fix LOCKS one gate across frames: a
    # lightweight track of the chosen gate's (range, camera bearing) that each new frame matches the
    # most-consistent candidate to -- and REJECTS a candidate that jumps implausibly (coast on the
    # track instead). When tracking is on, ``detect_gate_lever`` returns the TRACKED gate, not the
    # raw closest. Off => legacy closest-each-frame.
    use_gate_track: bool = True
    # FIRST acquisition (no track yet): prefer the most CENTERED gate (smallest camera bearing from
    # boresight) -- the gate we are flying the line at -- over merely the closest. Once a track
    # exists, continuity (below) selects, not centring.
    track_prefer_centered: bool = True
    # --- NEAREST-GATE FIRST ACQUISITION (the 2026-06-29 attempt-5 BLOCKER 2 fix) ---
    # A5 (2.0 m/s): "prefer-centered" alone locked a DISTANT off-axis gate (trk_range 37-60 m,
    # trk_az +0.32 = +18deg) over the NEAR start-line gate dead ahead -- the far gate's intermittent
    # detections starved the N=3 release and the frozen spawn-tilt drifted the drone into gate 0. The
    # near gate is the one we must fly FIRST, so first-acquisition must bias toward the NEAREST
    # plausible gate, not merely the most-centered. THE FIX: (1) REJECT any candidate beyond
    # ``max_acquire_range_m`` (a distant downrange gate is never the next gate to fly); (2) among the
    # admissible candidates, score by a blend of range AND bearing so the NEAR-and-reasonably-centered
    # gate wins over a far-but-perfectly-centered one. ``prefer_nearest`` gates the whole behaviour;
    # OFF => the legacy pure prefer-centered (or closest) selection.
    prefer_nearest: bool = True
    # First-acquisition candidates farther than this are REJECTED (not the next gate to fly). The near
    # start-line gate sits ~10 m at spawn; a 37-60 m lock is the far-gate trap. Generous enough to keep
    # an honest next gate, tight enough to reject the distant off-axis downrange gate.
    max_acquire_range_m: float = 22.0
    # First-acquisition score = range_m + nearest_bearing_weight_m_per_rad * |bearing|. The bearing
    # penalty is in METRES per radian so it trades off directly against range: at the default a 0.32 rad
    # (~18deg) off-axis gate carries a +6.4 m penalty, so the 9 m near gate (small bearing) beats the
    # 37 m far gate decisively. Larger => more centring weight; 0 => pure nearest.
    nearest_bearing_weight_m_per_rad: float = 20.0
    # A candidate is consistent with the track when BOTH its range and its camera bearing are within
    # these of the track's PREDICTED value. A candidate outside EITHER gate is a jump (a different
    # gate / a PnP-depth flip) and is REJECTED -- the track coasts (no detection this tick) rather
    # than locking onto the flapper. Generous enough to follow honest closing range between ticks at
    # the slow cruise; tight enough to reject the 10<->30 m flap.
    track_max_range_jump_m: float = 6.0
    track_max_bearing_jump_rad: float = 0.35     # ~20 deg of camera bearing step between frames
    # The track's range/bearing are smoothed (EMA) so a single noisy-but-accepted PnP depth does not
    # yank the prediction. 1.0 => snap to the new measurement; small => heavy smoothing.
    track_ema_alpha: float = 0.5
    # Drop the track after this many CONSECUTIVE ticks with no consistent candidate (gate genuinely
    # lost / between gates) so re-acquisition can re-centre on a fresh gate.
    track_max_coast_ticks: int = 8

    # --- POST-RELEASE PURSUIT RAMP + GUIDANCE RATE LIMIT (the LAYER-2b fix) ---
    # A short ramp on pursuit authority AFTER the anchor releases, so a noisy FIRST bearing can't
    # step-saturate roll the instant pursuit begins (A3: the first pursuit tick already commanded
    # +2.19 roll, the third +3.43 -> roll-over). Over this window the commanded body-rate is scaled
    # up from a small floor to full, easing into pursuit. 0.0 => no ramp (legacy).
    pursuit_ramp_s: float = 0.8
    pursuit_ramp_floor: float = 0.15            # authority scale at the instant of release (>0 so it still steers)
    # Rate-limit the per-tick CHANGE of the pursuit guidance HEADING (the yaw setpoint the controller
    # tracks): a noisy bearing can demand a large heading step that the controller turns into a
    # saturating roll/yaw. Capping the heading slew keeps the steering bearing SMOOTH frame-to-frame
    # (the proximate fix for the swing that saturated roll). rad per tick-second.
    pursuit_yaw_slew_rps: float = 1.0
    # Cap the ROLL (FRD body-rate X) command in pursuit so a residual bearing swing can never
    # saturate roll into a roll-over (A3's crash axis). Well below the controller's max_body_rate_rps.
    pursuit_roll_rate_cap_rps: float = 1.5

    # --- BOUNDED FEEDFORWARD FORWARD TILT (the 2026-06-29 attempt-4 PITCH-windup fix) ---
    # A4: with roll + yaw capped, PITCH was the last uncapped pursuit axis -- and it WOUND UP to the
    # -4.0 rad/s controller limit over ~1 s (cmd_pitch -0.69 -> -2.39 -> -3.999), then crashed. ROOT
    # CAUSE: the old pursuit asked the controller for a desired VELOCITY (cruise_speed*los) and the
    # decoupled controller closes it with a velocity-ERROR term ``kd_vel*(des_vel - vel)``. On the
    # MAP-FREE VQ2 wire velocity is UNOBSERVABLE (no position/velocity on the wire; the navigator's
    # dead-reckoned velocity stays ~0), so the velocity error NEVER closes -> the demanded forward
    # accel stays at ~kd_vel*cruise_speed -> the tilt -> the pitch rate saturates. You CANNOT use a
    # velocity setpoint for forward motion in the self-localizing regime.
    #
    # THE FIX (this block): command forward motion as a BOUNDED FEEDFORWARD horizontal acceleration
    # (a small fixed forward demand the controller adds as PURE feedforward, ``sp.accel_ned``, with NO
    # velocity_ned term to wind up), RAMPED in over ``forward_ramp_s`` and the pitch rate hard-CAPPED
    # symmetric to the roll cap. The forward tilt is therefore bounded + rate-limited + ramped and can
    # never saturate. ``use_feedforward_forward`` gates the whole behaviour (off => legacy velocity).
    use_feedforward_forward: bool = True
    # The bounded forward horizontal acceleration demand (m/s^2) along the (slewed) gate heading. Small
    # so the steady forward tilt is gentle: forward accel a -> lean ~atan(a/g), so 1.2 m/s^2 ~ 7 deg.
    # This is the CREEP demand for the slow lap; it does NOT depend on (unobservable) velocity.
    forward_accel_mps2: float = 1.2
    # Cap the PITCH (FRD body-rate Y) command in pursuit -- the A4 crash axis. Symmetric to the roll
    # cap; well below the controller's max_body_rate_rps so the forward lean can never wind to the
    # -4.0 limit even on a transient attitude error.
    pursuit_pitch_rate_cap_rps: float = 1.5
    # Ramp the forward-accel demand up from zero over this window from the anchor release, so the
    # forward lean eases IN (the first pursuit ticks don't step to the full forward tilt). Mirrors the
    # post-release authority ramp. 0.0 => forward demand applied at full from tick 1 (legacy step).
    forward_ramp_s: float = 1.0

    # --- SPAWN-GATE EGRESS (the 2026-06-29 attempt-4 start-gate-contact fix) ---
    # A4: the drone SPAWNS INSIDE the start gate (gate 0). The first forward motion drove it straight
    # into the start-gate structure (realized IMU: a 44-rps spike at impact). THE FIX: after release,
    # before full pursuit, run a brief EGRESS phase -- a small CAPPED forward creep along the START-GATE
    # NORMAL (the spawn heading, which is the direction OUT of the gate the drone is sitting in) -- to
    # clear gate 0, then transition to normal pursuit. Gentle + bounded (it reuses the same feedforward
    # tilt + pitch cap as pursuit, only the heading is FROZEN to the spawn heading, not slewed to the
    # downrange gate). use_spawn_egress gates it; egress_s is its duration.
    use_spawn_egress: bool = True
    egress_s: float = 0.8
    # The egress forward demand (m/s^2): a gentle straight creep out of the spawn gate. Small.
    egress_accel_mps2: float = 1.0
    # --- DISTANCE-BASED EGRESS (the 2026-06-29 attempt-5 BLOCKER 3 fix) ---
    # A5: the fixed-time 0.8 s egress did not reliably clear the gate-0 frame the drone spawns inside
    # (run4 contacted structure at close range). THE FIX: make the egress end on DISTANCE travelled out
    # of the spawn gate, not a wall-clock timer -- creep along the frozen spawn heading until clear of
    # the gate-0 frame by roughly the gate depth. We have no absolute self-position map-free, so we
    # integrate the seeker's OWN bounded forward demand (egress_accel_mps2, ramped) into a dead-reckoned
    # along-heading distance and end egress once it exceeds ``egress_clear_distance_m``. ``egress_s``
    # remains a hard UPPER bound (a timeout) so egress can never run forever. use_distance_egress gates
    # it; OFF => the legacy pure time-based egress.
    use_distance_egress: bool = True
    # Clear the spawn gate by this along-heading distance before transitioning to pursuit. ~the gate
    # depth + a margin so the chassis is fully past the gate-0 frame plane it spawned in.
    egress_clear_distance_m: float = 1.5
    # --- EGRESS THRUST FLOOR (the 2026-06-29 A7 close-range thrust-collapse fix) ---
    # A7 (n=3, speed-independent): the drone SPAWNS INSIDE gate 0, point-blank. The seeker's alt-hold
    # collective bottoms out at the controller floor ``alt_thrust_lo`` (0.05) -- a near-zero thrust ->
    # the drone FREE-FALLS at spawn (tlog |a|=0.1 m/s^2) and tumbles into the gate frame. (A6's 2.3 Hz
    # loop drifted clear before it could cut thrust; A7's 30 Hz loop cuts it immediately at t=0, so the
    # perf fix EXPOSED this latent collapse.) THE FIX: during the SPAWN-GATE EGRESS the collective is
    # FLOORED to at least hover-equivalent (``hover_thrust * egress_thrust_floor_frac``), so the drone
    # can only HOLD or CLIMB out of the spawn gate along the start normal -- it can never descend /
    # free-fall while egressing. Floor = 1.0 -> exactly hover (no sink); >1.0 -> a gentle climb out.
    # Bounds only the egress collective; once past egress the controller's own alt-hold owns thrust
    # (so post-egress flight is byte-identical to today). ``use_egress_thrust_floor`` gates it.
    use_egress_thrust_floor: bool = True
    egress_thrust_floor_frac: float = 1.0
    # --- EGRESS ATTITUDE FREEZE (the 2026-06-30 A10 acquisition-trap fix) ---
    # Hold (do not re-level) the frozen spawn pitch/roll during egress. The egress forward demand
    # ramps from ~0, so a frozen spawn tilt would otherwise drive a saturated nose-UP re-level that
    # points the +20deg camera OFF the spawn gate (the A10 acquisition trap: camera leaves frame ->
    # detector dark -> pursuit never acquires a pose). With this ON, egress zeroes its OWN roll/pitch
    # rate command (exactly like the hold's hold_freeze_attitude) so the camera stays on the gate;
    # only yaw(=frozen) + thrust-floor + forward feedforward remain. OFF (default) == today's egress.
    egress_freeze_attitude: bool = False
    # --- POINT-BLANK ELEVATION GUARD (the 2026-06-29 A7 close-range descent fix) ---
    # A7 root cause: at spawn the start gate is POINT-BLANK (range ~1 m). At that range the PnP lever is
    # degenerate and ``trk_el`` is garbage -- it read NEGATIVE (gate appears below boresight), so the
    # vertical-align channel commanded a DESCENT off a gate that is essentially on top of us, which cut
    # the alt-hold thrust to the floor (the free-fall above). THE FIX: the elevation/vertical-align
    # channel is only TRUSTWORTHY beyond a minimum range; below ``min_trust_elevation_range_m`` the
    # seeker commands NO vertical correction (vz_t=0 -> the alt-hold holds the current height) rather
    # than chasing a point-blank gate's garbage elevation down into the floor. Normal downrange pursuit
    # acquires gates well beyond this range, so post-egress flight is byte-identical (the guard only
    # fires at the point-blank spawn). ``use_min_trust_elevation`` gates it; OFF => legacy (trust any range).
    use_min_trust_elevation: bool = True
    min_trust_elevation_range_m: float = 2.0
    # --- VERTICAL ALIGNMENT to the gate-opening centre (the 2026-06-29 attempt-5 BLOCKER 1 fix) ---
    # A5 (3.0 m/s, the closest-approach failure): the seeker held a FIXED ALTITUDE while pursuing, but
    # the gate OPENING sits BELOW the held path (run4: the tracked-gate camera elevation trk_el drifted
    # +0.057 -> -0.064 as it closed). The drone rode too HIGH and CLIPPED THE GATE TOP BAR at 2.78 m
    # instead of threading the opening. THE FIX: during pursuit, drive the gate-opening vertical offset
    # toward ~zero -- descend/climb so the opening is centred on the flight path -- instead of holding
    # altitude. The offset is the VERTICAL (world-NED Z) component of the gate-centre lever
    # ``R_world_from_body @ R_camera_from_body().T @ t_cam_gate`` (the seen gate centre relative to the
    # drone, rotated body->world via the estimator's gravity-known attitude). This ACCOUNTS FOR THE
    # +20deg CAMERA MOUNT: R_camera_from_body() carries the mount tilt, so we do NOT naively null the
    # camera-frame trk_el (the camera points 20deg up); we null the TRUE world vertical offset. We
    # command it as a BOUNDED vertical VELOCITY target (vz_t in the alt-hold), proportional to the
    # offset, capped + ramped -- the same bounded/ramped discipline as the forward feedforward (never a
    # position step, never unbounded). ``use_vertical_align`` gates it; OFF => the legacy fixed-altitude
    # hold (z_t = current estimator z).
    use_vertical_align: bool = True
    # Proportional gain (1/s): commanded vertical velocity vz_t = clip(vertical_align_kp * offset_z_world,
    # +/-vertical_align_speed_cap). NED Z+ = down, so a gate BELOW the drone (offset_z_world > 0) -> a
    # positive (descend) vz_t -> the alt-hold reduces thrust to sink toward the opening. Gentle so the
    # descent is smooth at the slow cruise.
    vertical_align_kp: float = 0.8
    # Cap the commanded vertical velocity (m/s) so the descent/climb stays SLOW + bounded (mirrors the
    # forward speed cap). Small -- this is the slow-lap, we ease onto the opening height, never dive.
    vertical_align_speed_cap_mps: float = 1.0
    # A deadband (m) on the vertical offset: within this the opening is "centred enough" and no vertical
    # correction is commanded (avoids hunting on estimator noise near alignment).
    vertical_align_deadband_m: float = 0.1
    # Ramp the vertical-align authority in from zero over this window from the anchor release (mirrors
    # the forward-demand ramp), so the first pursuit ticks don't step to a full descent command. 0.0 =>
    # applied at full from tick 1.
    vertical_align_ramp_s: float = 1.0

    # --- HOLD-LAST-DEMAND BRIDGE (the 2026-06-30 A13 polygonal-motion fix) ---
    # A13: on the slow VQ2 cruise the detection->pose stage drops a USABLE pose on ~75% of ticks
    # (the track-continuity flap rejects a candidate whose PnP range/bearing jumps vs the EMA track,
    # plus the _valid_poses reproj/behind filters + the first-acquisition acquire-range reject). Each
    # such tick routes into regime 2 -> _hold_command, which ZEROES all body rates and re-levels (a
    # silent zero-coast). The result is POLYGONAL motion: a real pursuit command on ~25% of ticks with
    # 0.5-1.2 s zero/hold coasts between. THE FIX: bridge short pose gaps -- for this long after the
    # last USABLE pose, re-issue the LAST pursuit demand (the cached forward lean + frozen heading)
    # instead of regime-2's all-axes-zero re-level, so control is CONTINUOUS per tick (no zero-coast).
    # 0.0 == OFF == legacy (byte-identical: regime-2 falls straight through to _hold_command).
    hold_last_demand_s: float = 0.0
    # Over the hold window, scale the held forward-accel demand full->0 linearly (decay/level at the
    # end rather than cutting), so a genuinely-lost gate is not rammed forever -- the bridge eases the
    # forward lean out over the window. Only active when hold_last_demand_s > 0.
    hold_last_demand_decay: bool = True

    # --- TRUE-ATTITUDE FROM AHRS (the 2026-06-30 A14 yaw-mirror fix) ---
    # ROOT CAUSE (A14): under ``use_ahrs`` (vq2_case_c) the case-C Navigator re-encodes the TRUE
    # AHRS attitude into the legacy ODOMETRY conjugation before writing ``NavState`` (an R_y(pi)
    # conjugation, ``_ahrs_odo_quat = q_true * ODO_QUAT_TRUE_CONJ_WXYZ``). Empirically, every tick:
    # ``NavState.roll = -euler_roll(q_true)``, ``NavState.pitch = +euler_pitch(q_true)``,
    # ``NavState.yaw = -euler_yaw(q_true)``. So the seeker is handed a MIRRORED-yaw attitude:
    #   TRUE euler = (-nav.roll, nav.pitch, -nav.yaw).
    # The seeker's gate-direction geometry (``_gate_dir_world``/``_gate_lever_world``) used the RAW
    # conjugated euler, so for a gate on the RIGHT it computed a world LoS mirrored in yaw and
    # steered the nose LEFT (A14 confirmed). The controller's ``R_cur`` has the SAME mirror: its
    # ``odo_att_sign=[-1,1,1]`` un-conjugates ROLL (asign[0]=-1) but NOT yaw (asign[2]=1), so
    # ``R_cur`` yaw = ``nav.yaw = -true_yaw`` -- the controller half of the same bug.
    #
    # THE FIX (this flag, ON): the seeker consumes the TRUE euler ``(-nav.roll, nav.pitch, -nav.yaw)``
    # for ALL its attitude geometry + heading bookkeeping, and passes the controller a nav whose YAW
    # is NEGATED (``replace(nav, yaw=-nav.yaw)``) so ``R_cur`` becomes R_true WITHOUT touching the
    # controller's sign config: R_cur roll = (-true_roll)*-1 = +true_roll, pitch = +true_pitch,
    # yaw = (-nav.yaw)*1 = (+true_yaw). No sign knob (gyro_sign/body_rate_sign/odo_att_sign/vp_yaw)
    # is touched. OFF (default) == VQ1 / case-A byte-identical (raw ``nav`` euler everywhere).
    true_attitude_from_ahrs: bool = False

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
    # --- HOLD-ATTITUDE (the 2026-06-29 attempt-3 LAYER-1 fix) ---
    # FREEZE the spawn attitude during the launch/settle hold instead of force-LEVELLING off it.
    # A3 (2.0 m/s): the level-hold controller, fed the gravity-aligned SPAWN TILT, commands a
    # PERSISTENT clamped +0.6 rps pitch to drive the (correctly-estimated) tilt toward level. That
    # slow pitch-over tilts the +20deg camera OFF the gate -> the detector goes dark (meanBGR 27->9)
    # and the 3-consecutive-detection release stalls at 2/3 and never fires. The fix: in the hold the
    # seeker ZEROES its OWN roll/pitch rate command (holds the current camera attitude, keeping the
    # gate in view), only the alt-hold collective + a clamped yaw remain. We do NOT re-level off the
    # gate; the gate stays centred so the detection streak completes. (The estimate is gravity-aligned
    # so the spawn tilt is small; freezing it is safe -- and far safer than a slow pitch off-gate.)
    hold_freeze_attitude: bool = True

    # --- DEAD-RECKON THROUGH THE PASS + NEXT-GATE HANDOFF (the 2026-06-29 A5-footage fix) ---
    # A5 FOOTAGE (post-mortem of run4/run5): the drone THREADS the first gate dead-straight (lateral +
    # yaw perfect), then the POST-PASS transition kills it. At the pass the gate FILLS the frame and the
    # tracked range -> ~0, so the gate-relative geometry (t_cam_gate) is DEGENERATE: the close-range
    # vision fix corrupts (run4: est_pitch jumps to -1.08 rad at the 2.78 m pass tick, thr spikes 0.27->
    # 0.52) AND, the instant the gate is lost behind the drone, the seeker drops into the NO-DETECTION
    # hold -- which commands velocity=0 + launch_ramp=0 (re-level) and zeros the forward feedforward, so
    # the forward-leaned attitude SNAPS toward level => a PITCH-UP into the BACKSIDE of the gate (3.0
    # runs), or just an indefinite forward DRIFT that never re-acquires the next gate (2.0 runs). THE
    # FIX: a dedicated DEAD-RECKON-THROUGH-PASS regime. When the tracked gate range drops below
    # ``pass_arm_range_m`` the pass is ARMED; once the (about-to-be-passed) gate is then LOST/behind, or
    # ``active_gate_index`` increments, the seeker STOPS vision-servoing + STOPS vertical-align on the
    # degenerate near-zero lever, FREEZES the pre-pass pursuit heading, and commands a BOUNDED LEVEL
    # forward COAST (the same bounded feedforward forward tilt as pursuit, on the frozen heading, with NO
    # re-level snap) -- gliding straight through the opening on IMU. After the coast it enters ACQUIRE-
    # NEXT: it keeps the bounded forward coast while the track is reset so first-acquisition re-locks the
    # NEXT gate (active_gate_index now points to it); once re-acquired it resumes normal pursuit (a
    # smooth yaw toward an off-axis next gate, never a pitch-up, never an indefinite drift). This also
    # disarms the EARLY pitch-up: a transient mid-approach detection gap (range NOT yet armed) no longer
    # routes to a re-level hold -- it coasts level on the last heading instead.
    use_pass_dead_reckon: bool = True
    # ARM the pass once the tracked gate range drops below this (m). Below ~this the PnP lever is
    # degenerate (gate fills the frame) and we must NOT take a close-range vision fix from it. ~the gate
    # depth + a margin so we commit to the dead-reckon glide before the fix corrupts.
    pass_arm_range_m: float = 3.0
    # Below this even-tighter range (m) the vision fix is treated as DEGENERATE: while armed and this
    # close we stop servoing on the gate's bearing/vertical-offset (coast straight on the frozen heading)
    # even if a (corrupt) detection is still returned -- the run4 est_pitch -1.08 blow-out tick.
    pass_degenerate_range_m: float = 2.5
    # The bounded LEVEL forward coast after the pass is detected runs for at most this long (s) before
    # ACQUIRE-NEXT takes over (which itself keeps coasting until the next gate is seen). A hard upper
    # bound so the dead-reckon glide is a brief committed push through the opening, not an open loop.
    pass_coast_s: float = 1.2
    # Forward accel demand (m/s^2) for the dead-reckon / acquire-next coast: the same gentle bounded
    # feedforward creep discipline as pursuit (lean ~atan(a/g)), so the coast holds the pre-pass forward
    # attitude rather than snapping level. Small.
    pass_coast_accel_mps2: float = 1.2
    # ACQUIRE-NEXT: after the dead-reckon coast, keep the bounded forward coast for up to this long (s)
    # while re-acquiring the next gate (resetting the track so first-acquisition re-locks). If the next
    # gate is not seen within this window the seeker reverts to the gentle no-detection level coast (it
    # never pitches up or stalls). Generous -- the next gate may be briefly out of frame after the pass.
    acquire_next_s: float = 3.0


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
    # A25 gate-relative altitude (VerticalEstimator.latch_offset): the Navigator instance whose
    # ``_vert_est`` (the SAME VerticalEstimator the Navigator owns + steps at IMU rate) receives the
    # fresh-pose latch. Read live via ``getattr(nav_owner, "_vert_est", None)`` EACH tick (not cached
    # at construction time) because the Navigator lazily creates/reseeds ``_vert_est`` on
    # (re)initialize -- a sim-epoch restart swaps in a brand-new estimator instance. The seeker only
    # sees poses at control rate and cannot itself propagate ẑ_off between them or run the contact
    # gate, so it pushes the fresh measurement INTO the shared estimator instead of owning any state
    # of its own (spec §1.2 rationale). None (default) => latch_offset is never called -> byte-
    # identical when the estimator is off / not wired (VQ1 / case-A untouched).
    nav_owner: object | None = field(default=None, repr=False)
    _last_latched_pose_ns: int | None = field(default=None, repr=False)  # fresh-pose dedupe (§1.4)
    _t0_sim_ns: int | None = field(default=None, repr=False)   # first-command sim time (launch clock)
    _last_index: int | None = field(default=None, repr=False)  # last seen active_gate_index
    _anchored: bool = field(default=False, repr=False)         # first accepted vision fix seen?
    _last_yaw: float | None = field(default=None, repr=False)  # last commanded heading (no-detection hold)
    _last_yaw_des: float | None = field(default=None, repr=False)  # last PRE-SLEW desired yaw toward gate
                                                                  # (instrumentation only, A14 yaw-sign probe)
    # -- A25 instrumentation (spec §7): stashed so fly_rl's nav_estimate logger can read them
    # directly instead of reconstructing them from the thrust law. Instrumentation only -- never
    # consumed by control. None when unavailable (no pursuit tick yet / no pose this tick).
    _last_vz_t: float | None = field(default=None, repr=False)             # commanded vertical vz_t
    _last_offset_z_world: float | None = field(default=None, repr=False)   # raw pre-latency-comp offset
    _last_pose_age_s: float | None = field(default=None, repr=False)       # this tick's pose obs age
    _last_frame_id: int | None = field(default=None, repr=False)  # detector idempotence across re-feeds
    _last_pose: GatePose | None = field(default=None, repr=False)  # cached detected lever for re-fed frames
    _consec_detections: int = field(default=0, repr=False)     # consecutive own-detection ticks (anchor release)
    # -- temporal gate track (Layer 2a): the locked gate's smoothed (range, camera-bearing) --
    _track_range_m: float | None = field(default=None, repr=False)      # tracked gate range, EMA-smoothed
    _track_bearing: np.ndarray | None = field(default=None, repr=False)  # tracked gate camera bearing (az,el) rad
    _track_coast_ticks: int = field(default=0, repr=False)     # consecutive ticks with no consistent candidate
    # -- post-release pursuit ramp (Layer 2b): the sim-time the anchor released --
    _release_t_ns: int | None = field(default=None, repr=False)
    _last_pursuit_t_ns: int | None = field(default=None, repr=False)  # last pursuit tick (heading slew dt)
    # -- spawn-gate egress (A4 fix): the heading frozen at release = the direction OUT of the start gate --
    _spawn_heading: float | None = field(default=None, repr=False)
    # -- distance-based egress (A5 BLOCKER 3): dead-reckoned along-heading distance crept since release --
    _egress_dist_m: float = field(default=0.0, repr=False)
    _egress_v: float = field(default=0.0, repr=False)                 # dead-reckoned egress creep speed
    _last_egress_t_ns: int | None = field(default=None, repr=False)   # last egress tick (for the dt integral)
    _egress_done: bool = field(default=False, repr=False)             # distance target reached -> egress complete
    # -- dead-reckon through the pass + next-gate handoff (A5-footage fix) --
    _pass_armed: bool = field(default=False, repr=False)              # tracked range dropped below pass_arm_range_m
    _pass_min_range_m: float = field(default=float("inf"), repr=False)  # closest tracked range seen while pursuing
    _passing: bool = field(default=False, repr=False)                # PASS detected -> dead-reckon coast active
    _pass_t_ns: int | None = field(default=None, repr=False)         # sim time the pass dead-reckon began
    _pass_heading: float | None = field(default=None, repr=False)    # FROZEN pre-pass pursuit heading (coast direction)
    _pass_index: int | None = field(default=None, repr=False)        # active_gate_index at the moment the pass armed
    # -- hold-last-demand bridge (A13): cache the last good pursuit demand so a pose-None tick can
    #    re-issue it (continuous per-tick command) instead of regime-2's zero-coast hold --
    _last_demand_los: np.ndarray | None = field(default=None, repr=False)   # last pursuit world heading unit vec
    _last_demand_yaw: float | None = field(default=None, repr=False)        # last pursuit slewed yaw
    _last_demand_accel: float | None = field(default=None, repr=False)      # last pursuit effective forward accel
    _last_demand_t_ns: int | None = field(default=None, repr=False)         # sim time of the last pursuit demand
    # -- A13 instrumentation: per-flight pose-None breakdown + bridge coverage counters (logging only,
    #    no behaviour change). Accumulated in memory; fly_rl emits a one-line summary at loop exit. --
    _last_none_reason: str | None = field(default=None, repr=False)         # why detect_gate_lever returned None
    diag_counts: dict = field(default_factory=lambda: {
        "pursuit": 0,            # regime 3: real pose -> pursuit command
        "none_total": 0,         # pose=None ticks (after launch, post-settle/anchor)
        "none_valid_poses_empty": 0,   # no candidate survived _valid_poses
        "none_continuity_reject": 0,   # candidates existed but all dropped at the track-continuity gate
        "none_first_acq_reject": 0,    # first-acquisition acquire-range reject (no admissible candidate)
        "none_other": 0,         # pose=None for some other reason (no detector / no frame / coast-drop)
        "bridged": 0,            # pose=None ticks bridged by hold-last-demand
        "held_legacy": 0,        # pose=None ticks fallen through to the legacy zeroed hold
    }, repr=False)

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
    @staticmethod
    def _pose_bearing(pose: GatePose) -> np.ndarray:
        """Camera bearing (azimuth, elevation) of the gate centre, in radians, from the optical-frame
        lever. Camera optical frame: +x right, +y down, +z forward. az = atan2(x, z), el = atan2(y, z).
        This is the gate's ANGULAR position in the image -- the track-continuity coordinate."""
        t = np.asarray(pose.t_cam_gate, dtype=np.float64)
        z = max(float(t[2]), 1e-6)
        return np.array([np.arctan2(float(t[0]), z), np.arctan2(float(t[1]), z)], dtype=np.float64)

    def _valid_poses(self, frame: Frame) -> list[GatePose]:
        """All quality-gated candidate gate poses in ``frame`` (score + reproj + in-front), unsorted."""
        # A15/A17 double-detect fix: reuse the navigator's per-frame_id detections (same shared detector)
        # instead of running detect() a second time this frame. Identical observations, ~half the cost.
        observations: list[GateObservation] = list(detect_cached(self.detector, frame))
        out: list[GatePose] = []
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
            out.append(pose)
        return out

    def detect_gate_lever(self, frame: Frame | None) -> GatePose | None:
        """Run the injected detector + PnP on ``frame`` and return the camera-relative pose of the
        gate to chase (``GatePose.t_cam_gate`` = gate centre in the camera optical frame), or ``None``
        when nothing usable is seen.

        MAP-FREE: no association to any map gate, no self-position — just "which opening is in front
        of me, and where is it relative to the camera". Quality-gated by detection score + PnP reproj.

        TEMPORAL TRACK (Layer 2a, default ``use_gate_track``): the active gate is LOCKED across frames
        rather than re-chosen from scratch each tick. A track of the chosen gate's smoothed (range,
        camera bearing) is maintained; each new frame the candidate most CONSISTENT with the track's
        prediction is selected, and a candidate that JUMPS implausibly (range/bearing discontinuity)
        is REJECTED -- the track coasts (returns ``None`` this tick) instead of locking onto a flapper.
        On FIRST acquisition the most-CENTERED gate is preferred (the line we fly), not just the
        closest. This kills the A3 10<->30 m range-flap that swung the bearing and saturated roll.

        With tracking OFF the legacy behaviour returns: pick the CLOSEST (smallest PnP range) gate
        each frame, no continuity."""
        if self.detector is None or frame is None or getattr(frame, "image_bgr", None) is None:
            self._last_none_reason = "other"     # no detector / no frame -> nothing to localize
            return None
        poses = self._valid_poses(frame)

        if not self.config.use_gate_track:
            best: GatePose | None = None
            for pose in poses:
                if best is None or pose.range_m < best.range_m:
                    best = pose
            self._last_none_reason = None if best is not None else "valid_poses_empty"
            return best

        # --- temporal track: lock one gate across frames -------------------
        if not poses:
            self._last_none_reason = "valid_poses_empty"   # no candidate survived score/reproj/in-front
            self._track_coast_ticks += 1
            if self._track_coast_ticks > max(1, int(self.config.track_max_coast_ticks)):
                self._track_range_m, self._track_bearing = None, None
            return None

        if self._track_range_m is None or self._track_bearing is None:
            # FIRST acquisition: choose the gate we must fly FIRST. (A5 BLOCKER 2) Pure prefer-centered
            # locked a DISTANT off-axis gate over the NEAR start-line gate; the near gate is the next one
            # to fly, so bias toward the NEAREST plausible gate. REJECT candidates beyond
            # ``max_acquire_range_m`` (a distant downrange gate is never the next gate); among the rest,
            # score by range + a bearing penalty so a near-and-reasonably-centered gate beats a
            # far-but-perfectly-centered one. Falls back to the legacy selection when prefer_nearest is
            # off (or when every candidate is beyond the acquire range -> don't reject them all).
            chosen = self._first_acquisition(poses)
        else:
            # CONTINUITY: pick the candidate nearest the track in (range, bearing); REJECT a jump.
            pred_r = float(self._track_range_m)
            pred_b = np.asarray(self._track_bearing, dtype=np.float64)

            def _consistent(p: GatePose) -> bool:
                return (abs(p.range_m - pred_r) <= self.config.track_max_range_jump_m
                        and float(np.linalg.norm(self._pose_bearing(p) - pred_b))
                        <= self.config.track_max_bearing_jump_rad)

            cands = [p for p in poses if _consistent(p)]
            if not cands:
                # every candidate jumped -> COAST on the track (do not lock onto a flapper).
                self._last_none_reason = "continuity_reject"   # the dominant A13 pose=None source
                self._track_coast_ticks += 1
                if self._track_coast_ticks > max(1, int(self.config.track_max_coast_ticks)):
                    self._track_range_m, self._track_bearing = None, None
                return None
            # among the consistent candidates, the one closest to the predicted bearing+range.
            chosen = min(
                cands,
                key=lambda p: float(np.linalg.norm(self._pose_bearing(p) - pred_b))
                + abs(p.range_m - pred_r) / max(self.config.track_max_range_jump_m, 1e-6),
            )

        # accept -> update the smoothed track and reset the coast counter.
        a = float(np.clip(self.config.track_ema_alpha, 0.0, 1.0))
        b_meas = self._pose_bearing(chosen)
        if self._track_range_m is None or self._track_bearing is None:
            self._track_range_m, self._track_bearing = float(chosen.range_m), b_meas
        else:
            self._track_range_m = (1.0 - a) * float(self._track_range_m) + a * float(chosen.range_m)
            self._track_bearing = (1.0 - a) * np.asarray(self._track_bearing, dtype=np.float64) + a * b_meas
        self._track_coast_ticks = 0
        self._last_none_reason = None    # a usable pose this tick (clear the stale reason)
        return chosen

    def _first_acquisition(self, poses: list[GatePose]) -> GatePose:
        """Pick the gate to LOCK on first acquisition (no track yet). (A5 BLOCKER 2 fix.)

        With ``prefer_nearest`` (default): REJECT candidates beyond ``max_acquire_range_m`` (a distant
        downrange gate is never the next gate to fly -- the A5 far-gate trap), then among the admissible
        ones minimise ``range_m + nearest_bearing_weight_m_per_rad * |bearing|`` so a NEAR,
        reasonably-centered gate beats a far-but-perfectly-centered one. If EVERY candidate is beyond the
        acquire range we do NOT reject them all (keep the nearest admissible-by-fallback); the score then
        still favours the nearest. ``prefer_nearest`` off => the legacy prefer-centered / closest select.
        """
        if not self.config.prefer_nearest:
            if self.config.track_prefer_centered:
                return min(poses, key=lambda p: float(np.linalg.norm(self._pose_bearing(p))))
            return min(poses, key=lambda p: p.range_m)
        # reject the distant downrange gates; if that empties the set, fall back to ALL (never reject
        # every candidate -> we must still lock something to make progress).
        admissible = [p for p in poses if p.range_m <= self.config.max_acquire_range_m]
        if not admissible:
            admissible = poses
        w = float(self.config.nearest_bearing_weight_m_per_rad)
        return min(admissible,
                   key=lambda p: p.range_m + w * float(np.linalg.norm(self._pose_bearing(p))))

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
          2.5 **Dead-reckon through the pass + acquire-next** (the A5-footage fix): once the tracked
             gate range drops below ``pass_arm_range_m`` the pass is ARMED; when that gate is then
             lost/behind (or ``active_gate_index`` increments) the seeker STOPS vision-servoing on the
             degenerate near-zero lever, FREEZES the pre-pass heading and commands a BOUNDED LEVEL
             forward coast (no re-level pitch-up) straight through the opening, then re-acquires the
             NEXT gate before resuming pursuit. Eliminates the post-pass backside-clip pitch-up + the
             indefinite forward drift.
          3. **Pursuit** (anchored + gate seen): build a desired velocity toward the seen opening at
             the slow cap and a yaw that centers its bearing, capped so the turn is smooth.
        """
        if self._t0_sim_ns is None:
            self._t0_sim_ns = int(nav.sim_time_ns)
        # wire-driven pass signal: did RACE_STATUS advance the active gate since the last command?
        index_advanced = self._last_index is not None and int(active_gate_index) > self._last_index
        self._last_index = int(active_gate_index)
        if self._last_yaw is None:
            self._last_yaw = self._att_yaw(nav)

        # ACQUIRE-NEXT track reset: once we are past the dead-reckon GLIDE (the pass_coast_s window) the
        # just-passed gate is behind us; the temporal track may still be stale-locked on it (a fresh
        # downrange gate would read as a big range JUMP and be rejected as a flapper, blocking the
        # handoff). While in the acquire-next window we CLEAR the track each tick so first-acquisition
        # re-locks the NEXT gate cleanly. (Inside the glide window we leave it alone -- we are coasting
        # straight and deliberately not steering on any gate.)
        if (self.config.use_pass_dead_reckon and self._passing
                and self._pass_t_ns is not None
                and (int(nav.sim_time_ns) - self._pass_t_ns) / 1e9 >= self.config.pass_coast_s):
            self._track_range_m, self._track_bearing, self._track_coast_ticks = None, None, 0

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
            if not self._anchored:
                self._release_t_ns = int(nav.sim_time_ns)   # start the post-release pursuit ramp clock
                # FREEZE the spawn heading = the direction OUT of the start gate the drone sits in
                # (the egress phase creeps along it to clear gate 0 before re-aiming downrange).
                self._spawn_heading = self._att_yaw(nav)
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

        # --- regime 1.5: SPAWN-GATE EGRESS (just released, drone still inside gate 0) -> a brief,
        # CAPPED forward creep along the FROZEN spawn heading (the direction OUT of the start gate),
        # to clear the start-gate structure BEFORE re-aiming at the downrange gate. The forward
        # demand is a bounded feedforward tilt (NOT a velocity setpoint), pitch-rate capped, so it
        # cannot lunge -- it eases the drone out of the spawn gate. (A4: the first forward motion
        # drove straight into the start-gate frame; the egress departs the spawn gate first.) ---
        if self._in_egress(int(nav.sim_time_ns)):
            return self._egress_command(nav)

        # --- PASS DETECTION + DEAD-RECKON-THROUGH bookkeeping (the A5-footage fix) -------------------
        # Maintain the closest tracked range seen while pursuing and ARM the pass once it drops below
        # pass_arm_range_m (the gate is filling the frame; below this the PnP lever degenerates). When
        # an armed gate is then LOST/behind, or the wire advances active_gate_index, we COMMIT to the
        # dead-reckon coast: STOP servoing on the about-to-be-passed gate, FREEZE the heading, glide
        # straight through on IMU, then acquire the next gate. ``_update_pass_state`` evaluates the
        # arm/commit triggers; ``_in_pass_dead_reckon`` bounds the committed coast/acquire-next window.
        if self.config.use_pass_dead_reckon and self._anchored:
            self._update_pass_state(int(nav.sim_time_ns), pose, index_advanced)
            if self._passing:
                # PASS COMMITTED: glide straight on the FROZEN pre-pass heading (no re-level snap, no
                # vision-servo on the degenerate lever). After ``pass_coast_s`` (the dead-reckon glide)
                # a fresh NEXT gate re-acquisition ends the pass and resumes normal pursuit on it;
                # otherwise keep the bounded forward coast (bounded by the coast+acquire window).
                if self._pass_acquired_next(nav, pose):
                    self._end_pass()                # next gate re-acquired -> resume pursuit below
                else:
                    return self._pass_coast_command(nav)

        # --- regime 2: NO DETECTION -> coast level on the last heading, gentle re-acquire ---
        # (Not during a pass: a pass-time gate loss is handled by the dead-reckon coast above, which
        # holds the forward lean instead of re-levelling into a pitch-up.)
        if pose is None:
            self._count_none_tick()    # A13 instrumentation: pose=None breakdown by reason
            # HOLD-LAST-DEMAND BRIDGE (A13): re-issue the last pursuit demand across a short pose gap
            # (continuous per-tick command) instead of regime-2's all-axes-zero re-level zero-coast.
            # Default OFF (hold_last_demand_s=0.0 -> None -> legacy hold == byte-identical).
            bridged = self._maybe_hold_last_demand(nav)
            if bridged is not None:
                self.diag_counts["bridged"] += 1
                return bridged
            self.diag_counts["held_legacy"] += 1
            return self._hold_command(nav, yaw_rate_cap=self.config.reacquire_yaw_rate_rps,
                                      attitude_safe=True)

        # --- regime 3: PURSUIT -> bounded feedforward forward tilt toward the SEEN opening +
        # centering yaw, pitch + roll capped, forward demand ramped (never a velocity setpoint) ---
        self.diag_counts["pursuit"] += 1
        return self._visual_pursuit_command(nav, pose)

    # =======================================================================
    # DEAD-RECKON THROUGH THE PASS + NEXT-GATE HANDOFF  (the A5-footage fix)
    # =======================================================================
    def _update_pass_state(self, sim_time_ns: int, pose: GatePose | None,
                           index_advanced: bool) -> None:
        """Maintain the pass state machine each anchored tick (called before the no-detection /
        pursuit regimes). Three jobs:

          * ARM the pass once the tracked gate range drops below ``pass_arm_range_m`` (the gate is
            filling the frame; below this the PnP lever degenerates). Tracks the closest range seen.
          * COMMIT to the dead-reckon coast (``_passing``) when an armed gate is (a) within the even-
            tighter ``pass_degenerate_range_m`` (the run4 est_pitch -1.08 blow-out: a close-range fix
            is already corrupt -> stop servoing on it NOW), or (b) LOST/behind after being armed, or
            (c) the wire advanced ``active_gate_index`` (the authoritative pass signal). On commit it
            FREEZES the current heading as the straight-through coast direction.
          * A transient mid-approach detection gap that is NOT armed (range never went below
            pass_arm_range_m) does NOT commit -- it falls through to the gentle no-detection coast, so
            an early detection dropout can never trigger the pass pitch-up.
        """
        # ARM + track the closest range while we still have a usable (non-degenerate) sighting.
        if pose is not None:
            rng = float(pose.range_m)
            self._pass_min_range_m = min(self._pass_min_range_m, rng)
            if rng <= self.config.pass_arm_range_m:
                if not self._pass_armed:
                    self._pass_index = self._last_index
                self._pass_armed = True
        if self._passing:
            return                                  # already committed -> nothing more to arm
        # COMMIT triggers (only meaningful once anchored + past egress):
        degenerate_close = (self._pass_armed and pose is not None
                            and float(pose.range_m) <= self.config.pass_degenerate_range_m)
        lost_after_arm = self._pass_armed and pose is None
        if degenerate_close or lost_after_arm or index_advanced:
            self._begin_pass(sim_time_ns)

    def _begin_pass(self, sim_time_ns: int) -> None:
        """Commit to the dead-reckon-through-pass coast: FREEZE the current pursuit heading (the
        straight-through direction) and start the coast clock. Drops the temporal gate track so that,
        once the coast ends, ACQUIRE-NEXT re-runs first-acquisition on the NEXT gate."""
        self._passing = True
        self._pass_t_ns = int(sim_time_ns)
        self._pass_heading = self._last_yaw if self._last_yaw is not None else 0.0
        # reset the temporal track so the next-gate re-acquisition starts clean (a different gate).
        self._track_range_m, self._track_bearing, self._track_coast_ticks = None, None, 0

    def _end_pass(self) -> None:
        """End the pass regime (the NEXT gate has been re-acquired) -> resume normal pursuit on it.
        Re-arms the pass bookkeeping for the next gate."""
        self._passing = False
        self._pass_armed = False
        self._pass_min_range_m = float("inf")
        self._pass_t_ns = None
        self._pass_heading = None
        self._pass_index = self._last_index

    def _pass_acquired_next(self, nav: NavState, pose: GatePose) -> bool:
        """True iff the seeker is in the ACQUIRE-NEXT window (past the dead-reckon coast) AND a fresh
        NEXT gate has been re-acquired -- i.e. a usable pose that is NOT the just-passed gate (a real
        downrange range, past the degenerate band). During the initial dead-reckon coast (within
        ``pass_coast_s``) we IGNORE any pose (it is the gate we are passing through) and keep gliding;
        only after that window do we accept a re-acquired gate as the next one to pursue."""
        if self._pass_t_ns is None or pose is None:
            return False
        elapsed = (int(nav.sim_time_ns) - self._pass_t_ns) / 1e9
        if elapsed < self.config.pass_coast_s:
            return False                            # still gliding through the opening -> not yet
        # past the dead-reckon coast: a sighting at a real downrange range = the next gate re-acquired.
        return float(pose.range_m) > self.config.pass_degenerate_range_m

    def _in_pass_dead_reckon(self, sim_time_ns: int) -> bool:
        """True while the committed pass coast (dead-reckon + acquire-next) window is active. Bounded
        by ``pass_coast_s + acquire_next_s`` so the straight glide can never run forever; past that the
        seeker reverts to the gentle no-detection coast (never a pitch-up)."""
        if not self._passing or self._pass_t_ns is None:
            return False
        elapsed = (int(sim_time_ns) - self._pass_t_ns) / 1e9
        return elapsed < (self.config.pass_coast_s + self.config.acquire_next_s)

    def _pass_coast_command(self, nav: NavState) -> ControlCommand:
        """The bounded LEVEL forward COAST through the pass (dead-reckon) + while acquiring the next
        gate. Holds the FROZEN pre-pass heading and commands the same bounded feedforward forward tilt
        as pursuit (so the forward lean is MAINTAINED -- no re-level snap that pitches up into the gate
        backside) with the pitch/roll caps. NO vision-servo on the degenerate near-zero lever and NO
        vertical-align (the lever is degenerate at the pass). If the coast+acquire window has elapsed
        without re-acquiring the next gate, fall back to the gentle no-detection level coast (which can
        never pitch up), so the dead-reckon glide is always bounded."""
        if not self._in_pass_dead_reckon(int(nav.sim_time_ns)):
            # the bounded coast+acquire window elapsed -> end the pass, revert to the gentle level hold.
            self._end_pass()
            return self._hold_command(nav, yaw_rate_cap=self.config.reacquire_yaw_rate_rps,
                                      attitude_safe=True)
        yaw0 = self._pass_heading if self._pass_heading is not None else self._att_yaw(nav)
        self._last_yaw = yaw0
        los = np.array([np.cos(yaw0), np.sin(yaw0), 0.0])
        launch = self._launch_ramp(int(nav.sim_time_ns))
        # full forward authority for the coast (we are committed to the glide); no vertical-align.
        return self._feedforward_command(nav, los, yaw0, launch,
                                         self.config.pass_coast_accel_mps2, 1.0, vz_cmd=0.0)

    # =======================================================================
    # HOLD-LAST-DEMAND BRIDGE  (the A13 polygonal-motion fix)
    # =======================================================================
    def _record_last_demand(self, los: np.ndarray, yaw: float, accel: float,
                            sim_time_ns: int) -> None:
        """Cache the LAST good pursuit demand (slewed heading + yaw + effective forward accel + sim
        time) so a subsequent pose-None tick can re-issue it via :meth:`_maybe_hold_last_demand`
        instead of regime-2's zero-coast hold. Written ONLY on a fresh pursuit tick (regime 3); never
        cleared on a pose-None tick, so it persists across the gap (only ``reset`` drops it)."""
        self._last_demand_los = np.asarray(los, dtype=np.float64).copy()
        self._last_demand_yaw = float(yaw)
        self._last_demand_accel = float(accel)
        self._last_demand_t_ns = int(sim_time_ns)

    def _maybe_hold_last_demand(self, nav: NavState) -> ControlCommand | None:
        """Bridge a short pose gap: if ``hold_last_demand_s > 0`` AND a cached pursuit demand exists
        AND we are still within the hold window since that demand, RE-ISSUE the last pursuit demand
        (cached forward lean on the frozen heading) so control is CONTINUOUS per tick -- no regime-2
        zero-coast. Returns the bridged :class:`ControlCommand`, or ``None`` to fall through to the
        legacy zeroed hold (OFF, no cache, or window expired).

        The held forward accel is scaled full->0 linearly over the window when ``hold_last_demand_decay``
        (decay/level out rather than ram a genuinely-lost gate forever). The bridge reuses the SAME
        capped ``_feedforward_command`` (yaw/roll/pitch caps) so it is bounded + safe; vertical-align is
        OFF (no fresh elevation lever during a gap, the lever is stale). The yaw is the cached (frozen)
        pursuit heading, so the camera is not slewed during the gap."""
        if (self.config.hold_last_demand_s <= 0.0 or self._last_demand_t_ns is None
                or self._last_demand_los is None or self._last_demand_yaw is None
                or self._last_demand_accel is None):
            return None
        elapsed = (int(nav.sim_time_ns) - int(self._last_demand_t_ns)) / 1e9
        if elapsed < 0.0 or elapsed >= self.config.hold_last_demand_s:
            return None    # window expired (or a stale/negative clock) -> legacy zeroed hold
        decay_scale = (1.0 - elapsed / self.config.hold_last_demand_s
                       if self.config.hold_last_demand_decay else 1.0)
        yaw0 = float(self._last_demand_yaw)
        self._last_yaw = yaw0                     # keep the heading frozen across the bridge
        launch = self._launch_ramp(int(nav.sim_time_ns))
        # Re-issue the cached forward demand on the frozen heading; demand_ramp=1.0 (the accel was
        # already the EFFECTIVE post-ramp value at cache time), decayed over the window. vz_cmd=0.0:
        # vertical-align OFF during the gap (the gate lever is stale -> no fresh elevation correction).
        return self._feedforward_command(
            nav, self._last_demand_los, yaw0, launch,
            float(self._last_demand_accel) * float(decay_scale), 1.0, vz_cmd=0.0)

    def _count_none_tick(self) -> None:
        """A13 instrumentation: tally a regime-2 (pose=None) tick by the REASON detect_gate_lever
        recorded (``_last_none_reason``). Logging only -- no behaviour change. The reason split tells
        the next fly which gate (track-continuity vs valid-poses) is the dominant gap source so A14 can
        relax it with data. ``valid_poses_empty`` = no candidate survived score/reproj/in-front;
        ``continuity_reject`` = candidates existed but all jumped the track gate (the suspected
        dominant source); ``first_acq_reject`` = the acquire-range reject (rarely a None source since
        first-acquisition falls back rather than rejecting all); ``other`` = no detector/frame/coast."""
        self.diag_counts["none_total"] += 1
        reason = self._last_none_reason
        key = {
            "valid_poses_empty": "none_valid_poses_empty",
            "continuity_reject": "none_continuity_reject",
            "first_acq_reject": "none_first_acq_reject",
        }.get(reason, "none_other")
        self.diag_counts[key] += 1

    def _in_settle(self, sim_time_ns: int) -> bool:
        """True while within ``settle_s`` of the launch clock arming (the post-arm cold-AHRS settle)."""
        if self.config.settle_s <= 0.0 or self._t0_sim_ns is None:
            return False
        return (int(sim_time_ns) - self._t0_sim_ns) / 1e9 < self.config.settle_s

    # -- TRUE-ATTITUDE recovery (A14 yaw-mirror fix) ------------------------
    def _att_rpy(self, nav: NavState) -> tuple[float, float, float]:
        """The (roll, pitch, yaw) the seeker geometry + heading bookkeeping should use.

        When ``true_attitude_from_ahrs`` is ON, recover the TRUE euler from the ODO-conjugated
        ``NavState`` the case-C Navigator emits: ``(-nav.roll, nav.pitch, -nav.yaw)`` (roll + yaw
        negated, pitch kept -- the R_y(pi) conjugation). OFF (default) => the raw ``nav`` euler
        (VQ1 / case-A byte-identical)."""
        if self.config.true_attitude_from_ahrs:
            return (-float(nav.roll), float(nav.pitch), -float(nav.yaw))
        return (float(nav.roll), float(nav.pitch), float(nav.yaw))

    def _att_yaw(self, nav: NavState) -> float:
        """The TRUE (or raw, flag-off) yaw for heading bookkeeping (last-yaw / spawn-heading / holds)."""
        return self._att_rpy(nav)[2]

    def _controller_nav(self, nav: NavState) -> NavState:
        """The NavState to hand the controller so its ``R_cur`` becomes R_true WITHOUT touching the
        controller sign config. When the flag is ON, NEGATE yaw only (``replace(nav, yaw=-nav.yaw)``):
        the controller's ``odo_att_sign=[-1,1,1]`` then yields R_cur = R_world_from_body(
        nav.roll*-1, nav.pitch, (-nav.yaw)*1) = R_world_from_body(+true_roll, +true_pitch, +true_yaw).
        Roll/pitch are left as the conjugated values that asign already handles. OFF => raw ``nav``."""
        if self.config.true_attitude_from_ahrs:
            import dataclasses
            return dataclasses.replace(nav, yaw=-float(nav.yaw))
        return nav

    def _gate_dir_world(self, nav: NavState, pose: GatePose) -> np.ndarray:
        """Unit world-NED direction from the drone to the DETECTED gate centre, from the relative lever.

        ``pose.t_cam_gate`` is the gate centre in the camera optical frame. Rotate it into the body
        frame (the fixed camera mount), then into world NED with the estimator's gravity-known
        attitude (roll/pitch from the AHRS accel-levelled tilt, yaw from the vision-pinned heading).
        NO absolute self-position enters — only the DIRECTION to the seen gate."""
        d_cam = _unit(np.asarray(pose.t_cam_gate, dtype=np.float64))
        d_body = R_camera_from_body().T @ d_cam
        tr, tp, ty = self._att_rpy(nav)
        R_wb = R_world_from_body(tr, tp, ty)
        return _unit(R_wb @ d_body, fallback=np.array([np.cos(ty), np.sin(ty), 0.0]))

    def _gate_lever_world(self, nav: NavState, pose: GatePose) -> np.ndarray:
        """FULL (non-unit) world-NED vector from the drone to the DETECTED gate centre, from the lever.

        Same body->world rotation as :meth:`_gate_dir_world` but keeps the MAGNITUDE: the gate centre's
        position RELATIVE to the drone in world NED (metres). Its Z component (NED Z+ = down) is the true
        VERTICAL OFFSET between the drone's flight path and the gate-opening centre -- positive = the
        opening is BELOW the drone (descend), negative = ABOVE (climb). Because ``R_camera_from_body()``
        carries the +20deg mount tilt and ``R_wb`` is the estimator's gravity-known attitude, this is the
        TRUE world vertical offset, NOT the raw camera-frame elevation (the camera points 20deg up, so
        nulling the camera-frame trk_el would leave a residual world offset). [A5 BLOCKER 1]"""
        t_cam = np.asarray(pose.t_cam_gate, dtype=np.float64)
        v_body = R_camera_from_body().T @ t_cam
        tr, tp, ty = self._att_rpy(nav)
        R_wb = R_world_from_body(tr, tp, ty)
        return R_wb @ v_body

    def _vertical_align_vz(self, nav: NavState, pose: GatePose) -> float:
        """Bounded vertical VELOCITY target (m/s, NED Z+ = down) that nulls the gate-opening vertical
        offset, ramped in from release. (A5 BLOCKER 1.)

        ``vz_t = clip(vertical_align_kp * offset_z_world, +/-cap) * ramp``, with a deadband so a
        near-aligned opening commands no correction (no hunting on estimator noise). offset_z_world is
        the world-NED Z of the gate-centre lever (:meth:`_gate_lever_world`): a gate BELOW the drone
        (offset > 0) yields a positive (descend) vz_t, which the controller's alt-hold turns into reduced
        thrust to sink toward the opening height. Bounded + ramped (NEVER a position step) -- the same
        discipline as the forward feedforward, so the vertical command can't lurch.

        A25 §5: below min-trust range, vz_t still returns 0.0 (unchanged) -- but that no longer
        strands the loop, because ẑ_off (latched separately, see :meth:`_maybe_latch_z_off`) now
        carries the altitude memory through the close-in zone."""
        self._last_offset_z_world = None                     # instrumentation default (no pose / no align)
        if not self.config.use_vertical_align:
            return 0.0
        # POINT-BLANK ELEVATION GUARD (A7): below the min-trust range the PnP elevation is degenerate
        # (a point-blank spawn gate read trk_el<0 and drove a descent into the thrust floor -> free-fall).
        # Command NO vertical correction there -- the alt-hold holds the current height instead of chasing
        # a garbage elevation. Normal pursuit gates sit well beyond this range, so this never fires in
        # downrange flight (post-egress byte-identity preserved).
        if (self.config.use_min_trust_elevation
                and float(pose.range_m) < self.config.min_trust_elevation_range_m):
            return 0.0
        offset_z = float(self._gate_lever_world(nav, pose)[2])
        self._last_offset_z_world = offset_z                  # A25 §7 instrumentation (raw, pre-deadband)
        if abs(offset_z) <= self.config.vertical_align_deadband_m:
            return 0.0
        cap = abs(float(self.config.vertical_align_speed_cap_mps))
        vz = float(np.clip(self.config.vertical_align_kp * offset_z, -cap, cap))
        return vz * self._vertical_align_ramp(int(nav.sim_time_ns))

    def _maybe_latch_z_off(self, nav: NavState, pose: GatePose) -> None:
        """A25 §1.4/§5: latch the gate-relative vertical offset into the shared ``VerticalEstimator``
        (``nav_owner._vert_est``, read LIVE each call -- see :attr:`nav_owner`) on a FRESH pose -- a
        NEW detection, never a re-used cached one (an async ZOH re-feed of the same capture would
        otherwise re-inject stale data every tick). Suppressed below ``min_trust_elevation_range_m``
        (the point-blank PnP elevation is degenerate there; §5.2 -- ẑ_off HOLDS its last latched
        value and keeps propagating by vz instead of being overwritten by garbage). No-op when no
        estimator is wired/seeded -- VQ1/case-A byte-identical; the seeker's OWN pursuit behaviour
        never depends on this call.

        A26 pose_age_s FIX (2026-07-02): ``pose.sim_time_ns`` is on the CAMERA/server epoch (the
        JPEG-wire header, unix-wall-clock ns on the live sim) while ``nav.sim_time_ns`` is on the
        IMU master epoch (``HIGHRES_IMU.time_usec``, sim-uptime ns) -- two DIFFERENT, unreconciled
        clocks (see ``NavigatorConfig.reconcile_vision_clock``'s note in navigator.py). The camera
        epoch is always vastly LARGER than the IMU epoch on the live wire, so the raw subtraction
        ``nav.sim_time_ns - pose.sim_time_ns`` was always hugely NEGATIVE, and the very next
        ``max(0.0, ...)`` staleness guard silently clamped every tick to exactly 0.0 (confirmed:
        run 20260702_221235's nav_estimate.jsonl logs pose_age_s==0.0 on all 159 non-null ticks).
        Convert the pose's camera-epoch stamp onto the IMU epoch FIRST via the Navigator's learned
        ``delta_epoch`` (``nav_owner.camera_epoch_to_imu_ns``, the same reconciliation the KF's own
        OOSM fix-time already uses, see ``Navigator._vision_fix_time_imu_ns``) so the subtraction
        compares like-for-like. ``None`` (epoch not learned yet, or ``nav_owner`` unset -- e.g. the
        unit-test seeker built without a Navigator) falls back to the RAW stamp: on synthetic/VQ1
        tests both clocks are the same fabricated epoch (delta==0 either way), so this fallback is
        byte-identical there; it only under-corrects on a live wire before the first paired
        (frame, ds) has landed (a handful of startup ticks, harmless -- the estimator isn't seeded
        that early either)."""
        self._last_pose_age_s = None                          # instrumentation default (no pose this tick)
        vert_est = getattr(self.nav_owner, "_vert_est", None)
        if vert_est is None or not getattr(vert_est, "seeded", False):
            return
        _to_imu_ns = getattr(self.nav_owner, "camera_epoch_to_imu_ns", None)
        pose_imu_ns = _to_imu_ns(pose.sim_time_ns) if _to_imu_ns is not None else None
        if pose_imu_ns is None:
            pose_imu_ns = int(pose.sim_time_ns)     # fallback: same-epoch tests / pre-reconciliation
        obs_age_s = max(0.0, (int(nav.sim_time_ns) - int(pose_imu_ns)) / 1e9)
        obs_age_s = min(obs_age_s, 1.0)                        # obs_age_max_s (A25 §2.2): reject a
                                                                # garbage/negative delta or >1s stale pose
        self._last_pose_age_s = obs_age_s                      # A25 §7 instrumentation (every pursuit tick)
        if (self.config.use_min_trust_elevation
                and float(pose.range_m) < self.config.min_trust_elevation_range_m):
            return                                              # HOLD: do not latch (§5.2), do not re-mark fresh
        if int(pose.sim_time_ns) == self._last_latched_pose_ns:
            return                                              # same capture (async ZOH re-feed): not fresh
        offset_z_world = float(self._gate_lever_world(nav, pose)[2])
        vert_est.latch_offset(offset_z_world, obs_age_s)
        self._last_latched_pose_ns = int(pose.sim_time_ns)

    def _vertical_align_ramp(self, sim_time_ns: int) -> float:
        """Vertical-align authority ramp [0,1] over ``vertical_align_ramp_s`` from the anchor release, so
        the first pursuit ticks don't step to a full descent command (mirrors the forward-demand ramp)."""
        if self.config.vertical_align_ramp_s <= 0.0 or self._release_t_ns is None:
            return 1.0
        elapsed = (int(sim_time_ns) - self._release_t_ns) / 1e9
        return float(np.clip(elapsed / self.config.vertical_align_ramp_s, 0.0, 1.0))

    def _visual_pursuit_command(self, nav: NavState, pose: GatePose) -> ControlCommand:
        """Build the slow pursuit CTBR from the SEEN gate's relative bearing (no map, no abs position).

        FORWARD MOTION (A4 PITCH-windup fix): commanded as a BOUNDED FEEDFORWARD horizontal
        acceleration (``forward_accel_mps2`` along the slewed gate heading), RAMPED in over
        ``forward_ramp_s`` -- NOT a velocity setpoint. Map-free, velocity is unobservable, so a
        velocity-error term (the old ``cruise_speed*los`` path) never closes and winds the pitch to
        the controller limit. A fixed feedforward forward demand depends on no velocity estimate, so
        the forward tilt is BOUNDED by construction; the PITCH rate is then hard-capped symmetric to
        the roll cap. Altitude is held by the alt-hold (the forward demand is purely horizontal).
        ``use_feedforward_forward=False`` restores the legacy velocity-setpoint pursuit.

        LAYER-2b smoothing (the A3 roll-over fix, retained): the commanded heading is RATE-LIMITED so
        a noisy bearing can't STEP the yaw setpoint; a post-release PURSUIT RAMP scales lean authority
        up from a small floor over ``pursuit_ramp_s``; the ROLL command is capped below saturation."""
        gdir = self._gate_dir_world(nav, pose)
        horiz = np.array([gdir[0], gdir[1], 0.0])
        _ty = self._att_yaw(nav)
        los = _unit(horiz, fallback=np.array([np.cos(_ty), np.sin(_ty), 0.0]))
        yaw_des = float(np.arctan2(los[1], los[0]))
        # INSTRUMENTATION ONLY (A14 yaw-steer-sign probe): stash the PRE-SLEW desired yaw toward the
        # gate so the nav-estimate logger can read it. Not consumed by control — purely additive.
        self._last_yaw_des = yaw_des
        # RATE-LIMIT the heading slew: cap the per-tick change of the yaw setpoint so the steering
        # bearing stays SMOOTH (the proximate fix for the swing that saturated roll in A3).
        yaw = self._slew_heading(yaw_des, int(nav.sim_time_ns))
        self._last_yaw = yaw
        # PURSUIT RAMP: scale the translational lean authority up from a floor over the first
        # pursuit_ramp_s after release (a noisy first bearing can't step-saturate roll), composed with
        # the takeoff launch ramp.
        ramp = self._pursuit_ramp(int(nav.sim_time_ns))
        launch = self._launch_ramp(int(nav.sim_time_ns))
        eff_ramp = ramp if launch is None else min(ramp, launch)
        if self.config.use_feedforward_forward:
            # BOUNDED FEEDFORWARD forward tilt toward the seen gate: a fixed forward accel demand
            # (ramped), NO velocity term to wind up. Cross-track centering is owned by the yaw (the
            # heading points at the gate, so "forward" == toward the opening). VERTICAL ALIGNMENT (A5
            # BLOCKER 1): a bounded vertical-velocity target nulls the gate-opening vertical offset so
            # the drone descends/climbs onto the opening centre instead of holding altitude and clipping.
            vz = self._vertical_align_vz(nav, pose)
            self._last_vz_t = vz                          # A25 §7 instrumentation (raw command, pre-controller)
            # A25 §1.4/§5: latch the gate-relative vertical offset into the shared VerticalEstimator
            # (fresh-pose dedupe + min-range suppression live inside). This is INDEPENDENT of vz_t
            # above (which stays 0 below min range / deadband) -- ẑ_off is the loop's persistent
            # altitude memory; latching it is not gated on whether vz_t itself fired this tick.
            self._maybe_latch_z_off(nav, pose)
            fwd_ramp = self._forward_accel_ramp(int(nav.sim_time_ns))
            # HOLD-LAST-DEMAND BRIDGE (A13): cache this fresh pursuit demand so a subsequent pose-None
            # tick can re-issue it (continuous per-tick command) instead of regime-2's zero-coast. We
            # cache the slewed world heading + yaw + the EFFECTIVE forward accel (forward_accel * the
            # combined authority/forward ramps) + the demand sim time. Written ONLY on a fresh pursuit
            # tick; never cleared on a pose-None tick (so it persists across the gap).
            self._record_last_demand(los, yaw,
                                     self.config.forward_accel_mps2 * float(eff_ramp) * float(fwd_ramp),
                                     int(nav.sim_time_ns))
            return self._feedforward_command(nav, los, yaw, eff_ramp,
                                              self.config.forward_accel_mps2,
                                              fwd_ramp, vz_cmd=vz)
        # LEGACY: a desired-velocity setpoint (the controller closes it with a velocity-error term).
        sp = Setpoint(
            sim_time_ns=int(nav.sim_time_ns),
            velocity_ned=self.config.cruise_speed * los,    # level pursuit (altitude held by alt-hold)
            yaw=yaw,
            launch_ramp=eff_ramp,
        )
        cmd = self.controller.command(self._controller_nav(nav), sp)
        cmd = self._cap_yaw_rate(cmd, self.config.visual_yaw_rate_cap_rps)
        cmd = self._cap_roll_rate(cmd, self.config.pursuit_roll_rate_cap_rps)
        return self._cap_pitch_rate(cmd, self.config.pursuit_pitch_rate_cap_rps)

    def _feedforward_command(self, nav: NavState, los: np.ndarray, yaw: float,
                             launch_ramp: float | None, accel_mps2: float,
                             demand_ramp: float, vz_cmd: float = 0.0) -> ControlCommand:
        """Shared bounded-feedforward forward-tilt CTBR (pursuit + egress). Commands a horizontal
        acceleration ``accel_mps2 * demand_ramp`` along the unit world heading ``los`` via
        ``Setpoint.accel_ned`` -- the controller adds it as PURE feedforward (no velocity-error term
        that could wind up map-free) and turns it into a tilt. The pursuit/launch authority ramp,
        yaw cap, roll cap and PITCH cap are then applied so the forward lean is bounded + rate-limited
        + ramped and can NEVER saturate pitch (the A4 crash).

        VERTICAL (A5 BLOCKER 1): when ``vz_cmd`` != 0 a bounded vertical-velocity target is carried in
        ``Setpoint.velocity_ned`` (HORIZONTAL components ZERO -- only Z), so the controller's altitude
        hold tracks the commanded sink/climb rate (vz_t) toward the gate-opening height while the
        horizontal accel feedforward owns the forward/cross-track tilt. The horizontal velocity error
        the controller derives from velocity_ned[0:2]=0 is ~zero map-free (vel~0), so it does not
        disturb the forward feedforward; vz_cmd=0 leaves the legacy fixed-altitude hold (velocity_ned
        stays None -> vz_t=0 -> hold current z)."""
        a_fwd = float(max(accel_mps2, 0.0)) * float(np.clip(demand_ramp, 0.0, 1.0))
        velocity_ned = None
        if abs(float(vz_cmd)) > 0.0:
            velocity_ned = np.array([0.0, 0.0, float(vz_cmd)], dtype=np.float64)  # VERTICAL target only
        sp = Setpoint(
            sim_time_ns=int(nav.sim_time_ns),
            accel_ned=a_fwd * np.asarray(los, dtype=np.float64),   # bounded feedforward forward tilt
            velocity_ned=velocity_ned,                             # bounded vertical-align vz_t (Z only)
            yaw=yaw,
            launch_ramp=launch_ramp,
        )
        cmd = self.controller.command(self._controller_nav(nav), sp)
        cmd = self._cap_yaw_rate(cmd, self.config.visual_yaw_rate_cap_rps)
        cmd = self._cap_roll_rate(cmd, self.config.pursuit_roll_rate_cap_rps)
        return self._cap_pitch_rate(cmd, self.config.pursuit_pitch_rate_cap_rps)

    def _in_egress(self, sim_time_ns: int) -> bool:
        """True during the SPAWN-GATE EGRESS window: a straight creep along the frozen spawn heading that
        clears the start gate (the drone spawns inside gate 0) before normal downrange pursuit. Off when
        ``use_spawn_egress`` is False / not yet released.

        END CONDITION (A5 BLOCKER 3): with ``use_distance_egress`` the egress ends on DISTANCE crept out
        of the gate (``egress_clear_distance_m``, dead-reckoned from the seeker's own bounded forward
        demand) rather than a fixed timer -- the 0.8 s timer didn't reliably clear the gate-0 frame.
        ``egress_s`` remains a hard UPPER bound (timeout) so egress can never run forever. With
        ``use_distance_egress`` off it is the legacy pure time window (< ``egress_s``)."""
        if not self.config.use_spawn_egress or self.config.egress_s <= 0.0:
            return False
        if self._release_t_ns is None or self._spawn_heading is None:
            return False
        elapsed = (int(sim_time_ns) - self._release_t_ns) / 1e9
        if elapsed >= self.config.egress_s:        # hard timeout (also the legacy end condition)
            return False
        if self.config.use_distance_egress:
            # distance-based: still egressing until we've crept the clear distance out of the gate.
            if self._egress_done or self._egress_dist_m >= self.config.egress_clear_distance_m:
                return False
        return True

    def _egress_command(self, nav: NavState) -> ControlCommand:
        """SPAWN-GATE EGRESS: a small CAPPED forward creep along the FROZEN spawn heading (the
        direction OUT of the start gate the drone sits in), to clear gate 0 before re-aiming at the
        downrange gate. Reuses the bounded feedforward forward-tilt + pitch cap; the heading is the
        spawn heading (NOT slewed toward the downrange gate) so the drone departs straight out of the
        spawn gate rather than turning + lunging into its frame."""
        yaw0 = self._spawn_heading if self._spawn_heading is not None else self._att_yaw(nav)
        self._last_yaw = yaw0
        los = np.array([np.cos(yaw0), np.sin(yaw0), 0.0])
        launch = self._launch_ramp(int(nav.sim_time_ns))
        # ramp the egress creep in over its own window so even the egress lean eases (no step-lunge).
        if self.config.egress_s > 0.0 and self._release_t_ns is not None:
            elapsed = (int(nav.sim_time_ns) - self._release_t_ns) / 1e9
            demand_ramp = float(np.clip(elapsed / self.config.egress_s, 0.0, 1.0))
        else:
            demand_ramp = 1.0
        # DEAD-RECKON the along-heading distance crept since release (A5 BLOCKER 3): integrate the
        # seeker's OWN bounded forward demand (a = egress_accel * ramp) into a velocity then a distance
        # (v += a*dt ; dist += v*dt). Map-free we have no absolute self-position, so this is the only
        # observable "how far out of the gate am I" signal; _in_egress ends the phase once dist exceeds
        # egress_clear_distance_m. Bounded by construction (the forward demand is the same capped creep).
        if self.config.use_distance_egress:
            self._advance_egress_distance(int(nav.sim_time_ns), demand_ramp)
        cmd = self._feedforward_command(nav, los, yaw0, launch,
                                        self.config.egress_accel_mps2, demand_ramp)
        # EGRESS ATTITUDE FREEZE (A10): hold the frozen spawn pitch/roll -- zero the egress' OWN
        # roll/pitch rate command (like hold_freeze_attitude) so the +20deg camera stays ON the spawn
        # gate while forward demand ramps from ~0. Without this the level-target-vs-tilted-current
        # delta saturates the pitch-rate cap into a nose-UP re-level that points the camera off the
        # gate -> detector dark -> pursuit never acquires a pose (the acquisition trap). Yaw (frozen)
        # + thrust-floor + forward feedforward remain. OFF (default) == legacy egress.
        if self.config.egress_freeze_attitude:
            cmd = self._zero_rp_rate(cmd)
        # EGRESS THRUST FLOOR (A7): clamp the collective to at least hover-equivalent so the drone can
        # only HOLD or CLIMB out of the spawn gate -- it can never descend / free-fall while egressing
        # (the point-blank close-range thrust collapse). Only the egress command is floored; pursuit's
        # alt-hold owns thrust unchanged.
        return self._floor_egress_thrust(cmd)

    def _floor_egress_thrust(self, cmd: ControlCommand) -> ControlCommand:
        """Floor the egress collective to at least ``hover_thrust * egress_thrust_floor_frac`` (A7): a
        point-blank spawn gate can drive the alt-hold to the controller's 0.05 floor -> free-fall; the
        egress must HOLD or CLIMB out of the start gate, never descend. ``use_egress_thrust_floor`` off
        => the legacy egress collective passes through."""
        if (not self.config.use_egress_thrust_floor or cmd.thrust is None
                or self.config.egress_thrust_floor_frac <= 0.0):
            return cmd
        import dataclasses
        floor = float(self.controller.hover_thrust) * float(self.config.egress_thrust_floor_frac)
        if cmd.thrust >= floor:
            return cmd
        return dataclasses.replace(cmd, thrust=float(min(floor, 1.0)))

    def _advance_egress_distance(self, sim_time_ns: int, demand_ramp: float) -> None:
        """Integrate the dead-reckoned along-heading egress distance from the seeker's own bounded
        forward demand. Double-integrates a = egress_accel_mps2 * demand_ramp (the capped creep) over
        the tick dt: ``self._egress_v`` is folded into ``self._egress_dist_m``. Latches ``_egress_done``
        once the clear distance is reached so the phase can't re-enter. [A5 BLOCKER 3]"""
        if self._last_egress_t_ns is None:
            self._last_egress_t_ns = int(sim_time_ns)
            return
        dt = max((int(sim_time_ns) - self._last_egress_t_ns) / 1e9, 0.0)
        self._last_egress_t_ns = int(sim_time_ns)
        a = float(max(self.config.egress_accel_mps2, 0.0)) * float(np.clip(demand_ramp, 0.0, 1.0))
        # simple forward kinematics: v += a*dt, dist += v*dt (the creep speed grows under the bounded
        # accel). The speed accumulates on the instance across ticks.
        self._egress_v = self._egress_v + a * dt
        self._egress_dist_m += self._egress_v * dt
        if self._egress_dist_m >= self.config.egress_clear_distance_m:
            self._egress_done = True

    def _forward_accel_ramp(self, sim_time_ns: int) -> float:
        """Forward-demand ramp [0,1] over ``forward_ramp_s`` from the anchor release, so the forward
        tilt eases IN (the first pursuit ticks don't step to the full forward lean). The egress window
        sits inside this ramp, so when pursuit takes over the forward demand is already partly ramped;
        we measure the ramp from RELEASE (not from pursuit start) for a continuous build. 1.0 once the
        ramp completes / when disabled."""
        if self.config.forward_ramp_s <= 0.0 or self._release_t_ns is None:
            return 1.0
        elapsed = (int(sim_time_ns) - self._release_t_ns) / 1e9
        return float(np.clip(elapsed / self.config.forward_ramp_s, 0.0, 1.0))

    def _slew_heading(self, yaw_des: float, sim_time_ns: int) -> float:
        """Rate-limit the commanded heading: step ``_last_yaw`` toward ``yaw_des`` by at most
        ``pursuit_yaw_slew_rps`` * dt (shortest angular path). Keeps the steering bearing smooth so a
        noisy bearing can't demand a heading jump the controller turns into a saturating roll/yaw."""
        last = self._last_yaw if self._last_yaw is not None else yaw_des
        slew = float(self.config.pursuit_yaw_slew_rps)
        if slew <= 0.0 or self._last_pursuit_t_ns is None:
            self._last_pursuit_t_ns = int(sim_time_ns)
            return yaw_des
        dt = max((int(sim_time_ns) - self._last_pursuit_t_ns) / 1e9, 0.0)
        self._last_pursuit_t_ns = int(sim_time_ns)
        # shortest signed angular error in (-pi, pi]
        derr = float(np.arctan2(np.sin(yaw_des - last), np.cos(yaw_des - last)))
        max_step = slew * dt if dt > 0.0 else abs(derr)
        derr = float(np.clip(derr, -max_step, max_step))
        return float(np.arctan2(np.sin(last + derr), np.cos(last + derr)))

    def _pursuit_ramp(self, sim_time_ns: int) -> float:
        """Pursuit authority [floor, 1] ramping over ``pursuit_ramp_s`` from the anchor release, so
        the first pursuit ticks ease in (a noisy first bearing can't step-saturate roll). 1.0 once the
        ramp completes / when disabled."""
        if self.config.pursuit_ramp_s <= 0.0 or self._release_t_ns is None:
            return 1.0
        elapsed = (int(sim_time_ns) - self._release_t_ns) / 1e9
        if elapsed >= self.config.pursuit_ramp_s:
            return 1.0
        f = float(np.clip(self.config.pursuit_ramp_floor, 0.0, 1.0))
        return float(f + (1.0 - f) * np.clip(elapsed / self.config.pursuit_ramp_s, 0.0, 1.0))

    def _cap_roll_rate(self, cmd: ControlCommand, cap_rps: float) -> ControlCommand:
        """Clamp the ROLL (FRD body-rate X) command to +/-``cap_rps`` -- the pursuit roll-over guard
        (A3's crash axis: a residual bearing swing must never saturate roll)."""
        if cmd.body_rate is None or cap_rps <= 0.0:
            return cmd
        import dataclasses
        br = np.asarray(cmd.body_rate, dtype=np.float64).copy()
        c = abs(float(cap_rps))
        br[0] = float(np.clip(br[0], -c, c))
        return dataclasses.replace(cmd, body_rate=br)

    def _cap_pitch_rate(self, cmd: ControlCommand, cap_rps: float) -> ControlCommand:
        """Clamp the PITCH (FRD body-rate Y) command to +/-``cap_rps`` -- the pursuit PITCH-windup
        guard (A4's crash axis: the map-free forward lean must never wind to the -4.0 limit).
        Symmetric to ``_cap_roll_rate``; the second half of the bounded-feedforward-tilt fix."""
        if cmd.body_rate is None or cap_rps <= 0.0:
            return cmd
        import dataclasses
        br = np.asarray(cmd.body_rate, dtype=np.float64).copy()
        c = abs(float(cap_rps))
        br[1] = float(np.clip(br[1], -c, c))
        return dataclasses.replace(cmd, body_rate=br)

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
        hold_yaw = self._last_yaw if self._last_yaw is not None else self._att_yaw(nav)
        sp = Setpoint(
            sim_time_ns=int(nav.sim_time_ns),
            velocity_ned=np.zeros(3),       # no horizontal lean -> level hover
            yaw=hold_yaw,
            launch_ramp=0.0,                # full anti-lean: keep the attitude level while holding
        )
        cmd = self.controller.command(self._controller_nav(nav), sp)
        cmd = self._cap_yaw_rate(cmd, yaw_rate_cap)
        if attitude_safe:
            if self.config.hold_freeze_attitude:
                # LAYER 1 (A3 fix): FREEZE the spawn attitude -- zero the roll/pitch rate command so
                # the hold does NOT actively re-level off the gravity-aligned spawn TILT. The level-
                # hold controller, fed the (correct) spawn tilt, otherwise commands a persistent
                # clamped pitch that slowly tilts the camera OFF the gate -> the detector goes dark
                # and the release streak stalls. Holding the current attitude keeps the gate in view
                # so the detection streak completes. (Yaw clamp + bounded thrust still apply below.)
                cmd = self._zero_rp_rate(cmd)
            else:
                cmd = self._cap_rp_rate(cmd, self.config.hold_rp_rate_cap_rps)
            cmd = self._bound_hold_thrust(cmd)
        return cmd

    def _zero_rp_rate(self, cmd: ControlCommand) -> ControlCommand:
        """Zero the ROLL/PITCH (FRD body-rate X,Y) command -- the LAYER-1 attitude FREEZE: hold the
        current camera attitude (do not re-level off the gate) so the gate stays in view through the
        detection streak. Yaw is left untouched (clamped separately)."""
        if cmd.body_rate is None:
            return cmd
        import dataclasses
        br = np.asarray(cmd.body_rate, dtype=np.float64).copy()
        br[0] = 0.0
        br[1] = 0.0
        return dataclasses.replace(cmd, body_rate=br)

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
        self._track_range_m = None
        self._track_bearing = None
        self._track_coast_ticks = 0
        self._release_t_ns = None
        self._last_pursuit_t_ns = None
        self._spawn_heading = None
        self._egress_dist_m = 0.0
        self._egress_v = 0.0
        self._last_egress_t_ns = None
        self._egress_done = False
        self._pass_armed = False
        self._pass_min_range_m = float("inf")
        self._passing = False
        self._pass_t_ns = None
        self._pass_heading = None
        self._pass_index = None
        # hold-last-demand bridge (A13): drop the cached demand + the pose-None reason; zero the
        # per-flight diagnostic counters so a fresh epoch starts clean.
        self._last_demand_los = None
        self._last_demand_yaw = None
        self._last_demand_accel = None
        self._last_demand_t_ns = None
        self._last_none_reason = None
        for k in self.diag_counts:
            self.diag_counts[k] = 0

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
