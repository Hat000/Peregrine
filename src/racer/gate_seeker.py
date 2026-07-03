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


def _clip_norm(v: np.ndarray, cap: float) -> np.ndarray:
    """Scale ``v`` down so its norm is at most ``cap`` (direction preserved). cap<=0 => untouched."""
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if cap <= 0.0 or n <= cap:
        return v
    return v * (float(cap) / n)


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

    # --- LOS-RATE DAMPING / tangential-velocity kill (the A29 orbit fix, 2026-07-03) ---
    # A29 (run 20260703_024023): after clearing gate 0 the drone carried ~3 m/s of momentum mostly
    # PERPENDICULAR to the gate-1 line-of-sight -- and NO term in the whole control chain opposes
    # that real tangential velocity (velocity is unobservable on this wire; ff_owns_horizontal
    # correctly removed the damping of the FICTIONAL dead-reckoned velocity, and nothing replaced
    # it). Pure pursuit with a lateral velocity component is the classic tail-chase: the seeker
    # poured 21.1 m/s of commanded delta-v into a 392-deg orbit (net vector 2.6 m/s, coherence
    # 0.12) -- it yawed AT the gate while coasting AROUND it. THE FIX: the missing observable is
    # the LOS RATE. For a fixed gate, v_t = v.e_t = -r*theta_dot, where theta is the world LOS
    # angle (yaw_des, pre-slew -- already computed every pursuit tick) and r the tracked PnP range
    # (_track_range_m, already EMA-maintained). Classical proportional navigation, done entirely
    # with signals that already exist: estimate theta_dot by differencing yaw_des across FRESH
    # poses ON THE CAMERA-EPOCH STAMPS (same clock both sides, so the A29 epoch-rate skew
    # cancels), EMA-filter it, and add a lateral demand a_lat = -kd*v_t = +kd*r*theta_dot along
    # e_t = [-sin(theta), cos(theta), 0] that BRAKES the tangential drift. The bearing then stops
    # receding, the orbit never forms, and the forward feedforward integrates coherently into
    # closing speed. No velocity estimate is consumed anywhere -- r and theta_dot are pure vision
    # observables; the banned dead-reckoned-velocity servo stays banned (A15b/A23 stand).
    # OFF (default) => byte-identical (VQ1 / case-A untouched).
    use_los_rate_damping: bool = False
    # EMA on the per-fresh-pose LOS-rate sample (the sample cadence is the fresh-pose cadence,
    # median ~48 ms gap on the A29 run; 0.4 tracks a 0.85 rad/s sweep with little lag).
    los_rate_ema_alpha: float = 0.4
    # Reject a raw sample beyond this (rad/s): a gate-track hop is a STEP in theta -> a >>3 rad/s
    # one-tick sample; the honest orbit rate was 0.4-0.85 rad/s.
    los_rate_max_rps: float = 3.0
    # Reject/reset across a pose gap longer than this (s): differencing across a long gap mixes
    # geometry regimes (the drone moved), so restart the filter instead.
    los_rate_max_dt_s: float = 0.5
    # Velocity-damping gain (1/s): a_lat = kd * |v_t| against the tangential drift. Observed
    # v_t ~ 3 m/s => 2.4 m/s^2 demand (capped below) kills it in ~1.5-2 s.
    tangential_kd: float = 0.8
    # Cap the lateral demand (m/s^2): 2.0 alone is a ~11.5 deg lean -- well inside the 1.5 rad/s
    # pitch/roll caps and far from the controller's max tilt.
    lateral_accel_cap_mps2: float = 2.0
    # Norm-cap the COMPOSED horizontal demand (forward + lateral; m/s^2): 2.5 ~ 14.3 deg lean.
    total_accel_cap_mps2: float = 2.5
    # No correction below this |v_t| (m/s): ~the r*theta_dot noise floor for r~10 m and ~0.03
    # rad/s theta_dot jitter -- don't hunt on noise near a dead bearing.
    tangential_deadband_mps: float = 0.3
    # Clamp the range used in v_t (m): a mis-depthed far candidate can't demand a huge lateral.
    los_range_cap_m: float = 25.0

    # --- A30 IMAGE-SERVO LATERAL: roll toward the APPARENT gate (2026-07-03) ---
    # A30 (run 20260703_150755): the A29 LOS-rate damping went UNSTABLE -- it differentiated a
    # noisy, self-motion-contaminated bearing (corr(raw sample, own yaw rate) = 0.65) and
    # multiplied by an untrusted range (track pinned at the 25 m cap on 62% of ticks), producing
    # a |58| m/s vt_est and a saturating ~2.1 s lateral limit cycle (alat sign-flip every 1.15 s,
    # 80% saturated) that thrashed the drone over gate 0. THE REPLACEMENT (this block): the
    # operator's image-proportional law -- roll toward where the gate APPEARS in the frame.
    # Per pursuit tick: az = wrap(psi_world - yaw_now) (the gate's apparent horizontal offset,
    # +right), a_lat = clip(k_az * dead(az), +/-cap) along e_right(yaw_now), composed with the
    # forward drive scaled by max(cos(az),0)^2 (point before pushing). NO derivative, NO range,
    # NO filter state: the input is a per-frame-measured LEVEL bounded by the FOV -- it
    # structurally cannot produce a 58 m/s internal state because it has no internal state.
    # WHY IT BRAKES THE ORBIT: when an orbit tries to form the LOS sweeps and the yaw servo LAGS
    # (measured on A28: az = 0.208*psi_dot, corr 0.64), so the gate sits off-center IN THE SWEEP
    # DIRECTION and the lateral accel toward it points anti-tangential -- the "differentiation"
    # is done by the yaw-servo physics, not numerics on a noisy signal. Replayed over the real
    # A28 gate-1 kinematics: net tangential delta-v = -16.4 m/s of braking available vs the
    # ~3 m/s carried. Near-centered it is the SAME braking direction A29 wanted at gain
    # k_az*b ~= 1.7 instead of kd*r = 20 -- 12x cooler, underived, range-free.
    # OFF (default) => byte-identical (VQ1 / case-A untouched). If both this and
    # use_los_rate_damping are ON, the image servo WINS (the A29 branch is unreachable).
    use_image_servo_lateral: bool = False
    # Lateral gain (m/s^2 per rad of apparent azimuth): saturates the cap beyond ~10.7 deg az;
    # linear-regime brake tau ~= r/(k_az*b) ~= 6 s at r=10 m (b ~= 0.21 s measured yaw lag);
    # PnP bearing noise 1-2 deg -> 0.14-0.28 m/s^2 command noise (benign, below deadband most ticks).
    image_kaz_mps2_per_rad: float = 8.0
    # Deadband on the apparent azimuth (rad): ~= the PnP bearing noise floor (1.7 deg) so a
    # centered gate commands NO lateral hunting. Applied CONTINUOUSLY (sign(az)*max(|az|-db,0))
    # so the lateral demand has no jump at the deadband edge.
    image_az_deadband_rad: float = 0.03
    # Cap the image-servo lateral demand (m/s^2): 1.5 ~= an 8.7 deg lean. The worst possible
    # transient (a track hop across the whole FOV) clips here, sign-correct the next fresh frame.
    image_lat_cap_mps2: float = 1.5
    # Capture-time attitude ring buffer depth (ticks): ~1.2 s at 30 Hz -- covers the observed
    # pose-age spread (p50 138 ms / p90 265 ms / max 334 ms under the A29 clock fix).
    image_att_hist_len: int = 36
    # Fall back to the CURRENT attitude when the nearest buffered attitude is farther than this
    # from the pose capture instant (s): a stale buffer must not rotate the lever with garbage.
    image_att_max_gap_s: float = 0.5

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

    # ===================================================================
    # A31 — IMMEDIATE TURN ON THE WIRE PASS + ORBIT-BREAKER + IMU-CONSISTENCY
    # BEARING GATE + LATERAL-DEMAND SLEW (2026-07-03; spec
    # handoff/vq2_a31_immediate_turn_spec_2026-07-03.md). All defaults legacy/off
    # => VQ1 / case-A byte-identical; activated only via vq2_case_c seeker_overrides.
    # ===================================================================
    # --- FIX 1: wire-pass fast window. When the pass was committed (or confirmed while already
    # committed) by the AUTHORITATIVE wire signal (RACE_STATUS.active_gate_index increment), the
    # drone is PAST the gate plane -- the 1.2 s blind dead-reckon glide is pure lost time (run
    # 20260703_160715: gate 1 was detected DURING the glide; pursuit resumed at exactly coast
    # expiry, 1.22 s late, carrying ~3.9 m/s). Use this much shorter acquire-eligibility window
    # instead (just enough to physically clear the frame the camera is inside). A vision-committed
    # pass that the wire then confirms MID-GLIDE upgrades to this window (the _pass_wire seam).
    # None => pass_coast_s everywhere (legacy, byte-identical).
    pass_wire_coast_s: float | None = None
    # --- FIX 2b: re-ramp the forward feedforward from zero over forward_ramp_s again after EVERY
    # pass ends (next gate acquired), not just from the spawn release: the post-pass geometry is a
    # fresh acquisition (bearing typically 30-60 deg off) and the A30 orbit data shows feeding
    # forward drive while the yaw converges is what sustains the tail-chase ("point before
    # pushing", enforced). False => legacy (ramp measured from spawn release only; byte-identical).
    reramp_forward_after_pass: bool = False
    # --- FIX 3: ORBIT-BREAKER. Run 20260703_160715: after overflying gate 1 the world LOS rotated
    # 203 deg in 4.9 s and the yaw lag-followed it all the way to BACKWARDS -- while the
    # INSTANTANEOUS camera bearing stayed small (+0.17 rad mean, in-FOV the whole whip), so an
    # instantaneous-bearing guard is structurally blind to it. The guard observable is therefore
    # the CUMULATIVE unwrapped LOS rotation since (re)acquisition of the current gate: a pursuit
    # whose LOS has rotated this far is orbiting its gate, not approaching it -- following further
    # is always wrong. Trip => a bounded "orbit_break" regime: forward accel 0, lateral =
    # image_lat_cap_mps2 toward the CURRENT apparent gate (the brake, same e_right composition),
    # yaw setpoint HELD (stop following the sweep -- let the LOS come back as v_t dies). Early
    # exit when the gate re-centers with a low fresh-pose LOS drift; hard exit at orbit_break_s.
    # A SECOND trip on the same acquisition drops the track and falls to the no-detection hold:
    # refuse the spin, wait level, reacquire clean. 0.0 => guard off (byte-identical).
    orbit_guard_rad: float = 0.0          # trip when |unwrapped psi_world - psi_at_acquire| exceeds this
    orbit_break_s: float = 1.0            # bounded brake regime length
    orbit_break_exit_az_rad: float = 0.15   # early exit: gate re-centered ...
    orbit_break_exit_rate_rps: float = 0.3  # ... AND fresh-pose LOS drift below this
    # Hard clamp on the slewed pursuit yaw setpoint, relative to the yaw at (re)acquisition of the
    # current gate: the "never turn to backwards chasing a gate" pin. 0.0 => off (byte-identical).
    orbit_yaw_clamp_rad: float = 0.0
    # --- THE UPGRADE (operator directive, replaces the tightened-fixed-threshold band-aid): the
    # IMU-CONSISTENCY BEARING GATE -- the horizontal analog of the A28 vertical complementary
    # filter. Gates are STATIC: over one inter-frame dt the gate's bearing can only change as fast
    # as the drone's OWN MEASURED motion -- rotation (the AHRS attitude delta between the two
    # capture instants, which we measure cleanly via the A30 capture-time attitude ring buffer)
    # plus a translation-parallax term (∝ motion/range, larger at close range). We gate in the
    # WORLD frame: each accepted pose's camera lever is rotated with the attitude AT ITS CAPTURE
    # TIME (_rpy_at), so the measured rotation is compensated EXACTLY and a static gate's world
    # direction is quasi-constant -- the only honest change left is translation parallax, which we
    # bound explicitly (bearing_gate_trans_mps * dt / range, so the gate WIDENS ∝ dt and ∝ 1/range
    # and a legitimate close-range sweep or a short pose gap is never falsely rejected). A vision
    # bearing whose world-direction deviation from the prediction exceeds noise + parallax is
    # REJECTED (track coasts; the A13 bridge covers the gap) -- REGARDLESS of the deviation's
    # absolute size: no fixed threshold a right-sized hop can defeat (the t=3.25 0.33 rad hop slid
    # under the 0.35 rad fixed gate; against a ~0.1 rad motion-consistency allowance it is 3x out).
    # When ON this REPLACES the fixed track_max_bearing_jump_rad check; the fixed check remains
    # the fallback when no capture-time attitude is available (cold buffer / gap-guard miss) and
    # the entire path is skipped when OFF (VQ1 byte-identical). The range-jump check is unchanged.
    use_imu_bearing_gate: bool = False
    # Sensor-noise floor of the deviation (rad): PnP bearing noise 1-2 deg + AHRS attitude-delta
    # error margin => ~3.4 deg. Honest per-frame world-bearing change at zero translation is ~0.
    bearing_gate_noise_rad: float = 0.06
    # Translation bound (m/s) for the parallax allowance: the slow-lap never exceeds ~3 m/s
    # commanded build-up; 4.0 keeps margin so an honest close-range crossing sweep passes.
    bearing_gate_trans_mps: float = 4.0
    # Clamp the range used in the parallax term (m): 1/range blows up at point-blank. NOTE the
    # range here is the track's EMA prediction; close-range PnP OVER-reports range on this wire
    # (spec §1.3), which UNDER-sizes the allowance -- the strict direction (rejects route to the
    # bridge-covered coast, recoverable; a missed hop is not).
    bearing_gate_min_range_m: float = 1.0
    # --- FIX 4 (defense-in-depth): rate-limit the A30 image-servo lateral demand (m/s^3) so even
    # an ACCEPTED noisy bearing can't snap the roll to the rail in one tick (the t=3.25 hop put
    # alat 0 -> -1.5 in ONE tick). 6.0 => a full-scale reversal (-1.5 -> +1.5) takes 0.5 s; an
    # honest az ramp (<=0.7 rad/s sweep x k_az=8 = 5.6 m/s^3 worst) is never limited, a one-frame
    # rail-snap is. 0.0 => off (legacy snap, byte-identical).
    image_lat_slew_mps3: float = 0.0

    # ===================================================================
    # A32 — SOFT-WEIGHTED VISION FUSION (2026-07-03; operator directive "don't throw away
    # measurements that don't match -- take every one in and WEIGHT it"; spec
    # handoff/vq2_a32_robust_estimation_spec_2026-07-03.md §3.2). The A31 IMU-consistency
    # bearing gate above is the RIGHT physics with the WRONG consequence: binary reject. Run
    # 20260703_172104 measured 59.9% of evaluated frames hard-rejected, median rejected frame
    # only 1.87x over the allowance -- honest information thrown away wholesale, the seeker
    # starved (continuity_reject -> coast -> track drop -> re-acquisition churn), and an
    # accel-starved AHRS attitude error masqueraded as vision inconsistency that the gate then
    # AMPLIFIED. When ON, the dev/allow computation stays EXACTLY as-is but the consequence
    # becomes a continuous Cauchy weight
    #     nu = dev/allow; nu_r = |range jump|/track_max_range_jump_m; w = 1/(1 + nu^2 + nu_r^2)
    # scaling the track EMA update (alpha_eff = track_ema_alpha * w) and the downstream demand
    # steps (the image-servo az term; the z_off latch weight) -- EVERY frame contributes
    # (w(1)=0.5, w(1.87)~0.22, w(4)~0.06, never 0), starvation is impossible by construction,
    # and hard consequences (coast/track-drop) fire only on PERSISTENCE (w below the coast
    # threshold for track_max_coast_ticks). Requires use_imu_bearing_gate machinery for the
    # dev/allow physics; when the IMU gate is not live for a frame (no capture-time attitude /
    # no reference) the legacy fixed checks run unchanged. OFF (default) = the A31 binary path,
    # byte-identical (VQ1 / case-A untouched).
    use_soft_bearing_weight: bool = False
    # Tick the coast/track-drop counter only when the accepted frame's weight is below this
    # (nu ~ 3): a track whose EVERY frame is grossly inconsistent for track_max_coast_ticks is a
    # real track loss, not noise.
    bearing_w_coast_thresh: float = 0.1

    # ===================================================================
    # A33 — GATE-2 COORDINATED INTERCEPT (2026-07-03; spec
    # handoff/vq2_a33_gate2_intercept_spec_2026-07-03.md). All defaults legacy/off
    # => VQ1 / case-A byte-identical; activated only via vq2_case_c seeker_overrides.
    # ===================================================================
    # --- H-1(a): GEOMETRIC OLD-GATE EXCLUSION. The RACE_STATUS index leads the physical gate plane
    # by ~3 m on this wire (run 20260703_210632: index advanced at cmd 49 while the tracked gate was
    # still ~3 m ahead, dead-center). The A31 time-windows (pass_wire_coast_s / pass_coast_s) existed
    # only to keep the acquire-next re-lock from grabbing the gate being passed; replace that TIME
    # discrimination with a GEOMETRIC one so the windows can collapse (H-1b). While _passing and this
    # flag is ON, a candidate whose world direction lies within ``pass_prev_gate_excl_rad`` of the
    # passed gate's snapshot direction AND is no farther than snapshot-range + ``..._margin_m`` is
    # EXCLUDED from acquisition (it is the gate we just passed). A gate well off the pass heading (the
    # next gate, an ~85 deg turn away here) passes at any range; a straight-section next gate dead
    # ahead passes on the range margin. OFF (default) => no exclusion (byte-identical). Requires the
    # A30/A31 capture-time attitude ring (already on under vq2_case_c) for the world-direction rotate.
    pass_exclude_prev_gate: bool = False
    pass_prev_gate_excl_rad: float = 0.35      # world-direction cone half-angle around the pass heading
    pass_prev_gate_excl_margin_m: float = 3.0  # range slack beyond the snapshot tracked range
    # --- H-1(c): TURN-THROUGH-OCCLUSION. At the pass the gate-1 frame occludes much of the camera,
    # so a gate-2 pose may be absent for a beat -- the turn must NOT wait for one (operator amendment
    # 1). When ON, ``_begin_pass`` latches a BLIND TURN TARGET (``_pass_turn_yaw``) and
    # ``_pass_coast_command`` SLEWS toward it (at ``pursuit_yaw_slew_rps``) instead of freezing the
    # pre-pass heading, so the turn starts on the pass TRIGGER and rides blind through the occlusion;
    # gate-2 poses refine (not start) it once they clear. The target is the last pre-pass next-gate
    # world bearing when known (clamped to +/-``pass_blind_turn_cap_rad`` from the pass heading), else
    # the pass heading + sign(last az/chase drift) * the cap. OFF (default) => ``_pass_coast_command``
    # freezes the heading exactly as today (byte-identical).
    pass_turn_through: bool = False
    pass_blind_turn_cap_rad: float = 1.6   # max blind heading change during the coast (~92 deg)
    # --- S-1: sharpen the off-axis forward cut. fwd_scale = max(cos(az),0)^fwd_scale_pow. cos^2
    # (default) still drives 55% forward at az=42 deg while badly mis-pointed; cos^4 gives 30% --
    # cuts the overfly speed (which also drives the translational-lift up-bias) without touching
    # near-centered pace (az 0.2 rad: 0.92 vs 0.96). 2.0 (default) == today's cos^2 (byte-identical).
    fwd_scale_pow: float = 2.0
    # --- H-3: HARD RANGE-JUMP REJECT on the A32 soft track path. A candidate 8+ m off the track's
    # range prediction is a DIFFERENT PHYSICAL OBJECT (a gate cannot move 8 m between frames), so
    # Cauchy-weighting it is a category error -- it smeared the track gate-1 -> gate-2 over ~10 frames
    # (run 20260703_210632, cmd 60-72). Reinstate the RANGE leg of the continuity check as a HARD
    # candidate filter on the soft path (the BEARING leg stays soft -- that was the A31 starvation
    # source). OFF (default) => the A32 soft path takes all poses (byte-identical). Only meaningful
    # under ``use_soft_bearing_weight``.
    soft_range_hard_reject: bool = False


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
    # -- A33 H-1(a): old-gate exclusion snapshot (world dir + range of the gate being passed) --
    _pass_prev_dir_world: np.ndarray | None = field(default=None, repr=False)
    _pass_prev_range_m: float | None = field(default=None, repr=False)
    # -- A33 H-1(c): blind turn target latched at pass commit (turn-through-occlusion) --
    _pass_turn_yaw: float | None = field(default=None, repr=False)
    # -- hold-last-demand bridge (A13): cache the last good pursuit demand so a pose-None tick can
    #    re-issue it (continuous per-tick command) instead of regime-2's zero-coast hold --
    _last_demand_los: np.ndarray | None = field(default=None, repr=False)   # last pursuit world heading unit vec
    _last_demand_yaw: float | None = field(default=None, repr=False)        # last pursuit slewed yaw
    _last_demand_accel: float | None = field(default=None, repr=False)      # last pursuit effective forward accel
    _last_demand_t_ns: int | None = field(default=None, repr=False)         # sim time of the last pursuit demand
    # -- A29 LOS-rate damping state: the filtered world LOS angular rate of the TRACKED gate.
    #    Sampled by differencing yaw_des (pre-slew) across FRESH poses on their CAMERA-EPOCH stamps
    #    (same clock both sides -> the epoch-rate skew cancels). Describes ONE gate's geometry:
    #    reset at pass begin/end, track drop, and reset(). --
    _los_prev_angle: float | None = field(default=None, repr=False)   # last fresh-pose world LOS angle
    _los_prev_pose_ns: int | None = field(default=None, repr=False)   # its pose.sim_time_ns (camera epoch)
    _los_rate_ema: float = field(default=0.0, repr=False)             # filtered LOS rate (rad/s, world)
    _los_rate_valid: bool = field(default=False, repr=False)          # >=1 accepted sample since reset
    # -- A29 instrumentation (spec §4.4): stashed each flag-ON pursuit tick so the nav_estimate
    #    logger can read them (same stale-retention semantics as yaw_des_rad). Never consumed by
    #    control; None when the flag is off / no valid estimate yet. --
    _last_los_rate: float | None = field(default=None, repr=False)          # = _los_rate_ema when valid
    _last_vt_est: float | None = field(default=None, repr=False)            # v_t = -r*theta_dot (m/s)
    _last_alat: float | None = field(default=None, repr=False)              # applied lateral accel (m/s^2)
    _last_track_range_m: float | None = field(default=None, repr=False)     # range used in v_t (m)
    # -- A30 image-servo state: capture-time attitude ring buffer (a deque of
    #    (sim_time_ns, (roll, pitch, yaw)), created lazily on the first flag-ON tick so the OFF
    #    path allocates nothing) + instrumentation stashes (spec §4; logging only, None when the
    #    flag is off / no pursuit tick yet). _last_alat above is REUSED for the applied image-servo
    #    lateral when the flag is on. --
    _att_hist: object | None = field(default=None, repr=False)              # deque[(t_ns, rpy)]
    _last_az_err: float | None = field(default=None, repr=False)            # apparent azimuth az (rad, +right)
    _last_fwd_scale: float | None = field(default=None, repr=False)         # cos^2(az) forward-pointing scale
    # -- A31 state (all inert on the flag-off paths; see the A31 config block) --
    _pass_wire: bool = field(default=False, repr=False)            # pass committed/confirmed by the WIRE
    _fwd_ramp_t_ns: int | None = field(default=None, repr=False)   # forward re-ramp clock (reramp_forward_after_pass)
    # chase / orbit-breaker: cumulative unwrapped LOS rotation since (re)acquisition of the gate.
    _chase_psi0: float | None = field(default=None, repr=False)    # yaw_des baseline at acquisition
    _chase_yaw0: float | None = field(default=None, repr=False)    # slewed yaw at acquisition (the clamp anchor)
    _chase_dpsi: float = field(default=0.0, repr=False)            # accumulated shortest-path yaw_des delta
    _chase_prev_psi: float | None = field(default=None, repr=False)     # last fresh-pose yaw_des
    _chase_prev_pose_ns: int | None = field(default=None, repr=False)   # its camera-epoch stamp (dedupe)
    _chase_rate: float | None = field(default=None, repr=False)    # per-fresh-pose LOS drift (rad/s, break exit)
    _orbit_break_t_ns: int | None = field(default=None, repr=False)     # break regime start (None = not braking)
    _orbit_trips: int = field(default=0, repr=False)               # guard trips on the current acquisition
    # A30 lateral-demand slew state (image_lat_slew_mps3 > 0 only).
    _alat_slew_prev: float | None = field(default=None, repr=False)
    _alat_slew_t_ns: int | None = field(default=None, repr=False)
    # IMU-consistency bearing gate: the last ACCEPTED tracked pose's world direction (rotated with
    # the attitude AT ITS CAPTURE TIME) + its camera-epoch stamp -- the static-gate prediction.
    _bg_prev_dir_world: np.ndarray | None = field(default=None, repr=False)
    _bg_prev_pose_ns: int | None = field(default=None, repr=False)
    # A31 instrumentation stashes (logging only, never consumed by control).
    _last_chase_dpsi: float | None = field(default=None, repr=False)
    _last_bearing_dev_rad: float | None = field(default=None, repr=False)
    _last_bearing_allow_rad: float | None = field(default=None, repr=False)
    # A32 soft bearing weight of the LAST accepted (returned) pose: consumed by the image-servo
    # az term + passed into the z_off latch, and logged as ``bearing_w``. None when the soft
    # path is off / no pose yet; 1.0 on a soft-path pose the IMU gate could not evaluate.
    _last_bearing_w: float | None = field(default=None, repr=False)
    # -- A29 regime stash (spec §4.4): which command_visual regime returned this tick, first-class
    #    (settle/anchor/egress/pass/bridge/hold/pursuit) -- the A29 run's regime had to be
    #    reconstructed from field-change fingerprints. Logging only. --
    _last_regime: str | None = field(default=None, repr=False)
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

        # A33 H-1(a): while passing, EXCLUDE the just-passed gate GEOMETRICALLY (its snapshot world
        # direction + range) so the acquire-next re-lock (or the turn-through acquisition) cannot grab
        # the gate we are threading. Flag-off / not passing / no snapshot => no-op (byte-identical).
        if (self.config.pass_exclude_prev_gate and self._passing
                and self._pass_prev_dir_world is not None):
            poses = [p for p in poses if not self._is_prev_gate(p)]

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
                self._reset_los_rate()   # A29: tracked-gate identity gone -> LOS-rate history with it
                self._reset_bearing_gate()   # A31: the prediction reference dies with the track
            return None

        soft_w: float | None = None      # A32: the chosen candidate's Cauchy weight (soft path only)
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

            # A31 IMU-CONSISTENCY BEARING GATE (see the config block): when live, the bearing leg
            # of the continuity check becomes a MOTION-CONSISTENCY test against the static-gate
            # prediction instead of the fixed frame-to-frame threshold. All candidates in a frame
            # share the capture stamp, so one capture-time attitude lookup serves them all. Falls
            # back to the legacy fixed check when no capture-time attitude / no reference exists.
            rpy_cap = None
            imu_gate_live = False
            if self.config.use_imu_bearing_gate:
                rpy_cap = self._rpy_at(int(poses[0].sim_time_ns))
                imu_gate_live = (rpy_cap is not None
                                 and self._bg_prev_dir_world is not None
                                 and self._bg_prev_pose_ns is not None)

            # A32 SOFT PATH (use_soft_bearing_weight + the IMU gate live): consistency becomes a
            # continuous Cauchy WEIGHT on the chosen candidate, not a filter -- every frame
            # contributes, selection is unchanged (rank by closeness to the prediction), and the
            # coast/track-drop counter ticks only on w < bearing_w_coast_thresh (persistence).
            # The binary reject below is REMOVED on this path (the A31 starvation source); it
            # remains bit-identical when the flag is off or the gate is not live for this frame.
            soft = self.config.use_soft_bearing_weight and imu_gate_live
            if soft:
                # A33 H-3: keep the BEARING leg soft (the A31 starvation source) but reinstate the
                # RANGE leg as a HARD candidate filter -- an 8+ m range jump is a DIFFERENT gate, not
                # a noisy same-gate measurement (it smeared the track gate-1 -> gate-2 over ~10 frames
                # on run 20260703_210632). Flag-off => cands = poses, byte-identical A32 soft path.
                if self.config.soft_range_hard_reject:
                    cands = [p for p in poses
                             if abs(p.range_m - pred_r) <= self.config.track_max_range_jump_m]
                    if not cands:
                        # every candidate jumped in RANGE -> coast on the track (same bookkeeping as
                        # the binary continuity_reject branch below).
                        self._last_none_reason = "continuity_reject"
                        self._track_coast_ticks += 1
                        if self._track_coast_ticks > max(1, int(self.config.track_max_coast_ticks)):
                            self._track_range_m, self._track_bearing = None, None
                            self._reset_los_rate()
                            self._reset_bearing_gate()
                        return None
                else:
                    cands = poses
            else:
                def _consistent(p: GatePose) -> bool:
                    if abs(p.range_m - pred_r) > self.config.track_max_range_jump_m:
                        return False
                    if imu_gate_live:
                        return self._imu_bearing_consistent(p, rpy_cap, pred_r)
                    return (float(np.linalg.norm(self._pose_bearing(p) - pred_b))
                            <= self.config.track_max_bearing_jump_rad)

                cands = [p for p in poses if _consistent(p)]
                if not cands:
                    # every candidate jumped -> COAST on the track (do not lock onto a flapper).
                    self._last_none_reason = "continuity_reject"   # the dominant A13 pose=None source
                    self._track_coast_ticks += 1
                    if self._track_coast_ticks > max(1, int(self.config.track_max_coast_ticks)):
                        self._track_range_m, self._track_bearing = None, None
                        self._reset_los_rate()   # A29: tracked-gate identity gone -> LOS-rate history with it
                        self._reset_bearing_gate()   # A31: the prediction reference dies with the track
                    return None
            # among the consistent candidates, the one closest to the predicted bearing+range.
            chosen = min(
                cands,
                key=lambda p: float(np.linalg.norm(self._pose_bearing(p) - pred_b))
                + abs(p.range_m - pred_r) / max(self.config.track_max_range_jump_m, 1e-6),
            )
            if soft:
                dev, allow = self._imu_bearing_dev(chosen, rpy_cap, pred_r)
                nu = dev / max(allow, 1e-9)
                nu_r = (abs(float(chosen.range_m) - pred_r)
                        / max(self.config.track_max_range_jump_m, 1e-6))
                soft_w = 1.0 / (1.0 + nu * nu + nu_r * nu_r)   # Cauchy: w(0)=1, never 0

        # A32 soft-path persistence bookkeeping (BEFORE the EMA so a drop tick never half-updates
        # the track): a grossly-inconsistent frame (w < thresh) still nudges the track by its tiny
        # weight, but ticks the coast counter; track_max_coast_ticks of them = a REAL track loss.
        if soft_w is not None and soft_w < self.config.bearing_w_coast_thresh:
            self._track_coast_ticks += 1
            if self._track_coast_ticks > max(1, int(self.config.track_max_coast_ticks)):
                self._track_range_m, self._track_bearing = None, None
                self._reset_los_rate()
                self._reset_bearing_gate()
                self._last_none_reason = "continuity_reject"   # a real loss, same reason taxonomy
                return None
        # accept -> update the smoothed track (EMA scaled by the soft weight) and the coast counter.
        w_acc = 1.0 if soft_w is None else float(soft_w)
        a = float(np.clip(self.config.track_ema_alpha, 0.0, 1.0)) * w_acc
        b_meas = self._pose_bearing(chosen)
        if self._track_range_m is None or self._track_bearing is None:
            self._track_range_m, self._track_bearing = float(chosen.range_m), b_meas
        else:
            self._track_range_m = (1.0 - a) * float(self._track_range_m) + a * float(chosen.range_m)
            self._track_bearing = (1.0 - a) * np.asarray(self._track_bearing, dtype=np.float64) + a * b_meas
        if soft_w is None or soft_w >= self.config.bearing_w_coast_thresh:
            self._track_coast_ticks = 0
        self._last_none_reason = None    # a usable pose this tick (clear the stale reason)
        # A32: stash the weight for the downstream consumers (image-servo az term, z_off latch,
        # the ``bearing_w`` log key). 1.0 when the soft flag is on but this frame had no live IMU
        # gate (first acquisition / cold buffer -- nothing to weigh against); None when flag off.
        self._last_bearing_w = (w_acc if self.config.use_soft_bearing_weight else None)
        # A31: refresh the IMU-consistency reference on EVERY accepted pose (first acquisition
        # seeds it). The world direction is rotated with the attitude AT THE CAPTURE INSTANT so
        # the next frame's prediction carries the measured rotation exactly; no capture-time
        # attitude available => no reference => the next frame falls back to the legacy check.
        # A32 soft path: a low-weight frame (w < coast thresh) does NOT re-anchor the reference --
        # the prediction stays on the last trusted direction and the allowance keeps widening with
        # dt, so a garbage stretch cannot drag the prediction with it.
        if self.config.use_imu_bearing_gate:
            if soft_w is not None and soft_w < self.config.bearing_w_coast_thresh:
                return chosen
            rpy_acc = self._rpy_at(int(chosen.sim_time_ns))
            if rpy_acc is not None:
                self._bg_prev_dir_world = self._gate_dir_world_rpy(chosen, rpy_acc)
                self._bg_prev_pose_ns = int(chosen.sim_time_ns)
            else:
                self._reset_bearing_gate()
        return chosen

    def _imu_bearing_dev(self, p: GatePose, rpy_cap: tuple[float, float, float],
                         pred_r: float) -> tuple[float, float]:
        """A31/A32 shared physics: the candidate's motion-consistency ``(deviation, allowance)``.

        Predicted bearing = the last accepted pose's WORLD direction (a static gate's world
        bearing is quasi-constant; the capture-time attitude rotation has already compensated the
        measured rotation between the frames -- the IMU prediction). Deviation = the angle between
        the candidate's world direction (rotated with the attitude at ITS capture instant,
        ``rpy_cap``) and that prediction. Allowance = the sensor-noise floor + the translation-
        parallax bound ``trans_mps * dt / range`` (dt on the CAMERA-epoch stamps, same clock both
        sides so the A29 epoch-rate skew cancels; the allowance widens ∝ dt so it predicts THROUGH
        short pose gaps, and ∝ 1/range so an honest close-range sweep passes). Consumed as a
        binary test by :meth:`_imu_bearing_consistent` (A31) and as the normalized innovation of
        the Cauchy soft weight (A32). Writes the dev/allow instrumentation stashes."""
        d_world = self._gate_dir_world_rpy(p, rpy_cap)
        prev = np.asarray(self._bg_prev_dir_world, dtype=np.float64)
        dev = float(np.arccos(np.clip(float(d_world @ prev), -1.0, 1.0)))
        dt = max((int(p.sim_time_ns) - int(self._bg_prev_pose_ns)) / 1e9, 0.0)
        r = max(float(pred_r), float(self.config.bearing_gate_min_range_m))
        allow = (float(self.config.bearing_gate_noise_rad)
                 + float(self.config.bearing_gate_trans_mps) * dt / r)
        self._last_bearing_dev_rad = dev            # A31 instrumentation (logging only)
        self._last_bearing_allow_rad = allow
        return dev, allow

    def _imu_bearing_consistent(self, p: GatePose, rpy_cap: tuple[float, float, float],
                                pred_r: float) -> bool:
        """A31: is candidate ``p``'s bearing consistent with the drone's OWN measured motion since
        the last accepted pose? (See :meth:`_imu_bearing_dev` for the physics.) Rejects REGARDLESS
        of the deviation's absolute size -- no fixed threshold to slide under. The BINARY
        consequence this implements is replaced by the A32 Cauchy weight on the vq2_case_c path
        (``use_soft_bearing_weight``); this method remains the flag-off/VQ1 behaviour."""
        dev, allow = self._imu_bearing_dev(p, rpy_cap, pred_r)
        return dev <= allow

    def _bearing_w_ctl(self) -> float:
        """A32: the soft bearing weight the DOWNSTREAM demand steps consume for the current pose
        (the stash rides ZOH re-feeds of the same capture, so a bridged/re-fed tick keeps its
        frame's weight). 1.0 whenever the soft path is off / has not evaluated a pose yet --
        every flag-off consumer is byte-identical."""
        if not self.config.use_soft_bearing_weight or self._last_bearing_w is None:
            return 1.0
        return float(self._last_bearing_w)

    def _reset_bearing_gate(self) -> None:
        """Drop the A31 bearing-gate reference. The reference describes ONE tracked gate, so it is
        dropped wherever the track identity can change: track drop, pass begin, acquire-next
        reset, and :meth:`reset`. The next accepted pose re-seeds it."""
        self._bg_prev_dir_world = None
        self._bg_prev_pose_ns = None

    def _is_prev_gate(self, p: GatePose) -> bool:
        """A33 H-1(a): is candidate ``p`` the gate we are currently passing through? True when its
        world direction lies within ``pass_prev_gate_excl_rad`` of the passed gate's snapshot
        direction AND its range is no farther than the snapshot range + ``pass_prev_gate_excl_margin_m``.
        Uses the attitude AT THE POSE CAPTURE INSTANT (the A30/A31 ring buffer, ``_rpy_at``) to rotate
        the camera lever into world NED; falls back to the pass-heading direction (a level frame at the
        frozen heading) when no capture-time attitude is available. Only ever called while ``_passing``
        with a snapshot present (the caller guards both)."""
        rpy = self._rpy_at(int(p.sim_time_ns))
        if rpy is None:
            yaw0 = self._pass_heading if self._pass_heading is not None else 0.0
            rpy = (0.0, 0.0, float(yaw0))
        d = self._gate_dir_world_rpy(p, rpy)
        prev = np.asarray(self._pass_prev_dir_world, dtype=np.float64)
        ang = float(np.arccos(np.clip(float(d @ prev), -1.0, 1.0)))
        rng_cap = (float(self._pass_prev_range_m) if self._pass_prev_range_m is not None
                   else float(self.config.pass_arm_range_m))
        return (ang < float(self.config.pass_prev_gate_excl_rad)
                and float(p.range_m) <= rng_cap + float(self.config.pass_prev_gate_excl_margin_m))

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
        # A30: record this tick's attitude into the capture-time ring buffer (one entry per
        # command_visual tick), so a later pursuit tick can interpret an AGED pose's camera lever
        # with the attitude AT THE MOMENT THE FRAME WAS CAPTURED (kills the omega*age false
        # lateral, spec §3.3). A31: the IMU-consistency bearing gate consumes the SAME buffer
        # (its rotation prediction is the capture-time attitude delta), so it also gates the
        # append. Flag-gated append -> ZERO side effects on the OFF path.
        if self.config.use_image_servo_lateral or self.config.use_imu_bearing_gate:
            self._append_att_hist(int(nav.sim_time_ns), self._att_rpy(nav))

        # ACQUIRE-NEXT track reset: once we are past the dead-reckon GLIDE (the pass_coast_s window) the
        # just-passed gate is behind us; the temporal track may still be stale-locked on it (a fresh
        # downrange gate would read as a big range JUMP and be rejected as a flapper, blocking the
        # handoff). While in the acquire-next window we CLEAR the track each tick so first-acquisition
        # re-locks the NEXT gate cleanly. (Inside the glide window we leave it alone -- we are coasting
        # straight and deliberately not steering on any gate.)
        if (self.config.use_pass_dead_reckon and self._passing
                and self._pass_t_ns is not None
                and (int(nav.sim_time_ns) - self._pass_t_ns) / 1e9 >= self._effective_pass_coast_s()):
            self._track_range_m, self._track_bearing, self._track_coast_ticks = None, None, 0
            self._reset_bearing_gate()   # A31: the prediction reference dies with the track

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
            self._last_regime = "settle"        # A29 §4.4 instrumentation (logging only)
            return self._hold_command(nav, yaw_rate_cap=self.config.anchor_yaw_rate_rps,
                                      attitude_safe=True)

        # --- regime 1: LAUNCH ANCHOR (settled, no release yet) -> hold attitude, clamp ALL rates ---
        if not self._anchored:
            self._last_regime = "anchor"        # A29 §4.4 instrumentation (logging only)
            return self._hold_command(nav, yaw_rate_cap=self.config.anchor_yaw_rate_rps,
                                      attitude_safe=True)

        # --- regime 1.5: SPAWN-GATE EGRESS (just released, drone still inside gate 0) -> a brief,
        # CAPPED forward creep along the FROZEN spawn heading (the direction OUT of the start gate),
        # to clear the start-gate structure BEFORE re-aiming at the downrange gate. The forward
        # demand is a bounded feedforward tilt (NOT a velocity setpoint), pitch-rate capped, so it
        # cannot lunge -- it eases the drone out of the spawn gate. (A4: the first forward motion
        # drove straight into the start-gate frame; the egress departs the spawn gate first.) ---
        if self._in_egress(int(nav.sim_time_ns)):
            self._last_regime = "egress"        # A29 §4.4 instrumentation (logging only)
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
                    self._end_pass(int(nav.sim_time_ns))   # next gate re-acquired -> pursuit below
                else:
                    # A29 §4.4 instrumentation, A31-suffixed by the commit source (logging only):
                    # "pass_wire" = committed/confirmed by the RACE_STATUS index (fast window
                    # eligible), "pass_vis" = vision-committed (degenerate/lost-after-arm).
                    self._last_regime = "pass_wire" if self._pass_wire else "pass_vis"
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
                self._last_regime = "bridge"        # A29 §4.4 instrumentation (logging only)
                return bridged
            self.diag_counts["held_legacy"] += 1
            self._last_regime = "hold"              # A29 §4.4 instrumentation (logging only)
            return self._hold_command(nav, yaw_rate_cap=self.config.reacquire_yaw_rate_rps,
                                      attitude_safe=True)

        # --- regime 3: PURSUIT -> bounded feedforward forward tilt toward the SEEN opening +
        # centering yaw, pitch + roll capped, forward demand ramped (never a velocity setpoint) ---
        self.diag_counts["pursuit"] += 1
        self._last_regime = "pursuit"               # A29 §4.4 instrumentation (logging only)
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
            # A31 UPGRADE SEAM: a vision-committed pass (degenerate/lost-after-arm) that the wire
            # then confirms MID-GLIDE flips to the fast acquire window (spec §2.1). State write
            # only -- never read unless pass_wire_coast_s is configured (VQ1 byte-identical).
            self._pass_wire = self._pass_wire or bool(index_advanced)
            return                                  # already committed -> nothing more to arm
        # COMMIT triggers (only meaningful once anchored + past egress):
        degenerate_close = (self._pass_armed and pose is not None
                            and float(pose.range_m) <= self.config.pass_degenerate_range_m)
        lost_after_arm = self._pass_armed and pose is None
        if degenerate_close or lost_after_arm or index_advanced:
            self._begin_pass(sim_time_ns, wire=index_advanced)

    def _begin_pass(self, sim_time_ns: int, wire: bool = False) -> None:
        """Commit to the dead-reckon-through-pass coast: FREEZE the current pursuit heading (the
        straight-through direction) and start the coast clock. Drops the temporal gate track so that,
        once the coast ends, ACQUIRE-NEXT re-runs first-acquisition on the NEXT gate. ``wire``
        (A31): the commit came from the AUTHORITATIVE RACE_STATUS index increment -- the drone is
        past the gate plane, so the fast acquire window applies (when configured)."""
        self._passing = True
        self._pass_wire = bool(wire)
        self._pass_t_ns = int(sim_time_ns)
        self._pass_heading = self._last_yaw if self._last_yaw is not None else 0.0
        # A33 H-1(a): snapshot the passed gate's world direction + range BEFORE _reset_bearing_gate
        # nulls the reference, so _is_prev_gate can exclude it from re-acquisition.
        if self.config.pass_exclude_prev_gate:
            if self._bg_prev_dir_world is not None:
                self._pass_prev_dir_world = np.asarray(self._bg_prev_dir_world, dtype=np.float64).copy()
            else:
                yaw0 = self._pass_heading
                self._pass_prev_dir_world = np.array([np.cos(yaw0), np.sin(yaw0), 0.0])
            self._pass_prev_range_m = (float(self._track_range_m)
                                       if self._track_range_m is not None
                                       else float(self.config.pass_arm_range_m))
        # A33 H-1(c): latch the BLIND TURN TARGET so the coast slews toward the next gate through the
        # occlusion instead of freezing the heading. Aim at the last-known next-gate drift direction:
        # sign(last apparent azimuth az, else the chase LOS drift), the pass heading + sign*cap. This
        # runs BEFORE _reset_chase drops _last_az_err's companions -- read the stashes here.
        if self.config.pass_turn_through:
            drift = 0.0
            if self._last_az_err is not None and abs(float(self._last_az_err)) > 1e-3:
                drift = float(self._last_az_err)
            elif self._chase_rate is not None and abs(float(self._chase_rate)) > 1e-6:
                drift = float(self._chase_rate)
            cap = abs(float(self.config.pass_blind_turn_cap_rad))
            sgn = float(np.sign(drift)) if drift != 0.0 else 0.0
            turn = float(np.arctan2(np.sin(self._pass_heading + sgn * cap),
                                    np.cos(self._pass_heading + sgn * cap)))
            self._pass_turn_yaw = turn
        # reset the temporal track so the next-gate re-acquisition starts clean (a different gate).
        self._track_range_m, self._track_bearing, self._track_coast_ticks = None, None, 0
        self._reset_los_rate()   # A29: the LOS-rate state describes the JUST-PASSED gate -> drop it
        self._reset_bearing_gate()   # A31: the prediction reference describes the passed gate
        self._reset_chase()          # A31: the chase/orbit integrator describes the passed gate
        self._alat_slew_prev, self._alat_slew_t_ns = None, None   # A31: fresh lateral-slew history

    def _end_pass(self, sim_time_ns: int | None = None) -> None:
        """End the pass regime (the NEXT gate has been re-acquired) -> resume normal pursuit on it.
        Re-arms the pass bookkeeping for the next gate. ``sim_time_ns`` (A31): when
        ``reramp_forward_after_pass`` is on, restart the forward-accel ramp clock here so the
        forward drive re-ramps from zero into the fresh acquisition (point before pushing)."""
        self._passing = False
        self._pass_wire = False
        self._pass_armed = False
        self._pass_min_range_m = float("inf")
        self._pass_t_ns = None
        self._pass_heading = None
        self._pass_index = self._last_index
        # A33 H-1: the exclusion snapshot + blind turn target describe the JUST-PASSED gate -> drop.
        self._pass_prev_dir_world = None
        self._pass_prev_range_m = None
        self._pass_turn_yaw = None
        self._reset_los_rate()   # A29: a NEW gate begins here -- seed its LOS-rate filter fresh
        self._reset_chase()      # A31: a NEW acquisition -- fresh chase baseline (+ trips)
        if self.config.reramp_forward_after_pass and sim_time_ns is not None:
            self._fwd_ramp_t_ns = int(sim_time_ns)

    def _effective_pass_coast_s(self) -> float:
        """A31: the acquire-eligibility window for the CURRENT pass. The fast wire window applies
        only when the pass was committed/confirmed by the AUTHORITATIVE wire signal AND
        ``pass_wire_coast_s`` is configured; otherwise the legacy ``pass_coast_s`` (None default
        => byte-identical everywhere, including for a wire-committed pass)."""
        if self._pass_wire and self.config.pass_wire_coast_s is not None:
            return float(self.config.pass_wire_coast_s)
        return float(self.config.pass_coast_s)

    def _pass_acquired_next(self, nav: NavState, pose: GatePose) -> bool:
        """True iff the seeker is in the ACQUIRE-NEXT window (past the dead-reckon coast) AND a fresh
        NEXT gate has been re-acquired -- i.e. a usable pose that is NOT the just-passed gate (a real
        downrange range, past the degenerate band). During the initial dead-reckon coast (within
        ``pass_coast_s``) we IGNORE any pose (it is the gate we are passing through) and keep gliding;
        only after that window do we accept a re-acquired gate as the next one to pursue."""
        if self._pass_t_ns is None or pose is None:
            return False
        elapsed = (int(nav.sim_time_ns) - self._pass_t_ns) / 1e9
        if elapsed < self._effective_pass_coast_s():   # A31: 0.25 s on a wire-committed pass
            return False                            # still gliding through the opening -> not yet
        # past the dead-reckon coast: a sighting at a real downrange range = the next gate re-acquired.
        # (range_m > pass_degenerate_range_m keeps the just-passed gate out even on the fast
        # window: behind the image plane it is rejected upstream, and point-blank it is degenerate.)
        return float(pose.range_m) > self.config.pass_degenerate_range_m

    def _in_pass_dead_reckon(self, sim_time_ns: int) -> bool:
        """True while the committed pass coast (dead-reckon + acquire-next) window is active. Bounded
        by ``pass_coast_s + acquire_next_s`` so the straight glide can never run forever; past that the
        seeker reverts to the gentle no-detection coast (never a pitch-up)."""
        if not self._passing or self._pass_t_ns is None:
            return False
        elapsed = (int(sim_time_ns) - self._pass_t_ns) / 1e9
        return elapsed < (self._effective_pass_coast_s() + self.config.acquire_next_s)

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
            self._end_pass(int(nav.sim_time_ns))
            return self._hold_command(nav, yaw_rate_cap=self.config.reacquire_yaw_rate_rps,
                                      attitude_safe=True)
        yaw0 = self._pass_heading if self._pass_heading is not None else self._att_yaw(nav)
        # A33 H-1(c): TURN-THROUGH-OCCLUSION. When a blind turn target is latched, SLEW the coast
        # heading toward it (at pursuit_yaw_slew_rps, the same turn-rate cap pursuit uses) instead of
        # freezing yaw -- the turn begins on the pass trigger and rides blind through the occlusion;
        # gate-2 poses refine it once they clear (_pass_acquired_next -> _end_pass -> pursuit). Flag
        # off / no target => freeze the heading exactly as today (byte-identical).
        if self.config.pass_turn_through and self._pass_turn_yaw is not None:
            yaw0 = self._slew_heading(float(self._pass_turn_yaw), int(nav.sim_time_ns))
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
        return self._gate_dir_world_rpy(pose, self._att_rpy(nav))

    def _gate_dir_world_rpy(self, pose: GatePose, rpy: tuple[float, float, float]) -> np.ndarray:
        """:meth:`_gate_dir_world` with an EXPLICIT (roll, pitch, yaw) — the A30 seam that lets the
        image servo rotate the camera lever with the attitude AT CAPTURE TIME (:meth:`_rpy_at`)
        instead of the current-tick attitude (which injects a spurious +omega*age into the world
        azimuth of an aged pose). Same math, same order of operations — calling it with
        ``self._att_rpy(nav)`` is bit-identical to the pre-A30 ``_gate_dir_world`` body."""
        d_cam = _unit(np.asarray(pose.t_cam_gate, dtype=np.float64))
        d_body = R_camera_from_body().T @ d_cam
        tr, tp, ty = rpy
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
        # A32 (spec §3.2): thread the frame's soft bearing weight into the latch -- a bearing-
        # inconsistent frame's z_off measurement is suspect for the same reason. 1.0 (no-op)
        # whenever the soft path is off.
        vert_est.latch_offset(offset_z_world, obs_age_s, weight=self._bearing_w_ctl())
        self._last_latched_pose_ns = int(pose.sim_time_ns)

    def _vertical_align_ramp(self, sim_time_ns: int) -> float:
        """Vertical-align authority ramp [0,1] over ``vertical_align_ramp_s`` from the anchor release, so
        the first pursuit ticks don't step to a full descent command (mirrors the forward-demand ramp)."""
        if self.config.vertical_align_ramp_s <= 0.0 or self._release_t_ns is None:
            return 1.0
        elapsed = (int(sim_time_ns) - self._release_t_ns) / 1e9
        return float(np.clip(elapsed / self.config.vertical_align_ramp_s, 0.0, 1.0))

    # =======================================================================
    # LOS-RATE (TANGENTIAL-VELOCITY) DAMPING  (the A29 orbit fix)
    # =======================================================================
    def _reset_los_rate(self) -> None:
        """Drop the A29 LOS-rate filter state. The state describes ONE gate's geometry, so it is
        dropped whenever the tracked-gate identity can change: pass begin/end, track drop (coast-ticks
        expiry), and :meth:`reset`. The first fresh sample after a reset seeds the EMA directly."""
        self._los_prev_angle = None
        self._los_prev_pose_ns = None
        self._los_rate_ema = 0.0
        self._los_rate_valid = False

    def _update_los_rate(self, yaw_des: float, pose_ns: int) -> None:
        """Fold one fresh-pose world LOS angle into the filtered LOS rate (A29 §4.1).

        FRESH pose only: we difference two CAMERA-EPOCH stamps of the SAME clock, so the A29
        epoch-rate skew (camera epoch measured at 0.9449x wall while the IMU epoch tracked 1.0002x)
        cancels exactly; an async ZOH re-feed (same ``pose_ns``) is not a new geometry sample and is
        ignored. Sample gates: a pose gap longer than ``los_rate_max_dt_s`` (or a non-positive dt)
        RESTARTS the filter (differencing across a long gap mixes geometry regimes); a raw sample
        beyond ``los_rate_max_rps`` is REJECTED (a gate-track hop is a STEP in theta -> a >>3 rad/s
        one-tick sample). An accepted sample seeds the EMA directly when the filter is fresh."""
        if self._los_prev_pose_ns is not None and pose_ns == self._los_prev_pose_ns:
            return
        if self._los_prev_angle is not None and self._los_prev_pose_ns is not None:
            dt = (pose_ns - self._los_prev_pose_ns) / 1e9
            if 0.0 < dt <= self.config.los_rate_max_dt_s:
                d = float(np.arctan2(np.sin(yaw_des - self._los_prev_angle),
                                     np.cos(yaw_des - self._los_prev_angle)))
                sample = d / dt
                if abs(sample) <= self.config.los_rate_max_rps:
                    a = float(np.clip(self.config.los_rate_ema_alpha, 0.0, 1.0))
                    self._los_rate_ema = (a * sample + (1.0 - a) * self._los_rate_ema
                                          if self._los_rate_valid else sample)
                    self._los_rate_valid = True
            elif dt > self.config.los_rate_max_dt_s or dt <= 0.0:
                self._los_rate_valid = False          # gap/garbage: restart the filter
                self._los_rate_ema = 0.0
        self._los_prev_angle = float(yaw_des)
        self._los_prev_pose_ns = int(pose_ns)

    def _compose_los_damped_accel(self, los: np.ndarray, pose: GatePose,
                                  eff_ramp: float, fwd_ramp: float) -> np.ndarray:
        """Compose the pursuit horizontal-accel VECTOR: today's forward feedforward along the LOS
        plus the A29 LOS-rate lateral damping term that BRAKES the tangential drift (§4.1).

        SIGN (safety-critical -- pinned by test_orbit_demands_braking_accel): with
        ``e_r = [cos(theta), sin(theta)]`` (drone->gate) and ``e_t = [-sin(theta), cos(theta)]``,
        a FIXED gate gives ``theta_dot = -(v.e_t)/r``, so ``v_t = v.e_t = -r*theta_dot`` and the
        damping accel is ``a_lat_vec = -kd*(v.e_t)*e_t = +kd*r*theta_dot*e_t``. Cross-checked on
        run 20260703_024023: t=4-5.5 s the drone coasted north with the gate swinging right
        (theta_dot>0, theta~90deg => e_t~[-1,0]) => the demand points south, braking the coast.
        A FLIPPED sign feeds the tangential velocity instead -- it TIGHTENS the orbit.

        The lateral term rides the same authority ramp as the forward term; the composed vector is
        norm-capped at ``total_accel_cap_mps2``. The range is the EMA track range (fallback: this
        pose's PnP range), clamped to ``los_range_cap_m``. Instrumentation stashes (_last_los_rate /
        _last_vt_est / _last_alat / _last_track_range_m) are written here each flag-ON pursuit tick."""
        a_fwd = self.config.forward_accel_mps2 * float(eff_ramp) * float(fwd_ramp)
        a_vec = a_fwd * np.asarray(los, dtype=np.float64)
        if not self._los_rate_valid:
            self._last_los_rate = None
            self._last_vt_est = None
            self._last_alat = None
            self._last_track_range_m = None
            return a_vec
        r = float(np.clip(self._track_range_m if self._track_range_m is not None
                          else pose.range_m, 0.0, self.config.los_range_cap_m))
        v_t = -r * self._los_rate_ema                       # v.e_t (see the sign derivation above)
        self._last_los_rate = float(self._los_rate_ema)     # A29 §4.4 instrumentation
        self._last_vt_est = float(v_t)
        self._last_track_range_m = r
        a_lat_applied = 0.0
        if abs(v_t) > self.config.tangential_deadband_mps:
            e_t = np.array([-los[1], los[0], 0.0])          # horiz unit perpendicular to the LOS
            a_lat = float(np.clip(-self.config.tangential_kd * v_t,
                                  -self.config.lateral_accel_cap_mps2,
                                  self.config.lateral_accel_cap_mps2))
            a_vec = a_vec + (a_lat * float(eff_ramp)) * e_t  # same authority ramp as the fwd term
            a_lat_applied = a_lat
        self._last_alat = a_lat_applied
        return _clip_norm(a_vec, self.config.total_accel_cap_mps2)

    # =======================================================================
    # A30 IMAGE-SERVO LATERAL  (roll toward the APPARENT gate — the orbit killer)
    # =======================================================================
    def _append_att_hist(self, sim_time_ns: int, rpy: tuple[float, float, float]) -> None:
        """Record one (sim_time_ns, (roll, pitch, yaw)) into the capture-time attitude ring buffer
        (A30 §3.3). Created lazily at the configured depth so a flag-OFF seeker allocates nothing."""
        from collections import deque
        if self._att_hist is None:
            self._att_hist = deque(maxlen=max(1, int(self.config.image_att_hist_len)))
        self._att_hist.append((int(sim_time_ns),
                               (float(rpy[0]), float(rpy[1]), float(rpy[2]))))

    def _rpy_at(self, pose_sim_time_ns: int) -> tuple[float, float, float] | None:
        """The buffered attitude NEAREST the pose's capture instant, or ``None`` when unusable
        (empty buffer / nearest entry farther than ``image_att_max_gap_s`` — the caller then falls
        back to the current attitude, exactly the pre-A30 rotation).

        ``pose_sim_time_ns`` is on the CAMERA epoch (the JPEG-wire header) while the buffer is
        stamped on the IMU master epoch, so convert FIRST via the Navigator's learned delta
        (``nav_owner.camera_epoch_to_imu_ns`` — the A29 continuous reconciliation makes this stamp
        trustworthy; the same conversion :meth:`_maybe_latch_z_off` uses). No converter (unit-test
        seeker without a Navigator) => the raw stamp: on synthetic same-epoch tests delta==0, and
        on a mixed-epoch feed the gap guard rejects the garbage lookup -> current-attitude fallback."""
        if not self._att_hist:
            return None
        _to_imu = getattr(self.nav_owner, "camera_epoch_to_imu_ns", None)
        t_imu = _to_imu(int(pose_sim_time_ns)) if _to_imu is not None else None
        if t_imu is None:
            t_imu = int(pose_sim_time_ns)       # fallback: same-epoch tests / pre-reconciliation
        t_near, rpy = min(self._att_hist, key=lambda e: abs(e[0] - t_imu))
        if abs(int(t_near) - int(t_imu)) / 1e9 > self.config.image_att_max_gap_s:
            return None
        return rpy

    def _compose_image_servo_accel(self, nav: NavState, los: np.ndarray, psi_world: float,
                                   eff_ramp: float, fwd_ramp: float) -> np.ndarray:
        """Compose the A30 pursuit horizontal-accel VECTOR (spec §3.1):

        ``az    = wrap(psi_world - yaw_now)``  (the gate's APPARENT horizontal offset, rad, +right;
                  psi_world is capture-consistent, yaw_now is current -> az stays LIVE at tick rate
                  even between fresh poses — yaw motion updates it, a ZOH pose does not freeze it)
        ``a_lat = clip(k_az * dead(az, az_db), +/-image_lat_cap_mps2)`` along ``e_right(yaw_now)``
        ``a_fwd = forward_accel * eff_ramp * fwd_ramp * max(cos(az), 0)^2``  (point before pushing)
        ``a_vec = clip_norm(a_fwd * los + a_lat * e_right, total_accel_cap_mps2)``

        SIGN (safety-critical — pinned by test_gate_right_rolls_right + the true-kinematics orbit
        pins in test_vq2_a30_image_servo): ``e_right(psi) = [-sin(psi), cos(psi), 0]`` is the
        drone's RIGHT in world NED at yaw psi, and +az = gate appears RIGHT of the nose, so
        ``a_lat > 0`` accelerates RIGHT — toward the apparent gate. When an orbit sweeps the LOS,
        the yaw servo LAGS and the gate sits off-center in the SWEEP direction, so this same
        demand points anti-tangential: the brake. A flipped sign here pushes AWAY from the
        apparent gate and feeds the orbit.

        The deadband is CONTINUOUS (``sign(az)*max(|az|-db, 0)``) so the demand has no step at
        the deadband edge (no 0.24 m/s^2 chatter at |az|==db). NO derivative, NO range, NO filter
        state anywhere in this function. Instrumentation stashes (_last_az_err / _last_fwd_scale /
        _last_alat) are written every flag-ON pursuit tick."""
        yaw_now = self._att_yaw(nav)
        az = float(np.arctan2(np.sin(psi_world - yaw_now), np.cos(psi_world - yaw_now)))
        self._last_az_err = az                                  # A30 instrumentation (raw, pre-deadband)
        db = max(float(self.config.image_az_deadband_rad), 0.0)
        az_eff = float(np.sign(az)) * max(abs(az) - db, 0.0)    # continuous deadband
        # A32 (spec §3.2): scale the az error term by the frame's soft bearing weight -- a
        # low-consistency frame steers gently instead of being (A31) thrown away or (pre-A31)
        # trusted fully. 1.0 whenever the soft path is off (byte-identical).
        az_eff *= self._bearing_w_ctl()
        cap = abs(float(self.config.image_lat_cap_mps2))
        a_lat = float(np.clip(self.config.image_kaz_mps2_per_rad * az_eff, -cap, cap))
        # A31 LATERAL-DEMAND SLEW (defense-in-depth for hops that pass any gate): rate-limit the
        # per-tick CHANGE of a_lat so even an ACCEPTED one-frame bearing hop cannot snap the roll
        # demand to the rail in one tick (t=3.25: alat 0 -> -1.5 in ONE tick). An honest az ramp
        # is far below the limit; the very first slewed tick passes through (no dt reference yet
        # -- the pursuit ramp owns the acquisition transient). 0.0 => off (legacy, byte-identical).
        if self.config.image_lat_slew_mps3 > 0.0:
            _now = int(nav.sim_time_ns)
            if self._alat_slew_t_ns is not None and self._alat_slew_prev is not None:
                _dt = max((_now - self._alat_slew_t_ns) / 1e9, 0.0)
                _step = float(self.config.image_lat_slew_mps3) * _dt
                a_lat = float(np.clip(a_lat, float(self._alat_slew_prev) - _step,
                                      float(self._alat_slew_prev) + _step))
            self._alat_slew_prev = a_lat
            self._alat_slew_t_ns = _now
        fwd_scale = float(max(np.cos(az), 0.0)) ** float(self.config.fwd_scale_pow)  # push hardest centered, yield off-axis (A33 S-1: cos^pow)
        self._last_fwd_scale = fwd_scale
        a_fwd = self.config.forward_accel_mps2 * float(eff_ramp) * float(fwd_ramp) * fwd_scale
        e_right = np.array([-np.sin(yaw_now), np.cos(yaw_now), 0.0])
        a_vec = a_fwd * np.asarray(los, dtype=np.float64) + a_lat * e_right
        self._last_alat = a_lat                                 # reuse the A29 lateral stash (alat_mps2)
        return _clip_norm(a_vec, self.config.total_accel_cap_mps2)

    # =======================================================================
    # A31 ORBIT-BREAKER  (cumulative-LOS guard + hard yaw-excursion clamp)
    # =======================================================================
    def _reset_chase(self) -> None:
        """Drop the A31 chase state (the cumulative-LOS integrator + the yaw-clamp anchor + the
        trip counter). The state describes ONE gate acquisition, so it is dropped at pass
        begin/end and :meth:`reset`; the next pursuit tick re-seeds the baseline."""
        self._chase_psi0 = None
        self._chase_yaw0 = None
        self._chase_dpsi = 0.0
        self._chase_prev_psi = None
        self._chase_prev_pose_ns = None
        self._chase_rate = None
        self._orbit_break_t_ns = None
        self._orbit_trips = 0

    def _rebase_chase(self, yaw_des: float) -> None:
        """Re-anchor the chase baseline at the CURRENT geometry (an orbit-break exit): the
        integrator restarts from zero and the yaw clamp re-anchors at the held heading. The trip
        counter is KEPT -- a second trip on the same acquisition drops the track (spec §2.4.4)."""
        self._chase_psi0 = float(yaw_des)
        self._chase_prev_psi = float(yaw_des)
        self._chase_dpsi = 0.0
        self._chase_rate = None
        self._chase_yaw0 = self._last_yaw if self._last_yaw is not None else float(yaw_des)
        self._last_chase_dpsi = 0.0                 # instrumentation follows the rebase

    def _update_chase(self, yaw_des: float, pose_ns: int) -> None:
        """Accumulate the CUMULATIVE unwrapped LOS rotation since acquisition (§1.2's observable:
        the instantaneous az stays small during a whip -- the yaw obediently lag-follows -- so
        only the integrated sweep exposes the orbit). One sample per FRESH pose (camera-epoch
        stamp dedupe, same discipline as the A29 LOS-rate sampler); the shortest-path per-pose
        delta also yields the fresh-pose LOS drift used by the orbit-break early exit."""
        if self._chase_psi0 is None:
            self._chase_psi0 = float(yaw_des)
            self._chase_yaw0 = self._last_yaw if self._last_yaw is not None else float(yaw_des)
            self._chase_prev_psi = float(yaw_des)
            self._chase_prev_pose_ns = int(pose_ns)
            self._chase_dpsi = 0.0
            self._chase_rate = None
        elif int(pose_ns) != self._chase_prev_pose_ns:
            d = float(np.arctan2(np.sin(yaw_des - self._chase_prev_psi),
                                 np.cos(yaw_des - self._chase_prev_psi)))
            dt = (int(pose_ns) - int(self._chase_prev_pose_ns)) / 1e9
            self._chase_dpsi += d
            if dt > 0.0:
                self._chase_rate = d / dt
            self._chase_prev_psi = float(yaw_des)
            self._chase_prev_pose_ns = int(pose_ns)
        self._last_chase_dpsi = float(self._chase_dpsi)     # A31 instrumentation (logging only)

    def _maybe_orbit_break(self, nav: NavState, yaw_des: float) -> ControlCommand | None:
        """The orbit-breaker regime driver (spec §2.4), called each pursuit tick after
        :meth:`_update_chase` when ``orbit_guard_rad > 0``. Returns the bounded brake command
        while the break regime is active, the no-detection hold on a second trip (track dropped),
        or ``None`` to continue normal pursuit.

          * TRIP: |cumulative LOS rotation| exceeds ``orbit_guard_rad`` -> enter ``orbit_break``
            for at most ``orbit_break_s``: forward accel 0, lateral = the full image cap toward
            the CURRENT apparent gate (the anti-tangential brake), yaw setpoint HELD (stop
            following the sweep -- let the LOS come back as the tangential velocity dies).
          * EXIT: early when the gate re-centers (|az| < exit_az) with the fresh-pose LOS drift
            below exit_rate, else on the clock. On exit the chase baseline REBASES and pursuit
            resumes the same tick.
          * SECOND trip on one acquisition: refuse the spin -- drop the track + the ZOH pose and
            fall to the level no-detection hold until a NEW frame re-acquires cleanly."""
        now = int(nav.sim_time_ns)
        yaw_now = self._att_yaw(nav)
        az = float(np.arctan2(np.sin(yaw_des - yaw_now), np.cos(yaw_des - yaw_now)))
        if self._orbit_break_t_ns is None:
            if abs(self._chase_dpsi) <= self.config.orbit_guard_rad:
                return None                          # healthy pursuit -> continue
            self._orbit_trips += 1
            if self._orbit_trips >= 2:
                # SECOND trip on the same acquisition: drop the track, wait level, reacquire.
                self._track_range_m, self._track_bearing, self._track_coast_ticks = None, None, 0
                self._reset_los_rate()
                self._reset_bearing_gate()
                self._reset_chase()
                self._last_pose = None               # drop the ZOH too: wait for a NEW frame
                self._last_regime = "hold"
                return self._hold_command(nav, yaw_rate_cap=self.config.reacquire_yaw_rate_rps,
                                          attitude_safe=True)
            self._orbit_break_t_ns = now             # enter the bounded brake regime
        else:
            elapsed = (now - self._orbit_break_t_ns) / 1e9
            recentered = (abs(az) < self.config.orbit_break_exit_az_rad
                          and self._chase_rate is not None
                          and abs(self._chase_rate) < self.config.orbit_break_exit_rate_rps)
            if elapsed >= self.config.orbit_break_s or recentered:
                self._orbit_break_t_ns = None
                self._rebase_chase(yaw_des)          # re-anchor and resume pursuit THIS tick
                self._last_regime = "pursuit"        # idempotent with command_visual's stamp
                return None
        return self._orbit_break_command(nav, az)

    def _orbit_break_command(self, nav: NavState, az: float) -> ControlCommand:
        """The bounded whip-abort brake: ZERO forward drive, the lateral at the full image cap
        toward the CURRENT apparent gate (same ``e_right(yaw_now)`` composition as A30 -- during
        a sweep the gate sits off-center in the sweep direction, so this points anti-tangential),
        yaw setpoint HELD at the last commanded heading."""
        yaw_now = self._att_yaw(nav)
        yaw_hold = self._last_yaw if self._last_yaw is not None else yaw_now
        a_lat = float(np.sign(az)) * abs(float(self.config.image_lat_cap_mps2))
        e_right = np.array([-np.sin(yaw_now), np.cos(yaw_now), 0.0])
        a_vec = a_lat * e_right
        self._last_az_err = float(az)               # instrumentation (shared A30 stashes)
        self._last_fwd_scale = 0.0
        self._last_alat = a_lat
        self._last_regime = "orbit_break"           # overrides command_visual's "pursuit"
        self._last_yaw = yaw_hold
        los = np.array([np.cos(yaw_hold), np.sin(yaw_hold), 0.0])
        launch = self._launch_ramp(int(nav.sim_time_ns))
        return self._feedforward_command(nav, los, yaw_hold, launch, 0.0, 1.0,
                                         vz_cmd=0.0, accel_vec=a_vec)

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
        # A30 (§3.3): under the image servo, rotate the camera lever with the attitude AT CAPTURE
        # TIME (nearest ring-buffer entry to the pose stamp; current attitude when unusable), so
        # psi_world/yaw_des is exact for a fixed gate up to translation-over-age — the +omega*age
        # contamination (8-17 deg during a 1-2 rad/s yaw = a 1.2-2.3 m/s^2 false lateral at k_az=8)
        # never enters az, the slewed yaw setpoint, or the yaw_des instrumentation. Flag OFF =>
        # the pre-A30 current-attitude rotation, bit-identical.
        if self.config.use_image_servo_lateral:
            rpy_cap = self._rpy_at(int(pose.sim_time_ns))
            gdir = self._gate_dir_world_rpy(
                pose, rpy_cap if rpy_cap is not None else self._att_rpy(nav))
        else:
            gdir = self._gate_dir_world(nav, pose)
        horiz = np.array([gdir[0], gdir[1], 0.0])
        _ty = self._att_yaw(nav)
        los = _unit(horiz, fallback=np.array([np.cos(_ty), np.sin(_ty), 0.0]))
        yaw_des = float(np.arctan2(los[1], los[0]))
        # INSTRUMENTATION ONLY (A14 yaw-steer-sign probe): stash the PRE-SLEW desired yaw toward the
        # gate so the nav-estimate logger can read it. Not consumed by control — purely additive.
        self._last_yaw_des = yaw_des
        # A29 LOS-rate sampling: fold this FRESH pose's world LOS angle (yaw_des, PRE-slew) into the
        # LOS-rate filter, differenced on the pose's CAMERA-EPOCH stamp (same clock both sides -> the
        # epoch-rate skew cancels; a ZOH re-feed of the same pose_ns is not a new geometry sample).
        if self.config.use_los_rate_damping:
            self._update_los_rate(yaw_des, int(pose.sim_time_ns))
        # A31 §2.6 instrumentation fix: _last_track_range_m was written ONLY inside the A29 branch
        # (unreachable under A30 -> track_range_m NULL all flight, run 20260703_160715). Stash it
        # every pursuit tick -- flight-over-flight range history measures the overfly radius.
        self._last_track_range_m = (float(self._track_range_m) if self._track_range_m is not None
                                    else float(pose.range_m))
        # A31 ORBIT-BREAKER + yaw-excursion clamp (both off by default -> unreachable). The chase
        # integrator accumulates the unwrapped LOS rotation since acquisition; the guard aborts a
        # forming whip into a bounded brake instead of following the sweep to backwards.
        if self.config.orbit_guard_rad > 0.0 or self.config.orbit_yaw_clamp_rad > 0.0:
            self._update_chase(yaw_des, int(pose.sim_time_ns))
            if self.config.orbit_guard_rad > 0.0:
                brk = self._maybe_orbit_break(nav, yaw_des)
                if brk is not None:
                    return brk
        # RATE-LIMIT the heading slew: cap the per-tick change of the yaw setpoint so the steering
        # bearing stays SMOOTH (the proximate fix for the swing that saturated roll in A3).
        yaw = self._slew_heading(yaw_des, int(nav.sim_time_ns))
        # A31 hard yaw-excursion clamp: the slewed pursuit yaw cannot rotate past the clamp from
        # the yaw at (re)acquisition, whatever the bearing does -- never turn to backwards.
        if self.config.orbit_yaw_clamp_rad > 0.0 and self._chase_yaw0 is not None:
            c = abs(float(self.config.orbit_yaw_clamp_rad))
            d = float(np.arctan2(np.sin(yaw - self._chase_yaw0), np.cos(yaw - self._chase_yaw0)))
            if abs(d) > c:
                d = float(np.clip(d, -c, c))
                yaw = float(np.arctan2(np.sin(self._chase_yaw0 + d),
                                       np.cos(self._chase_yaw0 + d)))
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
            # A30 IMAGE-SERVO LATERAL (the orbit fix that replaced A29): roll toward the gate's
            # APPARENT azimuth + cos^2(az) forward-pointing scale, composed into ONE vector and
            # passed through the SAME accel_vec seam as A29 (ramps + norm cap already inside; the
            # bridge cache stores unit(a_vec) + |a_vec| so a bridged tick re-issues the decayed
            # COMPOSED vector on the frozen heading). Checked BEFORE the A29 branch: if both
            # flags are ever on, the image servo WINS (see the config note). Flag OFF => fall
            # through unchanged (byte-identical pre-A30 paths below).
            if self.config.use_image_servo_lateral:
                a_vec = self._compose_image_servo_accel(nav, los, yaw_des,
                                                        float(eff_ramp), float(fwd_ramp))
                self._record_last_demand(_unit(a_vec), yaw, float(np.linalg.norm(a_vec)),
                                         int(nav.sim_time_ns))
                return self._feedforward_command(nav, los, yaw, eff_ramp,
                                                 self.config.forward_accel_mps2,
                                                 fwd_ramp, vz_cmd=vz, accel_vec=a_vec)
            # A29 LOS-RATE DAMPING (the orbit fix): compose the forward feedforward + the lateral
            # tangential-velocity brake into ONE vector, then pass it through the SAME machinery
            # (accel_vec bypasses only the scalar accel*ramp*los product -- the ramps are already
            # inside a_vec). The bridge cache stores unit(a_vec) + |a_vec|, so a bridged tick
            # re-issues the decayed COMPOSED vector on the frozen heading -- the cache machinery is
            # untouched. Flag OFF => the scalar path below, byte-identical to pre-A29.
            if self.config.use_los_rate_damping:
                a_vec = self._compose_los_damped_accel(los, pose, float(eff_ramp), float(fwd_ramp))
                self._record_last_demand(_unit(a_vec), yaw, float(np.linalg.norm(a_vec)),
                                         int(nav.sim_time_ns))
                return self._feedforward_command(nav, los, yaw, eff_ramp,
                                                 self.config.forward_accel_mps2,
                                                 fwd_ramp, vz_cmd=vz, accel_vec=a_vec)
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
                             demand_ramp: float, vz_cmd: float = 0.0,
                             accel_vec: np.ndarray | None = None) -> ControlCommand:
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
        # A29: an explicit COMPOSED accel vector (forward + LOS-rate lateral damping, ramps + norm
        # cap already applied inside) bypasses the scalar accel*ramp*los product below. ``None``
        # (every pre-A29 call site, and the flag-OFF path) => the scalar path, byte-identical.
        if accel_vec is not None:
            accel_ned = np.asarray(accel_vec, dtype=np.float64)
        else:
            a_fwd = float(max(accel_mps2, 0.0)) * float(np.clip(demand_ramp, 0.0, 1.0))
            accel_ned = a_fwd * np.asarray(los, dtype=np.float64)  # bounded feedforward forward tilt
        velocity_ned = None
        if abs(float(vz_cmd)) > 0.0:
            velocity_ned = np.array([0.0, 0.0, float(vz_cmd)], dtype=np.float64)  # VERTICAL target only
        sp = Setpoint(
            sim_time_ns=int(nav.sim_time_ns),
            accel_ned=accel_ned,
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
        ramp completes / when disabled.

        A31 (``reramp_forward_after_pass``): the ramp clock restarts at EVERY pass end
        (``_end_pass`` writes ``_fwd_ramp_t_ns``), so the forward drive re-ramps from zero into
        each fresh acquisition instead of feeding full drive while the yaw converges (the
        tail-chase feed, spec §2.3). Flag off / no pass yet => the release clock (byte-identical)."""
        if self.config.forward_ramp_s <= 0.0 or self._release_t_ns is None:
            return 1.0
        t0 = self._release_t_ns
        if self.config.reramp_forward_after_pass and self._fwd_ramp_t_ns is not None:
            t0 = self._fwd_ramp_t_ns
        elapsed = (int(sim_time_ns) - t0) / 1e9
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
        # A33 H-1: drop the old-gate exclusion snapshot + the blind turn target.
        self._pass_prev_dir_world = None
        self._pass_prev_range_m = None
        self._pass_turn_yaw = None
        # hold-last-demand bridge (A13): drop the cached demand + the pose-None reason; zero the
        # per-flight diagnostic counters so a fresh epoch starts clean.
        self._last_demand_los = None
        self._last_demand_yaw = None
        self._last_demand_accel = None
        self._last_demand_t_ns = None
        self._last_none_reason = None
        # A29 LOS-rate damping: drop the filter state + the instrumentation stashes + the regime.
        self._reset_los_rate()
        self._last_los_rate = None
        self._last_vt_est = None
        self._last_alat = None
        self._last_track_range_m = None
        self._last_regime = None
        # A30 image servo: drop the capture-time attitude history (a fresh epoch's clocks differ)
        # + the instrumentation stashes.
        self._att_hist = None
        self._last_az_err = None
        self._last_fwd_scale = None
        # A31: wire-pass flag, forward re-ramp clock, chase/orbit state, lateral-slew history,
        # bearing-gate reference + the instrumentation stashes.
        self._pass_wire = False
        self._fwd_ramp_t_ns = None
        self._reset_chase()
        self._alat_slew_prev = None
        self._alat_slew_t_ns = None
        self._reset_bearing_gate()
        self._last_chase_dpsi = None
        self._last_bearing_dev_rad = None
        self._last_bearing_allow_rad = None
        self._last_bearing_w = None            # A32: the soft weight dies with the epoch
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
