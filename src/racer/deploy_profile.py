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
    ``gyro_sign`` -> ``MavlinkClient(gyro_sign=...)`` (live-wire per-axis gyro convention correction);
    ``seeker_overrides`` -> ``GateSeekerConfig(**seeker_overrides)`` at the seeker construction seam
    (None == no overrides == today's seeker behaviour).
    ``controller_overrides`` -> ``make_seeker_controller(**controller_overrides)`` at the seeker's
    controller construction seam (None == no overrides == today's controller gains).
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
    # Optional GateSeekerConfig field overrides, splatted as ``GateSeekerConfig(**seeker_overrides)``
    # at the seeker construction seam (make_seeker). None == no overrides == today's seeker behaviour
    # (VQ1 / case-A byte-identical). vq2_case_c sets egress_freeze_attitude=True (the A10 acquisition-
    # trap fix). Default keeps existing callers / pickles forward-compatible.
    seeker_overrides: dict | None = None
    # Optional Controller field overrides, splatted as ``make_seeker_controller(**controller_overrides)``
    # at the seeker's controller construction seam. None == no overrides == today's controller gains
    # (VQ1 / case-A byte-identical). vq2_case_c SOFTENS the attitude loop (kp_att 10->4 + a body-rate
    # slew limit) to de-saturate the A11 egress->pursuit handoff overshoot + the bang-bang clip.
    # Default keeps existing callers / pickles forward-compatible.
    controller_overrides: dict | None = None
    # A20 ASYNC-DETECT (2026-07-01): run the gate detector in its OWN daemon worker thread
    # (racer.vision.async_detect) so the ~30 Hz control loop NEVER blocks on the ~250 ms
    # GPU-arbitration detect stall (ShadowPC passthrough RTX 2000 Ada; the sim's render load
    # starves guest CUDA — confirmed not fixable guest-side). The loop consumes the freshest
    # COMPLETED detection (however old); the RewindKF applies it at CAPTURE time (OOSM) and the
    # ESKF gyro-propagates between fixes, so staleness is the designed-for case. Default OFF =
    # byte-identical synchronous path (the worker module is not even imported); vq2_case_c turns
    # it ON. fly_rl's ``--async-detect on|off`` overrides the profile either way (A/B seam).
    async_detect: bool = False
    # A24 VERTICAL-VELOCITY WASHOUT (2026-07-02, supersedes the A21 floor-pin KF): run the
    # dedicated 1-D washout filter (racer.vertical_estimator — IMU a_up integrated with an
    # exponential leak, structurally bounded, NO floor-pin correction) and have the ff-owns-vertical
    # alt-hold damp on ITS smooth, bounded vz instead of the 6-state KF's dead-reckoned z + its
    # finite difference. WHY: the KF z step-teleports when a tight vision fix lands (-1.435 m in one
    # tick, run 20260702_040036) and is blind to the real climb between fixes (integrated a_up hit
    # +4-5 m/s upward that est_z never showed) — the damper reacted to steps instead of the actual
    # vertical rate, and the drone climbed over gate 0 into the ceiling. The floor-height "fix" the
    # A21 KF leaned on was itself a false premise (no floor grid on this wire) -- the A24 washout
    # needs no external measurement at all: it is bounded by construction (an exponential leak, not
    # a Kalman gain). The fly_rl construction seam threads this into BOTH sides of the seam it
    # spans: NavigatorConfig.use_vertical_estimator (own + step the filter, export vz on NavState)
    # and Controller.use_vertical_estimator (consume the export). A SEPARATE flag from
    # ff_owns_vertical so the estimator can be A/B'd against the A19c fd-of-z path independently.
    # Default OFF = byte-identical (no filter constructed, NavState fields NaN, controller on the
    # fd path); vq2_case_c turns it ON. fly_rl's ``--vertical-estimator on|off`` overrides the
    # profile either way (A/B seam).
    vertical_estimator: bool = False


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
        seeker_overrides=None,          # no seeker overrides (byte-identical seeker config)
        controller_overrides=None,      # no controller overrides (byte-identical controller gains)
        async_detect=False,             # synchronous detect (byte-identical VQ1 loop scheduling)
        vertical_estimator=False,       # no vertical channel (byte-identical VQ1 alt-hold inputs)
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
        # --- A32 robust "always find down" AHRS (2026-07-03; spec
        # handoff/vq2_a32_robust_estimation_spec_2026-07-03.md §3.1). The A8 inflation above is
        # kept as the MECHANISM but v2 makes it BOUNDED + soft: replay of run 20260703_172104
        # proved it a self-locking distrust loop (median 671x inflation in NORMAL flight,
        # 4950-10000x railed in the inverted tail while |a| = 9.81 exactly implied 180 deg roll --
        # the gravity pull was OFF and "down" never recovered). v2 = 25x inflation cap @ 1.0 m/s^2
        # knee, R_ref free-run limit 1 s, chi2 hard gate -> Huber-soft, TOTAL deweight cap 100x
        # whenever |a|~g (structural ~0.6 s recovery), gravity-recovery watchdog (25 deg / 0.5 s
        # -> P bump + re-anchor). ahrs_imu_rate_ingest: step the ESKF on the FULL ~185 Hz
        # HIGHRES_IMU ring instead of latest-sample-per-tick (the +45 deg/tick contact-spike
        # aliasing, spec F3).
        ahrs_accel_trust_v2=True,
        ahrs_imu_rate_ingest=True,
        # --- map-free vision yaw + z (no mag / no baro) ---
        use_vp_yaw=True,
        use_gate_bearing_yaw=True,
        use_floor_height=True,
        # --- gate-relative +L position chain ---
        use_gate_relative=True,
        use_rewind_kf=True,
        use_range_channel=True,
        use_inplane_pos_floor=True,
        # --- A17 CV-backstop decimation (frame-starvation fix, 2026-07-01) ---
        # The per-frame vp_yaw (~67 ms VP RANSAC + Manhattan) + floor_height (~37 ms) choked the live
        # loop to ~10 Hz and starved the video receiver. Decimate them: vp_yaw every 5th tick (yaw
        # drifts slowly + the gate-bearing-yaw lock pins yaw per accepted detection + the ESKF gyro-
        # integrates between), floor_height every 3rd (the ONLY dedicated z pin -> keep N tighter).
        # Default (1) is byte-identical; these are the estimator-safety-study tuned values.
        vp_yaw_decimate=5,
        floor_height_decimate=3,
        # --- A26 gate-offset-rate washout fusion -- SUPERSEDED by A28 (2026-07-03) ---
        # Was True (2026-07-02). Run 20260703_013748 proved the fusion is a noise/bias injector:
        # the finite difference of two noisy poses ~0.1 s apart swings +/-5 m/s tick-to-tick, and
        # even ACCEPTED samples dragged vz_est to a standing -0.67 m/s while the truth was 0 (A28
        # spec §1.3b). The A28 complementary filter's beta position-innovations carry the SAME
        # information with the correct structure, so the fusion goes back OFF here (the flag stays
        # for byte-compat / A-B replay of the A26 behaviour).
        use_gate_vz_fusion=False,
        # --- A29 continuous epoch reconciliation (2026-07-03) ---
        # The camera epoch runs at 0.9449x wall under GPU load (run 20260703_024023) while the IMU
        # epoch tracks 1.0002x, so the learn-once delta_epoch drifted ~0.05-0.10 s/s: pose_age_s
        # ramped to the 1.0 s cap, the vertical latch latency-comp over-corrected by vz*(~0.9 s)
        # (a large chunk of the 50% zoff innovation rejection), and every OOSM fix rewound past
        # RewindKF.horizon_s=0.5 from ~t=5 s on -- the KF position channel silently dead. Re-track
        # the delta per vision tick with a slow EMA (alpha field default 0.10). VQ1 default OFF.
        reconcile_vision_clock_continuous=True,
        # --- A28 2-state complementary filter on (z_off, vz_rel) (2026-07-03) ---
        # Splatted into VerticalEstimator(...) at Navigator._initialize (same opt-in dict pattern
        # as controller_overrides). use_zoff_filter=True: IMU predicts BOTH states per tick (phase
        # lead -- the exported state is current-time, so the ~137 ms / p90 345 ms pose age stops
        # eating loop phase margin), each fresh pose corrects both via alpha-beta position
        # innovations (alpha=0.4 / beta=0.15 field defaults, sized for ~10 Hz fresh-pose cadence,
        # sigma_z ~0.3-0.5 m; the beta line bleeds off the sub-threshold contact-accel poison that
        # railed the washout, spec §1.3a), innovation gate 2 m + reseed-after-4 handles the
        # observed +/-8 m gate-track jumps, and the internal state is UNCLAMPED (the A25 +/-3
        # state clamp destroyed all back-half information; the clamp moves to the controller's
        # gate_pd_vertical consumption point). export_clip_mps 1.5 -> 2.5: the legitimate steady
        # descent at full clamped offset is (Kp/Kd)*3 = 2.0 m/s under the A28 gains, and with
        # kd=0.06 the worst damping contribution is +/-0.15 collective -- the old +/-1.5 anti-slam
        # rationale no longer binds. VQ1/case-A never construct the estimator -- byte-identical.
        # A32 (2026-07-03): use_soft_innov_weight — the A28 2 m innovation cliff (130 hard-rejected
        # ticks, |innov| p90 7.3 m, reseed-teleports on run 20260703_172104) becomes a Huber-
        # weighted correction (sigma 0.7 m, full weight <= 1.4 m, never 0) with the reseed-on-
        # persistence RETAINED (miss at nu > 4 ~ 2.8 m). Every pose now moves the filter, scaled
        # by its consistency x the seeker's bearing weight for the same frame.
        # A33 V-1 (2026-07-03): weight-qualified reseed. The A32 reseed-on-persistence teleported
        # z_off onto low-weight smear/garbage frames twice on run 20260703_210632 (cmd 67 w
        # 0.11-0.31, cmd 164 w 0.00-0.04), arming the balloon climb. zoff_reseed_min_w=0.3: a garbage
        # frame no longer counts toward the reseed; an honest handoff (fresh-track frames w=1.0) still
        # reseeds in reseed_after frames. See handoff/vq2_a33_gate2_intercept_spec_2026-07-03.md §V-1.
        # A35 Fix-2 (2026-07-03): magnitude-gated trust floor. Run 20260704_024434 climbed into the
        # ceiling because as the drone rose ABOVE the gate the +20 deg camera lost it -> bearing_w
        # collapsed to 0.04 -> the A32 soft weight crushed the honest +4.7 m z_off innovation to 0.02
        # and z_off_est drifted to the WRONG SIGN (-3.5 m) for 96 ticks; the PD held ~hover, never
        # using V-2's open floor. use_zoff_big_trust: a LARGE (>=2.5 m) + sign-consistent (3 frames)
        # offset floors the correction weight (>=0.5), reseeds even from low bearing weight (the
        # exception to V-1's block), and freezes the propagate while distrusted -- a NARROW exception
        # for real excursions; small/noisy offsets keep the A32/A28 behaviour exactly. See
        # handoff/vq2_a35_exposed_control_failures_spec_2026-07-03.md §2.
        vertical_estimator_overrides={"use_zoff_filter": True, "export_clip_mps": 2.5,
                                      "use_soft_innov_weight": True,
                                      "zoff_reseed_min_w": 0.3,
                                      "use_zoff_big_trust": True},
    )
    return DeployProfile(
        name="vq2_case_c",
        nav_config=cfg,
        cmd_rate_scale=VQ2_CMD_RATE_SCALE,
        self_localizing=True,
        gyro_sign=VQ2_GYRO_SIGN,   # live-confirmed HIGHRES_IMU PITCH-axis flip (x/z unconfirmed -> +1)
        # A10 acquisition-trap fix: freeze the spawn attitude through egress so the +20deg camera
        # stays ON the spawn gate while forward demand ramps from ~0 (no saturated nose-up re-level).
        # A13 hold-last-demand bridge (2026-06-30): the slow VQ2 cruise drops a usable pose on ~75% of
        # ticks (track-continuity flap), each previously a regime-2 zero-coast -> POLYGONAL motion. Hold
        # the last pursuit demand for 0.6 s across these gaps so control is CONTINUOUS per tick. (The
        # track-continuity gate widening is DEFERRED to A14 -- tune with the instrumentation data.)
        # A14 yaw-mirror fix (2026-06-30): under use_ahrs the case-C Navigator hands the seeker the
        # ODO-conjugated attitude (roll+yaw negated, pitch kept). The seeker geometry used the raw
        # conjugated euler and the controller's odo_att_sign un-conjugates roll but NOT yaw, so a gate
        # on the RIGHT steered the nose LEFT. true_attitude_from_ahrs makes the seeker consume the TRUE
        # euler (-nav.roll, nav.pitch, -nav.yaw) and pass the controller a yaw-negated nav so R_cur is
        # R_true -- no sign knob touched. VQ1 / case-A byte-identical (default OFF).
        # A30 image-servo pursuit (2026-07-03, replaces the A29 LOS-rate damping): run
        # 20260703_150755 proved the A29 lateral UNSTABLE -- it differentiated a noisy,
        # self-motion-contaminated bearing (corr with own yaw rate 0.65) and multiplied by an
        # untrusted range (track pinned at the 25 m cap on 62% of ticks) -> vt_est hit |58| m/s
        # and alat limit-cycled at ~2.1 s (sign flip per 1.15 s, 80% saturated), thrashing the
        # drone over gate 0. ``use_los_rate_damping`` is therefore REMOVED from this dict
        # (default False -> A29 damping DISABLED; the code + tests stay in the tree for replay).
        # THE REPLACEMENT: use_image_servo_lateral -- roll toward where the gate APPEARS in the
        # frame (a_lat = clip(8*az, +/-1.5) on the per-frame image azimuth: no derivative, no
        # range, no filter state) + a cos^2(az) forward-pointing scale + the capture-time-attitude
        # ring buffer that keeps omega*age out of az. Replayed over the real A28 gate-1
        # kinematics this law had -16.4 m/s of tangential braking available vs the ~3 m/s
        # carried. Sub-field defaults (k_az=8, az deadband 0.03, lateral cap 1.5, att hist 36
        # ticks / 0.5 s gap guard) live on GateSeekerConfig; total_accel_cap_mps2 is OVERRIDDEN
        # 2.5 -> 2.0 here (11.5 deg max composed lean; the field default stays 2.5 = A29's
        # shipped value, so flag-off byte-identity is untouched).
        # A29 KEPT: pursuit_yaw_slew_rps 1.5 (the delay trim) and the continuous clock
        # reconciliation in nav_config above (pose_age p50 138 ms / 62% fresh -- a confirmed win).
        # A31 (2026-07-03, run 20260703_160715 -- immediate turn on the wire gate-pass; see
        # handoff/vq2_a31_immediate_turn_spec_2026-07-03.md):
        #   * pass_wire_coast_s 0.25: the RACE_STATUS index increment means the drone is PAST the
        #     gate plane -- the legacy 1.2 s blind glide was pure lost time (gate 1 was DETECTED
        #     during it; the turn started 1.22 s late carrying ~3.9 m/s). Poses become eligible
        #     0.25 s after a wire-committed/confirmed pass -> turn latency <= ~0.4 s.
        #   * pass_coast_accel_mps2 0.0: spend momentum through the pass, don't build it (+1.5 m/s
        #     was added across the old accelerating glide, on a FROZEN blind heading).
        #   * reramp_forward_after_pass: the forward feedforward re-ramps from ZERO after every
        #     pass -- point before pushing; feeding forward drive while the yaw converges is what
        #     sustained the tail-chase.
        #   * forward_accel_mps2 1.2 -> 0.8 (slow the approach, commander directive): the whip is
        #     a geometry disease -- LOS sweep rate v_t/r ~ 0.7 rad/s outran the closed-loop yaw
        #     follow. Less carried speed (~sqrt(0.8/1.2) ~ 0.82x build-up, compounding with the
        #     zero coast accel and the ~1 s earlier turn to roughly HALVE v_t at acquisition)
        #     drops the sweep toward ~0.2-0.3 rad/s, inside what the yaw servo + the A30 lateral
        #     brake (saturated 63% of the A30 chase at +/-1.5) can actually kill.
        #   * orbit_guard_rad 1.75 (~100 deg) + orbit_break_s 1.0: the pre-registered safety net.
        #     The guard observable is CUMULATIVE unwrapped LOS rotation since acquisition -- the
        #     203 deg whip kept the instantaneous az SMALL (+0.17 rad mean, yaw lag-following),
        #     so only the integral exposes it. Trip = a <=1 s brake (forward 0, lateral cap
        #     toward the apparent gate, yaw held), second trip = drop the track and hold.
        #   * orbit_yaw_clamp_rad 2.4 (~137 deg): the slewed pursuit yaw physically cannot rotate
        #     past this from the yaw at acquisition -- never turn to BACKWARDS chasing a gate.
        #   * use_imu_bearing_gate: the IMU-consistency bearing gate (operator directive --
        #     REPLACES the "tighten the fixed jump threshold" band-aid; the horizontal analog of
        #     the A28 vertical complementary filter). Gates are static: predict this frame's
        #     bearing from the last tracked WORLD direction (capture-time AHRS attitudes
        #     compensate the measured rotation exactly) and reject a deviation beyond sensor
        #     noise (0.06 rad) + the translation-parallax bound (4 m/s * dt / range) --
        #     REGARDLESS of size. The t=3.25 close-range 0.33 rad hop (left roll + the poisoned
        #     +0.82 m vertical latch -> floor tap) slid UNDER the fixed 0.35 gate; against the
        #     motion-consistency allowance (~0.10 rad at that dt/range) it is 3x out.
        #     track_max_bearing_jump_rad deliberately NOT tightened: it stays the 0.35 legacy
        #     fallback for attitude-unavailable frames (and VQ1's untouched default).
        #   * image_lat_slew_mps3 6.0 (defense-in-depth): even an accepted noisy bearing can't
        #     snap the A30 lateral 0 -> +/-1.5 in one tick (full-scale reversal takes 0.5 s; an
        #     honest az ramp is never limited).
        #   * hold_thrust_lo/hi_frac 0.90/1.12: the settle hold is a conservative HOVER, not an
        #     alt-hold on an unseeded estimator -- the old [0.6, 1.4] band let the cold z estimate
        #     rail the collective (0.159 <-> 0.372) and inject ~+1.2 m/s upward before pursuit
        #     even began (the startup swell -> overshoot -> floor-tap chain). Worst-case injected
        #     velocity drops to ~0.3 m/s; the egress thrust floor (1.0x hover) is untouched.
        seeker_overrides={"egress_freeze_attitude": True, "hold_last_demand_s": 0.6,
                          "true_attitude_from_ahrs": True,
                          "use_image_servo_lateral": True,
                          "total_accel_cap_mps2": 2.0,
                          "pursuit_yaw_slew_rps": 1.5,
                          # --- A31 ---
                          # (pass_wire_coast_s was 0.25 here; A33 H-1b sets it 0.0 below)
                          "pass_coast_accel_mps2": 0.0,       # spend momentum through the pass
                          "reramp_forward_after_pass": True,  # point before pushing, enforced
                          # forward_accel_mps2 set to A36 Item-1's 0.65 in the A36 block below
                          # (was A33's 0.8; slower approach per operator "slow is smooth").
                          "orbit_guard_rad": 1.75,            # cumulative-LOS whip abort (~100 deg)
                          "orbit_break_s": 1.0,
                          "orbit_yaw_clamp_rad": 2.4,         # never chase a gate to backwards
                          "use_imu_bearing_gate": True,       # IMU-consistency bearing gate (hop killer)
                          # --- A32 (2026-07-03): the A31 gate's BINARY consequence hard-rejected
                          # 59.9% of evaluated frames (median rejected frame only 1.87x over the
                          # allowance) and starved the seeker. Soft Cauchy weight instead:
                          # w = 1/(1 + (dev/allow)^2 + (range jump/max)^2) scales the track EMA,
                          # the image-servo az term, and the z_off latch -- every frame
                          # contributes, coast/track-drop only on PERSISTENT w < 0.1. The
                          # dev/allow physics above is unchanged (the gate machinery computes it).
                          "use_soft_bearing_weight": True,
                          "image_lat_slew_mps3": 6.0,         # no one-frame lateral rail-snap
                          # A35 Fix-3 (2026-07-03; spec vq2_a35_exposed_control_failures_spec):
                          # 0.90 -> 1.00. Run 20260704_024434 tapped the ground on the line: during
                          # settle/anchor the cold AHRS levels a ~-18 deg spawn pitch (~0.5 s) and the
                          # [0.90,1.12]x hover band let collective sit at 0.239 (below hover 0.2656) --
                          # 32% of the first 1.6 s sub-hover -> net sink -> ground tap BEFORE egress.
                          # Floor the settle/anchor hold at hover (never sink; the 1.12 A31 anti-swell
                          # cap stays). A small climb off the line is recoverable; a ground tap is not.
                          "hold_thrust_lo_frac": 1.00,        # settle = HOLD hover, never sink (A35 Fix-3)
                          "hold_thrust_hi_frac": 1.12,        # ... 1.12 anti-swell cap (A31) kept
                          # --- A33 (2026-07-03; spec vq2_a33_gate2_intercept_spec) ---
                          # H-1(a) GEOMETRIC old-gate exclusion: the RACE_STATUS index leads the
                          # physical plane by ~3 m (run 20260703_210632 cmd 49), so the acquire-next
                          # re-lock must exclude the just-passed gate by GEOMETRY, not by a time
                          # window -- then the windows can collapse (H-1b) to kill the 1.2 s
                          # frozen-yaw stall that discarded well-detected gate-2 frames.
                          "pass_exclude_prev_gate": True,
                          # H-1(b) collapse the eligibility windows (the exclusion now rejects the
                          # old gate, so the clock no longer has to): wire 0.25 -> 0.0, vis 1.2 -> 0.3.
                          "pass_wire_coast_s": 0.0,           # (overrides the A31 0.25 above)
                          "pass_coast_s": 0.3,
                          # H-1(c) TURN-THROUGH-OCCLUSION: begin the yaw slew on the pass TRIGGER
                          # (blind turn target), ride through the gate-frame occlusion, refine when
                          # gate-2 clears -- do NOT wait for a gate-2 pose (operator amendment 1).
                          "pass_turn_through": True,
                          # A34 (2026-07-03; spec vq2_a34_pose_feed_regression_spec) REPLACES H-3:
                          # H-3's pred_r-RELATIVE range wall STARVED recovery on run 20260703_223956
                          # (once the track smeared to ~49 m via first-acq's permissive fallback
                          # locking a 52 m mis-depth, a fresh close 6 m pose is |6-49|=43 m > jump ->
                          # REJECTED forever) and never caught the original gradual A32 smear anyway.
                          # DROPPED (soft_range_hard_reject no longer set here -> field default False).
                          # ABSOLUTE cap instead: a >35 m reading is physically impossible on the
                          # course (max spacing ~30-38 m, usable PnP ~32 m) = garbage, HARD-discarded
                          # at candidate admission before any acquisition path (closes the permissive-
                          # fallback hole). 35 m keeps every legit close pose (6.4-10.7 m) with margin.
                          "track_abs_range_cap_m": 35.0,
                          # S-1 sharpen the off-axis forward cut cos^2 -> cos^4: cuts overfly speed
                          # (and the translational-lift up-bias it drives) without touching centered
                          # pace; keeps the immediate turn (H-1c) from also being an immediate lunge.
                          "fwd_scale_pow": 4.0,
                          # === A36 (2026-07-03; spec vq2_a36_turn_convergence_spec) ===
                          # Item 0 (body_rate_sign -> (1,1,-1), in controller_overrides) FLEW + is
                          # CONFIRMED (run 20260704_042616: yaw turns toward gate 2, range closed to
                          # 2.8 m). FLIGHT 2 exposed a YAW OVERSHOOT: the turn WHIPPED ~270 deg because
                          # the pass turn (controlled ~1 rad/s) reverted to PURSUIT after 0.3 s and
                          # pursuit rail-chased the rotating close-range LOS for -207 deg. This build
                          # (flight-3 profile) enables FIX A (Item 3 hold-until-pointed) + FIX B
                          # (close-range yaw taper) to kill the overshoot, plus Items 1 & 4 (slow +
                          # switch-lanes) per operator. FIX C (kd_att bump) is built but LEFT OFF (the
                          # attitude loop is not the dominant cause; enable only if A+B leave residual
                          # ring).
                          #
                          # FIX A / ITEM 3 -- PHYSICAL-PLANE TURN + HOLD-UNTIL-POINTED (ON): commit the
                          #   turn at the physical plane (rng ~pass_arm_range_m) not the ~9 m-early wire
                          #   (self-cancels via H-1b's 0.0 coast + acquire-next), and HOLD the turn coast
                          #   until POINTED (~10 deg of the target, bounded 1.5 s) so the turn COMPLETES
                          #   as a controlled slew and hands off pointed -- instead of dumping a half-turn
                          #   into the saturated pursuit chase (the -207 deg whip).
                          "pass_wire_requires_near": True,
                          "pass_turn_hold_until_pointed": True,
                          "pass_turn_coast_s": 1.5,
                          "pass_turn_point_tol_rad": 0.17,
                          # REFINE-TO-REAL-GATE: the blind sign*cap turn target was ~147 deg WRONG on
                          # run 20260704_042616 (LEFT off the passed gate's close az while gate 2 was
                          # RIGHT). Instead coast straight through the occlusion, RE-AIM at the first
                          # valid downrange gate-2 pose (past degenerate + <=35m, NOT the passed gate via
                          # H-1a, bearing_w>=0.3), latch-follow, hold-until-pointed to the REAL gate.
                          # Fallback if none seen = straight coast (never a committed wrong turn).
                          "pass_turn_refine": True,
                          "pass_refine_min_bw": 0.3,
                          #
                          # FIX B -- CLOSE-RANGE YAW-RATE TAPER (ON): scale the pursuit yaw setpoint-slew
                          #   authority by tracked range (floor 0.35 near, full by 10 m) so a fast
                          #   close-range LOS can't drive the post-release tail-chase rail. Proportional
                          #   on measured range, no derivative.
                          "yaw_slew_taper_lo_range_m": 3.0,
                          "yaw_slew_taper_hi_range_m": 10.0,
                          "yaw_slew_taper_floor": 0.35,
                          #
                          # ITEM 1 -- POINTING GATE on forward drive ("point before you push") + SLOWER
                          #   approach (operator: "even a tad slower would be good"): cut a_fwd to ~0
                          #   until the gate is roughly centered (sharp turn, not a wide arc), and drop
                          #   the cruise forward accel 0.8 -> 0.65 (~19% slower). "slow is smooth."
                          "fwd_point_gate_az_rad": 0.35,
                          "fwd_point_gate_full_az_rad": 0.05,
                          "forward_accel_mps2": 0.65,   # A36 Item 1: slower approach (was A33's 0.8)
                          #
                          # ITEM 4 -- LATERAL-FIRST "switch lanes" (ON): bank sideways onto the approach
                          #   line instead of yaw-to-face + being carried past (operator saw "barely any
                          #   roll"). Raise the lateral cap so a big cross-track drives a real
                          #   translation (3.0 ~= a 17 deg bank; the A31 lat-slew still rate-limits it).
                          "use_lateral_first_budget": True,
                          "image_lat_cap_mps2": 3.0,
                          },
        # A11 control-softening fix (2026-06-30): the seeker's stiff attitude loop (kp_att=10 vs the
        # +/-1.5 rad/s pursuit pitch-rate cap) saturates on ANY attitude error > ~8.6deg (1.5/10), so
        # the egress->pursuit HANDOFF (held ~-18deg nose-down vs ~-5deg cruise = ~13deg error) commands
        # ~2.3 rad/s, clips to 1.5, and OVERSHOOTS past level into nose-UP -> gate whips out of frame
        # (plus 15/18 bang-bang clip bursts). SOFTEN it for vq2_case_c only:
        #   * kp_att 10 -> 4: max non-saturating attitude error = cap/kp = 1.5/4 ~= 0.375 rad ~= 21.5deg
        #     (vs 8.6deg at kp=10), so the ~13deg handoff no longer saturates + corrections are gentler.
        #   * body_rate_slew_max_rps2 = 8.0: at ~12 Hz (dt~=0.083 s) the per-tick body-rate change is
        #     bounded to 8.0*0.083 ~= 0.66 rad/s, so a step (handoff, or a big correction after a
        #     detection gap) ramps 0->1.5 over ~2-3 ticks instead of whipping the camera in ONE tick.
        #   * ff_owns_horizontal = True: the A15b nose-up-into-the-ceiling ROOT fix. The pursuit
        #     feedforward (accel_ned) owns the horizontal tilt; the vertical-align velocity_ned=[0,0,vz]
        #     is consumed by the alt-hold for vz ONLY and its (zero) horizontal components no longer
        #     damp the DEAD-RECKONED (fictional) horizontal velocity. Without this, -kd_vel*vel_xy
        #     swamps the +1.2 feedforward, flips the tilt target nose-UP, and the drone climbs into the
        #     ceiling the instant pursuit engages (A15b flight 2; offline-repro-confirmed). Softening
        #     (kp_att/slew) could NOT fix it because the TARGET direction was wrong (nose-up), not stiff.
        #   * ff_owns_vertical = True: the A19c bang-bang ROOT fix (the vertical twin of the above). The
        #     alt-hold's kd_alt*(vel[2]-vz_t) damped the estimator's DEAD-RECKONED vel[2] (no baro /
        #     ODOMETRY blocked -> drifting IMU integration); in pursuit kp_alt*(z-z_t) is constant so that
        #     term was the SOLE thrust driver and rail-slammed the collective 0.05<->0.60 -> a net climb
        #     OVER the acquired gate -> gate lost -> the 360 yaw-search (A19c, offline-repro-confirmed:
        #     rail-slam flips 19->3). ON: a position loop on the trustworthy floor-corrected z + the
        #     vertical-align vz_t ramped THROUGH a moving z_target + damping on a low-passed finite-diff of
        #     z (NOT raw vel[2]). Without this, the vertical channel ends the flight regardless of control.
        #   * body_rate_sign = (1,1,1): the A22 gate-2 no-turn ROOT fix (2026-07-02). The seeker's
        #     default SEEKER_SIGNS carries the VQ1-MEASURED yaw command inversion (body_rate_sign
        #     z=-1: a right-turn FRD yaw command is emitted NEGATED because THAT sim inverted it).
        #     The VQ2 wire does NOT invert: run 20260702_152528_rl_s1_f1 (the gate-2 no-turn
        #     failure) shows the realized yaw rate tracks the POST-sign wire value directly —
        #     corr(wire, d(yaw_est)/dt) = +0.79 at the known ~2.1x realization gain, only 5% sign
        #     agreement with the pre-sign FRD command — while the estimator itself is exonerated
        #     twice over (yaw_est == integral of the A9 sign-corrected gyro, corr +0.96; AND the
        #     camera-measured gate bearing swept the SAME way as the estimated turn, +0.43 rad/s).
        #     So a gate seen on the RIGHT produced a wire command that physically turned the drone
        #     LEFT: a positive-feedback yaw loop that saturates and spins the gate out of frame —
        #     the "passed gate 1, never turned to the well-detected gate 2" signature. Roll/pitch
        #     are exonerated (attitude held stable all flight, |roll|<0.10 rad pre-collision), so
        #     only the yaw element changes: identity (1,1,1). The per-wire actuation convention
        #     lives HERE, in the profile, exactly like gyro_sign; VQ1 keeps the measured seeker
        #     default [1,1,-1] untouched (flight-proven on that wire, byte-identical off-path).
        #   * kp_alt = 0.0 (the A24 washout fix, R0, 2026-07-02): kill the position term. Before
        #     this, ``make_seeker_controller``'s kp_alt=2.0 default was LIVE on the ff-owns-vertical
        #     path, multiplying the diverging double-integrated z (a SECOND, LARGER poison than the
        #     vz term the A21->A24 rewrite fixes). With kp_alt=0, the effective law is purely
        #     ``thrust = hover + ff_vertical_kd_alt*(vz - vz_t)``, clamped -- the z_target-ramp
        #     machinery goes inert (only fed by the now-zeroed kp_alt term) but is left in place,
        #     unused.
        #   * gate_pd_vertical = True (the A28 vertical-stability fix, 2026-07-03 -- see
        #     handoff/vq2_a28_vertical_stability_spec_2026-07-03.md §2.2): the vertical law becomes
        #     the SINGLE PD ``thrust = hover - kp_gate*clip(z_off,+/-3) + ff_vertical_kd_alt*vz_lp``
        #     with vz_t REMOVED from the law. Diagnosis of run 20260703_013748 (reconstruction-
        #     exact, |err|~1e-3): the old law hid a SECOND position path inside the damping term
        #     (-kd*vz_t = -kd*clip(0.8*z_off,+/-1), slope 0.20 thrust/m vs the explicit 0.06 ->
        #     effective stiffness ~0.26 thrust/m, omega_n ~3.1 rad/s) whose ONLY damping rode a
        #     three-ways-corrupted vz_est (anti-damping surges reproduced exactly at t=9.17), AND
        #     the hold-last-demand bridge flicked the controller's ACTUAL vz_t 0<->+/-1 at 2.3 Hz
        #     (62 flips/27 s), stepping thrust by 0.25 = 94% of hover as a square wave. With vz_t
        #     out, both vanish with NO seeker change (the seeker keeps emitting vz_t for
        #     instrumentation; the bridge flicker becomes harmless).
        #   * kp_gate = 0.04 (A28, was 0.06 explicit + the 0.20 hidden path): with the plant gain
        #     c ~= 37 m/s^2 per unit collective (MEASURED this flight, t=1.3-2.0 burst) ->
        #     omega_n = sqrt(37*0.04) = 1.21 rad/s (T ~5 s to close a full 3 m clamped offset --
        #     slow-lap appropriate, 6.5x softer than the 3.1 rad/s that limit-cycled). Sign
        #     unchanged (-kp_gate*z_off: above-gate => sink), pinned by the A25 sign tests.
        #   * ff_vertical_kd_alt = 0.06 (A28, was 0.25 on a garbage signal): zeta = 37*0.06/
        #     (2*1.21) = 0.92 -- critically-ish damped, no overshoot to re-excite the floor<->
        #     ceiling swing. Steady descent at full clamped offset = (Kp/Kd)*3 = 2.0 m/s -- the
        #     velocity cap now EMERGES from the gain ratio instead of a saturating vz_t path. The
        #     damping rides the A28 filter's IMU-predicted vz_rel (effective delay ~1 tick ~7deg at
        #     the 2.3 rad/s PD crossover, vs 52deg for a vision-rate signal -- why the filter, not
        #     the raw washout, must carry it).
        #   * ff_vertical_vz_lp_alpha = 0.8 (unchanged from A26): light LP on the vz_meas branch,
        #     ~19 Hz @ 30 Hz control rate, negligible lag at the 1.21 rad/s loop.
        # See the A28 spec §2.3 for the full omega_n/zeta sizing; values are gated to vq2_case_c
        # only via this override dict (Controller field defaults untouched -> VQ1/case-A stay
        # byte-identical).
        #   * alt_thrust_lo = 0.15 + alt_thrust_slew_per_s = 2.0 (the A27 forward-decouple pair,
        #     2026-07-03): KEPT under A28, explicitly re-scoped as DORMANT SAFETY NETS, not
        #     stabilizers (A28 spec §2.4 verdict). The raised floor did NOT cause the 20260703_013748
        #     failure (the divergence was up-going anti-damping surges; the back-half floor-pinning
        #     was CORRECT commanded descent while lodged) and its forward-coupling rationale was
        #     validated (translation improved). At the floor the plant still yields -4.3 m/s^2 of
        #     descent authority ~= the A28 loop's max commanded descent accel -- not binding. The
        #     2.0/s slew needs at most ~0.5/s in linear A28 operation -- dormant, as it should be.
        #     PRE-REGISTERED REVERT CRITERION (spec §2.4): if a future flight shows z_off > 1 m
        #     with thrust pinned at 0.15 for > 2 s while vz_rel < 0.5 m/s of descent, lower
        #     alt_thrust_lo to 0.10 for vq2_case_c.
        controller_overrides={"kp_att": 4.0, "body_rate_slew_max_rps2": 8.0,
                              "ff_owns_horizontal": True, "ff_owns_vertical": True,
                              # A36 FIX C (built but LEFT OFF): yaw-rate DAMPING bump kd_att 0.30 ->
                              # ~0.7 for better-damped attitude tracking. The yaw-overshoot diagnosis
                              # (run 20260704_042616) attributes the whip to the pursuit tail-chase
                              # (Fixes A+B), NOT an intrinsically hot attitude loop, so this stays OFF
                              # to keep attribution clean. ENABLE only if a flight shows residual
                              # RINGING after A+B (attitude under-damping): uncomment the next line.
                              # "kd_att": 0.7,
                              # A36 ITEM 0 (2026-07-03; spec vq2_a36_turn_convergence_spec):
                              # (1,1,1) -> (1,1,-1) -- REVERT the A22 yaw actuation sign. A22
                              # (d1bca4b) set yaw +1 off ONE run (20260702_152528) that PREDATED
                              # its own commit by 36 min and flew on the seeker default -1; A22
                              # read the pre/post-sign relationship backwards and flipped a WORKING
                              # sign. Every one of the 16 flights since (all on +1) shows the SENT
                              # yaw command correctly aimed at the gate (run 20260704_032626:
                              # cmd-toward-gate 132/5) yet the drone yawing AWAY (realized
                              # toward-gate 23/141) -- a positive-feedback yaw loop that is the ROOT
                              # of the never-turns-to-gate-2 orbit (and the 203deg whip the A31
                              # orbit-breaker chased). The wire INVERTS the yaw-rate command, so we
                              # emit -1 (the VQ1-proven seeker default). The command PATH is
                              # unchanged since A22 (controller.py:513 omega*body_rate_sign); only
                              # this override value moved. VQ1/case-A untouched (seeker default -1).
                              "body_rate_sign": (1.0, 1.0, -1.0),
                              # A36 YAW-STEER SELECTOR (2026-07-04): full-package flight 1
                              # (20260704_051851) -- operator eyes: ROLL banks correctly toward the gate
                              # but YAW rotates the WRONG physical way ("rolled right, yawed left").
                              # STEP-0 seam: R_cur yaw uses odo_att_sign[2]=+1 (yaw NOT recovered) so
                              # R_cur yaw = nav.yaw = -true_yaw while R_des yaw (seeker setpoint) is
                              # +true -> inverted yaw error. Roll is CORRECT because its pairing recovers
                              # true (odo_roll=-1). FIX = a VQ2-only R_cur-yaw recovery selector:
                              #   "B" (THIS, fly first): recover R_cur yaw (odo_yaw->-1) and KEEP the
                              #       VQ1-PROVEN brs_yaw=-1 (the yaw WIRE genuinely inverts -- A22's
                              #       brs_yaw=+1 ORBITED). Minimal departure: proven baseline + one
                              #       VQ2-specific seam fix. body_rate_sign above stays (1,1,-1).
                              #   "A" (fallback if B flies inverted the other way): also revert
                              #       brs_yaw->+1 (set body_rate_sign (1,1,1)) to match roll's (odo,brs)
                              #       =(-1,+1) pairing literally. One-line flip: yaw_steer_mode "A" +
                              #       body_rate_sign (1,1,1).
                              # A/B produce OPPOSITE yaw direction; roll/pitch/thrust identical between
                              # them. Physical winner = OPERATOR-EYES-resolved on the confirm-fly
                              # (offline can't see the yaw mirror). VQ1/case-A untouched (mode "off").
                              "yaw_steer_mode": "B",
                              "kp_alt": 0.0, "gate_pd_vertical": True,
                              "ff_vertical_kd_alt": 0.06,
                              "ff_vertical_vz_lp_alpha": 0.8, "kp_gate": 0.04,
                              # A33 V-2a (2026-07-03): OPEN the floor 0.15 -> 0.05. Run 20260703_210632
                              # reconciliation: at the pinned floor the drone was near-NEUTRAL vertically
                              # (realized ~-0.55 m/s^2, still rising) with ~+7 m/s^2 translational lift
                              # and LARGE unused down-authority margin -- the FLOOR, not the plant, was
                              # the binding constraint (the gate-PD's own worst-case demand is
                              # hover-0.12-0.15 ~= 0.0). 0.05 lets the bounded PD (+/-3 z_off clip, fixed
                              # gains) reach its designed descent; the collective can never go below what
                              # the bounded PD asks, so no wide-open slam surface. The A27 slew (2.0/s,
                              # RETAINED symmetric) now RAMPS into the (sub-0.15, UNMEASURED) net-down
                              # region rather than stepping to it -- FC-10 watchdog guards it. See spec
                              # §V-2 + the flagged low-collective sysid gap (§V-2d).
                              "alt_thrust_lo": 0.05, "alt_thrust_slew_per_s": 2.0},
        # A20 async-detect (2026-07-01): decouple the ~250 ms GPU-stalled YOLO detect from the
        # control loop (worker thread + latest-wins snapshot; racer.vision.async_detect). Fixes
        # the ~3 Hz loop -> ~300 ms ZOH command-hold -> one held climb command flies into the
        # ceiling at gate 0. Loop holds --rate (~30 Hz); vision lands at whatever rate the GPU
        # allows (~4 Hz busy) and the OOSM/gyro-propagation chain absorbs the ~250 ms obs age.
        async_detect=True,
        # A24 vertical-velocity washout (2026-07-02): the egress->ceiling-climb fix. The alt-hold's
        # damping rate comes from the dedicated a_up-integrating washout channel — structurally
        # bounded, smooth, and live to a real climb between vision fixes (the two failure modes of
        # the dead-reckoned z the fd path read). See DeployProfile.vertical_estimator.
        vertical_estimator=True,
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
