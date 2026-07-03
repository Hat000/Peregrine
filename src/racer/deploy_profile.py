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
        vertical_estimator_overrides={"use_zoff_filter": True, "export_clip_mps": 2.5},
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
        # A29 orbit fix (2026-07-03): LOS-rate damping kills the tangential drift that made the
        # drone yaw at gate 1 while coasting a 392-deg orbit around it (21.1 m/s of commanded
        # delta-v, net vector 2.6 m/s, coherence 0.12 -- run 20260703_024023). The lateral term
        # v_t = -r*theta_dot is a pure vision observable (tracked PnP range x filtered LOS rate),
        # so no dead-reckoned velocity re-enters the loop (A15b/A23 stand). Sub-field defaults
        # (kd=0.8, lateral cap 2.0, total cap 2.5, deadband 0.3, EMA 0.4) live on GateSeekerConfig.
        # A29 delay trim: the 115-deg gate-1 acquire turn tracked a RECEDING bearing at ~1.05 rad/s
        # realized; pursuit_yaw_slew_rps 1.0 -> 1.5 matches visual_yaw_rate_cap so the SLEW is no
        # longer the binding constraint (the cap + slew-limit still bound steps; the A3
        # roll-saturation guard was about heading STEPS, which the 1.5 slew still rate-limits).
        seeker_overrides={"egress_freeze_attitude": True, "hold_last_demand_s": 0.6,
                          "true_attitude_from_ahrs": True,
                          "use_los_rate_damping": True,
                          "pursuit_yaw_slew_rps": 1.5},
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
                              "body_rate_sign": (1.0, 1.0, 1.0),
                              "kp_alt": 0.0, "gate_pd_vertical": True,
                              "ff_vertical_kd_alt": 0.06,
                              "ff_vertical_vz_lp_alpha": 0.8, "kp_gate": 0.04,
                              "alt_thrust_lo": 0.15, "alt_thrust_slew_per_s": 2.0},
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
