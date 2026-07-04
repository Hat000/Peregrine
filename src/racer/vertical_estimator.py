"""VerticalEstimator — the 1-D vertical-velocity WASHOUT filter for the state-denied VQ2 wire
(the A24 washout fix, replacing the A21 floor-pin 3-state KF, 2026-07-02).

THE BUG THIS KILLS (run 20260702_040036, quantified via scripts/analyze_flight.py): the wire is
POSITION/VELOCITY DENIED, so the 6-state KF's z is dead-reckoning between the sparse (~a few Hz,
latent) vision ``floor_height`` pins — and each accepted pin is a TIGHT z fix, so ``est_z``
TELEPORTS (−1.435 m in ONE tick at t=1569.41) while the true motion is smooth. The controller's
ff-owns-vertical damper reads a low-passed finite difference of that z: the teleport becomes a
~−4.4 m/s vz spike -> a thrust slam -> and between pins the damper is BLIND to the real climb
(integrated a_up shows +4-5 m/s upward over 2.7 s that est_z never tracked) -> the drone climbs
over gate 0 into the ceiling. Smooth true motion vs a step-function estimate = the bug.

THE A21 FIX (superseded) tried a tiny 3-state [z, vz, bias] KF, correcting the slow drift with the
absolute floor_height pins. THIS WAS A FALSE PREMISE (2026-07-02 spec review): the warehouse has no
floor grid, so the "correction" was never a real external fix — a KF with a bias state and NO
measurements is just a washout in a Kalman costume, and the bias is fundamentally unobservable on
this wire. The vertical channel is gate-RELATIVE; absolute height is neither obtainable nor needed.

THE A24 FIX is a first-order washout (leaky integrator), NOT a Kalman filter. It integrates the
IMU-derived kinematic upward acceleration ``a_up`` (fast ~140 Hz, smooth, but drifting) with an
exponential leak (time constant ``washout_tau_s``) so the state is STRUCTURALLY bounded: it cannot
diverge regardless of the input, because a sustained bias ``b`` only pulls it to a constant offset
``tau*b``, never a ramp. A short pre-arm window captures the accelerometer/attitude-projection bias
once (grounded, at rest) so residual pitch-projection error does not masquerade as vertical motion.

    vz <- exp(-dt/tau)*vz + (a_dn_clamped - b_hat)*dt      # a_dn = -a_up (NED down-positive)

Deliberately NOT fused with any vision measurement in v1 (gate-offset-rate fusion is a documented,
default-OFF v2 hook — see the spec, not implemented here): the only candidate measurement (floor
pins) is gone, and the gate-relative offset-rate the outer loop already tracks would add only noisy,
low-frequency information at v1's confidence level.

ACCEL DECODE (VERIFIED empirically 2026-07-01, docs/accel_investigation.md + analyze_flight.py):
HIGHRES_IMU accel is SPECIFIC FORCE in body FRD, gravity-INCLUDED (on-pad |f| = 9.810 exactly),
so the kinematic upward acceleration is  a_up = -( (R_body->ned @ f)_z + g ).

Consumed by the Navigator behind ``NavigatorConfig.use_vertical_estimator`` (default OFF =
byte-identical); exported on ``NavState.vert_vz_est`` (``vert_z_est`` is permanently NaN — no
absolute-altitude state remains in this filter); the controller's ff-owns-vertical alt-hold damps
on the exported vz behind ``Controller.use_vertical_estimator``.

THE A25 FIX (this file, 2026-07-02, extends A24 -- see
handoff/vq2_gate_relative_altitude_spec_2026-07-02.md): adds a GATE-RELATIVE vertical-offset state
``z_off`` (``z-hat_off`` in the spec prose) alongside the washout ``vz``. The washout alone has no
POSITION reference -- it can null a velocity but cannot seek a gate-relative height, and the
diagnosis (run 20260702_203428) showed the drone climbing monotonically into gate 0's ceiling
because nothing ever told the loop "you are above the gate, come down". ``z_off`` is latched from
the seeker's fresh pose (``offset_z_world``, world-NED, DOWN-positive: positive = gate opening is
BELOW the drone), propagated at IMU rate by the SAME washout ``vz`` used above (dead-reckoning the
gate-relative height between poses), latency-compensated for the pose's capture-to-use age,
clamped to +/-3 m, and FROZEN (along with ``vz``) during a contact event (a specific-force spike --
see ``contact_frozen``) so ceiling-scrape impacts never masquerade as fictional descent. The
controller's alt-hold then adds a proportional term ``-kp_gate*z_off`` (NEGATIVE: above-gate,
z_off>0, must REDUCE thrust to sink) -- see ``racer.controller._ff_owns_vertical_thrust``. Default
OFF (``Controller.kp_gate=0.0``, ``z_off`` never latched until a pose arrives) -- VQ1/case-A
byte-identical.

THE A26 FIX (this file, 2026-07-02, brake fix -- see
handoff/vq2_a26_vertical_brake_2026-07-02.md): A25 flew and worked directionally (reversed the
climb, sought gate height) but OVERSHOT past gate-1 height to the floor. Root cause: the washout
``vz`` RAILS to its export clip (18-28% of ticks) -- so the damping term ``kd_alt*(vz-vz_t)``
swings at 5x the authority of the A25 position term, and thrust bang-bangs the [0.05,0.6] clamps;
and ``vz`` is BLIND to real sustained descent (reads ~0), so the smooth overshoot into the floor
was never braked. TWO changes: (1) ``export_clip_mps`` 3.0 -> 1.5 (de-saturate the damper -- the
worst-case swing no longer dwarfs the position term). (2) the gate-offset-rate fusion this file's
A24 docstring called "deliberately NOT fused" and the A25 spec's §6 explicitly DEFERRED is now
BUILT: a real gate-relative descent rate ``vz_gate``, the finite difference of two consecutive
FRESH ``offset_z_world`` latches (``latch_offset``, called by the seeker only on new poses), is
blended into the washout state (``vz <- vz + k_gate_vz_fusion*(vz_gate-vz)``, see ``_fuse_gate_vz``
for the accept/reject gates and the sign derivation) so the damping term stops reading ~0 during a
genuine descent. Gated behind ``use_gate_vz_fusion`` (default False -- VQ1/case-A byte-identical;
True only on the vq2_case_c estimator, mirroring how ``NavigatorConfig.use_vertical_estimator`` is
threaded). No gate visible -> pure washout, no mode switch (the fusion only ever nudges an already-
running washout state; it never substitutes for it).

THE A28 FIX (this file, 2026-07-03, structural -- see
handoff/vq2_a28_vertical_stability_spec_2026-07-03.md): A26 flew (run 20260703_013748) and the
vertical loop DIVERGED floor->ceiling. Diagnosis (spec §1): ``vz`` was fiction three distinct
ways -- (a) sustained SUB-threshold contact accel (~-2.4 m/s^2 while scraping the floor; zero
samples over the 20 m/s^2 contact gate) poisoned the washout to its +1.5 rail, arming ANTI-damping
thrust surges during a real climb; (b) the A26 gate-offset-rate fusion finite-differences two
noisy poses ~0.1 s apart -- accepted samples were still garbage (a standing -0.67 m/s DC fiction
while the truth was 0); (c) the washout's DC is tau*(any accel residual) by construction, so a
0.33 m/s^2 attitude-projection residual = +/-0.65 m/s of standing fictional velocity. THE FIX is
the 2-STATE COMPLEMENTARY FILTER on ``(z_off, vz_rel)`` -- the vertical marginal of the roadmap's
gate-landmark VIO: the IMU PREDICTS both states (phase lead -- the exported state is current-time,
so pose age stops eating loop phase margin) and each fresh pose CORRECTS both states via a
POSITION innovation (alpha-beta): ``z_off += alpha*innov``, ``vz_rel -= beta*innov/dt_accept``.
The beta line bleeds off exactly the sustained sub-threshold contact poison of (a) within a few
pose periods (the washout's only defense was a 2 s leak; 5 s of -2.4 m/s^2 fully corrupted it),
and the velocity correction comes from position innovations instead of differencing two noisy
poses (kills (b) at the root -- same information as the A26 fusion, correct structure). An
INNOVATION GATE (2 m) rejects gate-track jumps (observed +/-8 m offset_z steps), with a
reseed-after-4-consecutive-rejects re-lock for REAL retargets (gate handoff). NO leak while
innovations flow (the vision carries the DC); after ``blind_coast_after_s`` without an accepted
innovation the washout leak resumes -- degrading to exactly today's bounded behaviour when vision
disappears. The internal state is UNCLAMPED (the A25 +/-3 state clamp pinned at +3.00 the entire
lodged back half, destroying all information; the clamp moves to the controller consumption
point, see ``controller.Controller.gate_pd_vertical``). Gated behind ``use_zoff_filter`` (default
False -- VQ1/case-A + today's vq2 path byte-identical; True only via vq2_case_c's
``NavigatorConfig.vertical_estimator_overrides``). ``use_gate_vz_fusion`` is SUPERSEDED by the
beta innovations and is skipped whenever ``use_zoff_filter`` is on (the profile also sets it
False; the flag itself stays for byte-compat).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_G = 9.80665


def a_up_from_specific_force(f_body: np.ndarray, R_wb: np.ndarray) -> float:
    """Kinematic UPWARD acceleration (m/s^2) from a specific-force sample + the body->NED rotation.

    ``a_ned = R_wb @ f + g_ned`` with ``g_ned = (0, 0, +G)`` (NED z is DOWN), so ``a_up = -a_ned_z``.
    The matrix form of the VERIFIED analyze_flight.py decode — the Navigator has the TRUE R_wb
    (AHRS) in hand, so no euler round-trip. At rest (attitude correct) this reads ~0.
    """
    a_ned_z = float(np.asarray(R_wb, dtype=np.float64)[2] @ np.asarray(f_body, dtype=np.float64)) + _G
    return -a_ned_z


@dataclass
class VerticalEstimator:
    """First-order washout (leaky integrator) vertical-velocity channel: ``(vz, b_hat)`` state.

    Call ``seed()`` once (on the pad, at rest), then ``predict(a_up, dt)`` per IMU tick. Read
    ``vz`` (NED down +, clipped to +/-3 m/s). Un-seeded, every call no-ops and ``vz`` is NaN — the
    consumer's absent-marker. There is no altitude state: ``z`` is permanently NaN (see the ``z``
    property) — this filter deliberately carries NO absolute-height estimate.
    """

    # Washout time constant (s): vz <- exp(-dt/tau)*vz + (a_dn-b_hat)*dt. Bounds the state by
    # tau*sup|a_dn-b_hat| regardless of input -- the structural cannot-diverge guarantee. Band
    # 1.5-3.0s; 2.0 keeps the outer gate-align loop well damped (zeta~=0.62 at Kd=0.25) while a
    # sustained bias residual (tau*b) stays small.
    washout_tau_s: float = 2.0
    # Reject implausibly large predict steps (sim reset / stutter) -- mirrors LinearKF.max_dt_s.
    max_dt_s: float = 0.2
    # A_UP INPUT OUTLIER CLAMP (COMMANDER ADDITION, 2026-07-02): clamp a_dn to +/-this (m/s^2)
    # BEFORE the washout update. A real slow-lap drone cannot sustain >3g vertical; this rejects
    # contact-impact spikes AND sensor glitches at the SOURCE, not just at the output clip.
    # Empirically material on the contaminated replay: full-flight rail 81%->49%, in-band 19%->52%.
    a_up_clamp_mps2: float = 30.0
    # Export clip (m/s), applied on read: the ABSOLUTE, unconditional cannot-diverge guarantee
    # regardless of what garbage the input stream carries.
    # A26 FIX 1 (2026-07-02, de-saturate the damper): 3.0 -> 1.5. At +/-3 the damping term
    # ``ff_vertical_kd_alt*(vz-vz_t)`` reaches +/-1.0 collective (4x hover) -> a guaranteed
    # clamp-slam that outguns the A25 position term (``-kp_gate*z_off``, +/-0.065 at kp_gate=
    # 0.06/z_off_clip=3) by 5x -- the A25 position term was correct-but-outgunned (thrust
    # bang-banged the [0.05,0.6] clamps 48% of ticks on the flown run). At +/-1.5 the worst-case
    # swing is ``0.25*(1.5-(-1))=+/-0.625`` -- still strong damping authority, no longer dwarfing
    # the position term. See handoff/vq2_a26_vertical_brake_2026-07-02.md FIX 1.
    export_clip_mps: float = 1.5
    # Pre-arm bias capture window (s): the first this-many seconds after seed (grounded -- true
    # vz=0), a running mean of a_dn (post input-clamp) is captured into b_hat, then frozen.
    bias_capture_s: float = 1.0
    # Captured bias clip (m/s^2): the frozen b_hat is clamped to +/- this after capture -- a sane
    # ceiling on a real attitude-projection residual (the AHRS is cold at arm, but not THAT cold).
    bias_clip_mps2: float = 0.5

    # --- A25 gate-relative vertical offset (z_off) ---------------------------------------------
    # Clamp on the ẑ_off state (m): the gate-relative vertical offset is physically bounded on a
    # slow lap; this is the structural cannot-run-away guard mirroring ``export_clip_mps`` for vz.
    z_off_clip_m: float = 3.0

    # --- A26 FIX 2: gate-offset-rate washout fusion (§6 of the A25 spec, previously DEFERRED) --
    # The washout ``vz`` rails to its export clip (18-28% of ticks on the flown run) and is BLIND
    # to real sustained descent (reads ~0 during genuine sinking) -- it is not a brakeable velocity
    # signal. Fuse a REAL gate-relative descent rate ``vz_gate`` (finite-difference of two FRESH
    # ``offset_z_world`` latches) into the washout state so the damping term stops reading ~0 on
    # descent. Gated behind ``use_gate_vz_fusion`` (default False -- VQ1/case-A byte-identical;
    # True only on the vq2_case_c estimator, mirroring how ``use_vertical_estimator`` is threaded).
    use_gate_vz_fusion: bool = False
    # Fusion blend gain: vz <- vz + k_gate_vz_fusion*(vz_gate - vz). 0.15 (spec-fixed) -- at the
    # ~118 ms TRT pose cadence this is an effective ~0.6 rad/s complementary crossover, just above
    # the washout's own leak (tau=2.0s -> ~0.5 rad/s), so it corrects the washout's DC descent blind
    # spot without out-voting the IMU-rate integration between poses.
    k_gate_vz_fusion: float = 0.15
    # Accept window for the finite-difference dt between two FRESH latched poses (s): too small (
    # <=0.05s) and the difference is dominated by pose-timestamp jitter/noise; too large (>0.8s) and
    # the two offsets no longer describe one coherent velocity (gate re-acquire after a long gap,
    # possible gate-index handoff). Outside this window, the fused sample is REJECTED (skip fusion
    # this latch, pure washout continues unchanged).
    gate_vz_fusion_dt_min_s: float = 0.05
    gate_vz_fusion_dt_max_s: float = 0.8
    # Reject a computed vz_gate whose magnitude exceeds this (m/s) -- a slow-lap drone cannot
    # sustain faster gate-relative closure than this; larger values are a bad pose / mismatched
    # latch (e.g. a gate-index handoff retargeting z_off to a different gate) rather than real
    # motion. Mirrors the spirit of ``a_up_clamp_mps2``/``export_clip_mps``: reject at the source.
    gate_vz_fusion_reject_mps: float = 2.5
    # CONTACT-GATE (A25 §4, LOAD-BEARING): a |specific-force| spike above this (m/s^2, ~2g of
    # kinematic vertical accel a slow-lap drone cannot sustain) marks a contact tick (impact /
    # ceiling scrape). On a contact tick -- and for ``contact_hold_s`` after the LAST spike (a
    # refractory hold; real contact events are multi-tick) -- BOTH the vz washout update AND the
    # z_off propagate are SKIPPED (held), not merely clamped: a clamped-but-still-huge spike would
    # still inject a large Δvz. This is what stops corrupted post-impact accel from being read as
    # a fictional descent (the diagnosis's 8.4 s ceiling-lock).
    a_contact_mps2: float = 20.0
    contact_hold_s: float = 0.25

    # --- A28: 2-state complementary filter on (z_off, vz_rel) -- see the module docstring -------
    # Master gate. False (default) = the A24/A25/A26 washout path, byte-identical (VQ1/case-A AND
    # every pre-A28 vq2 path). True (vq2_case_c via NavigatorConfig.vertical_estimator_overrides):
    # IMU predicts both states (no leak while innovations flow, z_off UNCLAMPED), fresh poses
    # correct both states via alpha-beta position innovations, innovation-gated + reseed-on-retarget.
    use_zoff_filter: bool = False
    # alpha-beta correction gains (spec §2.1, sized for ~10 Hz fresh-pose cadence and
    # sigma_z ~= 0.3-0.5 m; retune +/-50% via offline latch-stream replay if cadence/noise differ):
    #   z_off  += alpha * innov
    #   vz_rel += -beta * innov / max(dt_since_last_accept, zoff_beta_dt_floor_s)
    # SIGN of the beta line: positive innovation => the state under-predicted z_off growth =>
    # zdot_off = -vz_rel was under-predicted => vz_rel was too POSITIVE => subtract. Pinned by
    # test_zoff_filter_sign_descending_toward_gate_drives_vz_positive (mirrors the A26 sign test).
    zoff_alpha: float = 0.4
    zoff_beta: float = 0.15
    # Floor (s) on the beta line's dt divisor: two accepts can land ~one control tick apart; an
    # unfloored divisor would turn a modest innovation into a huge velocity kick.
    zoff_beta_dt_floor_s: float = 0.15
    # Innovation gate (m): |z_meas - z_off| beyond this is a gate-track jump / bad pose (the
    # observed +/-8 m offset_z steps, e.g. 6.9 -> 15.5 in 0.4 s), REJECTED (state untouched)...
    innov_gate_m: float = 2.0
    # ...unless it persists: reseed_after CONSECUTIVE rejects = a REAL retarget (gate handoff),
    # not noise -> re-lock z_off = z_meas (vz_rel untouched -- the drone's motion didn't jump).
    reseed_after: int = 4
    # No ACCEPTED innovation for longer than this (s) => blind coast: re-apply the washout leak
    # (tau=washout_tau_s) to vz_rel so the filter degrades to exactly today's bounded behaviour
    # when vision disappears. While innovations flow there is NO leak (the vision owns the DC).
    blind_coast_after_s: float = 1.0

    # --- A32: soft (Huber) innovation weighting on the A28 correct step (2026-07-03; spec
    # handoff/vq2_a32_robust_estimation_spec_2026-07-03.md §3.2). The A28 innovation gate above is
    # a HARD 2 m cliff: run 20260703_172104 rejected 130 ticks (|innov| p90 7.3 m) and the reseed
    # teleported z_off repeatedly -- a 2.1 m innovation (possibly honest after a blind stretch)
    # contributed NOTHING while a 1.9 m one contributed fully. When ON, every latch corrects the
    # state, scaled by a Huber weight on the normalized innovation nu = |innov|/zoff_sigma_z_m:
    # full weight for nu <= zoff_huber_k, k/nu beyond (4 m -> ~0.35, 8 m -> ~0.18, never 0), and
    # further scaled by the seeker's bearing weight for the same frame (a bearing-inconsistent
    # frame's z_off is suspect for the same reason). RESEED-ON-PERSISTENCE RETAINED: the miss
    # counter increments when nu > zoff_miss_nu (~2.8 m, near-today's cliff) and reseed_after
    # consecutive misses still re-lock z_off = z_meas -- but those miss latches each still nudged
    # the state by their (small) weight, so the filter degrades gracefully INTO the reseed
    # instead of freezing then teleporting. Only meaningful under use_zoff_filter. OFF (default)
    # = the A28 binary gate, byte-identical (VQ1/case-A + today's vq2 path untouched).
    use_soft_innov_weight: bool = False
    zoff_sigma_z_m: float = 0.7    # honest pose 1-sigma incl. latency comp (median innov 0.60 m)
    zoff_huber_k: float = 2.0      # full weight for nu <= k (<=1.4 m: today's typical innovations)
    zoff_miss_nu: float = 4.0      # miss/reseed persistence threshold (2.8 m at sigma=0.7)

    # --- A33 V-1: WEIGHT-QUALIFIED RESEED (2026-07-03; spec
    # handoff/vq2_a33_gate2_intercept_spec_2026-07-03.md §V-1). The A32 reseed-on-persistence never
    # consulted the frame WEIGHT: 4 consecutive big innovations re-lock z_off = z_meas whether they
    # came from honest OR garbage frames. Run 20260703_210632 teleported z_off twice onto low-weight
    # smear/garbage frames (cmd 67 on w 0.11-0.31, cmd 164 on w 0.00-0.04) -> a -5 m fictional
    # "gate above" that armed the balloon climb. When ``zoff_reseed_min_w > 0`` the miss counter (and
    # thus the reseed) only advances on a frame whose applied weight ``w >= zoff_reseed_min_w`` -- a
    # garbage frame is NOT retarget evidence. A REAL retarget (gate handoff) arrives on well-scored
    # frames (a fresh track's first frames carry w=1.0), so an honest handoff still reseeds in
    # ``reseed_after`` frames. 0.0 (default) => any weight counts == byte-identical A32 reseed. Only
    # meaningful under ``use_soft_innov_weight``.
    zoff_reseed_min_w: float = 0.0

    # --- A35 Fix-2: MAGNITUDE-GATED TRUST FLOOR (2026-07-03; spec
    # handoff/vq2_a35_exposed_control_failures_spec_2026-07-03.md §2). Run 20260704_024434: as the
    # drone climbed ABOVE the gate the +20 deg camera lost it -> bearing_w collapsed to 0.04 -> the
    # A32 soft weight crushed the HONEST +4.7 m z_off innovation to weight 0.02 and flagged it
    # rejected -> the estimator IGNORED vision and propagated z_off to the WRONG SIGN (-3.5 m, "climb
    # more") on a drifting IMU vz for 96 ticks, so the PD held ~hover and never used V-2's open floor.
    # THE FIX (a NARROW exception -- small/noisy offsets keep the A32/A28 behaviour EXACTLY): a z_off
    # innovation that is BOTH large (|innov| >= ``zoff_big_innov_m``) AND sign-consistent for
    # ``zoff_big_sign_consec`` consecutive frames is a REAL vertical excursion, not a noisy frame:
    #   (a) FLOOR the applied correction weight to at least ``zoff_big_innov_min_w`` (the bearing
    #       distrust can no longer crush a large honest offset to ~0);
    #   (b) allow a RESEED even from low bearing weight (the explicit exception to A33 V-1's
    #       ``zoff_reseed_min_w`` block -- a large+persistent+consistent offset IS retarget-grade);
    #   (c) while distrusted-and-large, FREEZE the predict-side z_off propagate (do not drift z_off on
    #       the unfed IMU vz -- a large KNOWN offset held beats the same offset drifted to the opposite
    #       sign). See ``_big_persistent_offset`` (the predict step reads it).
    # Only meaningful under ``use_soft_innov_weight``. use_zoff_big_trust=False (default) => the A32
    # soft path unchanged, byte-identical (VQ1/case-A + every pre-A35 vq2 path). vq2_case_c: True.
    use_zoff_big_trust: bool = False
    zoff_big_innov_m: float = 2.5       # "this is a real excursion, not noise" magnitude threshold
    zoff_big_innov_min_w: float = 0.5   # floor on the applied correction weight for a big+consistent offset
    zoff_big_sign_consec: int = 3       # consecutive same-sign big frames before the exception engages

    _vz: float = field(default=float("nan"), repr=False)
    _b_hat: float = field(default=float("nan"), repr=False)
    _seeded: bool = field(default=False, repr=False)
    # Pre-arm bias-capture running state: elapsed time since seed + the running mean accumulator.
    _bias_capture_done: bool = field(default=False, repr=False)
    _bias_capture_elapsed_s: float = field(default=0.0, repr=False)
    _bias_capture_sum: float = field(default=0.0, repr=False)
    _bias_capture_n: int = field(default=0, repr=False)
    # -- A25 z_off state --
    _z_off: float = field(default=float("nan"), repr=False)
    _z_off_seen: bool = field(default=False, repr=False)
    # -- A25 contact-gate state: sim-time-like elapsed clock (seconds), monotonic via predict's dt --
    _contact_elapsed_s: float = field(default=0.0, repr=False)
    _contact_until_s: float = field(default=-1.0, repr=False)
    # -- A26 gate-offset-rate fusion state: the previous FRESH-pose offset_z_world (pre-latency-
    # comp, raw captured value) + the elapsed-clock timestamp it was latched at (same monotonic
    # clock as ``_contact_elapsed_s``, advanced by ``predict``'s dt -- NOT wall time). None until
    # the first fresh latch.
    _prev_fresh_offset_z: float | None = field(default=None, repr=False)
    _prev_fresh_offset_t_s: float | None = field(default=None, repr=False)
    # -- A28 complementary-filter state (only ever written when use_zoff_filter is True) --
    # Elapsed-clock time (same monotonic clock as _contact_elapsed_s) of the last ACCEPTED
    # innovation (or initial lock / reseed). None until the first latch -> blind coast (leak on).
    _zoff_last_accept_t_s: float | None = field(default=None, repr=False)
    # Consecutive innovation-gate rejects (the reseed counter).
    _zoff_miss: int = field(default=0, repr=False)
    # Instrumentation (spec §2.6, read by the nav-estimate logger; never fed back into the filter):
    # the last latch's innovation (m) and whether it was accepted (None until the first post-lock
    # latch; a reseed logs accepted=False -- it was gated, then force-relocked).
    _zoff_last_innov: float = field(default=float("nan"), repr=False)
    _zoff_last_accepted: bool | None = field(default=None, repr=False)
    # A32 instrumentation: the last latch's TOTAL applied weight (huber * bearing), NaN until the
    # first post-lock latch under use_soft_innov_weight. Logged as ``zoff_w``; never fed back.
    _zoff_last_w: float = field(default=float("nan"), repr=False)
    # -- A35 Fix-2 state: consecutive same-sign LARGE innovations (>= zoff_big_innov_m). Counts up
    # while the sign is stable, resets on a sign flip or a sub-threshold offset. The trust floor /
    # reseed exception / propagate-freeze engage once it reaches zoff_big_sign_consec. --
    _zoff_big_consec: int = field(default=0, repr=False)
    _zoff_big_sign: float = field(default=0.0, repr=False)
    # True while a big+persistent+consistent offset is being distrusted by bearing weight -> the
    # predict step FREEZES the z_off propagate (does not drift on the unfed IMU vz). Set in the
    # correct step, read in predict; cleared once the offset is corrected / vision recovers.
    _big_persistent_offset: bool = field(default=False, repr=False)

    # -- lifecycle ------------------------------------------------------------
    def seed(self) -> None:
        """(Re)initialise at rest: vz=0, b_hat=0, and re-arm the pre-arm bias-capture window.
        Also resets the A25 z_off state (un-seen, NaN), the contact-gate clock, and the A26
        gate-offset-rate fusion history -- a re-seed (sim epoch restart) must forget any latched
        gate-relative offset (and its fusion history) from the prior epoch."""
        self._vz = 0.0
        self._b_hat = 0.0
        self._seeded = True
        self._bias_capture_done = False
        self._bias_capture_elapsed_s = 0.0
        self._bias_capture_sum = 0.0
        self._bias_capture_n = 0
        self._z_off = float("nan")
        self._z_off_seen = False
        self._contact_elapsed_s = 0.0
        self._contact_until_s = -1.0
        self._prev_fresh_offset_z = None
        self._prev_fresh_offset_t_s = None
        self._zoff_last_accept_t_s = None
        self._zoff_miss = 0
        self._zoff_last_innov = float("nan")
        self._zoff_last_accepted = None
        self._zoff_last_w = float("nan")
        self._zoff_big_consec = 0
        self._zoff_big_sign = 0.0
        self._big_persistent_offset = False

    @property
    def seeded(self) -> bool:
        return self._seeded

    @property
    def z(self) -> float:
        """Permanently NaN -- this filter carries NO absolute-altitude state (A24 washout rip-out
        of the floor-pin z-correction path; the vertical channel is gate-relative only)."""
        return float("nan")

    @property
    def z_off(self) -> float:
        """Gate-relative vertical offset ẑ_off (m, NED down-positive: drone→gate; positive = drone
        ABOVE the gate). NaN until the first :meth:`latch_offset` call (no gate ever seen) -- the
        consumer's absent-marker, mirrored exactly on ``NavState.z_off_est`` (A25).

        A28 (``use_zoff_filter``): exported UNCLAMPED -- the A25 +/-3 read clip destroyed all
        back-half information when the true offset exceeded it (pinned at +3.00 the entire lodged
        phase of run 20260703_013748). The clamp moves to the CONTROLLER consumption point
        (``Controller.gate_pd_vertical``: ``clip(z_off, +/-3)`` inside the PD law), so the
        estimate stays honest while the control authority stays bounded. Legacy (flag OFF) keeps
        the A25 clip -- byte-identical."""
        if not self._z_off_seen:
            return float("nan")
        if self.use_zoff_filter:
            return float(self._z_off)
        return float(np.clip(self._z_off, -self.z_off_clip_m, self.z_off_clip_m))

    @property
    def zoff_miss(self) -> int:
        """A28 instrumentation: consecutive innovation-gate rejects (the reseed counter)."""
        return self._zoff_miss

    @property
    def zoff_last_innov(self) -> float:
        """A28 instrumentation: the last latch's innovation (m); NaN until the first post-lock
        latch under ``use_zoff_filter``."""
        return self._zoff_last_innov

    @property
    def zoff_last_accepted(self) -> bool | None:
        """A28 instrumentation: whether the last latch's innovation was accepted (None until the
        first post-lock latch under ``use_zoff_filter``)."""
        return self._zoff_last_accepted

    @property
    def zoff_last_w(self) -> float:
        """A32 instrumentation: the last latch's TOTAL applied correction weight (huber * bearing);
        NaN until the first post-lock latch under ``use_soft_innov_weight``."""
        return self._zoff_last_w

    def contact_frozen(self) -> bool:
        """True while a contact event (or its refractory hold) is active -- the vz washout update
        AND the z_off propagate must both be SKIPPED this tick (A25 §4)."""
        return self._contact_elapsed_s < self._contact_until_s

    @property
    def vz(self) -> float:
        """Estimated NED world-down velocity zdot (m/s), clipped to +/-``export_clip_mps``; NaN
        until seeded. The structural cannot-diverge guarantee: absolute, regardless of input."""
        if not self._seeded:
            return float("nan")
        return float(np.clip(self._vz, -self.export_clip_mps, self.export_clip_mps))

    # -- per-IMU-tick propagation ----------------------------------------------
    def predict(self, a_up: float, dt: float) -> None:
        """Integrate one accel sample with an exponential leak: ``a_dn = -a_up`` (a_up UP-positive,
        NED down-positive), clamped to +/-``a_up_clamp_mps2`` BEFORE the update (rejects contact-
        impact spikes / sensor glitches at the source). During the pre-arm bias-capture window
        (the first ``bias_capture_s`` seconds after seed), accumulate a running mean of the
        clamped a_dn into ``b_hat`` instead of advancing vz (grounded -- true vz is 0); once the
        window closes, b_hat freezes (clipped to +/-``bias_clip_mps2``) and the washout recurrence
        runs. ``dt <= 0`` (a between-IMU control tick) and ``dt > max_dt_s`` (sim reset / stutter --
        integrating a garbage step corrupts vz worse than skipping one) are no-ops, mirroring
        LinearKF.

        A25: AFTER the vz washout update (same tick, same dt), propagate the gate-relative offset
        ``z_off`` by the just-updated vz: ``z_off <- z_off - vz*dt`` (vz is NED down-positive; a
        real CLIMB reads vz<0 in that convention, so ``-vz*dt`` is positive on a climb -- the gate
        gets further below as the drone rises, growing z_off MORE positive; see the spec §1.3.1 and
        the sign-pinning test ``test_propagate_sign_climb_grows_z_off_positive``). Also runs the
        CONTACT-GATE (§4): a |specific-force| spike (``|a_dn_raw - b_hat| > a_contact_mps2``) marks
        this tick (and the ``contact_hold_s`` refractory window after it) as CONTACT-FROZEN -- on
        such a tick BOTH the vz washout update and the z_off propagate are SKIPPED entirely (held),
        never merely clamped, so corrupted post-impact accel can never read as fictional descent."""
        if not self._seeded or not (0.0 < dt <= self.max_dt_s):
            return
        self._contact_elapsed_s += dt
        a_dn_raw = -float(a_up)
        a_dn = float(np.clip(a_dn_raw, -self.a_up_clamp_mps2, self.a_up_clamp_mps2))

        if not self._bias_capture_done:
            self._bias_capture_sum += a_dn
            self._bias_capture_n += 1
            self._bias_capture_elapsed_s += dt
            if self._bias_capture_elapsed_s >= self.bias_capture_s:
                mean_a_dn = self._bias_capture_sum / max(self._bias_capture_n, 1)
                self._b_hat = float(np.clip(mean_a_dn, -self.bias_clip_mps2, self.bias_clip_mps2))
                self._bias_capture_done = True
            return   # grounded during capture: vz stays 0, no washout advance yet, no contact gate

        # CONTACT-GATE (A25 §4): detect on the RAW (unclamped) specific-force-derived accel vs the
        # frozen bias -- a clamped 30 m/s^2 spike still corrupts the integrator, so the detector
        # must see the true magnitude before the a_up_clamp_mps2 clip is applied.
        if abs(a_dn_raw - self._b_hat) > self.a_contact_mps2:
            self._contact_until_s = self._contact_elapsed_s + self.contact_hold_s
        if self.contact_frozen():
            return   # HOLD: skip both the vz washout update and the z_off propagate this tick

        if self.use_zoff_filter:
            # A28 complementary-filter PREDICT: pure integration (NO leak) while accepted
            # innovations are flowing -- the beta corrections own the DC, so leaking here would
            # fight them (and the leak's tau*b standing fiction is exactly failure (c) of the
            # diagnosis). Once blind for > blind_coast_after_s (or never latched), the washout
            # leak resumes and the filter degrades to exactly today's bounded behaviour.
            if self._zoff_meas_fresh():
                self._vz = self._vz + (a_dn - self._b_hat) * dt
            else:
                leak = float(np.exp(-dt / self.washout_tau_s))
                self._vz = leak * self._vz + (a_dn - self._b_hat) * dt
            # z_off propagate, UNCLAMPED (same sign/lock-step discipline as the A25 line below;
            # the clamp lives at the controller consumption point now -- see the z_off property).
            # A35 Fix-2 (c): FREEZE the propagate while a large offset is being distrusted by bearing
            # weight -- drifting z_off on the unfed IMU vz is what drove it to the WRONG SIGN (-3.5 m)
            # on run 20260704_024434. A large KNOWN offset held beats the same offset drifted to the
            # opposite sign. (_big_persistent_offset is only ever True under use_zoff_big_trust; the
            # default path never freezes -> byte-identical.)
            if self._z_off_seen and not self._big_persistent_offset:
                self._z_off = float(self._z_off - self._vz * dt)
            return

        alpha = float(np.exp(-dt / self.washout_tau_s))
        self._vz = alpha * self._vz + (a_dn - self._b_hat) * dt

        # A25 z_off propagate: dead-reckon the gate-relative height by the just-updated vz, using
        # the SAME (post-update) vz and the SAME dt, so z_off and vz stay in lock-step. Only once a
        # gate has ever been latched (else z_off has no reference to propagate).
        if self._z_off_seen:
            self._z_off = float(np.clip(self._z_off - self._vz * dt,
                                        -self.z_off_clip_m, self.z_off_clip_m))

    def _zoff_meas_fresh(self) -> bool:
        """A28: True while the innovation stream is FRESH -- an accepted innovation (or initial
        lock / reseed) landed within ``blind_coast_after_s`` of the current elapsed clock. Governs
        the predict-side leak (fresh => no leak, the vision owns the DC; stale/never => washout
        leak, today's bounded blind-coast behaviour)."""
        return (self._zoff_last_accept_t_s is not None
                and (self._contact_elapsed_s - self._zoff_last_accept_t_s)
                <= self.blind_coast_after_s)

    # -- A25 gate-relative latch -----------------------------------------------
    def latch_offset(self, offset_z_world: float, obs_age_s: float,
                     weight: float = 1.0) -> None:
        """Latch a FRESH gate-relative vertical-offset measurement (called by the seeker ONLY on a
        newly-detected pose, never a re-used cached one -- see the spec §1.4).

        Latency-compensates for the pose's capture-to-use age: over ``obs_age_s`` the drone moved
        vertically by ``-vz*obs_age_s`` (up-positive metres, since vz is down-positive and a climb
        reads vz<0), so the gate is now that much further below (or above) than the CAPTURED
        ``offset_z_world`` says:

            z_off_latched = offset_z_world_captured - vz * obs_age_s

        (spec §2.3, sign-pinned by ``test_latency_comp_climb_makes_latched_offset_more_positive``).
        Uses the CURRENT washout ``vz`` as a first-order proxy for the (unobserved) velocity during
        the blind interval -- acceptable over the sub-1s obs ages on this wire; no full OOSM buffer.
        No-op while un-seeded (mirrors ``predict``).

        A26 FIX 2 (gate-offset-rate washout fusion, spec §6, previously DEFERRED): when
        ``use_gate_vz_fusion`` is on, ALSO derive a real gate-relative descent rate ``vz_gate``
        from the finite difference of this fresh RAW (pre-latency-comp) ``offset_z_world`` against
        the previous fresh RAW offset, and blend it into the washout ``vz`` -- see
        :meth:`_fuse_gate_vz` for the sign derivation and the accept/reject gates. This runs BEFORE
        the z_off latch above is overwritten (so the "previous" offset is still the prior one), and
        the previous-offset bookkeeping is updated unconditionally on every fresh latch (whether or
        not this particular sample passed the fusion gates) so the NEXT latch always compares
        against the immediately-prior fresh pose.

        A28 (``use_zoff_filter``): the latch becomes the complementary filter's CORRECT step --
        see :meth:`_zoff_filter_correct`. The A26 fusion is SUPERSEDED by the beta innovations and
        skipped entirely on this path (differencing two noisy poses ~0.1 s apart was diagnosis
        failure (b)); the latency comp is shared and unchanged.

        A32 ``weight`` (default 1.0 == byte-identical): the seeker's soft bearing weight for this
        frame, consumed ONLY by the ``use_soft_innov_weight`` correct step (a bearing-inconsistent
        frame's z_off measurement is suspect for the same reason); every other path ignores it."""
        if not self._seeded:
            return
        if self.use_zoff_filter:
            self._zoff_filter_correct(float(offset_z_world), float(obs_age_s), float(weight))
            return
        if self.use_gate_vz_fusion:
            self._fuse_gate_vz(float(offset_z_world))
        vz_for_latency = self._vz if np.isfinite(self._vz) else 0.0
        self._z_off = float(np.clip(
            float(offset_z_world) - vz_for_latency * float(obs_age_s),
            -self.z_off_clip_m, self.z_off_clip_m))
        self._z_off_seen = True

    # -- A28 complementary-filter correct ----------------------------------------
    def _zoff_filter_correct(self, offset_z_world: float, obs_age_s: float,
                             weight: float = 1.0) -> None:
        """One fresh-pose CORRECT step of the A28 (z_off, vz_rel) complementary filter.

        ``z_meas`` is the latency-compensated measurement (same comp as the legacy latch: over
        ``obs_age_s`` the drone moved ``-vz*obs_age_s``, so the captured offset is adjusted to
        now-time). Then:

          * FIRST-EVER latch: initial lock -- ``z_off = z_meas`` (there is no prior state to
            innovate against), start the accept clock.
          * ``|innov| <= innov_gate_m``: alpha-beta correct BOTH states::

                z_off  += zoff_alpha * innov
                vz_rel += -zoff_beta * innov / max(dt_since_last_accept, zoff_beta_dt_floor_s)

            (beta SIGN: positive innov => z_off grew faster than the state predicted => the
            propagate zdot_off = -vz_rel under-predicted that growth => vz_rel too positive =>
            subtract. Equivalently: DESCENDING toward the gate makes the measured offsets shrink
            faster than predicted => negative innov => vz_rel is pushed MORE POSITIVE, matching
            the down-positive "descending = +" convention. Pinned by
            ``test_zoff_filter_sign_descending_toward_gate_drives_vz_positive``.)
          * ``|innov| > innov_gate_m``: REJECT (state untouched -- a gate-track jump / bad pose),
            bump the consecutive-miss counter; at ``reseed_after`` consecutive rejects this is a
            REAL retarget (gate handoff), so RE-LOCK ``z_off = z_meas`` with ``vz_rel`` untouched
            (the drone's physical motion did not jump with the track).

        The state is UNCLAMPED throughout (see the ``z_off`` property).

        A32 (``use_soft_innov_weight``, spec §3.2): the binary gate above becomes a HUBER-WEIGHTED
        correction -- EVERY latch corrects the state, scaled by ``w = min(1, k/nu)`` on the
        normalized innovation ``nu = |innov|/zoff_sigma_z_m`` and by the seeker's bearing
        ``weight`` for the same frame; the miss counter (and the retained reseed-on-persistence)
        keys on ``nu > zoff_miss_nu`` instead of the 2 m cliff. A 2.1 m innovation after a blind
        stretch now contributes ~65% weight instead of 0; a gross 8 m gate-track jump contributes
        ~18% per latch while the reseed persistence machinery decides whether it is a REAL
        retarget -- the filter degrades gracefully INTO the reseed instead of freezing then
        teleporting."""
        vz_for_latency = self._vz if np.isfinite(self._vz) else 0.0
        z_meas = float(offset_z_world) - vz_for_latency * float(obs_age_s)
        now_t = self._contact_elapsed_s
        if not self._z_off_seen:
            self._z_off = z_meas                       # initial lock: nothing to innovate against
            self._z_off_seen = True
            self._zoff_last_accept_t_s = now_t
            self._zoff_miss = 0
            return
        innov = z_meas - float(self._z_off)
        self._zoff_last_innov = innov
        if self.use_soft_innov_weight:
            self._zoff_soft_correct(innov, z_meas, now_t, weight)
            return
        if abs(innov) > self.innov_gate_m:
            self._zoff_last_accepted = False
            self._zoff_miss += 1
            if self._zoff_miss >= self.reseed_after:   # persistent => a REAL retarget: re-lock
                self._z_off = z_meas                   # vz_rel untouched (motion didn't jump)
                self._zoff_miss = 0
                self._zoff_last_accept_t_s = now_t     # measurements flow again: leak stays off
            return
        self._zoff_last_accepted = True
        dt_accept = (now_t - self._zoff_last_accept_t_s
                     if self._zoff_last_accept_t_s is not None else self.zoff_beta_dt_floor_s)
        dt_div = max(float(dt_accept), self.zoff_beta_dt_floor_s)
        self._z_off = float(self._z_off + self.zoff_alpha * innov)
        if np.isfinite(self._vz):
            self._vz = float(self._vz - self.zoff_beta * innov / dt_div)
        self._zoff_miss = 0
        self._zoff_last_accept_t_s = now_t

    def _zoff_soft_correct(self, innov: float, z_meas: float, now_t: float,
                           weight: float) -> None:
        """A32 Huber-weighted alpha-beta correct (see :meth:`_zoff_filter_correct`). The applied
        correction is ALWAYS non-zero (w > 0 everywhere -- soft-weight, never hard-reject); the
        accept clock (which governs the predict-side leak) and the miss counter (which governs
        the retained reseed) key on the persistence threshold ``zoff_miss_nu``."""
        nu = abs(innov) / max(self.zoff_sigma_z_m, 1e-6)
        w_huber = 1.0 if nu <= self.zoff_huber_k else self.zoff_huber_k / nu
        w = float(w_huber) * float(np.clip(weight, 0.0, 1.0))

        # --- A35 Fix-2: MAGNITUDE-GATED TRUST FLOOR ---------------------------------------------
        # Track consecutive SAME-SIGN LARGE innovations. A z_off innovation that is both large and
        # sign-consistent for a few frames is a REAL vertical excursion the loop MUST act on --
        # NOT a noisy frame the bearing weight should crush. (Small/noisy offsets skip this entirely
        # and keep the A32 soft behaviour byte-for-byte.)
        big_persistent = False
        if self.use_zoff_big_trust and abs(innov) >= self.zoff_big_innov_m:
            s = float(np.sign(innov))
            if s == self._zoff_big_sign and s != 0.0:
                self._zoff_big_consec += 1
            else:
                self._zoff_big_sign = s
                self._zoff_big_consec = 1
            big_persistent = self._zoff_big_consec >= self.zoff_big_sign_consec
        elif self.use_zoff_big_trust:
            # sub-threshold offset: the excursion is (being) corrected -> reset the streak + freeze.
            self._zoff_big_consec = 0
            self._zoff_big_sign = 0.0
            self._big_persistent_offset = False

        if big_persistent:
            # (a) FLOOR the applied correction weight so bearing distrust cannot crush a large honest
            # offset to ~0 -- the loop corrects toward the true offset (kills the wrong-sign drift).
            w = max(w, float(self.zoff_big_innov_min_w))
            # (c) mark distrusted-and-large so the predict step FREEZES the z_off propagate (no drift
            # on the unfed IMU vz). Only when the frame is ACTUALLY distrusted (bearing weight low);
            # if vision is trusted the normal correction handles it and no freeze is needed.
            self._big_persistent_offset = (float(np.clip(weight, 0.0, 1.0))
                                           < float(self.zoff_big_innov_min_w))
        self._zoff_last_w = w
        # weighted correction: every measurement contributes, scaled by its consistency.
        dt_accept = (now_t - self._zoff_last_accept_t_s
                     if self._zoff_last_accept_t_s is not None else self.zoff_beta_dt_floor_s)
        dt_div = max(float(dt_accept), self.zoff_beta_dt_floor_s)
        self._z_off = float(self._z_off + self.zoff_alpha * w * innov)
        if np.isfinite(self._vz):
            self._vz = float(self._vz - self.zoff_beta * w * innov / dt_div)
        if nu > self.zoff_miss_nu:
            # gross outlier: nudged above, but counted toward the RETAINED reseed persistence
            # (a gate handoff still needs the re-lock; the accept/leak clock is NOT refreshed --
            # a suspect stream must not keep the blind-coast leak off forever).
            self._zoff_last_accepted = False
            # A33 V-1: a garbage frame is NOT retarget evidence -- only advance the miss/reseed
            # counter when this frame's applied weight is credible. zoff_reseed_min_w=0 (default) =>
            # any weight counts == byte-identical A32. A real handoff arrives on w=1.0 frames.
            # A35 Fix-2 (b): a large+persistent+sign-consistent offset IS retarget-grade even from
            # LOW bearing weight -- the explicit exception to the V-1 block (that is exactly the
            # +4.7 m sustained excursion that must re-lock, which V-1 alone would strand).
            if w >= self.zoff_reseed_min_w or big_persistent:
                self._zoff_miss += 1
                if self._zoff_miss >= self.reseed_after:
                    self._z_off = z_meas               # REAL retarget: re-lock (vz_rel untouched)
                    self._zoff_miss = 0
                    self._zoff_last_accept_t_s = now_t
                    self._big_persistent_offset = False   # re-locked -> excursion resolved
            return
        self._zoff_last_accepted = True
        self._zoff_miss = 0
        self._zoff_last_accept_t_s = now_t

    # -- A26 gate-offset-rate fusion --------------------------------------------
    def _fuse_gate_vz(self, offset_z_world: float) -> None:
        """Derive ``vz_gate`` from two consecutive FRESH ``offset_z_world`` latches and blend it
        into the washout ``vz``.

        SIGN DERIVATION (must match ``vz``'s NED down-positive, "drone descending = positive"
        convention): ``offset_z_world`` is down-positive drone->gate (positive = gate is BELOW the
        drone). As the drone DESCENDS TOWARD a static gate, the drone gets closer to the gate, so
        the below-positive distance offset_z_world SHRINKS (decreases) over time --
        ``offset_z_now < offset_z_prev``. We need ``vz_gate > 0`` on that same descent, so:

            vz_gate = (offset_z_prev - offset_z_now) / dt          # NOT (now - prev)/dt

        (descending -> offset_z shrinks -> prev>now -> vz_gate>0 ✓). This is exactly the same
        relationship as the existing ``predict`` propagate (``z_off <- z_off - vz*dt``, i.e.
        ``vz = (z_off_old - z_off_new)/dt``) -- same sign convention, same filter. Pinned by
        ``test_vz_gate_sign_descending_toward_gate_is_positive``.

        Gates (reject -> leave ``_vz`` untouched, pure washout unchanged this latch):
          * no previous fresh latch yet (first-ever latch has nothing to difference against).
          * dt outside ``(gate_vz_fusion_dt_min_s, gate_vz_fusion_dt_max_s]`` -- too small is
            dominated by pose-timestamp jitter, too large no longer describes one coherent
            velocity (long blind gap / gate handoff).
          * ``|vz_gate| > gate_vz_fusion_reject_mps`` -- a slow-lap drone cannot sustain faster
            gate-relative closure; larger values are a bad pose / mismatched latch, not real motion.
          * NOTE: the spec also calls for a point-blank (<2 m range) reject. Range is NOT available
            at this layer -- ``VerticalEstimator`` only ever sees ``offset_z_world`` + ``obs_age_s``
            (see ``latch_offset``'s signature and the module docstring's §1.2 rationale: the
            estimator is deliberately kept pose/range-agnostic, that guard lives at the SEEKER,
            which already suppresses the z_off LATCH itself below ``min_trust_elevation_range_m``
            -- see ``gate_seeker._maybe_latch_z_off`` §5.2). Since a point-blank pose never reaches
            ``latch_offset`` in the first place below that range, the fusion input is already
            range-guarded one layer up; this method does not duplicate a range check it cannot see.

        On acceptance: blend ``vz <- vz + k_gate_vz_fusion*(vz_gate - vz)`` (a first-order
        complementary step toward the measured rate, NOT a hard overwrite -- keeps the IMU-rate
        washout as the primary signal between poses)."""
        prev_z = self._prev_fresh_offset_z
        prev_t = self._prev_fresh_offset_t_s
        now_t = self._contact_elapsed_s
        # unconditionally roll the "previous fresh" bookkeeping forward so the NEXT latch compares
        # against THIS one, regardless of whether this sample passes the fusion gates below.
        self._prev_fresh_offset_z = offset_z_world
        self._prev_fresh_offset_t_s = now_t

        if prev_z is None or prev_t is None:
            return   # first-ever fresh latch: nothing to difference against yet
        dt = now_t - prev_t
        if not (self.gate_vz_fusion_dt_min_s < dt <= self.gate_vz_fusion_dt_max_s):
            return   # outside the accept window: reject this sample, pure washout continues
        vz_gate = (prev_z - offset_z_world) / dt
        if abs(vz_gate) > self.gate_vz_fusion_reject_mps:
            return   # implausible gate-relative rate: reject, do not corrupt the washout
        if np.isfinite(self._vz):
            self._vz = float(self._vz + self.k_gate_vz_fusion * (vz_gate - self._vz))
