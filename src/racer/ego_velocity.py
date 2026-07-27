"""Vision-referenced LATERAL velocity correction for the egocentric deploy obs (D1).

WHY
---
``EgoObsBuilder`` feeds obs[0:3] from ``nav_state.velocity_ned`` -- the deploy KF's world
velocity. On the VQ2 wire that KF has NO position reference of any kind: there is no GPS, no
mag, no baro, and (measured over 94,991 ticks of 624 logged flights) the vision channel never
once produced a world fix -- ``time_since_vision_update_s`` is non-finite on 100% of ticks. So
obs[0:3] is pure IMU strapdown dead reckoning whose error is free to wander.

Training's velocity, by contrast, IS corrected by vision. v1.9/v2.0 both run the stage
``dual_gate_fullstack_floor_pef16`` (rl/launch_v19.sh:123, rl/launch_v20.sh:130), whose stage
dict sets ``ego_faithful=True`` (rl/vq2_ego_curriculum.py) plus the ``++dynamics.
capture_specific_force=true`` / ``n_substeps=5`` the faithful path requires. It sets no
per-channel override, so ``ego_vel_model`` takes its faithful default ``'kf'``
(rl/peregrine_racing_ego.py:1353) -- the translated deploy KF, NOT the ``'legacy'`` truth-pull
surrogate. On that path ``rl/ego_estimator.py:777-806`` folds a stream of vision POSITION fixes
into the KF (``z_datum = gate_pos_datum - R_datum @ fix_body``) and the velocity is corrected
through the pos/vel cross-covariance.

THE GAP IS THEREFORE NOT "truth vs dead reckoning" -- it is a CORRECTED KF vs an UNCORRECTED
one. And the reason the wire cannot form that fix is structural: ``z_datum`` needs
``gate_pos_datum``, i.e. a MAP, and the deployed profile is self-localizing/map-free
(fly_rl.py builds ``gates = []``), so no world position fix is ever constructed and the KF
runs open loop for the whole flight. This module supplies the map-FREE analog of the
correction training gets: a gate-RELATIVE displacement measurement, which needs no map.

(Superseded 2026-07-27: an earlier revision of this docstring claimed training ran
``vel_model='legacy'`` with a ~0.01 m/s error. That was wrong -- it was grepped from the
launchers, which never mention ``ego_faithful`` because the STAGE arms it. The training-side
error magnitude is NOT re-derivable from source; read the ``kf_vel_err_mean`` scalar that
peregrine_racing_ego.py:2393-2397 logs on every ``vel_model=='kf'`` run.)

WHAT THIS DOES
--------------
The tracked gate is a world-FIXED landmark, so the body-frame lever obeys

    r(t) = R(t<-ta) r(ta) - INT_{ta}^{t} R(t<-s) v_body(s) ds

with R(.) coming from the gyro, which IS trustworthy. Rearranged, the drone's own displacement
over a baseline is directly OBSERVED by vision:

    D_vis = R(t<-ta) r(ta) - r(t)            (anchor lever, gyro-rotated, minus the current fix)

Accumulating the dead-reckoned displacement over the SAME window in the SAME (current) body
frame, ``D_dr <- dR @ D_dr + v_dr*dt``, the mean DR velocity error over the window is

    b_meas = (D_vis - D_dr) / T

which is world-referenced (it is anchored to a fixed landmark), not dead-reckoned. A slow
first-order tracker on ``b_meas`` becomes the correction added to obs[0:3].

OBSERVABILITY -- what this can and cannot see
---------------------------------------------
Only the component of ``b_meas`` PERPENDICULAR to the line of sight is trustworthy. The
perpendicular part is (range x bearing change); the parallel part is a difference of two
monocular RANGES. Measured over 18,700 fix triples in the corpus (second difference of the
de-rotated lever, which annihilates constant velocity):

    LOS-perpendicular lever noise   median 0.036 m   (5.1 mrad -- ~1.6 px at fx=320)
    LOS-parallel (range) noise      median 0.089 m, p90 0.360 m   -- 2.5x to 10x worse

so this module estimates and applies ONLY the LOS-perpendicular part, in both directions:
the innovation is projected before it is absorbed, and the stored bias is projected again
before it is applied. Consequences, stated plainly:

  * lateral velocity with the gate near boresight (the racing case) is WELL observed;
  * velocity ALONG the line of sight (closing speed with the gate ahead) is NOT observed
    here at all and is left entirely to the KF -- obs[0] is essentially untouched on a
    head-on approach;
  * with the gate at a large off-boresight bearing the roles swap and the drone's LEFT
    velocity becomes the poorly observed one; the projection degrades gracefully (it simply
    stops correcting that axis) instead of injecting range noise into it;
  * it is also structurally immune to the range channel's known contaminants (the track EMA,
    the bbox-range fallback, ``--seeker-propagate-range``): those move the estimate along the
    LOS, which is exactly what gets projected out.

DEGENERATE CASES (all explicit, none can emit NaN/Inf)
  no track / cold slot        -> no anchor, no update; the bias holds and keeps being applied
  gate advance                -> anchor dropped (the new gate is a different landmark); the
                                 bias is KEPT (it is a property of the IMU, not of the gate)
  coasted / stale track       -> the builder only calls ``on_fix`` on a REAL accepted fix, so a
                                 coasted tick simply does not update
  close-in blackout           -> ``hold_s`` of frozen bias (the error is slowly varying:
                                 measured lag-1..lag-4 autocorrelation +0.58/+0.50/+0.43/+0.38
                                 on 0.5 s windows), then an exponential decay back to pure DR
  gate straight ahead         -> the GOOD case for lateral (see above)
  bad range / wild baseline / oversized tick gap / non-finite input -> update skipped, anchor
                                 dropped; ``correct()`` returns the input velocity unchanged

DEFAULT-OFF: ``gain=0.0`` disables the tracker completely and ``EgoObsBuilder`` never even
constructs it, so the deploy obs is byte-identical unless the pilot passes --ego-vel-fuse.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class VelocityFusionConfig:
    """All knobs. ``gain <= 0`` == OFF (the builder does not even construct the fuser)."""
    gain: float = 0.0            # first-order tracker gain per accepted update. ~0.15 == a ~4-update
                                 # (~2 s) averaging window, matched to the measured ~2 s correlation
                                 # time of the DR lateral error.
    min_baseline_s: float = 0.35  # shorter windows are noise-dominated (noise ~ sigma_perp / T)
    max_baseline_s: float = 1.00  # longer windows outrun the error's correlation time; also re-anchor
    min_range_m: float = 1.5     # inside this the gate fills the frame; the lever is unreliable
    max_range_m: float = 25.0    # beyond the seeker's 22 m acquire cap the fit is not trustworthy
    max_tick_dt_s: float = 0.30  # a bigger control-tick gap breaks the gyro/DR accumulation
    innov_clip_mps: float = 2.0  # robust clip on the per-update innovation (heavy-tailed PnP outliers)
    max_bias_mps: float = 2.0    # hard clamp on the emitted correction
    max_fix_step_m: float = 1.5  # TRACK-DISCONTINUITY gate. The seeker can re-lock onto a DIFFERENT
                                 # gate while RACE_STATUS still reports the old index (the pass-drop /
                                 # re-acquire seam), which teleports the lever and reads as a huge
                                 # velocity. Measured: 1.57% of same-gate consecutive fix pairs move
                                 # >3 m more than any plausible velocity explains (p99 4.4 m, max 29 m).
                                 #
                                 # The gate is a DISTANCE, not a speed, and that choice is measured, not
                                 # stylistic. Over 43,885 pairs, leak (teleports missed) / reject (all
                                 # pairs dropped):
                                 #     |d|/dt > 20 m/s      1.45% / 9.76%
                                 #     |d|    > 1.5 m       0.14% / 3.72%   <-- dominates on BOTH axes
                                 #     |d| > 20*dt + 1.0 m  4.78% / 2.58%
                                 # Dividing by dt dilutes a teleport that lands across a long detection
                                 # gap, while its DISTANCE stays large -- so the speed form leaks ~10x
                                 # more while rejecting 2.6x more good data. 1.5 m is also ~the physical
                                 # ceiling on honest inter-fix motion: the corpus p99 speed (14.2 m/s)
                                 # over the p99 fix gap (124 ms) is 1.76 m. Sweep: 1.0 m -> 0.00%/7.59%,
                                 # 1.5 -> 0.14%/3.72%, 2.0 -> 0.60%/2.50%, 3.0 -> 11.3%/1.49%.
                                 # (Form credit: the v21-release-dive session; thresholds measured here.)
                                 #
                                 # Judged on VISION+GYRO only -- never on the KF velocity under test --
                                 # so it cannot select for windows where dead reckoning happens to agree.
    hold_s: float = 0.8          # blackout: hold the bias frozen this long after the last fix ...
    decay_tau_s: float = 1.0     # ... then decay it back to zero (fall back to the raw KF velocity)


def _rot_body(w_flu: np.ndarray, dt: float) -> np.ndarray:
    """exp([-w*dt]x): how a WORLD-FIXED vector's body-frame coordinates change over dt -- the SAME
    convention ``EgoObsBuilder`` propagates the held lever with (``Rotation.from_rotvec(-w*dt)``)."""
    th = float(np.linalg.norm(w_flu) * dt)
    if th < 1e-12:
        return np.eye(3)
    k = -np.asarray(w_flu, dtype=np.float64) * dt / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


class LateralVelocityFuser:
    """Tracks the DR velocity error in the LOS-perpendicular subspace (see module docstring).

    Per control tick the owner calls, in order:
        ``propagate(w_flu, v_flu, dt)``      -- always
        ``on_fix(rel_flu)``                  -- only on an ACCEPTED fresh fix
        ``correct(v_flu)``                   -- returns the velocity to put in the obs
    All vectors are TRUE (UNflipped) body FLU [fwd, left, up]."""

    def __init__(self, config: VelocityFusionConfig | None = None):
        self.cfg = config or VelocityFusionConfig()
        self.reset()

    # -- state ------------------------------------------------------------------------------
    def reset(self) -> None:
        """Full reset (new flight)."""
        self.bias = np.zeros(3)
        self._anchor: np.ndarray | None = None   # anchor lever, rotated into the CURRENT body frame
        self._disp = np.zeros(3)                 # DR displacement over the window, current body frame
        self._baseline_s = 0.0
        self._prev_fix: np.ndarray | None = None  # previous fix, rotated into the CURRENT body frame
        self._prev_fix_dt = 0.0                  # time since that fix (track-continuity gate)
        self.n_jumps = 0                         # track discontinuities rejected
        self._age_s = 0.0                        # time since the last accepted fix
        self._u_last: np.ndarray | None = None   # last line-of-sight unit vector (body FLU)
        self.n_updates = 0
        self.n_rejected = 0
        self.last_meas: np.ndarray | None = None

    def drop_anchor(self) -> None:
        """Forget the current measurement window (keeps the bias)."""
        self._anchor = None
        self._disp = np.zeros(3)
        self._baseline_s = 0.0
        self._prev_fix = None
        self._prev_fix_dt = 0.0

    def on_gate_change(self) -> None:
        """Active gate advanced: the landmark changed, so the window is void. The BIAS is kept --
        it describes the IMU, not the gate."""
        self.drop_anchor()

    # -- per control tick -------------------------------------------------------------------
    def propagate(self, w_flu, v_flu, dt: float) -> None:
        """Roll the bias, the anchor and the DR displacement forward one control tick."""
        w = np.asarray(w_flu, dtype=np.float64).reshape(3)
        v = np.asarray(v_flu, dtype=np.float64).reshape(3)
        if not (np.isfinite(w).all() and np.isfinite(v).all()):
            self.drop_anchor()
            return
        cfg = self.cfg
        if not np.isfinite(dt) or dt <= 0.0:
            return                                  # no time passed (or a broken clock): nothing to do
        if dt > cfg.max_tick_dt_s:
            self.drop_anchor()                      # the window cannot be trusted across a big gap
            dt = cfg.max_tick_dt_s
        dR = _rot_body(w, dt)
        self.bias = dR @ self.bias                  # world-fixed vector held in body coordinates
        if self._u_last is not None:
            self._u_last = dR @ self._u_last
        if self._prev_fix is not None:
            self._prev_fix = dR @ self._prev_fix
            self._prev_fix_dt += dt
        if self._anchor is not None:
            self._anchor = dR @ self._anchor
            self._disp = dR @ self._disp + v * dt
            self._baseline_s += dt
            if self._baseline_s > 4.0 * max(cfg.max_baseline_s, 1e-3):
                self.drop_anchor()                  # a window that never closed: drop it
        self._age_s += dt
        # BLACKOUT policy: hold, then decay back to the raw KF velocity.
        if self._age_s > cfg.hold_s:
            self.bias = self.bias * math.exp(-dt / max(cfg.decay_tau_s, 1e-6))
        if not np.isfinite(self.bias).all():        # belt-and-braces: never latch a NaN
            self.bias = np.zeros(3)

    def on_fix(self, rel_flu) -> None:
        """An ACCEPTED fresh gate fix (TRUE body FLU, the same vector the builder snaps into
        ``_rel_flu[0]``). Closes the measurement window when it is usable, then re-anchors."""
        cfg = self.cfg
        r = np.asarray(rel_flu, dtype=np.float64).reshape(3)
        self._age_s = 0.0
        rng = float(np.linalg.norm(r))
        if not np.isfinite(r).all() or not (cfg.min_range_m <= rng <= cfg.max_range_m):
            self.drop_anchor()                      # unusable landmark: no window at all
            return
        u = r / rng
        self._u_last = u
        # TRACK-CONTINUITY GATE (the gate-seam teleport, v21-release-dive 2026-07-27): if the lever
        # moved faster than any drone can fly, the seeker re-locked onto a DIFFERENT gate and this is
        # not a velocity measurement at all. Judged on vision+gyro alone -- never on the KF velocity
        # under test -- so it cannot bias the estimate toward agreeing with dead reckoning.
        if self._prev_fix is not None:
            # A DISTANCE test, deliberately not a speed one -- see max_fix_step_m. It needs no dt,
            # so there is no divide-by-zero branch and two fixes inside one tick are handled by the
            # same arithmetic as any other pair.
            if float(np.linalg.norm(self._prev_fix - r)) > cfg.max_fix_step_m:
                self.n_jumps += 1
                self.drop_anchor()
                self._prev_fix = r.copy()
                self._prev_fix_dt = 0.0
                return
        self._prev_fix = r.copy()
        self._prev_fix_dt = 0.0
        if self._anchor is not None and cfg.min_baseline_s <= self._baseline_s <= cfg.max_baseline_s:
            b_meas = (self._anchor - r - self._disp) / self._baseline_s
            if np.isfinite(b_meas).all():
                self.last_meas = b_meas
                # keep ONLY the LOS-perpendicular part of the measurement (see module docstring)
                perp = b_meas - u * float(u @ b_meas)
                innov = perp - (self.bias - u * float(u @ self.bias))
                n = float(np.linalg.norm(innov))
                if n > cfg.innov_clip_mps:          # robust: heavy-tailed PnP outliers
                    innov = innov * (cfg.innov_clip_mps / n)
                    self.n_rejected += 1
                cand = self.bias + float(cfg.gain) * innov
                nb = float(np.linalg.norm(cand))
                if nb > cfg.max_bias_mps:
                    cand = cand * (cfg.max_bias_mps / nb)
                if np.isfinite(cand).all():
                    self.bias = cand
                    self.n_updates += 1
        if self._anchor is None or self._baseline_s >= cfg.max_baseline_s:
            self._anchor = r.copy()                 # (re-)anchor the next window on this fix
            self._disp = np.zeros(3)
            self._baseline_s = 0.0

    # -- output -----------------------------------------------------------------------------
    def correction(self) -> np.ndarray:
        """The body-FLU velocity correction actually applied: the stored bias projected onto the
        subspace PERPENDICULAR to the last line of sight, so an unobservable along-LOS component
        can never reach obs[0:3]. No track ever seen -> zeros."""
        if self._u_last is None or not np.isfinite(self.bias).all():
            return np.zeros(3)
        u = self._u_last
        n = float(np.linalg.norm(u))
        if not np.isfinite(n) or n < 1e-9:
            return np.zeros(3)
        u = u / n
        c = self.bias - u * float(u @ self.bias)
        return c if np.isfinite(c).all() else np.zeros(3)

    def correct(self, v_flu) -> np.ndarray:
        """``v_flu`` + the applied correction. Returns the INPUT unchanged if anything is
        non-finite (never degrade the obs into garbage)."""
        v = np.asarray(v_flu, dtype=np.float64).reshape(3)
        if not np.isfinite(v).all():
            return v
        out = v + self.correction()
        return out if np.isfinite(out).all() else v

    def diag(self) -> dict:
        c = self.correction()
        return {
            "bias": [round(float(x), 4) for x in self.bias],
            "corr": [round(float(x), 4) for x in c],
            "baseline_s": round(float(self._baseline_s), 3),
            "anchored": self._anchor is not None,
            "age_s": round(float(self._age_s), 3),
            "n_upd": int(self.n_updates),
            "n_clip": int(self.n_rejected),
            "n_jump": int(self.n_jumps),
        }
