"""Out-of-sequence-measurement (OOSM) wrapper around :class:`racer.state_estimator.LinearKF`.

COMPONENT C2 (case-C VQ2 estimator chain, BLUEPRINT §1.5 / §1.4). A vision fix arrives stamped at a
CAPTURE time ``t_fix`` that is older than ``t_now`` by the in-loop vision latency (frame-age + detect +
PnP + queue). The naive :class:`Navigator` applies that fix to the CURRENT KF state. At VQ1 speed
(~5 m/s) the staleness costs ~v*L ~ a few cm; at VQ2 speed (15-37 m/s) it is 0.5-2 m of pure
latency-induced position error injected at every fix -- a systematic bias toward where the drone WAS,
not where it is.

This wrapper fixes that without touching the filter physics. It keeps a ring buffer of the recent
filter HISTORY -- every operation (an IMU ``predict`` or a given/vision pos/vel update) with its
sim-time and inputs, plus a state SNAPSHOT taken immediately BEFORE each operation. When an
out-of-sequence fix arrives:

    1. find the buffered op boundary at-or-just-after t_fix (the insertion point);
    2. RESTORE the KF to the snapshot at that boundary (rewind);
    3. apply the position fix there (the correction the drone actually saw);
    4. REPLAY every buffered op from that boundary forward to t_now (re-propagate the IMU and
       re-apply the pos/vel updates that have streamed in since the capture).

Because every replayed op is the SAME ``LinearKF.predict`` / ``update`` call with the SAME inputs, the
only thing that changes vs. the naive path is WHERE in the timeline the fix lands. The covariance is
re-derived correctly by the replay (the fix's information is propagated forward through the buffered Q),
so the result is the exact KF estimate you would have computed had the fix arrived on time -- the
textbook OOSM solution (full re-propagation; no Bar-Shalom approximation needed, the horizon is tiny).

Design invariants (all pinned by tests/test_kf_rewind.py):
- ``update_position_at(t_fix, ...)`` with ``t_fix >= t_now`` (or L==0) is BIT-IDENTICAL to the naive
  ``LinearKF.update_position`` -- a zero-latency fix degenerates to in-place. (G2)
- the replay keeps P symmetric positive-definite (the wrapped Joseph-form update is SPD-preserving,
  and a rewind only re-runs those same updates).
- horizon MUST be STRICTLY > the maximum fix age L. ``horizon == L`` drops 100% of fixes (a 1-ns
  quantization edge) -> dead-reckon -> divergence (~21 m over a track). 0.5 s covers the measured edge
  L (~16 ms) AND the CPU-class L (~125 ms). ``assert_horizon_gt(L)`` makes the rule explicit. (G2)
- a fix older than the buffer horizon CANNOT rewind safely: it is reported as DROPPED (the buffered IMU
  to re-propagate is gone). We never silently apply a too-old fix to the current state (that is the very
  bug this class exists to prevent).

The wrapper COMPOSES the real filter; it does not subclass-and-override the math. ``predict`` and the
``update_*`` conveniences forward to the wrapped ``LinearKF`` and ALSO record the op + a pre-op
snapshot. ``x`` / ``P`` / ``position`` / ``velocity`` proxy the wrapped filter so the navigator reads
the estimate exactly as before.

Memory / compute: one buffer slot = 1 state snapshot (x:6 + P:36 = 42 f64 = 336 B) + op inputs ~ 0.5
KB/slot. A 0.5 s horizon at 90 Hz IMU + 30 Hz fixes ~ 60 slots ~ 30 KB. Per-fix rewind replays at most
``horizon_s * imu_hz`` predicts (~45 for 0.5 s @ 90 Hz); each predict is two 6x6 matmuls -> a fix costs
<~0.5 ms on the laptop. Negligible at 30 Hz.

[C2-ESTIMATOR-CHAIN 2026-06-13; productionized from handoff/ultracode-vision-case-c kf_rewind_buffer.py]
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from racer.state_estimator import LinearKF

# Operation kinds recorded in the history buffer.
_OP_PREDICT: Literal["predict"] = "predict"
_OP_POS: Literal["pos"] = "pos"
_OP_VEL: Literal["vel"] = "vel"
_OP_FIX: Literal["fix"] = "fix"  # an applied vision fix (recorded so a LATER OOSM fix replays over it)


@dataclass
class _Op:
    """One recorded filter operation + the state snapshot taken immediately BEFORE it."""

    sim_time_ns: int
    kind: str
    # snapshot of (x, P) BEFORE this op ran -- rewinding to this op restores these.
    x_before: np.ndarray
    P_before: np.ndarray
    # op payload (only the fields the op kind needs are populated)
    accel_body: np.ndarray | None = None
    R_wb: np.ndarray | None = None
    dt: float = 0.0
    z: np.ndarray | None = None
    cov: np.ndarray | None = None


@dataclass
class RewindResult:
    """What ``update_position_at`` did, for the caller / diagnostics."""

    applied: bool
    rewound: bool                 # True if it actually rewound+replayed (vs. degenerate in-place)
    dropped_reason: str | None = None
    n_replayed_ops: int = 0
    rewind_depth_s: float = 0.0   # t_now - t_fix actually rewound (0 if in-place / dropped)


@dataclass
class RewindKF:
    """Buffered-state OOSM wrapper around :class:`racer.state_estimator.LinearKF`.

    Build with an already-initialised ``LinearKF`` (the navigator's ``LinearKF.initialize(...)``) and a
    buffer horizon. Drive it exactly like the bare filter -- ``predict`` / ``update_position`` /
    ``update_velocity`` forward to the wrapped filter AND record history -- but for a late vision fix
    call ``update_position_at(t_fix, z, cov)`` instead of ``update_position(z, cov)``.
    """

    kf: LinearKF
    horizon_s: float = 0.5                          # how far back a fix may rewind (MUST be > L)
    _ops: "deque[_Op]" = field(default_factory=deque, repr=False)
    # current sim-time (set by predict from the IMU clock; the master timeline)
    _now_ns: int = field(default=0, repr=False)

    # ---- state proxies (so the navigator reads the estimate unchanged) ----
    @property
    def x(self) -> np.ndarray:
        return self.kf.x

    @x.setter
    def x(self, value: np.ndarray) -> None:
        self.kf.x = value

    @property
    def P(self) -> np.ndarray:
        return self.kf.P

    @P.setter
    def P(self, value: np.ndarray) -> None:
        self.kf.P = value

    @property
    def position(self) -> np.ndarray:
        return self.kf.position

    @property
    def velocity(self) -> np.ndarray:
        return self.kf.velocity

    @property
    def now_ns(self) -> int:
        return self._now_ns

    def buffer_horizon_ns(self) -> int:
        return int(self.horizon_s * 1e9)

    def oldest_buffered_ns(self) -> int | None:
        return self._ops[0].sim_time_ns if self._ops else None

    def assert_horizon_gt(self, latency_s: float, *, where: str = "RewindKF") -> None:
        """LOUD guard for the horizon<=L divergence trap (BLUEPRINT §1.5 / G2).

        ``horizon == L`` drops 100% of fixes (a 1-ns quantization edge: the oldest buffered op is
        newer than ``now - horizon == t_fix``), so the filter dead-reckons and diverges (~21 m over a
        track). The horizon MUST be STRICTLY greater than the maximum fix age L. Call this once when
        wiring the RewindKF with the known L (the predict-forward ``vision_latency_const_s``, or the
        measured per-fix age bound)."""
        if not (self.horizon_s > latency_s):
            raise AssertionError(
                f"[{where}] RewindKF horizon_s={self.horizon_s:.4f} s is NOT strictly > the fix "
                f"latency L={latency_s:.4f} s. horizon<=L drops 100% of fixes (1-ns quantization "
                f"edge) -> dead-reckon -> divergence (~21 m). Use horizon_s > L (recommend 0.5 s, "
                f"which covers the CPU-class L~125 ms).")

    # ---- internal: snapshot + record + prune ------------------------------
    def _snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        return self.kf.x.copy(), self.kf.P.copy()

    def _record(self, op: _Op) -> None:
        self._ops.append(op)
        self._prune(op.sim_time_ns)

    def _prune(self, ref_ns: int) -> None:
        """Drop ops older than the horizon relative to the newest sim-time seen."""
        horizon = self.buffer_horizon_ns()
        while self._ops and (ref_ns - self._ops[0].sim_time_ns) > horizon:
            self._ops.popleft()

    # ---- driven exactly like LinearKF, but recorded -----------------------
    def predict(self, accel_body: np.ndarray, R_wb: np.ndarray, dt: float, sim_time_ns: int) -> None:
        """IMU predict. ``sim_time_ns`` is the IMU stamp AFTER this step (the new now).

        Snapshot BEFORE so a fix that lands during the interval (t-dt, t] can rewind to the start of
        this step. We forward the *exact same* call to the wrapped filter (its own dt<=0 / dt>max_dt_s
        guards apply identically), so the history replays bit-for-bit.
        """
        x0, P0 = self._snapshot()
        self.kf.predict(accel_body, R_wb, dt)
        self._now_ns = int(sim_time_ns)
        self._record(
            _Op(
                sim_time_ns=int(sim_time_ns),
                kind=_OP_PREDICT,
                x_before=x0,
                P_before=P0,
                accel_body=np.asarray(accel_body, dtype=np.float64).copy(),
                R_wb=np.asarray(R_wb, dtype=np.float64).copy(),
                dt=float(dt),
            )
        )

    def update_position(self, z: np.ndarray, cov: np.ndarray, sim_time_ns: int | None = None) -> None:
        """In-sequence position fix (given pos at VQ1, or a zero-latency vision fix)."""
        t = self._now_ns if sim_time_ns is None else int(sim_time_ns)
        x0, P0 = self._snapshot()
        self.kf.update_position(z, cov)
        self._record(
            _Op(sim_time_ns=t, kind=_OP_POS, x_before=x0, P_before=P0,
                z=np.asarray(z, dtype=np.float64).copy(),
                cov=np.asarray(cov, dtype=np.float64).copy())
        )

    def update_velocity(self, z: np.ndarray, cov: np.ndarray, sim_time_ns: int | None = None) -> None:
        t = self._now_ns if sim_time_ns is None else int(sim_time_ns)
        x0, P0 = self._snapshot()
        self.kf.update_velocity(z, cov)
        self._record(
            _Op(sim_time_ns=t, kind=_OP_VEL, x_before=x0, P_before=P0,
                z=np.asarray(z, dtype=np.float64).copy(),
                cov=np.asarray(cov, dtype=np.float64).copy())
        )

    # ---- the OOSM correction ---------------------------------------------
    def update_position_at(self, t_fix_ns: int, z: np.ndarray, cov: np.ndarray) -> RewindResult:
        """Apply a position fix stamped at capture-time ``t_fix_ns`` (typically < now).

        REWIND to t_fix, apply the fix there, REPLAY buffered ops forward to now. If t_fix is
        at-or-after now (a zero/negative-latency fix, or L==0) this degenerates to an in-place
        ``update_position`` at now -- BIT-IDENTICAL to the naive path (G2). If t_fix is older than the
        buffer horizon the fix is DROPPED (the IMU to re-propagate is gone); the caller decides what to
        do (the safe default is to skip it -- never apply a too-old fix to the current state, which is
        the bug this class exists to prevent).
        """
        z = np.asarray(z, dtype=np.float64)
        cov = np.asarray(cov, dtype=np.float64)
        t_fix = int(t_fix_ns)

        # Degenerate: fix is current (or future) -> ordinary in-place update at now. This MUST be
        # bit-identical to bare LinearKF.update_position (the zero-latency / predict-forward-at-now
        # path), so we forward straight to the wrapped filter's conveniences.
        if not self._ops or t_fix >= self._now_ns:
            self.update_position(z, cov, sim_time_ns=max(t_fix, self._now_ns))
            return RewindResult(applied=True, rewound=False, n_replayed_ops=0, rewind_depth_s=0.0)

        # Too old to rewind: the oldest buffered op is newer than the fix -> we lack the IMU history to
        # re-propagate from t_fix. Drop (do NOT apply to current state).
        oldest = self._ops[0].sim_time_ns
        if t_fix < oldest:
            return RewindResult(
                applied=False, rewound=False,
                dropped_reason=f"fix t={t_fix} older than buffer horizon (oldest={oldest})",
                n_replayed_ops=0, rewind_depth_s=(self._now_ns - t_fix) / 1e9,
            )

        # Find the insertion index: the first op whose sim_time_ns > t_fix. Every op at index >= idx
        # happened AFTER the fix and must be replayed; we rewind to op[idx].x_before (the state as it
        # was right before that op ran -> i.e. the state AT t_fix, having already integrated up through
        # the predict that ended at-or-before t_fix).
        ops = list(self._ops)
        idx = next((i for i, op in enumerate(ops) if op.sim_time_ns > t_fix), len(ops))
        base = ops[idx] if idx < len(ops) else ops[0]
        # restore KF to the pre-state of the first post-fix op (the state at t_fix)
        self.kf.x = base.x_before.copy()
        self.kf.P = base.P_before.copy()

        # apply the fix AT t_fix (this is the correction the drone actually observed)
        self.kf.update_position(z, cov)

        # replay every op from idx forward, re-deriving their pre-snapshots so a still-later OOSM fix
        # can rewind through THIS one too.
        replay = ops[idx:]
        new_ops: list[_Op] = ops[:idx]
        # insert the fix as a recorded op at t_fix so future rewinds see it
        new_ops.append(
            _Op(sim_time_ns=t_fix, kind=_OP_FIX, x_before=base.x_before.copy(),
                P_before=base.P_before.copy(), z=z.copy(), cov=cov.copy())
        )
        for op in replay:
            x0, P0 = self._snapshot()
            op2 = _Op(sim_time_ns=op.sim_time_ns, kind=op.kind, x_before=x0, P_before=P0,
                      accel_body=op.accel_body, R_wb=op.R_wb, dt=op.dt, z=op.z, cov=op.cov)
            if op.kind == _OP_PREDICT:
                self.kf.predict(op.accel_body, op.R_wb, op.dt)
            elif op.kind == _OP_POS or op.kind == _OP_FIX:
                self.kf.update_position(op.z, op.cov)
            elif op.kind == _OP_VEL:
                self.kf.update_velocity(op.z, op.cov)
            new_ops.append(op2)

        self._ops = deque(new_ops)
        return RewindResult(
            applied=True, rewound=True, n_replayed_ops=len(replay),
            rewind_depth_s=(self._now_ns - t_fix) / 1e9,
        )

    # ---- convenience accessor for the navigator output --------------------
    def snapshot_state(self) -> tuple[np.ndarray, np.ndarray]:
        return self.kf.x.copy(), self.kf.P.copy()
