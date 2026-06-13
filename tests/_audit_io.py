"""SHARED numerical-test substrate for the adversarial frame-convention audit.

Promoted into tests/ from handoff/ultracode-substrate-audit-2026-06-13/scratch/_audit_io.py
(LAPTOP-PROMOTE-REGRESSION-SUITE, 2026-06-13). The ONLY change from the audit copy is the
repo-root discovery (``_find_root`` below) so the module no longer depends on its directory
depth -- everything else (loader, force model, Test-A harness) is byte-for-byte the audit
substrate. NOT named ``test_*`` so pytest will not try to collect it as a test module.

BANKED METHODOLOGY (this project bitten 4x by this bug class -- R_y(pi) ODOMETRY-quat
conjugation, body-vs-world velocity-frame mix, corner_to_center 180deg flip,
ATTITUDE.pitch sign inversion):

  * A proper-rotation conjugation / sign-alias is INVISIBLE to internal-consistency
    checks (quat-vs-rate finite-difference, level-flight correlation, twist round-trip
    ALL PASS while the convention is WRONG).
  * ONLY EXTERNAL invariants discriminate. Canonical one: finite-difference of the
    PRISTINE state.velocity_ned (world NED, the trusted anchor) vs attitude-derived
    specific force = K*L*(R@[0,0,-1]) + quad_drag(R) + [0,0,g], evaluated in the
    TILTED bin (tilt > 35 deg). Level flight is INADMISSIBLE -- a proper rotation
    looks identical to its conjugate when the body z-axis is aligned with world down.
  * Treat every "// verified" / convention comment as a CLAIM to be externally refuted,
    NOT a fact. Prefer EXTERNAL/NUMERICAL evidence over the comment that asserts a
    convention.

This module is pure I/O + the Test-A force model, factored out of the reference
implementation handoff/laptop-frame-audit-2026-06-12/scripts/audit_candidates.py so
every downstream verification agent shares ONE loader and ONE force model.

  WARNING -- LAPSE (thrust-lapse-vs-airspeed) is VOIDED in production (S18 lapse VOIDED,
  do NOT use --plant lapse / dr_lapse). It is retained here ONLY to bit-match the
  reference Test-A probe (audit_candidates.py:90). force_model() exposes a
  use_lapse flag (default True for reference parity); set use_lapse=False to drop it.

DO NOT import or audit rl/contact_true_eval.py (out of scope, edited elsewhere).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def _find_root(start: Path) -> Path:
    """Repo root = nearest ancestor that holds BOTH ``src/`` and ``handoff/``.

    Depth-independent so this file works whether it lives in tests/ or in the
    audit scratch dir. Falls back to ``start.parents[1]`` (tests/ -> repo root)
    if no marker ancestor is found.
    """
    p = start.resolve()
    for cand in (p, *p.parents):
        if (cand / "src").is_dir() and (cand / "handoff").is_dir():
            return cand
    return p.parents[1]


ROOT = _find_root(Path(__file__))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# Constants imported from the production plant (single source of truth).  QUAD_DRAG_C2_MEASURED
# is shape (3,2): per body axis, column 0 = forward-velocity coeff, column 1 = backward.
from racer.rl_plant import (  # noqa: E402
    COLL_MAP_THR_MEASURED,
    COLL_MAP_ACCEL_MEASURED,
    QUAD_DRAG_C2_MEASURED,
    LAPSE_SPEED_MEASURED,
    LAPSE_FACTOR_MEASURED,
)
from scipy.spatial.transform import Rotation  # noqa: E402

G = 9.80665  # world NED, gravity is +9.80665 on +Z (Z-down)

# ---------------------------------------------------------------------------------------
# Dataset roots.  Both carry rl_inc6 frame-audit recordings with the FULL deployed schema.
#   postfix = frame-CLEAN runs (POST VISION-FRAME-FIX 8d7b0b3; the navigator/world-fix bug
#             was fixed -- these are the negative control: TRUE conjugation should win cleanly).
#   refit   = the dataset the reference audit_candidates.py loads (DATA constant). Same schema.
# NOTE: "postfix"/"refit" name the *dataset capture session*, NOT presence/absence of the
# quat mirror in the raw telemetry. The R_y(pi) ODOMETRY-quat conjugation is a property of the
# SIM TELEMETRY (q_raw is ALWAYS mirrored), present in BOTH datasets -- which is exactly why the
# AS-IS [1,1,1,1] candidate anti-correlates and the TRUE [1,-1,1,-1] candidate wins in BOTH.
# Use that mirror as a POSITIVE CONTROL: AS-IS must fail in any honest run, pre- or post-fix.
# ---------------------------------------------------------------------------------------
_DATASET_DIRS = {
    "postfix": ROOT / "handoff/shadowpc-postfix-dataset-2026-06-12/extracted",
    "refit": ROOT / "handoff/shadowpc-refit-dataset-2026-06-12/extracted",
}

# Candidate per-component quat signs (wxyz) -- the only re-interpretations that preserve
# hover/identity (proper-rotation conjugations).  TRUE = R_y(pi)-conjugation = [1,-1,1,-1].
CAND_TRUE = np.array([1.0, -1.0, 1.0, -1.0])   # R_y(pi)-conj: negate x,z  (the winner)
CAND_ASIS = np.array([1.0, 1.0, 1.0, 1.0])     # raw / bcc93f9 assumption (the loser; positive control)


def dataset_present(name: str) -> bool:
    """True iff the named dataset's ``extracted/`` dir exists on disk.

    The ``postfix`` recordings are git-tracked, but ``refit/extracted`` is gitignored
    (only its .zip is tracked). On a checkout that has not unzipped refit, the promoted
    tests ``skip`` (rather than ERROR) -- see the per-module ``pytestmark`` skipif.
    """
    return _DATASET_DIRS.get(name, Path("/nonexistent")).is_dir()


def datasets_present() -> bool:
    """True iff at least one configured dataset dir exists on disk."""
    return any(d.is_dir() for d in _DATASET_DIRS.values())


# =======================================================================================
# Run discovery + loading
# =======================================================================================
def list_runs(dataset: str) -> list[Path]:
    """Return sorted run directories for dataset in {"postfix","refit"}."""
    if dataset not in _DATASET_DIRS:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {sorted(_DATASET_DIRS)}")
    base = _DATASET_DIRS[dataset]
    if not base.is_dir():
        raise FileNotFoundError(f"dataset dir missing: {base}")
    return sorted((p for p in base.iterdir() if p.is_dir()), key=lambda p: p.name)


# Field name in the jsonl  ->  output key in the returned dict.  Missing fields are omitted
# (tolerant), so a loader can be reused across schema revisions.
_FIELD_MAP = {
    "t_mono": "t",
    "pos_ned": "pos",
    "vel_ned": "vel",
    "q_raw_wxyz": "q_raw",
    "w_raw": "w_raw",
    "collective": "coll",
    "obs": "obs",
    "actor_mean": "actor_mean",
    "tanh": "tanh",
    "act_rescaled": "act_rescaled",
    "rate_frd": "rate_frd",
    "normed_thrust": "normed_thrust",
    "gate_index": "gate_index",
    "n_coll": "n_coll",
}


def load_run(run_dir) -> dict:
    """Load one run -> dict of numpy arrays + 'meta' (the header dict).

    Skips the header row (stored under returned dict key 'meta'). Each present field in
    _FIELD_MAP becomes a numpy array; absent fields are silently omitted (tolerant).
    """
    run_dir = Path(run_dir)
    path = run_dir / "debug_obs.jsonl"
    rows = []
    meta: dict = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("type") == "header":
                meta = r
                continue
            rows.append(r)
    if not rows:
        raise ValueError(f"no data rows in {path}")

    out: dict = {"meta": meta, "run_dir": run_dir, "name": run_dir.name, "n": len(rows)}
    for src, dst in _FIELD_MAP.items():
        if all(src in r for r in rows):
            out[dst] = np.array([r[src] for r in rows], dtype=float)
        elif any(src in r for r in rows):
            # partial presence -> omit (tolerant), but flag once for the caller's awareness.
            out.setdefault("_partial_fields", []).append(src)
    return out


# =======================================================================================
# Attitude / force model (mirrors audit_candidates.py Test A)
# =======================================================================================
def Rmats(q_wxyz_array, sign=(1, 1, 1, 1)) -> np.ndarray:
    """Rotation matrices for an (N,4) wxyz quat array, applying optional per-component sign.

    sign is a length-4 (w,x,y,z) multiplier used to test conjugation candidates:
      [1,1,1,1]   -> AS-IS (raw)
      [1,-1,1,-1] -> R_y(pi)-conjugation (TRUE)
    Returns (N,3,3) world<-body rotation matrices. scipy wants xyzw order.
    """
    q = np.asarray(q_wxyz_array, dtype=float)
    if q.ndim == 1:
        q = q[None, :]
    s = np.asarray(sign, dtype=float)
    qs = q * s[None, :]
    # wxyz -> xyzw for scipy
    return Rotation.from_quat(qs[:, [1, 2, 3, 0]]).as_matrix()


def tilt_deg(R) -> np.ndarray:
    """Tilt angle (deg) = angle between body-down and world-down = arccos(R[2,2]).

    Accepts a single (3,3) matrix or an (N,3,3) stack.
    """
    R = np.asarray(R, dtype=float)
    if R.ndim == 2:
        return float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))
    return np.degrees(np.arccos(np.clip(R[:, 2, 2], -1.0, 1.0)))


def force_model(R, v, coll_delayed, use_lapse: bool = True) -> np.ndarray:
    """Attitude-derived specific force in world NED (m/s^2), mirroring Test A.

      model = K * L * (R @ [0,0,-1])  +  quad_drag(R, v)  +  [0,0,g]

    Args:
      R            : (3,3) world<-body rotation for this tick.
      v            : (3,) world-NED velocity at this tick (the PRISTINE anchor).
      coll_delayed : scalar collective used for the thrust map. The reference uses the
                     collective 2 ticks earlier (coll[k-2]) to model command->thrust delay;
                     the CALLER is responsible for passing the delayed value.
      use_lapse    : multiply K by the airspeed lapse factor L. Default True for reference
                     parity ONLY -- LAPSE IS VOIDED IN PRODUCTION (S18). Set False to drop it.

    Returns (3,) world-NED specific force to compare against central-FD of vel_ned.
    """
    R = np.asarray(R, dtype=float)
    v = np.asarray(v, dtype=float)
    K = float(np.interp(coll_delayed, COLL_MAP_THR_MEASURED, COLL_MAP_ACCEL_MEASURED))
    if use_lapse:
        L = float(np.interp(np.linalg.norm(v), LAPSE_SPEED_MEASURED, LAPSE_FACTOR_MEASURED))
    else:
        L = 1.0
    thr = K * L * (R @ np.array([0.0, 0.0, -1.0]))
    vb = R.T @ v  # body-frame velocity
    cc = np.where(vb >= 0, QUAD_DRAG_C2_MEASURED[:, 0], QUAD_DRAG_C2_MEASURED[:, 1])
    drag = R @ (-cc * np.abs(vb) * vb)
    return thr + drag + np.array([0.0, 0.0, G])


# =======================================================================================
# Convenience: Test-A force-vs-FD residual for a candidate sign in the tilt>thr bin
# =======================================================================================
def test_a_force_vs_fd(run, sign, tilt_min=35.0, use_lapse: bool = True):
    """Run the canonical EXTERNAL-invariant Test A on one loaded run for one candidate sign.

    Compares central-FD of pristine vel_ned against force_model(R(sign), ...) in the
    tilt>tilt_min bin, per axis. Returns dict with per-axis median|residual| and corr,
    plus sample count. This is the discriminating test -- AS-IS [1,1,1,1] will show an
    anti-correlated East axis (~ -0.84) while TRUE [1,-1,1,-1] correlates ~ +0.97..+0.99.
    """
    t, pos, vel, q = run["t"], run["pos"], run["vel"], run["q_raw"]
    coll = run["coll"]
    n = len(t)
    R = Rmats(q, sign=sign)
    tilt_raw = tilt_deg(Rmats(q, sign=CAND_ASIS))  # tilt is invariant across candidates; use raw
    model, meas, mask_tilt = [], [], []
    for k in range(2, n - 2):
        dt = t[k + 1] - t[k - 1]
        if not (0.05 < dt < 0.09):
            continue
        a_meas = (vel[k + 1] - vel[k - 1]) / dt
        if not np.all(np.isfinite(a_meas)) or np.max(np.abs(a_meas)) > 90:
            continue
        coll_d = coll[k - 2]
        model.append(force_model(R[k], vel[k], coll_d, use_lapse=use_lapse))
        meas.append(a_meas)
        mask_tilt.append(tilt_raw[k])
    model = np.array(model)
    meas = np.array(meas)
    tl = np.array(mask_tilt)
    m = tl > tilt_min
    res = {"n_total": len(meas), "n_tilted": int(m.sum()), "axes": {}}
    AX = ["N", "E", "D"]
    for ax in range(3):
        if m.sum() < 5:
            res["axes"][AX[ax]] = {"med_abs_res": np.nan, "corr": np.nan}
            continue
        md = float(np.median(np.abs(model[m, ax] - meas[m, ax])))
        cor = (float(np.corrcoef(model[m, ax], meas[m, ax])[0, 1])
               if model[m, ax].std() > 1e-6 else np.nan)
        res["axes"][AX[ax]] = {"med_abs_res": md, "corr": cor}
    return res
