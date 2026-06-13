"""EXTERNAL-INVARIANT REGRESSION: sim-wire velocity is single-rotation world-NED.

BUG CLASS GUARDED (one line):
    body-vs-world velocity-frame MIX -- the stored `state.velocity_ned` must equal the
    pristine LOCAL_POSITION_NED world velocity (ODOMETRY twist rotated body->world EXACTLY
    ONCE; not zero times = raw body twist, not twice = double-rotated).

This is one of the FOUR convention bugs that have bitten this project (R_y(pi) ODOMETRY-quat
conjugation; body-vs-world velocity-frame mix [THIS ONE, fixed in c3b5a8e -- twist is BODY
frame, was stored as world]; corner_to_center 180deg flip; ATTITUDE.pitch sign inversion).

WHY INTERNAL-CONSISTENCY CHECKS MISS IT
---------------------------------------
A frame-mix is a per-tick rotation applied to a vector. It is INVISIBLE to every
internal-consistency check:
  * |v| (speed magnitude) is rotation-invariant, so any norm/energy check passes.
  * A "twist round-trip" (rotate body->world->body) tautologically passes -- it never
    compares against an independent truth.
  * Near-LEVEL flight is inadmissible: when the body z-axis aligns with world-down, R ~= I,
    so body-frame and world-frame velocity coincide and the bug is masked. (In this dataset
    the median banked separation |v_world - v_body| ~= 14.7 m/s, but it collapses to ~0 at
    level -- which is exactly why we MUST restrict to the TILTED bin.)
Only an EXTERNAL invariant discriminates: the time-derivative of the SEPARATE, trusted
world-NED position channel (`pos_ned`, from LOCAL_POSITION_NED) must equal `vel_ned`. pos
and vel are independent sim-wire fields; if vel were stored in the wrong frame (or rotated
the wrong number of times), the FD of pos would still be true world-NED and the two would
DECORRELATE through bank. Comments asserting "twist is world frame" are CLAIMS to refute
numerically, not facts -- this project has shipped self-mirrored convention comments before.

THE INVARIANT (tilted bin, tilt > 35 deg)
-----------------------------------------
    central-FD(pos_ned)  ==  vel_ned    (per axis: corr ~= +0.997, gain ~= +0.997,
                                         median |residual| <~ 0.25 m/s)

NEGATIVE CONTROLS (prove the test has discriminating power)
-----------------------------------------------------------
The recordings store only the CORRECT world velocity, so we reconstruct the body twist
v_body = R_true^T @ vel_ned and synthesize the two frame-mix bugs:
  * ZERO-rotation  (store raw body twist):     v = v_body          -> corr collapses to ~ -0.17..+0.17
  * DOUBLE-rotation (apply world<-body twice): v = R_true @ R_true @ v_body -> corr ~ -0.12..+0.05
Both blow the per-axis residual to ~4.5-7.8 m/s. The test asserts the live data matches the
WORLD candidate AND that both bug candidates would FAIL the same gate (so a regression cannot
slip through).

DATA: handoff/shadowpc-{refit,postfix}-dataset-2026-06-12/extracted via the shared substrate
handoff/ultracode-substrate-audit-2026-06-13/scratch/_audit_io.py. Does NOT import
rl/contact_true_eval.py (out of scope).

Run BOTH ways:
  .venv/Scripts/python.exe handoff/ultracode-substrate-audit-2026-06-13/regression_suite/test_mavlink_velocity_single_rotation.py
  pytest handoff/ultracode-substrate-audit-2026-06-13/regression_suite/test_mavlink_velocity_single_rotation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# --- locate the shared substrate (sibling 'scratch' dir) and import it ---------------------
_HERE = Path(__file__).resolve().parent                       # .../regression_suite
_SCRATCH = _HERE.parent / "scratch"                           # .../scratch
if str(_SCRATCH) not in sys.path:
    sys.path.insert(0, str(_SCRATCH))
import _audit_io as A  # noqa: E402  (shared loader + Rmats + tilt_deg; single source of truth)

# --- gate thresholds -----------------------------------------------------------------------
TILT_MIN_DEG = 35.0      # tilted bin: below this, body~=world frame and the bug is masked
DT_LO, DT_HI = 0.05, 0.09  # central-FD dt window (30 Hz -> 2*0.0333 ~= 0.0667 s)
ACCEL_CAP = 90.0         # reject FD glitches / reset discontinuities
MIN_TILTED = 2000        # need a substantial tilted pool for a meaningful corr
# PASS gates for the WORLD (correct) candidate, applied per axis:
CORR_MIN = 0.98          # live ~= 0.996; bug candidates < 0.2.  0.98 is a wide safety margin.
RESID_MAX = 0.60         # m/s; live median ~= 0.24; bug candidates ~= 4.5-7.8.
# FAIL gates the bug candidates MUST trip (negative-control assertions):
BUG_CORR_MAX = 0.85      # a frame-mix must drop SOME axis below this (live min axis ~= 0.994)


def _gather():
    """Pool tilted-bin samples across both datasets.

    Returns dict of (N,3) arrays: pos_fd (central-FD of pristine pos_ned), v_world (stored
    vel_ned), v_body (reconstructed body twist = R_true^T @ v_world), v_double (R_true twice).
    Tilt is measured from the candidate-invariant raw rotation. We use the SEPARATE pos_ned
    channel as the external truth -- pos and vel are independent sim-wire fields.
    """
    pos_fd, v_world, v_body, v_double = [], [], [], []
    n_runs = n_skipped = 0
    for dataset in ("refit", "postfix"):
        for rd in A.list_runs(dataset):
            try:
                d = A.load_run(rd)
            except ValueError:
                n_skipped += 1          # header-only run (e.g. ..._std_f2); tolerate + skip
                continue
            if not all(k in d for k in ("t", "pos", "vel", "q_raw")):
                n_skipped += 1
                continue
            n_runs += 1
            t, pos, vel, q = d["t"], d["pos"], d["vel"], d["q_raw"]
            R = A.Rmats(q, sign=A.CAND_TRUE)         # TRUE world<-body (only used to *reconstruct*
            tl = A.tilt_deg(R)                       # the bug controls + the tilt mask)
            n = len(t)
            for k in range(2, n - 2):
                dt = t[k + 1] - t[k - 1]
                if not (DT_LO < dt < DT_HI):
                    continue
                pf = (pos[k + 1] - pos[k - 1]) / dt          # FD of the EXTERNAL truth channel
                if not np.all(np.isfinite(pf)) or np.max(np.abs(pf)) > ACCEL_CAP:
                    continue
                if tl[k] <= TILT_MIN_DEG:
                    continue
                vw = vel[k]
                vb = R[k].T @ vw                  # reconstructed raw body twist (zero-rot bug)
                vd = R[k] @ (R[k] @ vb)           # double world<-body (two-rot bug)
                pos_fd.append(pf); v_world.append(vw); v_body.append(vb); v_double.append(vd)
    return {
        "pos_fd": np.array(pos_fd), "v_world": np.array(v_world),
        "v_body": np.array(v_body), "v_double": np.array(v_double),
        "n_runs": n_runs, "n_skipped": n_skipped,
    }


def _per_axis(pos_fd, cand):
    """Per-axis corr + median |residual| of a velocity candidate vs FD-of-pos truth."""
    cors, resid = [], []
    for ax in range(3):
        a, b = pos_fd[:, ax], cand[:, ax]
        cors.append(float(np.corrcoef(a, b)[0, 1]) if b.std() > 1e-9 else np.nan)
        resid.append(float(np.median(np.abs(a - b))))
    return cors, resid


def evaluate():
    """Compute the live result + both negative controls. Returns a result dict."""
    g = _gather()
    n = len(g["pos_fd"])
    if n < MIN_TILTED:
        raise AssertionError(
            f"insufficient tilted samples ({n} < {MIN_TILTED}); cannot test the invariant "
            f"(n_runs={g['n_runs']}, n_skipped={g['n_skipped']})"
        )
    # geometric non-vacuity: body and world frames must genuinely differ in this bin
    sep = np.median(np.linalg.norm(g["v_world"] - g["v_body"], axis=1))
    world_cor, world_res = _per_axis(g["pos_fd"], g["v_world"])
    body_cor, body_res = _per_axis(g["pos_fd"], g["v_body"])
    dbl_cor, dbl_res = _per_axis(g["pos_fd"], g["v_double"])
    return {
        "n": n, "n_runs": g["n_runs"], "n_skipped": g["n_skipped"], "frame_sep": float(sep),
        "world": {"corr": world_cor, "resid": world_res},
        "body": {"corr": body_cor, "resid": body_res},
        "double": {"corr": dbl_cor, "resid": dbl_res},
    }


# ============================ pytest entry point ============================================
def test_mavlink_velocity_single_rotation():
    """stored velocity_ned == FD of pristine pos_ned through bank (single body->world rotation)."""
    r = evaluate()
    AX = "NED"

    # (0) non-vacuity: the tilted bin must actually separate the body & world frames.
    assert r["frame_sep"] > 1.0, (
        f"tilted bin does not separate frames (median |v_world-v_body|={r['frame_sep']:.2f} "
        f"<= 1 m/s) -- the invariant would be untestable"
    )

    # (1) LIVE INVARIANT: world candidate must correlate + small residual on EVERY axis.
    for ax in range(3):
        assert r["world"]["corr"][ax] >= CORR_MIN, (
            f"velocity_ned decorrelates from FD(pos_ned) on {AX[ax]}: "
            f"corr={r['world']['corr'][ax]:+.4f} < {CORR_MIN} "
            f"-- velocity is NOT single-rotation world-NED (frame-mix regression)"
        )
        assert r["world"]["resid"][ax] <= RESID_MAX, (
            f"velocity_ned residual too large on {AX[ax]}: "
            f"med|res|={r['world']['resid'][ax]:.3f} > {RESID_MAX} m/s"
        )

    # (2) NEGATIVE CONTROLS: each frame-mix bug MUST trip the corr gate on some axis,
    #     proving the test would catch a (re)introduced body-vs-world mix.
    for bug in ("body", "double"):
        assert min(r[bug]["corr"]) < BUG_CORR_MAX, (
            f"negative control '{bug}' did NOT fail the gate (min corr "
            f"{min(r[bug]['corr']):+.3f} >= {BUG_CORR_MAX}) -- the test lacks discriminating "
            f"power; do not trust a PASS"
        )


# ============================ standalone CLI ===============================================
def _fmt(triplet, w=7, p=4):
    return "[" + ", ".join(f"{x:+{w}.{p}f}" for x in triplet) + "]"


def main() -> int:
    print("=" * 78)
    print("REGRESSION: mavlink_velocity_single_rotation")
    print("  invariant: FD(pos_ned) == vel_ned through bank (single body->world rotation)")
    print("  bug class: body-vs-world velocity-frame mix (c3b5a8e)")
    print("=" * 78)
    try:
        r = evaluate()
    except AssertionError as e:
        print(f"\nFAIL (setup): {e}")
        return 1

    print(f"\ndatasets refit+postfix: {r['n_runs']} runs loaded, {r['n_skipped']} skipped "
          f"(header-only/missing-field)")
    print(f"tilted-bin samples (tilt>{TILT_MIN_DEG:.0f}deg): n={r['n']}")
    print(f"frame separation (median |v_world - v_body|): {r['frame_sep']:.2f} m/s  "
          f"(>1 => bin is non-vacuous)")

    print("\n  candidate          corr(N,E,D)                     median|residual|(N,E,D) m/s")
    for name, key in (("WORLD (live/good)", "world"),
                      ("BODY  (0x rot bug)", "body"),
                      ("DOUBLE(2x rot bug)", "double")):
        print(f"  {name:18s} {_fmt(r[key]['corr']):32s} {_fmt(r[key]['resid'], 6, 2)}")

    # apply the same gates the pytest test asserts
    ok = True
    world_min_corr = min(r["world"]["corr"]); world_max_res = max(r["world"]["resid"])
    if not (world_min_corr >= CORR_MIN and world_max_res <= RESID_MAX):
        ok = False
    if r["frame_sep"] <= 1.0:
        ok = False
    bug_caught = min(r["body"]["corr"]) < BUG_CORR_MAX and min(r["double"]["corr"]) < BUG_CORR_MAX
    if not bug_caught:
        ok = False

    print(f"\n  live  : min-axis corr={world_min_corr:+.4f} (>= {CORR_MIN}), "
          f"max-axis med|res|={world_max_res:.3f} (<= {RESID_MAX})")
    print(f"  controls caught: body min-corr={min(r['body']['corr']):+.3f}, "
          f"double min-corr={min(r['double']['corr']):+.3f} (both < {BUG_CORR_MAX})")
    print("\n" + ("PASS" if ok else "FAIL")
          + " -- velocity_ned is single-rotation world-NED; frame-mix would be caught"
          if ok else "\nFAIL -- invariant violated OR negative control did not trip")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
