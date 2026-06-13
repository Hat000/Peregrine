"""TRAIN<->DEPLOY OBS SEAM -- element-wise reconstruction regression test.

================================================================================
BUG CLASS THIS CATCHES
================================================================================
Drift between the TRAINING observation builder (peregrine_racing.get_observations)
and the DEPLOY observation builder (fly_rl.build_obs / obs_from_zup) at the seam
where recorded physical state -> the 17-dim policy obs. Concretely it guards the
*structural* half of the seam -- the part that has a single correct answer that is
NOT subject to a proper-rotation sign-alias:

  * gate-relative POSITION   obs[0:3]  = R_w2g @ (gate_pos - pos)   (NED->Z-up FLIP)
  * gate-relative VELOCITY   obs[3:6]  = R_w2g @ vel
  * NEXT-GATE relpos/relyaw  obs[13:17] (precomputed lookup, gate-index indexed)
  * COLLECTIVE action-memory obs[12]   = previous-tick RESCALED normed_thrust,
                                          and EXACTLY 0.0 at episode start.

If any of these drift -- wrong gate-relative frame (R_w2g sign / NED<->Z-up FLIP),
a mis-indexed next-gate lookup, a stale gate table, or a broken collective-memory
init (obs[12] not zeroed at reset, or fed the unrescaled value) -- the train policy
and the deploy policy see DIFFERENT inputs and the checkpoint silently mis-behaves.
This is the same family that has bitten this project four times:
    1. R_y(pi) ODOMETRY-quat conjugation,
    2. body-vs-world velocity-frame mix,
    3. corner_to_center 180deg flip,
    4. ATTITUDE.pitch sign inversion.

The ATTITUDE/RATE half of the obs (obs[6:12]) carries the quat-conjugation /
rate-sign alias, which a pure element-wise reconstruction CANNOT pin down from a
recording PRODUCED UNDER A SUPERSEDED CONVENTION (see "WHY ..." below). That half is
guarded TWO ways:
  * deterministically, by a FULL-17-dim element-wise bit-exact check on the POSTFIX
    recordings (captured AFTER the 93023cf frame-audit fix, i.e. UNDER the current
    convention) -- there the current deploy build_obs reproduces every logged obs
    dim to <=2e-4 (recording round-off only), so any future seam drift in the
    attitude/rate/quat-conjugation path trips immediately and deterministically; and
  * statistically / convention-independently, by the EXTERNAL force invariant
    (Test A, East-axis discriminator) on the larger refit pool, reusing the shared
    substrate -- which holds regardless of which convention captured the recording.

================================================================================
WHY INTERNAL-CONSISTENCY / ELEMENT-WISE-ONLY CHECKS MISS THE ATTITUDE HALF
================================================================================
A proper-rotation conjugation (R_y(pi): q_true = q_raw*[1,-1,1,-1]) is a
self-consistent relabeling of the body frame. A recording's logged obs[6:12]
(attitude + body rate) is INTERNALLY consistent with whatever convention produced
it -- so reconstructing obs[6:12] "from the recording" and comparing element-wise
only proves the recording agrees with ITSELF, never that the convention is the
TRUE one. Worse: the inc6 recordings used here were produced by the bcc93f9-era
deploy mapping (quat AS-IS [1,1,1,1] + rate sign [+1,-1,+1]); the CURRENT tree
(93023cf) deliberately SUPERSEDED that to (q_raw*[1,-1,1,-1], rate -w_raw). So a
naive `assert fly_rl.build_obs(...) == logged_obs` would FAIL on HEAD for a known,
intended reason -- a moving target, not a bug. We therefore:
  (a) reconstruct the STRUCTURAL axes (pos_g/vel_g/next/collective) INDEPENDENTLY
      of fly_rl.py -- re-derived from the documented Z-up gate geometry -- because
      those axes are convention-version-INVARIANT (no quat enters them); and
  (b) guard the ATTITUDE/RATE axes with the EXTERNAL invariant only: finite-
      difference the PRISTINE state.velocity_ned (trusted world-NED anchor, NOT
      derived from the possibly-mirrored attitude) and compare to attitude-derived
      specific force in the TILTED bin. East is the discriminating axis; TRUE
      conjugation correlates ~+0.99, the AS-IS mirror goes negative (~ -0.84).
      Level flight is INADMISSIBLE (a rotation and its conjugate give identical
      thrust direction when body-down == world-down).

NEGATIVE CONTROL (built in): test_negative_control_structural_axes() flips the
world->gate / NED->Z-up sign in the reconstruction and asserts the element-wise
equality would BREAK -- proving the structural assertion actually has teeth and is
not trivially satisfied.

================================================================================
DATA / METHOD
================================================================================
Recordings: handoff/shadowpc-{refit,postfix}-dataset-2026-06-12/extracted, each a
per-tick debug_obs.jsonl carrying the FULL deployed schema (pos_ned/vel_ned/
q_raw_wxyz/w_raw, the produced obs + obs_labels, the action pipeline, gate_index,
normed_thrust, n_coll). Loaded via the shared substrate
handoff/ultracode-substrate-audit-2026-06-13/scratch/_audit_io.py.

Reconstruction precision target: the dumps are rounded to 5 decimals
(np.round(...,5) in fly_rl), so element-wise agreement to ~1e-4 is exact-to-dump.

Tolerant of empty/header-only runs (e.g. 20260612_184029_..._std_f2 is header-only;
load_run raises ValueError -> we skip it, no retry loop).

DOES NOT import rl/contact_true_eval.py (out of scope; edited elsewhere).

Run directly:   .venv/Scripts/python.exe <this file>     (prints PASS/FAIL + numbers)
Run via pytest: pytest <this file>                        (test_*() with assertions)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import pytest

# --- shared substrate (tests/_audit_io.py) lives beside this file --------------------
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

import _audit_io as A  # noqa: E402  (shared loader + Test-A force model)

# refit/extracted is gitignored (only its .zip is tracked); postfix/extracted IS tracked.
# Skip (don't ERROR) on a checkout lacking either set -- on the canonical laptop both exist.
pytestmark = pytest.mark.skipif(
    not (A.dataset_present("postfix") and A.dataset_present("refit")),
    reason="ShadowPC audit recordings (postfix+refit extracted) not present",
)


# =====================================================================================
# Course geometry -- re-derived INDEPENDENTLY of fly_rl.py / peregrine_racing.py.
#
# These are the 6 VQ1 gates in the DiffAero Z-up frame (Z-up = NED * [1,-1,-1]) with
# all gate yaws = pi, copied from rl/peregrine_course_diffaero.json (the SAME source
# fly_rl and the training course load). We deliberately do NOT import the deploy
# module's constants: the recordings encode the bcc93f9 attitude convention, which the
# current tree has superseded, so coupling this test to HEAD's _ODO_* constants would
# make a convention-version mismatch masquerade as a structural-seam failure. The
# STRUCTURAL axes (pos_g/vel_g/next/collective) are convention-version-INVARIANT, so we
# pin them against an independent re-derivation of the geometry instead.
# =====================================================================================
N_GATES = 6
_GATE_POS_ZUP = np.array([
    [-23.2979679107666,    0.39990234375,       1.3919580206274986],
    [-46.89374923706055,   2.499990224838257,  -3.708041787147522],
    [-74.59375,           -1.2000097036361694, -12.308041214942932],
    [-111.49374389648438,  5.099989891052246,  -23.208040833473206],
    [-135.49374389648438,  0.7999902367591858, -23.995653748512268],
    [-159.19374084472656,  4.399990081787109,  -24.60804045200348],
], dtype=np.float64)

# NED <-> Z-up world (and FRD <-> FLU body) frame flip: R_x(pi) = diag(1,-1,-1).
_FLIP = np.array([1.0, -1.0, -1.0], dtype=np.float64)

# World->gate rotation for the uniform gate yaw = pi: R_z(pi) = diag(-1,-1,1).
_R_W2G = np.diag([-1.0, -1.0, 1.0])

# Precomputed next-gate relative position (peregrine_racing __init__ /
# fly_rl _GATE_REL_POS): gate_rel_pos[i] = R_w2g(yaw[i-1]) @ (gate_pos[i]-gate_pos[i-1]).
# All yaws identical -> single R_w2g. Python [-1] wraps to gate 5 for i=0.
_GATE_REL_POS = np.array(
    [_R_W2G @ (_GATE_POS_ZUP[i] - _GATE_POS_ZUP[i - 1]) for i in range(N_GATES)],
    dtype=np.float64,
)
_GATE_YAW_REL = np.zeros(N_GATES, dtype=np.float64)   # 0 everywhere (uniform gate yaw)


# =====================================================================================
# Independent reconstruction of the STRUCTURAL obs axes from a recorded physical state.
# =====================================================================================
def reconstruct_structural(pos_ned, vel_ned, gate_index, last_normed_thrust,
                           *, w2g_sign: float = 1.0):
    """Rebuild the convention-invariant obs axes from pristine world-NED state.

    Returns a dict of the reconstructed sub-vectors so the caller can diff them
    against the logged obs slices:
        pos_g  -> obs[0:3]
        vel_g  -> obs[3:6]
        nxt    -> obs[13:17]   (next-gate relpos[3] + relyaw[1])
        coll   -> obs[12]      (previous-tick rescaled normed_thrust; 0.0 at reset)

    ``w2g_sign``: NEGATIVE-CONTROL knob. 1.0 = the true world->gate / NED->Z-up
    transform. Flipping it to -1.0 corrupts the gate-relative frame exactly the way
    a re-introduced R_w2g / FLIP sign bug would, so the negative-control test can
    confirm the element-wise assertion has teeth.
    """
    flip = _FLIP * w2g_sign
    pos_zup = np.asarray(pos_ned, dtype=np.float64) * flip
    vel_zup = np.asarray(vel_ned, dtype=np.float64) * flip
    gi = int(gate_index)
    nxt_i = min(gi + 1, N_GATES - 1)
    R_w2g = _R_W2G * w2g_sign
    pos_g = R_w2g @ (_GATE_POS_ZUP[gi] - pos_zup)
    vel_g = R_w2g @ vel_zup
    nxt = np.concatenate([_GATE_REL_POS[nxt_i], [_GATE_YAW_REL[nxt_i]]])
    return {
        "pos_g": pos_g,
        "vel_g": vel_g,
        "nxt": nxt,
        "coll": float(last_normed_thrust),
    }


# =====================================================================================
# Per-run / pooled element-wise residual of the structural axes.
# =====================================================================================
# The fly_rl debug dump rounds every logged field to 5 decimals (np.round(...,5)), so
# the floor on achievable agreement is ~5e-5 per element plus a touch of accumulation
# in the matrix product. 1e-3 is exact-to-dump with comfortable margin and far below
# any real convention bug (which flips a SIGN -> O(1..100) residual).
STRUCT_TOL = 1.0e-3
MIN_TOTAL_TICKS = 1000          # require a meaningful pool before asserting PASS

# External-invariant (attitude/rate half) thresholds -- mirror the force-vs-FD canary.
TRUE_EAST_CORR_MIN = 0.90       # TRUE R_y(pi) conjugation correlates strongly +ve on East
ASIS_EAST_CORR_MAX = -0.40      # AS-IS mirror (positive control) must be clearly negative
TILT_MIN_DEG = 35.0
MIN_TILTED_SAMPLES = 200


def _struct_residuals_for_run(run, *, w2g_sign: float = 1.0):
    """Return per-element max |reconstructed - logged| for the structural axes of one run.

    obs[12] (collective memory) is reconstructed as the PREVIOUS tick's logged
    normed_thrust, with the episode-start value forced to 0.0 -- this is exactly the
    training contract (BaseEnv.last_action zeroed in reset_idx; obs[12] = rescaled
    last normed_thrust). A regression that fails to zero obs[12] at reset, or that
    feeds the unrescaled collective, shows up here.
    """
    obs = run["obs"]
    pos = run["pos"]
    vel = run["vel"]
    gi = run["gate_index"]
    nt = run["normed_thrust"]
    n = len(obs)

    worst = {"pos_g": 0.0, "vel_g": 0.0, "nxt": 0.0, "coll": 0.0}
    for k in range(n):
        last_nt = 0.0 if k == 0 else float(nt[k - 1])   # zeroed at episode start
        rec = reconstruct_structural(pos[k], vel[k], gi[k], last_nt, w2g_sign=w2g_sign)
        lo = obs[k]
        worst["pos_g"] = max(worst["pos_g"], float(np.max(np.abs(rec["pos_g"] - lo[0:3]))))
        worst["vel_g"] = max(worst["vel_g"], float(np.max(np.abs(rec["vel_g"] - lo[3:6]))))
        worst["nxt"] = max(worst["nxt"], float(np.max(np.abs(rec["nxt"] - lo[13:17]))))
        worst["coll"] = max(worst["coll"], abs(rec["coll"] - float(lo[12])))
    return worst, n


def _pooled_structural(dataset: str, *, w2g_sign: float = 1.0):
    """Worst-case structural residual across every loadable run in a dataset.

    Returns dict: per-axis worst |residual|, total ticks, n_runs, skipped run names.
    """
    runs = A.list_runs(dataset)
    agg = {"pos_g": 0.0, "vel_g": 0.0, "nxt": 0.0, "coll": 0.0}
    total_ticks = 0
    n_runs = 0
    skipped = []
    for run_dir in runs:
        try:
            run = A.load_run(run_dir)
        except ValueError:
            skipped.append(run_dir.name)        # header-only / empty -> tolerate & skip
            continue
        # require the structural fields to be present (tolerant of schema variation)
        if not all(k in run for k in ("obs", "pos", "vel", "gate_index", "normed_thrust")):
            skipped.append(run_dir.name + " (missing fields)")
            continue
        worst, n = _struct_residuals_for_run(run, w2g_sign=w2g_sign)
        for key in agg:
            agg[key] = max(agg[key], worst[key])
        total_ticks += n
        n_runs += 1
    return {"axes": agg, "total_ticks": total_ticks, "n_runs": n_runs, "skipped": skipped}


# =====================================================================================
# External-invariant (attitude/rate half) -- pooled Test-A East discriminator.
# =====================================================================================
def _pooled_force_vs_fd(dataset: str, sign, *, use_lapse: bool, tilt_min: float = TILT_MIN_DEG):
    """Pool the Test-A force-vs-FD residuals (tilted bin) across every loadable run.

    Mirrors the force-vs-fd canary: finite-difference of pristine vel_ned vs the
    attitude-derived specific force, per axis, in the tilt>tilt_min bin. East is the
    discriminating axis. Returns per-axis corr + the pooled tilted-sample count.
    """
    pooled_model = {0: [], 1: [], 2: []}
    pooled_meas = {0: [], 1: [], 2: []}
    for run_dir in A.list_runs(dataset):
        try:
            run = A.load_run(run_dir)
        except ValueError:
            continue
        if not all(k in run for k in ("t", "vel", "q_raw", "coll")):
            continue
        t, vel, q, coll = run["t"], run["vel"], run["q_raw"], run["coll"]
        n = len(t)
        if n < 6:
            continue
        R = A.Rmats(q, sign=sign)
        tilt_raw = A.tilt_deg(A.Rmats(q, sign=A.CAND_ASIS))  # tilt invariant across candidates
        for k in range(2, n - 2):
            dt = t[k + 1] - t[k - 1]
            if not (0.05 < dt < 0.09):
                continue
            a_meas = (vel[k + 1] - vel[k - 1]) / dt
            if not np.all(np.isfinite(a_meas)) or np.max(np.abs(a_meas)) > 90:
                continue
            if tilt_raw[k] <= tilt_min:
                continue
            a_model = A.force_model(R[k], vel[k], coll[k - 2], use_lapse=use_lapse)
            for ax in range(3):
                pooled_model[ax].append(a_model[ax])
                pooled_meas[ax].append(a_meas[ax])
    AX = ["N", "E", "D"]
    out = {"n_tilted": len(pooled_model[1]), "axes": {}}
    for ax in range(3):
        md = np.array(pooled_model[ax])
        ms = np.array(pooled_meas[ax])
        if md.size < 5 or md.std() < 1e-6:
            out["axes"][AX[ax]] = float("nan")
        else:
            out["axes"][AX[ax]] = float(np.corrcoef(md, ms)[0, 1])
    return out


# =====================================================================================
# pytest entry points
# =====================================================================================
def test_train_deploy_structural_obs_elementwise():
    """STRUCTURAL seam: pos_g/vel_g/next-gate/collective reconstruct element-wise.

    Asserted over BOTH datasets pooled. The reconstruction is independent of the
    deploy module's attitude convention (re-derived geometry), so it is a stable
    regression target across convention revisions. A drift in the gate-relative
    frame, gate-index lookup, gate table, or collective-memory init -> O(1..100)
    residual -> trips this assertion.
    """
    res_refit = _pooled_structural("refit")
    res_post = _pooled_structural("postfix")
    total = res_refit["total_ticks"] + res_post["total_ticks"]

    assert total >= MIN_TOTAL_TICKS, (
        f"too few reconstructable ticks to judge the seam: {total} < {MIN_TOTAL_TICKS}. "
        f"Check the recordings/datasets (skipped refit={res_refit['skipped']}, "
        f"postfix={res_post['skipped']})."
    )

    for name, res in (("refit", res_refit), ("postfix", res_post)):
        ax = res["axes"]
        for axis_name, val in ax.items():
            assert val <= STRUCT_TOL, (
                f"TRAIN<->DEPLOY OBS SEAM TRIPPED [{name}/{axis_name}]: worst element-wise "
                f"|reconstructed - logged| = {val:.4g} > {STRUCT_TOL:g}. The deploy obs no "
                f"longer matches an independent reconstruction of the training "
                f"get_observations layout. Likely a gate-relative-frame (R_w2g / NED<->Z-up "
                f"FLIP) sign flip, a mis-indexed/stale next-gate lookup, or a broken "
                f"collective-memory init (obs[12] not zeroed at reset / unrescaled). "
                f"({res['n_runs']} runs, {res['total_ticks']} ticks)."
            )


def test_negative_control_structural_axes():
    """NEGATIVE CONTROL: flipping the world->gate / NED->Z-up sign MUST break the
    element-wise equality -- proving the structural assertion has teeth and is not
    trivially satisfied (e.g. by all-zero or degenerate reconstructions).

    NOTE (symmetry caveat, verified numerically): flipping w2g_sign negates BOTH
    R_w2g and the NED->Z-up FLIP, and for VELOCITY those two negations cancel
    (vel_g = (-R_w2g)@(-FLIP*vel) = R_w2g@(FLIP*vel)) -- so vel_g is INVARIANT under
    this particular flip and is NOT a valid tripwire here. POSITION does not cancel
    (the gate-position term is not flipped), so pos_g blows up to race-scale tens of
    metres. We therefore assert on pos_g, which is the decisive, non-degenerate tell.
    """
    flipped = _pooled_structural("refit", w2g_sign=-1.0)
    pos_g = flipped["axes"]["pos_g"]
    # A sign-flipped gate-relative frame must blow the position residual far past the
    # tolerance (gate positions are tens of metres downrange at race scale).
    assert pos_g > STRUCT_TOL * 100, (
        f"NEGATIVE CONTROL FAILED: sign-flipping the world->gate transform left "
        f"pos_g residual {pos_g:.4g} within {STRUCT_TOL * 100:g}. The structural "
        f"assertion is not discriminating -- it would pass even with a corrupted "
        f"gate-relative frame. Re-examine the reconstruction."
    )


def test_attitude_rate_external_invariant():
    """ATTITUDE/RATE half of the obs seam: guarded by the EXTERNAL force invariant.

    The attitude+rate axes (obs[6:12]) carry the quat-conjugation / rate-sign alias
    that an element-wise reconstruction cannot pin from a recording (it would only
    prove self-consistency). The discriminating external invariant is the force-vs-FD
    East-axis test. Production-faithful force (use_lapse=False), 'refit' (largest pool):
      (1) TRUE  [1,-1,1,-1] East corr >= TRUE_EAST_CORR_MIN   (the truth holds)
      (2) AS-IS [1, 1,1, 1] East corr <= ASIS_EAST_CORR_MAX   (positive control: the
          raw-telemetry mirror is anti-correlated -> the probe still discriminates).
    """
    true_res = _pooled_force_vs_fd("refit", A.CAND_TRUE, use_lapse=False)
    asis_res = _pooled_force_vs_fd("refit", A.CAND_ASIS, use_lapse=False)

    assert true_res["n_tilted"] >= MIN_TILTED_SAMPLES, (
        f"too few tilted (>{TILT_MIN_DEG:.0f} deg) samples to judge attitude: "
        f"{true_res['n_tilted']} < {MIN_TILTED_SAMPLES}."
    )
    true_e = true_res["axes"]["E"]
    asis_e = asis_res["axes"]["E"]
    assert true_e >= TRUE_EAST_CORR_MIN, (
        f"ATTITUDE SEAM TRIPPED: TRUE [1,-1,1,-1] East corr {true_e:+.3f} "
        f"< {TRUE_EAST_CORR_MIN:+.2f}. The external invariant no longer favors the "
        f"R_y(pi) conjugation -- a quat conjugation / rate-sign mirror may be back in "
        f"the deploy/vision attitude path. Re-derive the convention."
    )
    assert asis_e <= ASIS_EAST_CORR_MAX, (
        f"POSITIVE-CONTROL LOST: AS-IS [1,1,1,1] East corr {asis_e:+.3f} "
        f"> {ASIS_EAST_CORR_MAX:+.2f}. The discriminating signal is gone (telemetry "
        f"mirror may no longer be present) -- this test can no longer catch a "
        f"re-introduced conjugation. Re-derive the test."
    )


# =====================================================================================
# FULL-17-dim deterministic guard on the POSTFIX set (captured under the CURRENT
# convention). Here -- and ONLY here -- the deploy build_obs reproduces every logged
# dim bit-exactly, so we assert the WHOLE seam (incl. obs[6:12] attitude/rate/quat) at
# once. We compare the recorded obs against (a) fly_rl.build_obs (the deploy builder
# under test) AND (b) the independent structural reconstruction, so neither code path
# is trusted alone. The refit set is the convention-supersession positive control.
# =====================================================================================
FULL_TOL = 2.0e-4               # round-off ceiling (obs rounded to 5 dp, telem to 6 dp)
_ANGLE_DIMS = (6, 7, 8, 16)     # rpy + next-gate relyaw: compare modulo 2*pi
_YAW_BODYZ_DIMS = {8, 11}       # rpy_g_yaw + w_fluz: the axes the bcc93f9 mirror flips


_REPO_ROOT = A.ROOT                      # repo root (depth-independent, from _audit_io)
if str(_REPO_ROOT / "rl") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "rl"))


def _build_obs_deploy(run, k, last_normed):
    """Run the DEPLOY builder fly_rl.build_obs on tick k of a loaded run (duck-typed state)."""
    import fly_rl  # imported lazily so the structural tests stay fly_rl-free

    class _S:
        pass
    s = _S()
    s.position_ned = run["pos"][k]
    s.velocity_ned = run["vel"][k]
    s.orientation_ned_wxyz = run["q_raw"][k]
    s.angular_rate_body = run["w_raw"][k]
    vf = bool(run["meta"].get("virtual_flip", True))
    return fly_rl.build_obs(s, int(run["gate_index"][k]), float(last_normed),
                            virtual_flip=vf).astype(np.float64)


def _full_obs_maxerr(run):
    """Per-dim max |build_obs(raw telemetry) - logged obs| (angle dims modulo 2*pi)."""
    obs = run["obs"]
    nt = run["normed_thrust"]
    dmax = np.zeros(17)
    for k in range(len(obs)):
        last_nt = 0.0 if k == 0 else float(nt[k - 1])    # obs[12] action-memory, 0 at reset
        rec = _build_obs_deploy(run, k, last_nt)
        e = rec - obs[k]
        for j in _ANGLE_DIMS:
            e[j] = (e[j] + np.pi) % (2 * np.pi) - np.pi
        dmax = np.maximum(dmax, np.abs(e))
    return dmax


def test_full_obs_bitexact_postfix():
    """FULL-17-dim deterministic seam guard on the current-convention POSTFIX set.

    Reconstructs EVERY obs dim (including the attitude/rate/quat-conjugation half
    obs[6:12]) from raw telemetry via the deploy build_obs and asserts element-wise
    equality with the logged obs. This is the deterministic complement to the
    statistical force invariant: it trips immediately on ANY future drift of the obs
    layout, gate frame, NED<->Z-up flip, virtual flip, quat conjugation, rate sign, or
    collective memory -- on data captured under the same (current) convention.
    """
    runs = []
    for rd in A.list_runs("postfix"):
        try:
            runs.append(A.load_run(rd))
        except ValueError:
            continue
    assert runs, "no non-empty postfix runs -- dataset missing?"
    worst = 0.0
    for run in runs:
        dmax = _full_obs_maxerr(run)
        bad = np.where(dmax > FULL_TOL)[0].tolist()
        assert not bad, (
            f"FULL OBS SEAM TRIPPED [{run['name']}]: dims {bad} drift "
            f"(max|err|={np.round(dmax[bad], 4).tolist()} > {FULL_TOL:g}). The deploy "
            f"build_obs no longer matches the logged obs on current-convention data -- "
            f"a regression in obs layout / gate frame / quat conjugation / rate sign / "
            f"virtual flip / collective memory."
        )
        worst = max(worst, float(dmax.max()))
    assert worst <= FULL_TOL


def test_refit_convention_supersession_control():
    """CONVENTION-SUPERSESSION control: the superseded bcc93f9 'refit' recordings must
    DISAGREE with the current deploy convention on exactly the yaw/body-z dims
    (8=rpy_g_yaw, 11=w_fluz; the rollfix runs add their roll axis 6/9). This confirms
    the full-obs comparison fingerprints a real convention flip -- it is the very mirror
    this project chased -- and documents that current-tree faithfulness lives on POSTFIX,
    not on the pre-fix refit set. If a future refit-era recording ever matched HEAD
    cleanly, the convention story banked in fly_rl's docstring would be wrong.
    """
    matched, mismatched_without_yaw = [], []
    n_checked = 0
    for rd in A.list_runs("refit"):
        try:
            run = A.load_run(rd)
        except ValueError:
            continue
        n_checked += 1
        dmax = _full_obs_maxerr(run)
        bad = set(np.where(dmax > FULL_TOL)[0].tolist())
        if not bad:
            matched.append(run["name"])
        elif not (_YAW_BODYZ_DIMS & bad):
            mismatched_without_yaw.append((run["name"], sorted(bad)))
    assert n_checked > 0, "no non-empty refit runs found"
    assert not matched, (
        f"convention story BROKEN: refit (superseded bcc93f9) runs {matched} now match the "
        f"CURRENT deploy convention bit-exactly -- the documented bcc93f9->93023cf supersession "
        f"would be wrong."
    )
    assert not mismatched_without_yaw, (
        f"refit mismatch did NOT include the expected yaw/body-z dims {sorted(_YAW_BODYZ_DIMS)}: "
        f"{mismatched_without_yaw}. The mirror fingerprint changed -- re-derive the convention."
    )


# =====================================================================================
# Standalone runner
# =====================================================================================
def _fmt_struct(res) -> str:
    ax = res["axes"]
    return (f"pos_g={ax['pos_g']:.2e}  vel_g={ax['vel_g']:.2e}  "
            f"nxt={ax['nxt']:.2e}  coll_prev={ax['coll']:.2e}")


def main() -> int:
    print("=" * 78)
    print("TRAIN<->DEPLOY OBS SEAM  --  element-wise reconstruction of the 17-dim obs")
    print("Structural axes (pos_g/vel_g/next/collective): re-derived geometry, "
          "convention-INVARIANT.")
    print("Attitude/rate axes (obs[6:12]): external force-vs-FD East invariant.")
    print("=" * 78)

    ok = True

    # ---- structural (the element-wise seam) ----
    print("\n[STRUCTURAL]  worst element-wise |reconstructed - logged|  "
          f"(tol {STRUCT_TOL:g})")
    res_refit = _pooled_structural("refit")
    res_post = _pooled_structural("postfix")
    for name, res in (("refit", res_refit), ("postfix", res_post)):
        print(f"  [{name}] n_runs={res['n_runs']} ticks={res['total_ticks']}"
              + (f" skipped={res['skipped']}" if res["skipped"] else ""))
        print(f"          {_fmt_struct(res)}")
        worst = max(res["axes"].values())
        passed = worst <= STRUCT_TOL
        ok = ok and passed
        print(f"          -> worst {worst:.2e} <= {STRUCT_TOL:g} ? "
              f"{'PASS' if passed else 'FAIL'}")
    total = res_refit["total_ticks"] + res_post["total_ticks"]
    ok = ok and (total >= MIN_TOTAL_TICKS)

    # ---- negative control (teeth check) ----
    print("\n[NEGATIVE CONTROL]  flip world->gate sign -> structural eq MUST break")
    flipped = _pooled_structural("refit", w2g_sign=-1.0)
    pos_g = flipped["axes"]["pos_g"]
    vel_g = flipped["axes"]["vel_g"]
    nc_ok = pos_g > STRUCT_TOL * 100
    ok = ok and nc_ok
    print(f"  flipped pos_g={pos_g:.2e}  vel_g={vel_g:.2e} (vel_g invariant by "
          f"R_w2g/FLIP cancellation -- pos_g is the tripwire)")
    print(f"  -> pos_g breaks (> {STRUCT_TOL * 100:g}) ? {'PASS' if nc_ok else 'FAIL'}")

    # ---- attitude/rate external invariant ----
    print("\n[ATTITUDE/RATE]  external force-vs-FD East invariant (production force) "
          "[refit]")
    true_p = _pooled_force_vs_fd("refit", A.CAND_TRUE, use_lapse=False)
    asis_p = _pooled_force_vs_fd("refit", A.CAND_ASIS, use_lapse=False)
    true_r = _pooled_force_vs_fd("refit", A.CAND_TRUE, use_lapse=True)
    asis_r = _pooled_force_vs_fd("refit", A.CAND_ASIS, use_lapse=True)
    te, ae = true_p["axes"]["E"], asis_p["axes"]["E"]
    print(f"  n_tilted(East)={true_p['n_tilted']}")
    print(f"  production (use_lapse=False) [ASSERTED]:  "
          f"TRUE E={te:+.3f}   AS-IS E={ae:+.3f}")
    print(f"  reference (use_lapse=True)   [report]:    "
          f"TRUE E={true_r['axes']['E']:+.3f}   AS-IS E={asis_r['axes']['E']:+.3f}")
    att_true_ok = te >= TRUE_EAST_CORR_MIN
    att_asis_ok = ae <= ASIS_EAST_CORR_MAX
    att_n_ok = true_p["n_tilted"] >= MIN_TILTED_SAMPLES
    ok = ok and att_true_ok and att_asis_ok and att_n_ok
    print(f"  -> TRUE East {te:+.3f} >= {TRUE_EAST_CORR_MIN:+.2f} ? "
          f"{'PASS' if att_true_ok else 'FAIL'}")
    print(f"  -> AS-IS East {ae:+.3f} <= {ASIS_EAST_CORR_MAX:+.2f} (pos ctrl) ? "
          f"{'PASS' if att_asis_ok else 'FAIL'}")

    # ---- FULL-17-dim deterministic guard on the current-convention postfix set ----
    print("\n[FULL OBS / POSTFIX]  deploy build_obs(raw telemetry) == logged obs, ALL 17 "
          f"dims (tol {FULL_TOL:g})")
    postfix_full = []
    for rd in A.list_runs("postfix"):
        try:
            postfix_full.append(A.load_run(rd))
        except ValueError:
            pass
    full_worst = 0.0
    full_ok = bool(postfix_full)
    for run in postfix_full:
        dmax = _full_obs_maxerr(run)
        bad = np.where(dmax > FULL_TOL)[0].tolist()
        full_worst = max(full_worst, float(dmax.max()))
        full_ok = full_ok and not bad
        print(f"  {'OK  ' if not bad else 'FAIL'} {run['name'][:46]:46s} "
              f"max|err|={dmax.max():.2e} bad_dims={bad}")
    ok = ok and full_ok
    print(f"  -> all postfix dims <= {FULL_TOL:g} ? {'PASS' if full_ok else 'FAIL'} "
          f"(worst {full_worst:.2e})")

    # ---- convention-supersession control (refit mismatches on yaw/body-z dims) ----
    print("\n[CONVENTION CONTROL]  superseded bcc93f9 'refit' set must mismatch HEAD on "
          "yaw/body-z dims {8,11}:")
    conv_ok = True
    n_conv = 0
    for rd in A.list_runs("refit"):
        try:
            run = A.load_run(rd)
        except ValueError:
            continue
        n_conv += 1
        dmax = _full_obs_maxerr(run)
        bad = sorted(np.where(dmax > FULL_TOL)[0].tolist())
        has_yaw = bool(_YAW_BODYZ_DIMS & set(bad))
        conv_ok = conv_ok and bool(bad) and has_yaw
        if n_conv <= 4:
            print(f"  {'OK  ' if (bad and has_yaw) else 'FAIL'} {run['name'][:42]:42s} "
                  f"bad_dims={bad}  yaw/body-z flagged={has_yaw}")
    conv_ok = conv_ok and (n_conv > 0)
    print(f"  -> all {n_conv} refit runs fingerprint the mirror ? {'PASS' if conv_ok else 'FAIL'}")
    ok = ok and conv_ok

    print("\n" + "=" * 78)
    print(f"VERDICT: {'PASS' if ok else 'FAIL'}")
    print("Guards: train<->deploy obs-seam drift -- gate-relative frame / NED<->Z-up "
          "FLIP, next-gate lookup, collective-memory init (structural, element-wise), "
          "and quat-conjugation / rate-sign mirror (attitude, external invariant).")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
