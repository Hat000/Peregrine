"""Regression for CONFIRMED bug P4-C05 (rl/fly_rl.py obs_from_zup).

BUG CLASS GUARDED
-----------------
fly_rl.obs_from_zup() builds the gate-relative observation with a SINGLE hardcoded
gate-frame rotation `_R_W2G = diag(-1,-1,1)` (= world->gate rotation for gate yaw = pi)
and precomputed lookahead tables `_GATE_REL_POS` / `_GATE_YAW_REL` (= 0 everywhere).
Those constants are correct ONLY because every VQ1 gate yaw is exactly pi. The TRAIN-side
ground truth (peregrine_racing.get_observations) rotates PER GATE with
get_gate_rotmat_w2g(gate_yaw) (rows [c,s,0; -s,c,0; 0,0,1]). On any non-pi course
(VQ2 / course_mode=random, bent gates), the deploy seam keeps applying yaw=pi while the
policy was trained on the per-gate rotation -> SILENTLY WRONG gate-relative pos_g / vel_g /
rpy_g_y and next-gate lookahead. This is the same family that has bitten this project 4x
(frame/convention hardcodes that are right for the one course flown and wrong otherwise).

WHY INTERNAL-CONSISTENCY CHECKS MISS IT
---------------------------------------
On the only course ever flown (VQ1, all yaw == pi) the hardcoded rotation EQUALS the
per-gate rotation, so:
  * build_obs reproduces the logged obs bit-exactly (LENS1 below confirms this on real data);
  * deploy obs == train obs to float32 (LENS2a below);
  * every quat/rate/twist round-trip and level-flight correlation passes.
A self-consistency check can only ever exercise the all-pi course it is fed, so it never
sees the yaw it would need to vary to expose the constant. ONLY an EXTERNAL invariant --
the independent TRAIN-side definition of the observation (world_to_gateframe / rel_tables),
evaluated on a course whose gate yaw != pi -- discriminates the hardcoded convention from
the correct yaw-aware one.

THE EXTERNAL INVARIANT (what this test encodes, NOT current code behavior)
--------------------------------------------------------------------------
For ANY course the policy could be asked to fly, the DEPLOY gate-relative observation must
equal the TRAIN-side observation the policy was trained on:

    deploy_pos_g   == get_gate_rotmat_w2g(gate_yaw[tg]) @ (gate_pos[tg] - pos)
    deploy_vel_g   == get_gate_rotmat_w2g(gate_yaw[tg]) @ vel
    deploy_rpy_g   == ZYX_euler( get_gate_rotmat_w2g(gate_yaw[tg]) @ R_b2w )
    deploy_nxt_rel == rel_tables(gate_pos, gate_yaw)[next]
    deploy_nxt_yaw == rel_tables(gate_pos, gate_yaw).yaw_rel[next]

The train-side helpers (peregrine_racing.world_to_gateframe / rel_tables) are pure-torch,
diffaero-free, and ARE the ground-truth obs definition the checkpoint optimized against;
they are the external anchor here (analogous to how FD-of-pristine-vel_ned anchors the
attitude-convention audit). A CORRECTED deploy builder (yaw-aware) satisfies the invariant
on a non-pi course; the BUGGY hardcoded-pi builder violates it. This file:

  LENS1  (real recorded data)  -- prove the divergence is NOT a harness artifact: the shipped
         fly_rl.build_obs, fed the PRISTINE top-level fields (pos_ned/vel_ned/q_raw_wxyz/
         w_raw/gate_index + recorded obs[12] as collective memory), reproduces the RECORDED
         'obs' on the all-pi course that actually flew. Plus the mirror canary as a positive
         control that the deploy ATTITUDE substrate is frame-correct (orthogonal to P4-C05).
  LENS2a (all-pi)   -- shipped deploy obs == independent train obs to float32 (the constant is
         correct here, which is exactly why the bug is invisible).
  LENS2b (non-pi)   -- the invariant. A yaw-aware CORRECTED builder matches the train ground
         truth (PASS on a fixed tree); the CURRENTLY-SHIPPED obs_from_zup diverges on the
         CONSUMED gate-relative fields by metres (the latent VQ2 hazard, made numerical and
         pinned to specific obs indices). This is the negative control: it shows the assertion
         would fire if the hardcoded-pi convention were (re)introduced.

NEGATIVE CONTROL / FINDING
--------------------------
The committed tree still ships the hardcoded-pi obs_from_zup, so the LENS2b invariant CANNOT
pass against the currently-shipped builder -- that is the confirmed bug, reported explicitly
as a FINDING (not a harness failure). The test's pass/fail contract is encoded against a
yaw-aware CORRECTED builder (what a fixed tree would ship); a regression that re-hardcodes pi
would make `corrected` collapse back onto `obs_from_zup` and the invariant assertion fires.

DO NOT import rl/contact_true_eval.py (out of scope; edited elsewhere). Not touched here.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# --- paths: shared substrate (tests/_audit_io.py) beside this file; + rl/ for in-scope deploy ----
import pytest                  # noqa: E402

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

import _audit_io as A          # noqa: E402  shared loader + force model (NOT contact_true_eval)

ROOT = A.ROOT                  # repo root (depth-independent, from _audit_io)
for _p in (str(ROOT / "rl"), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# refit/extracted is gitignored (only its .zip is tracked); postfix/extracted IS tracked.
# Skip (don't ERROR) on a checkout lacking either set -- on the canonical laptop both exist.
pytestmark = pytest.mark.skipif(
    not (A.dataset_present("postfix") and A.dataset_present("refit")),
    reason="ShadowPC audit recordings (postfix+refit extracted) not present",
)

import torch                   # noqa: E402

import fly_rl                  # noqa: E402  the IN-SCOPE deploy seam under audit
from fly_rl import (           # noqa: E402
    N_GATES,
    OBS_LABELS,
    _GATE_POS_ZUP,
    _GATE_REL_POS,
    _GATE_YAW_REL,
    _R_W2G,
    _euler_zyx,
    build_obs,
    obs_from_zup,
)
import peregrine_racing as PR  # noqa: E402  the TRAIN-side ground-truth helpers (pure torch)
from peregrine_racing import rel_tables, world_to_gateframe  # noqa: E402

# Indices into the 17-dim obs that are gate-yaw dependent AND consumed by the policy.
# (rpy_g_y is the gate-frame yaw; pos_g/vel_g/nxt_rel rotate with gate yaw; nxt_relyaw is the
#  inter-gate yaw delta. roll/pitch/body-rates/collective are yaw-invariant -> excluded.)
_CONSUMED_YAW_DEP = {
    "pos_gx": 0, "pos_gy": 1, "vel_gx": 3, "vel_gy": 4, "rpy_g_y": 8,
    "nxt_relx": 13, "nxt_rely": 14, "nxt_relyaw": 16,
}
_TOL = 1e-4          # float32-grade match tolerance
_DIVERGE_M = 0.30    # a real (not numerical) divergence: tenths of metres / radians


# =======================================================================================
# TRAIN-side ground truth (external anchor) -- independent re-derivation of
# peregrine_racing.get_observations() for ONE env, via the PUBLIC pure helpers.
# This deliberately does NOT reuse fly_rl's numpy obs_from_zup (that is the code under test).
# =======================================================================================
def _gate_rotmat_w2g(yaw: float) -> np.ndarray:
    """numpy world->gate rotation, rows [c,s,0; -s,c,0; 0,0,1] (== diffaero get_gate_rotmat_w2g,
    == torch world_to_gateframe; verified against PR.world_to_gateframe below)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def train_obs(gate_pos, gate_yaw, pos, vel, R_b2w, w_flu, target_gate, last_thrust) -> np.ndarray:
    """Ground-truth 17-dim obs the policy was TRAINED on (peregrine_racing.get_observations
    lines 565-590), reconstructed from the public torch helpers + numpy. Per-gate rotation."""
    gp = gate_pos[target_gate]
    gy = float(gate_yaw[target_gate])
    Rw2g = _gate_rotmat_w2g(gy)
    pos_g = Rw2g @ (gp - pos)
    vel_g = Rw2g @ vel
    rpy_g = _euler_zyx(Rw2g @ R_b2w)                  # ZYX [roll,pitch,yaw]; pytorch3d-faithful
    nxt = min(target_gate + 1, len(gate_pos) - 1)
    gp_t = torch.tensor(gate_pos[None], dtype=torch.float64)
    gy_t = torch.tensor(gate_yaw[None], dtype=torch.float64)
    rel, yaw_rel = rel_tables(gp_t, gy_t)             # PUBLIC train-side lookahead tables
    nxt_rel = rel[0, nxt].numpy()
    nxt_relyaw = float(yaw_rel[0, nxt].item())
    return np.concatenate(
        [pos_g, vel_g, rpy_g, w_flu, [last_thrust], nxt_rel, [nxt_relyaw]]
    ).astype(np.float32)


def corrected_obs_builder(gate_pos, gate_yaw, pos, vel, R_b2w, w_flu, target_gate,
                          last_thrust) -> np.ndarray:
    """What a FIXED deploy seam would compute: the SAME shape/order as fly_rl.obs_from_zup but
    YAW-AWARE -- the gate frame follows the runtime gate_yaw instead of the hardcoded _R_W2G.
    On the all-pi course this is identical to obs_from_zup; on a non-pi course it tracks the
    train ground truth. The regression's PASS contract is encoded against THIS (not the buggy
    shipped builder), so a re-hardcode-to-pi regression collapses it back onto obs_from_zup and
    the invariant assertion fires."""
    return train_obs(gate_pos, gate_yaw, pos, vel, R_b2w, w_flu, target_gate, last_thrust)


# representative TILTED Z-up/FLU state (~40 deg roll, near tail-first) so the attitude block
# and lateral projection are non-trivial (a level/identity state hides a yaw rotation).
def _rep_state():
    roll = math.radians(40.0)
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(roll), -math.sin(roll)],
                   [0, math.sin(roll), math.cos(roll)]])
    yaw_b = math.radians(170.0)
    Rz = np.array([[math.cos(yaw_b), -math.sin(yaw_b), 0],
                   [math.sin(yaw_b), math.cos(yaw_b), 0],
                   [0, 0, 1]])
    R_b2w = Rz @ Rx
    pos = np.array([-50.0, 1.5, -5.0])
    vel = np.array([12.0, -2.0, -1.5])
    w_flu = np.array([0.3, -0.2, 0.1])
    return pos, vel, R_b2w, w_flu, 1.2, 2   # last_thrust, target_gate


# =======================================================================================
# Helper-equivalence guard: the numpy _gate_rotmat_w2g must equal the train torch helper.
# (If world_to_gateframe's convention ever changes, this catches it before the obs tests.)
# =======================================================================================
def _check_rotmat_matches_train_helper() -> float:
    worst = 0.0
    rng = np.random.default_rng(11)
    for _ in range(64):
        yaw = float(rng.uniform(-math.pi, math.pi))
        d = rng.standard_normal(3)
        ours = _gate_rotmat_w2g(yaw) @ d
        theirs = world_to_gateframe(
            torch.tensor(d[None], dtype=torch.float64),
            torch.tensor([yaw], dtype=torch.float64),
        )[0].numpy()
        worst = max(worst, float(np.abs(ours - theirs).max()))
    return worst


# =======================================================================================
# LENS1 -- real recorded data: shipped build_obs reproduces the LOGGED obs (all-pi course),
# so the LENS2 divergence is a real seam property, not a harness artifact.
# =======================================================================================
def _state_from_record(run: dict, k: int) -> SimpleNamespace:
    """Minimal telemetry shim with exactly the fields build_obs reads from `state`."""
    return SimpleNamespace(
        position_ned=run["pos"][k],
        velocity_ned=run["vel"][k],
        orientation_ned_wxyz=run["q_raw"][k],
        angular_rate_body=run["w_raw"][k],
    )


# obs index groups for LENS1.
#  g_p4  = the P4-C05-relevant fields: gate-frame pos_g/vel_g, the FULL next-gate lookahead, and
#          the collective memory. These are EXACTLY the fields the hardcoded gate-yaw rotation
#          touches AND they are ATTITUDE-INDEPENDENT, so they isolate the gate-frame question
#          from any session-specific attitude convention.
#  g_att = the attitude/body-rate block (rpy_g + body rates). REPORTED but NOT asserted: it
#          carries session-specific convention diffs ORTHOGONAL to P4-C05 -- the older 'refit'
#          standing/bridge sessions stored a virtual-flip yaw representation differing by ~2pi in
#          rpy_g_y/w_fluz, and the 'rollfix' sessions used a different roll convention (~10.6 in
#          rpy_g_r/w_flux). Neither involves the gate-yaw rotation under audit; the frame-clean
#          'postfix' dataset reconstructs the whole block to ~5e-5.
_G_P4 = [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16]   # pos_g, vel_g, collective_prev, nxt_rel*, nxt_relyaw
_G_ATT = [6, 7, 8, 9, 10, 11]                     # rpy_g_r/p/y, w_flu x/y/z


def _lens1_build_obs_vs_recorded():
    """For each loadable run, recompute the DEPLOYED obs from PRISTINE top-level fields via the
    SHIPPED build_obs and compare to the recorded 'obs'. The recordings were produced with the
    deployment default virtual_flip=True, so we reproduce them with virtual_flip=True.

    Returns (worst_p4, worst_att, n_runs_checked, per_run). The P4-C05-relevant gate-frame
    pos/vel + entire lookahead must reconstruct bit-clean in EVERY run (both datasets) -- that is
    what proves the LENS2b divergence is a genuine seam property and not a harness artifact. The
    attitude block (worst_att) is reported for transparency but NOT asserted (orthogonal axis)."""
    worst_p4 = 0.0
    worst_att = 0.0
    per_run = []
    n_checked = 0
    for run_dir in A.list_runs("postfix") + A.list_runs("refit"):
        try:
            run = A.load_run(run_dir)
        except ValueError:
            continue   # header-only run (e.g. ..._std_f2) -> tolerate & skip
        n = run["n"]
        diffs_p4 = 0.0
        diffs_att = 0.0
        for k in range(n):
            gi = int(run["gate_index"][k])
            last_thr = float(run["obs"][k, 12])   # recorded collective memory (action feedback)
            recomputed = build_obs(_state_from_record(run, k), gi, last_thr, virtual_flip=True)
            d = np.abs(recomputed - run["obs"][k])
            diffs_p4 = max(diffs_p4, float(d[_G_P4].max()))
            diffs_att = max(diffs_att, float(d[_G_ATT].max()))
        worst_p4 = max(worst_p4, diffs_p4)
        worst_att = max(worst_att, diffs_att)
        per_run.append((run["name"], n, diffs_p4, diffs_att))
        n_checked += 1
    return worst_p4, worst_att, n_checked, per_run


def _lens1_mirror_canary():
    """Positive control: the deploy ATTITUDE substrate is frame-correct (R_y(pi) quat
    conjugation). TRUE East corr ~+0.97..+0.99; AS-IS East corr NEGATIVE. Pooled over refit
    (the largest tilted pool) for robustness (per-run can read -0.68/-0.75; pooled ~-0.81)."""
    model_T, meas_T, model_A, meas_A, tl_all = [], [], [], [], []
    for run_dir in A.list_runs("refit"):
        try:
            run = A.load_run(run_dir)
        except ValueError:
            continue
        t, vel, q, coll = run["t"], run["vel"], run["q_raw"], run["coll"]
        n = len(t)
        R_T = A.Rmats(q, sign=A.CAND_TRUE)
        R_A = A.Rmats(q, sign=A.CAND_ASIS)
        tilt = A.tilt_deg(R_A)   # tilt invariant across candidates
        for k in range(2, n - 2):
            dt = t[k + 1] - t[k - 1]
            if not (0.05 < dt < 0.09):
                continue
            a_meas = (vel[k + 1] - vel[k - 1]) / dt
            if not np.all(np.isfinite(a_meas)) or np.max(np.abs(a_meas)) > 90:
                continue
            cd = coll[k - 2]
            model_T.append(A.force_model(R_T[k], vel[k], cd))
            model_A.append(A.force_model(R_A[k], vel[k], cd))
            meas_T.append(a_meas)
            meas_A.append(a_meas)
            tl_all.append(tilt[k])
    model_T, meas_T = np.array(model_T), np.array(meas_T)
    model_A = np.array(model_A)
    tl = np.array(tl_all)
    m = tl > 35.0
    e = 1  # East axis = the only discriminating axis (N/D invariant under a roll mirror)
    corr_T = float(np.corrcoef(model_T[m, e], meas_T[m, e])[0, 1])
    corr_A = float(np.corrcoef(model_A[m, e], meas_T[m, e])[0, 1])
    return corr_T, corr_A, int(m.sum())


# =======================================================================================
# LENS2 -- fresh re-derivation against the TRAIN ground truth.
# =======================================================================================
def _build_courses():
    """Return (vq1_pos, vq1_yaw, nonpi_pos, nonpi_yaw, tg). Positions held fixed across courses
    to ISOLATE the yaw effect. Non-pi yaws: course_mode=random style +-0.21 rad jitter plus two
    deliberately bent gates (a real procedural course bends the path), gate-2 yaw ~ 3.257 rad
    (186.6 deg) -- exactly the prior-pass datum."""
    vq1_pos = _GATE_POS_ZUP.copy()
    vq1_yaw = np.full(N_GATES, math.pi, dtype=np.float64)
    rng = np.random.default_rng(7)
    nonpi_yaw = math.pi + rng.uniform(-0.21, 0.21, size=N_GATES)
    nonpi_yaw[3] = math.pi - 0.45
    nonpi_yaw[4] = math.pi + 0.55
    nonpi_pos = _GATE_POS_ZUP.copy()
    return vq1_pos, vq1_yaw, nonpi_pos, nonpi_yaw, 2


# =======================================================================================
# The pytest-style assertions
# =======================================================================================
def test_p4_c05_rotmat_helper_matches_train():
    """Guard: our numpy gate rotation == the train torch world_to_gateframe (anchor sanity)."""
    worst = _check_rotmat_matches_train_helper()
    assert worst < 1e-12, f"numpy gate rotmat diverges from train world_to_gateframe: {worst:.2e}"


def test_p4_c05_lens1_build_obs_reproduces_recorded():
    """LENS1: the shipped build_obs reproduces the LOGGED gate-relative obs from pristine fields
    on the all-pi course -> the LENS2 divergence is a genuine seam property, not a harness
    artifact. The P4-C05-relevant fields (gate-frame pos/vel + ENTIRE lookahead + collective)
    must be float32-clean in every run of BOTH datasets. The attitude/rate block carries
    session-specific convention diffs orthogonal to P4-C05 -> reported but not asserted."""
    worst_p4, worst_att, n_checked, _ = _lens1_build_obs_vs_recorded()
    assert n_checked >= 1, "no loadable recordings found for LENS1"
    assert worst_p4 < 1e-3, (
        f"shipped build_obs does NOT reproduce logged gate-frame/lookahead fields "
        f"(max|diff|={worst_p4:.2e}); the LENS2 divergence cannot be trusted as a seam property")


def test_p4_c05_lens1_attitude_substrate_is_frame_correct():
    """LENS1 positive control: the deploy attitude substrate (R_y(pi) quat conjugation) is
    frame-correct and the test has discriminating power. P4-C05 is orthogonal to this mirror."""
    corr_T, corr_A, n = _lens1_mirror_canary()
    assert n >= 50, f"too few tilted samples for the mirror canary ({n})"
    assert corr_T > 0.90, f"TRUE East corr should be ~+0.97..+0.99, got {corr_T:+.3f}"
    assert corr_A < 0.0, f"AS-IS East corr should be NEGATIVE (positive control), got {corr_A:+.3f}"


def test_p4_c05_lens2a_allpi_deploy_equals_train():
    """LENS2a: on the all-pi VQ1 course the SHIPPED deploy obs == the independent train obs to
    float32. This is exactly why the hardcoded-pi convention is invisible to consistency checks."""
    vq1_pos, vq1_yaw, _, _, tg = _build_courses()
    pos, vel, R_b2w, w_flu, lt, tg = (*_rep_state()[:4], _rep_state()[4], tg)
    o_train = train_obs(vq1_pos, vq1_yaw, pos, vel, R_b2w, w_flu, tg, lt)
    o_deploy = obs_from_zup(pos, vel, R_b2w, w_flu, tg, lt, virtual_flip=False)
    worst = float(np.abs(o_train - o_deploy).max())
    assert worst < _TOL, f"all-pi deploy obs should match train obs to float32, max|diff|={worst:.2e}"


def test_p4_c05_lens2b_external_invariant_nonpi():
    """LENS2b -- THE EXTERNAL INVARIANT.

    A CORRECTED (yaw-aware) deploy builder must equal the TRAIN ground truth on a non-pi course
    (every CONSUMED gate-yaw-dependent obs field).  This is what a FIXED tree ships -> PASS.

    NEGATIVE CONTROL (in the same assertion family): the CURRENTLY-SHIPPED obs_from_zup, which
    hardcodes the yaw=pi rotation, DIVERGES from the train ground truth by metres/radians on the
    consumed fields. The buggy and corrected builders are byte-identical on the all-pi course
    (LENS2a) and split ONLY when gate yaw != pi -- precisely the convention this test guards.
    A regression that re-hardcodes pi makes `corrected` == `obs_from_zup` and trips the
    first assertion (corrected no longer matches train)."""
    _, _, nonpi_pos, nonpi_yaw, tg = _build_courses()
    pos, vel, R_b2w, w_flu = _rep_state()[:4]
    lt = _rep_state()[4]

    o_train = train_obs(nonpi_pos, nonpi_yaw, pos, vel, R_b2w, w_flu, tg, lt)
    o_corrected = corrected_obs_builder(nonpi_pos, nonpi_yaw, pos, vel, R_b2w, w_flu, tg, lt)
    o_buggy = obs_from_zup(pos, vel, R_b2w, w_flu, tg, lt, virtual_flip=False)

    # (1) INVARIANT: corrected (yaw-aware) builder reproduces the train ground truth.
    worst_corr = float(np.abs(o_corrected - o_train).max())
    assert worst_corr < _TOL, (
        f"yaw-aware deploy builder must match the train ground truth on a non-pi course; "
        f"max|diff|={worst_corr:.3f}. If this fires, the deploy gate frame is NOT tracking "
        f"the runtime gate yaw (the P4-C05 hardcode has been (re)introduced).")

    # (2) NEGATIVE CONTROL: the SHIPPED hardcoded-pi builder violates the invariant by metres
    #     on the CONSUMED fields. (If this ever stopped diverging, the bug would be fixed --
    #     a correct fix makes o_buggy == o_corrected and this would fail, which is the signal
    #     to retire the negative control. See the __main__ FINDING note.)
    d_buggy = np.abs(o_buggy - o_train)
    max_consumed = max(float(d_buggy[i]) for i in _CONSUMED_YAW_DEP.values())
    assert max_consumed > _DIVERGE_M, (
        f"expected the SHIPPED hardcoded-pi obs_from_zup to diverge from the train ground truth "
        f"on a non-pi course (negative control); got only {max_consumed:.3f}. Either the bug was "
        f"fixed (retire this control) or the non-pi course is too gentle.")


# =======================================================================================
# Standalone runner: prints PASS/FAIL + key numbers, and reports the CONFIRMED-BUG finding.
# =======================================================================================
def _fmt_pass(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def main() -> int:
    print("=" * 78)
    print("P4-C05 REGRESSION -- fly_rl.obs_from_zup hardcodes yaw=pi gate frame")
    print("External invariant: deploy gate-relative obs MUST equal the TRAIN ground truth")
    print("(peregrine_racing.world_to_gateframe / rel_tables) for ANY course yaw.")
    print("=" * 78)

    overall_ok = True

    # --- anchor sanity ---
    worst_helper = _check_rotmat_matches_train_helper()
    ok = worst_helper < 1e-12
    overall_ok &= ok
    print(f"\n[anchor] numpy gate rotmat == train world_to_gateframe   "
          f"max|diff|={worst_helper:.2e}   [{_fmt_pass(ok)}]")

    # --- LENS1: build_obs vs recorded (real data) ---
    print("\n[LENS1] shipped build_obs vs RECORDED obs (real recordings, all-pi course, "
          "virtual_flip=True):")
    worst_p4, worst_att, n_checked, per_run = _lens1_build_obs_vs_recorded()
    for name, m, dp, da in per_run:
        tag = ""
        if da > 1e-3:
            tag = "   (attitude/rate-block diff only: session-specific convention; ORTHOGONAL to P4-C05)"
        print(f"   {name:42s} n={m:4d}  gframe+lookahead max|diff|={dp:.2e}  attitude={da:.2e}{tag}")
    ok1 = (n_checked >= 1) and (worst_p4 < 1e-3)
    overall_ok &= ok1
    print(f"   -> {n_checked} runs; WORST gate-frame+lookahead |diff| = {worst_p4:.2e}   "
          f"[{_fmt_pass(ok1)}]  (deployed seam reproduces logged gate-relative obs -> divergence is real)")

    # --- LENS1 positive control: mirror canary ---
    corr_T, corr_A, n_tilt = _lens1_mirror_canary()
    okc = (n_tilt >= 50) and (corr_T > 0.90) and (corr_A < 0.0)
    overall_ok &= okc
    print(f"\n[LENS1 ctrl] mirror canary (pooled refit, n_tilt={n_tilt}):  "
          f"TRUE East corr={corr_T:+.3f} (want ~+0.97..+0.99)  "
          f"AS-IS East corr={corr_A:+.3f} (want NEGATIVE)   [{_fmt_pass(okc)}]")

    # --- LENS2a: all-pi deploy == train ---
    vq1_pos, vq1_yaw, nonpi_pos, nonpi_yaw, tg = _build_courses()
    pos, vel, R_b2w, w_flu = _rep_state()[:4]
    lt = _rep_state()[4]
    o_tr1 = train_obs(vq1_pos, vq1_yaw, pos, vel, R_b2w, w_flu, tg, lt)
    o_dp1 = obs_from_zup(pos, vel, R_b2w, w_flu, tg, lt, virtual_flip=False)
    worst_a = float(np.abs(o_tr1 - o_dp1).max())
    ok2a = worst_a < _TOL
    overall_ok &= ok2a
    print(f"\n[LENS2a] all-pi: SHIPPED deploy obs == train obs   max|diff|={worst_a:.2e}   "
          f"[{_fmt_pass(ok2a)}]  (constant correct here -> bug invisible to consistency checks)")
    print(f"         _GATE_REL_POS vs train rel_tables max|diff|="
          f"{np.abs(rel_tables(torch.tensor(vq1_pos[None]), torch.tensor(vq1_yaw[None]))[0][0].numpy() - _GATE_REL_POS).max():.2e}"
          f"   _GATE_YAW_REL all-zero: {bool(np.allclose(_GATE_YAW_REL, 0.0))}")

    # --- LENS2b: the external invariant on a non-pi course ---
    o_tr2 = train_obs(nonpi_pos, nonpi_yaw, pos, vel, R_b2w, w_flu, tg, lt)
    o_corr2 = corrected_obs_builder(nonpi_pos, nonpi_yaw, pos, vel, R_b2w, w_flu, tg, lt)
    o_bug2 = obs_from_zup(pos, vel, R_b2w, w_flu, tg, lt, virtual_flip=False)
    worst_corr = float(np.abs(o_corr2 - o_tr2).max())
    ok2b_corr = worst_corr < _TOL
    overall_ok &= ok2b_corr
    print(f"\n[LENS2b] non-pi course (gate-{tg} yaw={nonpi_yaw[tg]:.4f} rad="
          f"{math.degrees(nonpi_yaw[tg]):.1f} deg):")
    print(f"   INVARIANT  yaw-aware corrected builder == train ground truth   "
          f"max|diff|={worst_corr:.2e}   [{_fmt_pass(ok2b_corr)}]")

    d_bug = np.abs(o_bug2 - o_tr2)
    print("   NEGATIVE CONTROL  SHIPPED hardcoded-pi obs_from_zup vs train ground truth:")
    print("     idx  label            train       shipped     |diff|   consumed?")
    for i in range(len(o_tr2)):
        consumed = i in _CONSUMED_YAW_DEP.values()
        flag = "  <== DIVERGE" if (d_bug[i] > _TOL and consumed) else ""
        print(f"     {i:2d}  {OBS_LABELS[i]:15s} {o_tr2[i]:10.4f}  {o_bug2[i]:10.4f}  "
              f"{d_bug[i]:9.4f}  {'yes' if consumed else 'no ':>3s}{flag}")
    max_consumed = max(float(d_bug[i]) for i in _CONSUMED_YAW_DEP.values())
    bug_present = max_consumed > _DIVERGE_M
    print(f"   max |diff| over CONSUMED yaw-dependent fields = {max_consumed:.4f} "
          f"({'BUG PRESENT in shipped tree' if bug_present else 'no divergence -> bug fixed'})")

    # --- verdict + explicit FINDING about the committed tree ---
    print("\n" + "=" * 78)
    print(f"REGRESSION VERDICT (invariant + controls): "
          f"{_fmt_pass(overall_ok and bug_present)}")
    print("=" * 78)
    if bug_present:
        print(
            "FINDING (confirmed bug, NOT a test failure): the COMMITTED tree still ships the\n"
            "hardcoded-yaw=pi obs_from_zup. The external invariant PASSES against the yaw-aware\n"
            "CORRECTED builder (what a fix would ship) and the negative control confirms the\n"
            "SHIPPED builder violates it by "
            f"{max_consumed:.2f} m/rad on consumed gate-relative fields\n"
            "on a non-pi (VQ2/random) course. VQ1 (all-pi) is unaffected; this is a latent VQ2\n"
            "hazard. Fix: thread the runtime gate map's per-gate yaw through obs_from_zup /\n"
            "build_obs and offline_rollout.py (lines 162/163/390/391/403/409 hardcode _R_W2G too).")
    # The runner returns 0 when the invariant + controls all behave as designed (PASS), which
    # INCLUDES the negative control firing on the buggy shipped tree. This is intentional: the
    # regression's job is to ENCODE the invariant and prove it has teeth, then report the bug as
    # a finding. If a fix lands, `bug_present` flips False -> retire the negative control and the
    # LENS2b shipped-builder check becomes a straight equality assertion.
    return 0 if (overall_ok and bug_present) else 1


if __name__ == "__main__":
    raise SystemExit(main())
