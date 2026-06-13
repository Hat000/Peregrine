"""REGRESSION for CONFIRMED bug CR4-01 -- stale-yaw producer artifact in the
refit recordings, and the EXTERNAL invariant that pins the live yaw convention.

================================================================================
BUG CLASS THIS GUARDS (one line)
================================================================================
A YAW-CHANNEL frame/sign alias on the ODOMETRY-quat -> TRUE-attitude conjugation
(the R_y(pi) family: q_true = q_raw * [1,-1,1,-1]). Re-introducing the wrong yaw
sign -- e.g. reverting to the bcc93f9 reading [1,1,1,1], or a yaw-only alias
[1,-1,1,1] -- silently mis-orients every gate-relative observation and every
wire rate command while passing all INTERNAL-consistency checks.

CR4-01 specifically: the `refit` dataset's RECORDED obs[8] (gate-frame yaw) and
rate_frd[2] (wire yaw command) are STALE bcc93f9-PRODUCER artifacts (a hybrid
roll/yaw split: obs[6] roll matches the CURRENT 93023cf encoder, obs[8] yaw is
the negation -- the OLD convention). They are diagnostic-only: the live loop
(fly_rl.py:765) recomputes obs FRESH via build_obs() from live telemetry every
tick and never feeds the recorded obs/rate_frd back into the command path, so the
artifact cannot reach deployment. This test encodes the EXTERNAL invariant that
proves the *current committed* yaw convention is right, plus the provenance fact
that makes the recorded artifact harmless.

================================================================================
WHY INTERNAL-CONSISTENCY CHECKS MISS IT (the banked methodology, 4x-bitten)
================================================================================
A proper-rotation conjugation / sign-alias is INVISIBLE to internal checks:
quat-vs-rate finite-difference, level-flight correlation, and twist round-trip
ALL PASS while the convention is WRONG -- the conjugate of a proper rotation is
still a proper rotation, internally self-consistent (this is exactly how the
bcc93f9 reading survived its own tilted-phase validation as a self-mirrored
alias). ONLY an EXTERNAL invariant discriminates:

  finite-difference of the PRISTINE state.velocity_ned (world NED, the trusted
  anchor) vs the attitude-derived specific force
      f = K * (R @ [0,0,-1]) + quad_drag(R, v) + g
  evaluated in the TILTED bin (tilt > 35 deg). Level flight is INADMISSIBLE:
  when body-down ~ world-down a rotation looks identical to its conjugate.

DISCRIMINATING AXIS = EAST. A roll mirror leaves North/Down nearly invariant, so
they are blind; the East-axis force correlation is the signal. The committed
TRUE conjugation [1,-1,1,-1] gives East corr ~ +0.97..+0.99; the bcc/as-is
reading [1,1,1,1] gives East corr ~ -0.71..-0.84 (positive control: it MUST go
negative, proving the anchor has discriminating power).

To prove the anchor is NOT YAW-BLIND (so it actually guards CR4-01's yaw channel,
not just roll), this test also flips ONLY the yaw component to the alias
[1,-1,1,1] and asserts the anchor's force correlation COLLAPSES -- a built-in
NEGATIVE CONTROL for the exact bug class.

================================================================================
WHAT WOULD MAKE THIS TEST FAIL (i.e. the bug being re-introduced)
================================================================================
 * Reverting the quat conjugation to [1,1,1,1] (bcc93f9) -> AS-IS East corr would
   stop being negative / TRUE East corr would stop being ~ +0.98 in the substrate.
 * Any yaw-only re-alias -> the yaw-sensitivity negative control would stop
   collapsing (anchor would appear yaw-blind), or TRUE would lose its East win.
 * Refactoring the live loop to feed RECORDED obs/rate_frd into the command path
   instead of recomputing via build_obs() -> the provenance assertion fails,
   because then the stale bcc yaw artifact in the refit recordings WOULD reach
   deployment.

The test encodes the EXTERNAL invariant + the provenance structure, NOT the
current code's possibly-buggy recorded values. It deliberately does NOT trust any
"// verified" convention comment -- comments have been wrong (and self-mirrored)
here repeatedly.

Substrate: handoff/ultracode-substrate-audit-2026-06-13/scratch/_audit_io.py
Datasets: handoff/shadowpc-{refit,postfix}-dataset-2026-06-12/extracted
Out of scope (NOT imported): rl/contact_true_eval.py.
================================================================================
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

import pytest

# --- shared substrate (tests/_audit_io.py) lives beside this file --------------
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))
import _audit_io as A  # noqa: E402  -- the ONE shared loader + Test-A force model

# refit/extracted is gitignored (only its .zip is tracked); postfix/extracted IS tracked.
# Skip (don't ERROR) on a checkout lacking either set -- on the canonical laptop both exist.
pytestmark = pytest.mark.skipif(
    not (A.dataset_present("postfix") and A.dataset_present("refit")),
    reason="ShadowPC audit recordings (postfix+refit extracted) not present",
)

# Repo root (substrate exposes it). Used for the source-level provenance check.
ROOT = A.ROOT
FLY_RL = ROOT / "rl" / "fly_rl.py"

# --- candidate quat-sign conventions (wxyz per-component multipliers) ----------
# TRUE  = committed R_y(pi) conjugation (the winner): q_true = q_raw*[1,-1,1,-1].
# AS-IS = bcc93f9 raw reading (positive control / loser): East force anti-corr.
# YAWFLIP = yaw-only alias off TRUE -> negate ONLY the yaw component. This is the
#   built-in NEGATIVE CONTROL: if the force anchor truly senses yaw, flipping yaw
#   alone must COLLAPSE the correlation. (Relative to TRUE [1,-1,1,-1], negating
#   yaw means flipping the quat x AND z signs back -> [1,1,1,1] is the roll+yaw
#   double-flip; the *pure-yaw* alias is [1,-1,1,1], i.e. TRUE with z un-negated.)
CAND_TRUE = A.CAND_TRUE                          # [1,-1,1,-1]
CAND_ASIS = A.CAND_ASIS                          # [1, 1,1, 1]
CAND_YAWFLIP = np.array([1.0, -1.0, 1.0, 1.0])   # pure-yaw alias off TRUE

# --- acceptance thresholds (set with margin around the MEASURED substrate
#     numbers; verified pooled on 2026-06-13 against real live data) ------------
#   refit pooled (use_lapse=False, tilt>35, n=30556):
#     TRUE  : N+0.986 E+0.989 D+0.956 | AS-IS: N+0.982 E-0.838 D+0.926
#     YAWFLIP: N+0.012 E+0.083 D+0.931  (anchor collapses on N&E when yaw flipped)
#   postfix pooled (NEW dataset; n=1346):
#     TRUE  : N+0.927 E+0.967 D+0.914 | AS-IS: N+0.926 E-0.714 D+0.902
#     YAWFLIP: N-0.534 E-0.210 D+0.821 (collapses)
TRUE_EAST_CORR_MIN = 0.90    # committed convention must strongly correlate East
ASIS_EAST_CORR_MAX = -0.40   # bcc/as-is reading must ANTI-correlate (positive control)
# yaw-flip negative control: the anchor's East AND North force corr must collapse
# (lose the strong-positive lock) -- proves the anchor is NOT yaw-blind.
YAWFLIP_CORR_MAX = 0.60      # collapsed corr must drop well below TRUE's >0.90 lock
TILT_MIN = 35.0              # tilted-phase bin; level flight is inadmissible
USE_LAPSE = False            # LAPSE is VOIDED in production (S18); production-faithful


# ===============================================================================
# helpers
# ===============================================================================
def _pooled_force_corr(dataset: str, sign, tilt_min: float = TILT_MIN,
                       use_lapse: bool = USE_LAPSE) -> dict:
    """Pool the Test-A force-vs-vel_ned-FD anchor across every loadable run in a
    dataset for one candidate quat sign. Returns per-axis correlation + count.

    This is the EXTERNAL invariant: model specific force (attitude-derived) vs
    central finite-difference of the PRISTINE world-NED velocity, in the tilted
    bin only. Empty (header-only) runs are tolerated/skipped.
    """
    model, meas, tl = [], [], []
    for rd in A.list_runs(dataset):
        try:
            run = A.load_run(rd)
        except ValueError:
            # header-only run (e.g. ..._std_f2) -> no data rows; skip, no retry.
            continue
        t, vel, q, coll = run["t"], run["vel"], run["q_raw"], run["coll"]
        R = A.Rmats(q, sign=sign)
        # tilt is invariant across candidate signs -> compute once from raw.
        tilt_raw = A.tilt_deg(A.Rmats(q, sign=CAND_ASIS))
        n = len(t)
        for k in range(2, n - 2):
            dt = t[k + 1] - t[k - 1]
            if not (0.05 < dt < 0.09):       # gate on a sane 2-tick dt (~67 ms)
                continue
            a_meas = (vel[k + 1] - vel[k - 1]) / dt
            if not np.all(np.isfinite(a_meas)) or np.max(np.abs(a_meas)) > 90:
                continue
            model.append(A.force_model(R[k], vel[k], coll[k - 2], use_lapse=use_lapse))
            meas.append(a_meas)
            tl.append(tilt_raw[k])
    model = np.asarray(model)
    meas = np.asarray(meas)
    tl = np.asarray(tl)
    m = tl > tilt_min
    out = {"n_tilted": int(m.sum())}
    for ax, name in enumerate(("N", "E", "D")):
        if m.sum() < 20 or model[m, ax].std() < 1e-6:
            out[name] = float("nan")
        else:
            out[name] = float(np.corrcoef(model[m, ax], meas[m, ax])[0, 1])
    return out


# ===============================================================================
# THE EXTERNAL-INVARIANT CHECKS
# ===============================================================================
def check_force_anchor(dataset: str) -> tuple[bool, dict]:
    """External invariant #1: TRUE conjugation wins the East force anchor and the
    AS-IS (bcc) reading anti-correlates (positive control fires)."""
    true_c = _pooled_force_corr(dataset, CAND_TRUE)
    asis_c = _pooled_force_corr(dataset, CAND_ASIS)
    ok = (
        true_c["n_tilted"] >= 20
        and true_c["E"] >= TRUE_EAST_CORR_MIN
        and asis_c["E"] <= ASIS_EAST_CORR_MAX     # positive control: MUST go negative
    )
    return ok, {"true": true_c, "asis": asis_c}


def check_yaw_sensitivity(dataset: str) -> tuple[bool, dict]:
    """External invariant #2 (NEGATIVE CONTROL): flip ONLY the yaw component off
    the TRUE convention -> the force anchor's strong-positive lock must COLLAPSE.
    This proves the anchor senses the YAW channel (so it genuinely guards the
    CR4-01 yaw bug class), and is not merely a roll discriminator."""
    yaw_c = _pooled_force_corr(dataset, CAND_YAWFLIP)
    # Both the (otherwise-blind) North axis and the discriminating East axis must
    # lose their >0.90 lock when yaw alone is flipped.
    ok = (
        yaw_c["n_tilted"] >= 20
        and yaw_c["E"] <= YAWFLIP_CORR_MAX
        and yaw_c["N"] <= YAWFLIP_CORR_MAX
    )
    return ok, {"yawflip": yaw_c}


def check_provenance_recorded_obs_unused() -> tuple[bool, dict]:
    """External invariant #3 (PROVENANCE / source-level): the live command path
    recomputes obs FRESH via build_obs() and never feeds RECORDED obs/rate_frd
    back into the wire command. This is what makes CR4-01's stale-yaw recording
    artifact harmless -- the artifact lives only in the diagnostic dump.

    Source-text invariant (durable against torch/network import weight):
      * the command-send call (`client.send_command(`) is preceded by an
        `obs = build_obs(` recompute and a `policy_step(actor, obs` call;
      * `body_rate=rate_frd` comes from policy_step's return, and the recorded
        `"obs"` / `"rate_frd"` JSON keys appear ONLY inside the debug dump
        (`dbg.update(`), i.e. AFTER the send -- never as an input to send_command.
    """
    src = FLY_RL.read_text(encoding="utf-8")
    info: dict = {"fly_rl": str(FLY_RL)}

    has_build_obs = re.search(r"\bobs\s*=\s*build_obs\(", src) is not None
    has_policy_step = re.search(r"policy_step\(\s*actor\s*,\s*obs\b", src) is not None
    has_send = "client.send_command(" in src
    info.update(build_obs=has_build_obs, policy_step=has_policy_step, send=has_send)

    # The recorded "obs" / "rate_frd" JSON keys (the artifact carriers) must be
    # confined to the diagnostic dump. Locate the dbg.update( block and require
    # the quoted keys to live downstream of the send, inside that update.
    send_idx = src.find("client.send_command(")
    dbg_idx = src.find("dbg.update(")
    obs_key_idx = src.find('"obs"')
    rate_key_idx = src.find('"rate_frd"')
    artifact_keys_are_diagnostic = (
        send_idx != -1 and dbg_idx != -1
        and obs_key_idx != -1 and rate_key_idx != -1
        and dbg_idx > send_idx                  # debug dump is AFTER the command send
        and obs_key_idx > dbg_idx                # recorded obs key is inside the dump
        and rate_key_idx > dbg_idx               # recorded rate_frd key is inside the dump
    )
    info["artifact_keys_are_diagnostic"] = artifact_keys_are_diagnostic

    # And the wire body_rate must come from the policy_step return (rate_frd), NOT
    # from any recorded field.
    wire_from_policy = re.search(r"body_rate\s*=\s*rate_frd\b", src) is not None
    info["wire_body_rate_from_rate_frd"] = wire_from_policy

    ok = (has_build_obs and has_policy_step and has_send
          and artifact_keys_are_diagnostic and wire_from_policy)
    return ok, info


# ===============================================================================
# pytest entry points
# ===============================================================================
def test_force_anchor_true_wins_east_refit():
    ok, d = check_force_anchor("refit")
    assert ok, (f"refit force anchor: TRUE East={d['true']['E']:+.3f} "
                f"(need >= {TRUE_EAST_CORR_MIN}), AS-IS East={d['asis']['E']:+.3f} "
                f"(need <= {ASIS_EAST_CORR_MAX}); n_tilt={d['true']['n_tilted']}")


def test_force_anchor_true_wins_east_postfix():
    ok, d = check_force_anchor("postfix")
    assert ok, (f"postfix force anchor: TRUE East={d['true']['E']:+.3f} "
                f"(need >= {TRUE_EAST_CORR_MIN}), AS-IS East={d['asis']['E']:+.3f} "
                f"(need <= {ASIS_EAST_CORR_MAX}); n_tilt={d['true']['n_tilted']}")


def test_yaw_sensitivity_negative_control_refit():
    ok, d = check_yaw_sensitivity("refit")
    y = d["yawflip"]
    assert ok, (f"refit yaw-flip control did NOT collapse (anchor would be "
                f"yaw-blind): N={y['N']:+.3f} E={y['E']:+.3f} "
                f"(both need <= {YAWFLIP_CORR_MAX})")


def test_yaw_sensitivity_negative_control_postfix():
    ok, d = check_yaw_sensitivity("postfix")
    y = d["yawflip"]
    assert ok, (f"postfix yaw-flip control did NOT collapse: N={y['N']:+.3f} "
                f"E={y['E']:+.3f} (both need <= {YAWFLIP_CORR_MAX})")


def test_recorded_obs_artifact_is_diagnostic_only():
    ok, info = check_provenance_recorded_obs_unused()
    assert ok, ("PROVENANCE BROKEN: the live command path no longer recomputes obs "
                "fresh / or recorded obs|rate_frd reached the wire command path. "
                f"info={info}")


# ===============================================================================
# standalone runner: prints PASS/FAIL + key numbers
# ===============================================================================
def _fmt(c: dict) -> str:
    return "  ".join(f"{ax}={c[ax]:+.3f}" for ax in ("N", "E", "D")) + f"  (n_tilt={c['n_tilted']})"


def main() -> int:
    print("=" * 78)
    print("REGRESSION test_confirmed_cr4_01 -- EXTERNAL invariant for the yaw-channel")
    print("quat conjugation + provenance of the stale-yaw refit recording artifact")
    print("=" * 78)

    all_ok = True

    # ---- invariant #1 + positive control: force anchor, both datasets ----------
    for ds in ("refit", "postfix"):
        ok, d = check_force_anchor(ds)
        print(f"\n[{ds}] Test-A force-vs-vel_ned-FD anchor (tilt>{TILT_MIN:.0f}, "
              f"use_lapse={USE_LAPSE}):")
        print(f"   TRUE  [1,-1,1,-1] : {_fmt(d['true'])}")
        print(f"   AS-IS [1, 1,1, 1] : {_fmt(d['asis'])}   <- positive control (East must be NEG)")
        print(f"   => TRUE East >= {TRUE_EAST_CORR_MIN} AND AS-IS East <= {ASIS_EAST_CORR_MAX} : "
              f"{'PASS' if ok else 'FAIL'}")
        all_ok &= ok

    # ---- invariant #2: yaw-sensitivity negative control ------------------------
    for ds in ("refit", "postfix"):
        ok, d = check_yaw_sensitivity(ds)
        y = d["yawflip"]
        print(f"\n[{ds}] yaw-only alias [1,-1,1,1] NEGATIVE CONTROL "
              f"(anchor must COLLAPSE -> not yaw-blind):")
        print(f"   YAWFLIP            : {_fmt(y)}")
        print(f"   => N & E both <= {YAWFLIP_CORR_MAX} (collapsed) : {'PASS' if ok else 'FAIL'}")
        all_ok &= ok

    # ---- invariant #3: provenance (recorded artifact is diagnostic-only) -------
    ok, info = check_provenance_recorded_obs_unused()
    print(f"\n[source] provenance -- recorded obs/rate_frd are diagnostic-only "
          f"(live loop recomputes via build_obs):")
    print(f"   obs=build_obs(): {info.get('build_obs')}  "
          f"policy_step(actor,obs): {info.get('policy_step')}  "
          f"send_command: {info.get('send')}")
    print(f"   wire body_rate=rate_frd (from policy_step): "
          f"{info.get('wire_body_rate_from_rate_frd')}")
    print(f"   recorded obs/rate_frd keys confined to debug dump AFTER send: "
          f"{info.get('artifact_keys_are_diagnostic')}")
    print(f"   => {'PASS' if ok else 'FAIL'}")
    all_ok &= ok

    print("\n" + "=" * 78)
    print(f"OVERALL: {'PASS' if all_ok else 'FAIL'}  -- guards CR4-01 yaw-channel quat "
          f"conjugation + stale-yaw recording artifact provenance")
    print("=" * 78)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
