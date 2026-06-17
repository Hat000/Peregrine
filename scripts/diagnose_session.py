"""scripts/diagnose_session.py -- unified telemetry AUTO-DIAGNOSER (the burn's "auto-judge").

The 2-day burn launches many autonomous sim flights / replays; a multi-agent triage wave needs a
TRUSTWORTHY machine-readable verdict per run so no human has to eyeball logs. This tool does NOT
re-derive anything -- it COMPOSES the existing, individually-trusted primitives into ONE verdict:

  1. RUN GRADE        racer.race_outcome (authoritative per-gate PASS-vs-COLLISION from RACE_STATUS
                      + COLLISION); when the tlog is unreadable (replay-only bundle, or pymavlink's
                      serial dep absent) it FALLS BACK to the recorder's own meta.json verdict.
  2. FRAME-RESIDUAL   scripts/frame_residual_report (the MANDATORY R_y(pi) conjugation canary). The
      CANARY          ODOMETRY quat is R_y(pi)-conjugated; internal consistency CANNOT catch a
                      conjugation, so the mirror canary -- East force-residual correlation at bank
                      under the TRUE vs the AS-IS attitude -- must be re-run after EVERY session
                      (healthy: TRUE ~ +0.97, AS-IS ~ -0.81). This tool turns that forget-prone
                      manual step into an ALWAYS-ON gate: a tripped canary FORCES ATTENTION.
  3. BUNDLE INTEGRITY scripts/verify_bundle (every required stream present at rate + GT velocity live).
  4. OBS / ACTION     the debug_obs.jsonl per-tick reader (fly_rl --debug-obs): non-finite obs or
      HEALTH          action FORCES ATTENTION; also surfaces action saturation, ODO staleness,
                      mid-run sim resets, and collision counts.

Output: a machine-readable JSON verdict ({PASS | ATTENTION} + per-check detail + a single
failure-taxonomy tag) plus a short human summary. Runs STANDALONE on any recording dir; degrades
gracefully when a primitive's input is unavailable (UNAVAILABLE check, never a crash, never a
silent pass).

Failure-taxonomy tags (one, highest-priority):
  NO_GO ARM_REFUSED SIM_RESET ODO_STALE NON_FINITE SPIN_ABORT CRASH MISS TIMEOUT
  (+ OK when the run is a clean pass, and ATTENTION_OTHER as a catch-all for an unrecognised abort)

Usage:
  .venv\\Scripts\\python scripts/diagnose_session.py data/runs/<stamp>_<label> [--json OUT] [--quiet]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))   # so we can import the sibling primitives by name

import numpy as np

# ----------------------------------------------------------------------------------------------
# The recorder's own final_state vocabulary (rl/fly_rl.py). These ARE our abort-family taxonomy
# tags; meta.json carries one verbatim. We trust it for the abort family and only DERIVE a grade
# (CRASH/MISS/OK) when the recorder said FINISHED or did not record a terminal abort.
# ----------------------------------------------------------------------------------------------
_CLEAN_STATES = {"FINISHED"}
# abort-family states that map straight to a taxonomy tag (recorder is authoritative here)
_ABORT_TAGS = {
    "NO_GO": "NO_GO", "ARM_REFUSED": "ARM_REFUSED", "SIM_RESET": "SIM_RESET",
    "ODO_STALE": "ODO_STALE", "NON_FINITE": "NON_FINITE", "SPIN_ABORT": "SPIN_ABORT",
    "CRASH": "CRASH", "TIMEOUT": "TIMEOUT", "BRIDGE_TIMEOUT": "TIMEOUT",
}
# states that are NOT a clean pass but lack a dedicated tag -> generic ATTENTION
_OTHER_ATTENTION_STATES = {"STALLED", "IDLE", "INTERRUPTED", "EXCEPTION", "ERROR"}


def _check(status: str, detail: dict | None = None, summary: str = "") -> dict:
    """One composed-check result. status in {PASS, ATTENTION, UNAVAILABLE}."""
    return {"status": status, "summary": summary, **(detail or {})}


# ==============================================================================================
# CHECK 1 -- RUN GRADE (compose racer.race_outcome; fall back to meta.json)
# ==============================================================================================
def _grade(session: Path, meta: dict, contact_window_s: float) -> dict:
    """Authoritative per-gate verdict. Tries the tlog (race_outcome) first; on any I/O / missing-dep
    failure falls back to the recorder's meta.json verdict so a replay-only bundle still grades."""
    tlog = session / "mavlink.tlog"
    if tlog.exists():
        try:
            from racer.race_outcome import analyze_outcome, load_from_recording
            rs, cols = load_from_recording(session)
            out = analyze_outcome(rs, cols, contact_window_s=contact_window_s)
            if "error" in out:
                return _check("ATTENTION", {"source": "tlog", "outcome": out},
                              f"no RACE_STATUS in tlog ({out.get('n_gate_collisions',0)} gate / "
                              f"{out.get('n_env_collisions',0)} env collisions)")
            clean = bool(out.get("clean_finish"))
            return _check("PASS" if clean else "ATTENTION",
                          {"source": "tlog", "outcome": out},
                          f"gates {out['gates_passed']} (clean {out['n_pass_clean']}/contact "
                          f"{out['n_pass_contact']}), finished={out['finished']}, "
                          f"clean_finish={clean}, env_coll={out['n_env_collisions']}")
        except Exception as exc:   # missing serial/pymavlink, corrupt tlog, etc. -> fall back
            tlog_err = f"{type(exc).__name__}: {exc}"
    else:
        tlog_err = "no mavlink.tlog in bundle"

    # --- fallback: the recorder's own verdict in meta.json ---
    fs = meta.get("final_state")
    if fs is None:
        return _check("ATTENTION", {"source": "meta", "tlog_unavailable": tlog_err},
                      "no tlog AND no final_state in meta.json -- ungradeable")
    rstat = meta.get("race_status") or {}
    clean = fs in _CLEAN_STATES and int(meta.get("collisions", 0) or 0) == 0
    return _check("PASS" if clean else "ATTENTION",
                  {"source": "meta", "final_state": fs,
                   "gate_index": meta.get("gate_index"),
                   "collisions": meta.get("collisions"),
                   "race_status": rstat, "tlog_unavailable": tlog_err},
                  f"meta final_state={fs}, gate_index={meta.get('gate_index')}, "
                  f"collisions={meta.get('collisions')} (graded from meta -- {tlog_err})")


# ==============================================================================================
# CHECK 2 -- FRAME-RESIDUAL R_y(pi) CANARY (compose scripts/frame_residual_report)
# ==============================================================================================
def _frame_residual(session: Path) -> dict:
    """Re-run the load-bearing mirror canary from frame_residual_report WITHOUT changing it: import
    its loader + model + measured constants and recompute the exact two scalars its report prints --
    the TRUE-vs-AS-IS East force-residual correlation at bank (the R_y(pi) conjugation gate) and the
    median TRUE-attitude residual magnitude (a plant-gap sniff). A tripped canary -> ATTENTION."""
    if not (session / "debug_obs.jsonl").exists():
        return _check("UNAVAILABLE", summary="no debug_obs.jsonl (canary needs --debug-obs ticks)")
    try:
        from scipy.spatial.transform import Rotation

        import frame_residual_report as frr
        from racer.frames import ODO_QUAT_TRUE_CONJ_WXYZ

        d = frr.load(session)
        if d is None:
            return _check("UNAVAILABLE", summary="debug_obs.jsonl present but < 10 usable ticks")

        n = len(d["t"])
        q_true = d["q"] * ODO_QUAT_TRUE_CONJ_WXYZ[None, :]
        Rt = Rotation.from_quat(q_true[:, [1, 2, 3, 0]]).as_matrix()
        Rr = Rotation.from_quat(d["q"][:, [1, 2, 3, 0]]).as_matrix()
        res_t, meas_e, mdl_te, mdl_re, tilts = [], [], [], [], []
        for k in range(2, n - 2):
            dt = d["t"][k + 1] - d["t"][k - 1]
            if not (0.05 < dt < 0.09):
                continue
            a = (d["vel"][k + 1] - d["vel"][k - 1]) / dt
            if not np.all(np.isfinite(a)) or np.max(np.abs(a)) > 90:
                continue
            v = d["vel"][k]
            c = d["coll"][max(k - 2, 0)]
            tilt = np.degrees(np.arccos(np.clip(Rt[k][2, 2], -1, 1)))
            mt = frr.model_accel(Rt[k], c, v)
            mr = frr.model_accel(Rr[k], c, v)
            res_t.append(mt - a)
            meas_e.append(a[1]); mdl_te.append(mt[1]); mdl_re.append(mr[1]); tilts.append(tilt)
        res_t = np.array(res_t)
        tilts = np.array(tilts)
        usable = len(res_t)
        if usable < 20:
            return _check("UNAVAILABLE",
                          {"usable_ticks": int(usable)},
                          f"only {usable} usable ticks (need >=20 for the canary) -- likely a hover "
                          f"/ near-level replay with no banked flight to test the mirror")

        # mirror canary -- the SAME flag frame_residual_report.report prints
        bank = tilts > 30
        if bank.sum() < 20:
            res_med = np.median(res_t, axis=0)
            return _check("UNAVAILABLE",
                          {"usable_ticks": int(usable), "n_bank": int(bank.sum()),
                           "true_residual_median_NED": [round(float(x), 3) for x in res_med]},
                          f"only {int(bank.sum())} banked ticks (tilt>30) -- canary needs banked "
                          f"flight; cannot test the R_y(pi) mirror on near-level data")
        me = np.array(meas_e)[bank]
        ct = float(np.corrcoef(np.array(mdl_te)[bank], me)[0, 1])
        cr = float(np.corrcoef(np.array(mdl_re)[bank], me)[0, 1])
        canary_ok = ct > max(cr + 0.2, 0.6)   # identical predicate to the primitive
        res_med = np.median(res_t, axis=0)
        # a sustained TRUE-attitude residual (|.|>3 m/s^2 per the primitive's own annotation) is a
        # plant gap, not a mirror -- we surface it but it does NOT by itself trip the conjugation gate.
        plant_gap = bool(np.max(np.abs(res_med)) > 3.0)
        detail = {
            "usable_ticks": int(usable), "n_bank": int(bank.sum()),
            "mirror_canary": {"east_corr_TRUE": round(ct, 3), "east_corr_ASIS": round(cr, 3),
                              "ok": canary_ok},
            "true_residual_median_NED": [round(float(x), 3) for x in res_med],
            "plant_gap_flag": plant_gap,
        }
        if not canary_ok:
            return _check("ATTENTION", detail,
                          f"!! R_y(pi) MIRROR CANARY TRIPPED: East corr TRUE {ct:+.2f} "
                          f"AS-IS {cr:+.2f} (healthy TRUE>>AS-IS) -- a mirror crept back into "
                          f"the frame convention; RE-AUDIT before trusting this run")
        return _check("PASS", detail,
                      f"R_y(pi) mirror canary OK (East corr TRUE {ct:+.2f} >> AS-IS {cr:+.2f})"
                      + ("  [note: TRUE-attitude residual >3 m/s^2 -> possible plant gap]"
                         if plant_gap else ""))
    except Exception as exc:
        return _check("UNAVAILABLE", {"error": f"{type(exc).__name__}: {exc}"},
                      f"frame-residual canary errored: {type(exc).__name__}: {exc}")


# ==============================================================================================
# CHECK 3 -- BUNDLE INTEGRITY (compose scripts/verify_bundle)
# ==============================================================================================
def _bundle(session: Path) -> dict:
    """Stream-presence + GT-velocity-live integrity. Needs the tlog + video index; UNAVAILABLE on a
    replay-only bundle or when pymavlink's serial dep is missing."""
    if not (session / "mavlink.tlog").exists():
        return _check("UNAVAILABLE", summary="no mavlink.tlog (replay-only bundle; nothing to verify)")
    try:
        import verify_bundle
        r = verify_bundle.verify(session)
        failed = [k for k, c in r["checks"].items() if not c["pass"]]
        return _check("PASS" if r["ALL_PASS"] else "ATTENTION",
                      {"report": r},
                      "all required streams present + GT velocity live"
                      if r["ALL_PASS"] else f"bundle integrity FAILED: {', '.join(failed)}")
    except Exception as exc:
        return _check("UNAVAILABLE", {"error": f"{type(exc).__name__}: {exc}"},
                      f"verify_bundle unavailable: {type(exc).__name__}: {exc}")


# ==============================================================================================
# CHECK 4 -- OBS / ACTION HEALTH (direct debug_obs.jsonl reader)
# ==============================================================================================
def _obs_health(session: Path) -> dict:
    """Per-tick obs/action sanity from debug_obs.jsonl: NON-FINITE obs or action is the hard gate
    (a NaN/inf on the wire = a corrupt run). Also reports action-saturation fraction, ODO staleness,
    mid-run sim resets, and the in-run collision count -- all surfaced for triage, only non-finite
    forces ATTENTION here (the abort family is owned by the grade/taxonomy)."""
    f = session / "debug_obs.jsonl"
    if not f.exists():
        return _check("UNAVAILABLE", summary="no debug_obs.jsonl")
    header = None
    n = 0
    n_nonfinite_obs = 0
    n_nonfinite_act = 0
    first_bad = None
    act_min = act_max = None
    n_sat = 0
    max_odo_age = 0.0
    n_stale_odo = 0
    reset_counters = set()
    max_n_coll = 0
    SAT_EPS = 1e-3
    STALE_MS = 100.0   # odo_age beyond this is a freshness concern (fly_rl odo gate is ~0.5 s)

    try:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("type") == "header":
                    header = row
                    act_min = row.get("act_min")
                    act_max = row.get("act_max")
                    continue
                n += 1
                obs = row.get("obs")
                if obs is not None and not all(isinstance(x, (int, float)) and math.isfinite(x) for x in obs):
                    n_nonfinite_obs += 1
                    if first_bad is None:
                        first_bad = {"k": row.get("k"), "where": "obs", "value": obs}
                act = row.get("act_rescaled")
                if act is None:
                    act = row.get("actor_mean")
                if act is not None:
                    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in act):
                        n_nonfinite_act += 1
                        if first_bad is None:
                            first_bad = {"k": row.get("k"), "where": "action", "value": act}
                    elif act_min is not None and act_max is not None and len(act) == len(act_min):
                        # count a tick "saturated" if any actuator pins an action bound
                        if any(a <= lo + SAT_EPS or a >= hi - SAT_EPS
                               for a, lo, hi in zip(act, act_min, act_max)):
                            n_sat += 1
                age = row.get("odo_age_ms")
                if isinstance(age, (int, float)):
                    max_odo_age = max(max_odo_age, float(age))
                    if age > STALE_MS:
                        n_stale_odo += 1
                rc = row.get("reset_counter")
                if rc is not None:
                    reset_counters.add(int(rc))
                nc = row.get("n_coll")
                if isinstance(nc, (int, float)):
                    max_n_coll = max(max_n_coll, int(nc))
    except Exception as exc:
        return _check("UNAVAILABLE", {"error": f"{type(exc).__name__}: {exc}"},
                      f"debug_obs.jsonl unreadable: {type(exc).__name__}: {exc}")

    if n == 0:
        return _check("UNAVAILABLE", summary="debug_obs.jsonl has a header but no tick rows")

    n_nonfinite = n_nonfinite_obs + n_nonfinite_act
    detail = {
        "n_ticks": n,
        "obs_dim": len(header["obs_labels"]) if header and header.get("obs_labels") else None,
        "checkpoint": header.get("checkpoint") if header else None,
        "n_nonfinite_obs": n_nonfinite_obs,
        "n_nonfinite_action": n_nonfinite_act,
        "first_nonfinite": first_bad,
        "action_saturation_frac": round(n_sat / n, 3),
        "max_odo_age_ms": round(max_odo_age, 1),
        "n_stale_odo_ticks": n_stale_odo,
        "reset_counters_seen": sorted(reset_counters),
        "n_resets": max(0, len(reset_counters) - 1),
        "max_in_run_collisions": max_n_coll,
    }
    if n_nonfinite > 0:
        return _check("ATTENTION", detail,
                      f"!! NON-FINITE on the wire: {n_nonfinite_obs} obs + {n_nonfinite_act} action "
                      f"tick(s) over {n} (first @k={first_bad.get('k') if first_bad else '?'})")
    extra = []
    if detail["n_resets"] > 0:
        extra.append(f"{detail['n_resets']} mid-run reset(s)")
    if max_n_coll > 0:
        extra.append(f"{max_n_coll} collision(s)")
    if n_stale_odo > 0:
        extra.append(f"{n_stale_odo} stale-ODO tick(s) (max {max_odo_age:.0f} ms)")
    return _check("PASS", detail,
                  f"{n} ticks all finite; action sat {detail['action_saturation_frac']*100:.0f}%"
                  + (("; " + ", ".join(extra)) if extra else ""))


# ==============================================================================================
# TAXONOMY + OVERALL VERDICT
# ==============================================================================================
def _taxonomy_tag(grade: dict, frame: dict, obs: dict, meta: dict) -> str:
    """Pick the single highest-priority failure tag. The recorder's terminal abort (meta.final_state)
    is authoritative for the abort family; otherwise we derive from the composed signals.

    Priority (most-actionable / earliest-failing first):
      ARM_REFUSED > NO_GO > SIM_RESET > NON_FINITE > ODO_STALE > SPIN_ABORT > CRASH > TIMEOUT > MISS
    """
    fs = (meta.get("final_state") or "").upper()
    # 1) the recorder named a terminal abort -> trust it (it saw the live wire)
    if fs in _ABORT_TAGS:
        return _ABORT_TAGS[fs]
    # 2) non-finite on the wire (obs check is the source of truth even if recorder didn't abort)
    if obs.get("status") == "ATTENTION" and obs.get("n_nonfinite_obs", 0) + obs.get("n_nonfinite_action", 0) > 0:
        return "NON_FINITE"
    # 3) mid-run reset seen in the per-tick stream but not flagged by the recorder
    if obs.get("status") != "UNAVAILABLE" and obs.get("n_resets", 0) > 0:
        return "SIM_RESET"
    # 4) graded outcome: a finish with contact / env-collision OR a crash-by-collision
    out = grade.get("outcome") if grade.get("source") == "tlog" else None
    if out is not None:
        if out.get("n_env_collisions", 0) > 0 or out.get("n_gate_collisions_no_pass", 0) > 0:
            return "CRASH"
        if out.get("finished") and not out.get("clean_finish"):
            return "MISS"   # finished but with gate contact
        if not out.get("finished"):
            return "MISS"   # started, never finished, no recorded collision -> incomplete lap
    else:
        # meta-only grade
        if int(meta.get("collisions", 0) or 0) > 0:
            return "CRASH"
        if fs in _OTHER_ATTENTION_STATES:
            return "ATTENTION_OTHER"
        if fs and fs not in _CLEAN_STATES:
            return "MISS"
    # 5) the canary tripped but nothing else did
    if frame.get("status") == "ATTENTION":
        return "ATTENTION_OTHER"
    return "ATTENTION_OTHER"


def diagnose(recording_dir: str | Path, contact_window_s: float = 0.5) -> dict:
    """Compose the four primitives into one verdict for ``recording_dir``. PURE w.r.t. its result
    (returns the dict); the only side effects are reading the bundle's files."""
    session = Path(recording_dir)
    result: dict = {"session": str(session), "exists": session.is_dir()}
    if not session.is_dir():
        result.update({"verdict": "ATTENTION", "taxonomy_tag": "ATTENTION_OTHER",
                       "summary": f"not a directory: {session}", "checks": {}})
        return result

    meta = {}
    mpath = session / "meta.json"
    if mpath.exists():
        try:
            meta = json.loads(mpath.read_text(encoding="utf-8"))
        except Exception as exc:
            meta = {"_meta_error": f"{type(exc).__name__}: {exc}"}

    checks = {
        "grade":          _grade(session, meta, contact_window_s),
        "frame_residual": _frame_residual(session),
        "bundle":         _bundle(session),
        "obs_health":     _obs_health(session),
    }

    # Overall verdict: PASS iff NO composed check is ATTENTION (UNAVAILABLE does not block, but is
    # surfaced). The R_y(pi) canary is load-bearing -- when it can run, an ATTENTION there forces
    # the whole verdict to ATTENTION.
    attentions = [k for k, c in checks.items() if c["status"] == "ATTENTION"]
    unavailable = [k for k, c in checks.items() if c["status"] == "UNAVAILABLE"]
    verdict = "ATTENTION" if attentions else "PASS"
    tag = "OK" if verdict == "PASS" else _taxonomy_tag(
        checks["grade"], checks["frame_residual"], checks["obs_health"], meta)

    result.update({
        "verdict": verdict,
        "taxonomy_tag": tag,
        "attention_checks": attentions,
        "unavailable_checks": unavailable,
        "meta": {k: meta.get(k) for k in ("label", "final_state", "gate_index", "collisions",
                                          "checkpoint", "duration_s")},
        "checks": checks,
    })
    return result


# ==============================================================================================
# CLI
# ==============================================================================================
def _print_human(r: dict) -> None:
    sess = Path(r["session"]).name
    icon = "PASS " if r["verdict"] == "PASS" else "ATTN!"
    print(f"\n=== diagnose: {sess}")
    print(f"  VERDICT: [{icon}] {r['verdict']}   tag={r['taxonomy_tag']}")
    m = r.get("meta", {})
    if m:
        print(f"  meta: final_state={m.get('final_state')}  gate_index={m.get('gate_index')}  "
              f"collisions={m.get('collisions')}  label={m.get('label')}")
    for name, c in r.get("checks", {}).items():
        mark = {"PASS": "ok  ", "ATTENTION": "ATTN", "UNAVAILABLE": "n/a "}.get(c["status"], "??  ")
        print(f"    [{mark}] {name:<15} {c.get('summary','')}")
    if r["verdict"] != "PASS":
        print(f"  -> ATTENTION on: {', '.join(r.get('attention_checks', [])) or '(taxonomy only)'}")
    if r.get("unavailable_checks"):
        print(f"  (i) unavailable (input absent in this bundle): {', '.join(r['unavailable_checks'])}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording_dir")
    ap.add_argument("--contact-window", type=float, default=0.5,
                    help="s; a gate collision within this of a pass = passed WITH CONTACT")
    ap.add_argument("--json", default=None, help="write the full machine-readable verdict here")
    ap.add_argument("--quiet", action="store_true", help="suppress the human summary (JSON/exit only)")
    args = ap.parse_args()

    r = diagnose(args.recording_dir, contact_window_s=args.contact_window)
    if not args.quiet:
        _print_human(r)
    if args.json:
        Path(args.json).write_text(json.dumps(r, indent=2), encoding="utf-8")
        if not args.quiet:
            print(f"\n  -> {args.json}")
    # exit 0 = PASS, 1 = ATTENTION (so a triage loop can branch on the process result)
    return 0 if r["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
