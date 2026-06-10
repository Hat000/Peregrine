"""scripts/race_outcome.py -- authoritative per-gate PASS-vs-COLLISION verdict for a recording.

Reports, per gate, whether it was passed CLEAN, passed WITH CONTACT, or hit-without-passing, from
the SIM's AUTHORITATIVE signals (``RACE_STATUS.active_gate_index`` transitions + ``COLLISION``
events) -- NOT our geometric plane-crossing heuristic or eyeballing the GUI (both historically
unreliable). The RL validation oracle + the run-grading tool. Logic lives in
``racer.race_outcome`` (unit-tested); this is the CLI.

Usage: .venv\\Scripts\\python scripts/race_outcome.py data/runs/<stamp>_<label>
       [--contact-window 0.5] [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from racer.race_outcome import analyze_outcome, load_from_recording


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--contact-window", type=float, default=0.5,
                    help="s; a gate collision within this of a pass = passed WITH CONTACT")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    race_samples, collisions = load_from_recording(args.session)
    out = analyze_outcome(race_samples, collisions, contact_window_s=args.contact_window)
    out["session"] = str(args.session)

    print(f"=== race outcome: {Path(args.session).name}")
    if "error" in out:
        print(f"  !! {out['error']}  (gate collisions={out['n_gate_collisions']}, env={out['n_env_collisions']})")
        return 0
    if out.get("had_pre_race_residue"):
        print("  (i) discarded leading RACE_STATUS residue from a PRIOR race; scored the final epoch only")
    print(f"  started={out['started']}  finished={out['finished']}  CLEAN FINISH={out['clean_finish']}")
    if out["recognized_time_ns"] is not None and out["recognized_time_ns"] >= 0:
        print(f"  sim-recognized time: {out['recognized_time_ns'] / 1e9:.2f} s")
    print(f"  gates passed: {out['gates_passed']}  (clean {out['n_pass_clean']} / "
          f"contact {out['n_pass_contact']})   next-active-index reached: {out['max_active_gate_index']}")
    for p in out["passes"]:
        tag = "" if not p["contact"] else f"  <-- CONTACT (threat {p['threat_level']})"
        print(f"    gate {p['gate']:>2}: {p['verdict']}{tag}")
    if out["n_gate_collisions_no_pass"]:
        print(f"  !! {out['n_gate_collisions_no_pass']} gate collision(s) with NO pass (hit, did not advance)")
    if out["n_env_collisions"]:
        print(f"  !! {out['n_env_collisions']} ENVIRONMENT collision(s) (id 1002 = wall/terrain)")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"  -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
