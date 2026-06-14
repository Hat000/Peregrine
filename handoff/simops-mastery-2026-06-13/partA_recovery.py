"""partA_recovery.py — PART A failure-mode recovery demonstrations (SIMOPS-MASTERY).

Each known sim failure mode (reference_sim_ops.md) gets an explicit, probe-verified recovery.
The 10-cycle clean loop (partA_cycle.py) already proves the OFF-RACE-PAUSE -> full-reset -> re-enter
path 13x; this tool covers the rest. Safe/fast modes are demonstrated LIVE; the spawn-artefact (which
requires a gate-3 HARD COLLISION to induce, excluded by the zero-contact rule) is documented with its
wired detection.

  pause    : off-race physics pause — sim_time FROZEN at HOME, advances once a race is active.
  stale    : a STARTED-but-parked race (the precursor to the video-stopping stale state) recovered
             by the full ESC+Down x3+Enter+Enter chain -> fresh waiting room (verified).
  zombie   : dual-instance detection (2 DCGame on 14550 = arms-but-deaf) + the documented recovery
             (kill ALL + relaunch ONE fresh, since killing one kills both). TEARS DOWN the sim.
  artefact : spawn-artefact 0-tick crash — detection (meta video_frames<10 & not FINISHED) + the
             sleep-before-arm recovery; documented (not induced — needs a gate-3 collision).

Usage: .venv\\Scripts\\python handoff/simops-mastery-2026-06-13/partA_recovery.py --demo pause,stale,artefact
       .venv\\Scripts\\python handoff/simops-mastery-2026-06-13/partA_recovery.py --demo zombie   # teardown
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from partA_cycle import (SIM_EXE, classify, drive_to_waiting, find_sim_window, n_sim_procs,  # noqa
                         probe, send_keys)

RESULTS: list = []


def _log(demo, **kw):
    e = {"demo": demo, "t": round(time.time(), 2), **kw}
    RESULTS.append(e)
    print(f"  [{demo}] {kw}")


def demo_pause() -> None:
    print("\n=== DEMO: off-race physics PAUSE ===")
    # get to HOME (off-race): full chain to HOME, stop before the waiting room
    send_keys("esc:1.0,down:0.4,down:0.4,down:0.4,enter:2.5")
    p1 = probe(2.0)
    time.sleep(2.0)
    p2 = probe(2.0)
    frozen = (p1["sim_time_ns"] == p2["sim_time_ns"]) or p1["sim_time_ns"] == 0
    _log("pause", state=classify(p1), sim_time_1=p1["sim_time_ns"], sim_time_2=p2["sim_time_ns"],
         physics_frozen_offrace=frozen)
    # now drive to a race and show sim_time advancing
    pw = drive_to_waiting(RESULTS)
    send_keys("enter:1.5")            # GO
    pa1 = probe(1.5)
    time.sleep(1.5)
    pa2 = probe(1.5)
    advancing = pa2["sim_time_ns"] > pa1["sim_time_ns"] > 0
    _log("pause", after_GO_state=classify(pa2), sim_time_a=pa1["sim_time_ns"],
         sim_time_b=pa2["sim_time_ns"], physics_advancing_inrace=advancing,
         RECOVERY="off-race pause is EXPECTED; live control/sysid needs an ACTIVE race "
                  "(drive HOME->waiting->GO).")


def demo_stale() -> None:
    print("\n=== DEMO: STALE started=True recovery (full chain) ===")
    # induce a STARTED-but-parked race: drive to waiting, GO, but never arm -> started=True,
    # drone idle at origin. (Left for minutes this becomes the video-stopping stale state; the
    # recovery chain is identical, so we verify the chain here without the minutes-long wait.)
    drive_to_waiting(RESULTS)
    send_keys("enter:1.5")            # GO (no stack attached -> nothing arms)
    time.sleep(2.0)
    ps = probe(2.0)
    _log("stale", induced_state=classify(ps), started=ps["started"], to_go_s=ps["to_go_s"],
         pos_off=ps["pos_off_m"], note="started=True, drone parked at origin (stale precursor)")
    # RECOVERY: the full ESC+Down x3+Enter+Enter chain -> fresh waiting room
    send_keys("esc:1.0,down:0.4,down:0.4,down:0.4,enter:2.5,enter:1.8")
    pr = probe(2.0)
    recovered = classify(pr) == "WAITING"
    _log("stale", recovered_state=classify(pr), started=pr["started"], pos_off=pr["pos_off_m"],
         RECOVERED=recovered,
         RECOVERY="full chain esc,down*3,enter,enter -> fresh waiting room (started=False). The "
                  "real stale state (video stopped after idle minutes) uses this SAME chain.")


def demo_artefact() -> None:
    print("\n=== DEMO: spawn-artefact 0-tick crash (detection + documented recovery) ===")
    # The artefact only follows a gate-3 HARD COLLISION (excluded by the zero-contact rule), so we
    # do NOT induce it. Show the wired detection on real recordings: a healthy flight has
    # video_frames >> 10 and final_state FINISHED; an artefact flight has video_frames < 10 / 0 ticks.
    runs = sorted(Path("data/runs").glob("*_simopsA_c*_f1"))
    sample = []
    for r in runs[:3]:
        try:
            m = json.loads((r / "meta.json").read_text())
            sample.append({"run": r.name, "video_frames": m.get("video_frames"),
                           "final_state": m.get("final_state"),
                           "artefact_flag": (m.get("video_frames", 0) < 10
                                             and m.get("final_state") != "FINISHED")})
        except Exception:
            pass
    _log("artefact", healthy_samples=sample,
         DETECTION="meta.video_frames < 10 AND final_state != FINISHED (obs header only, 0 usable "
                   "ticks) -> spawn-artefact.",
         RECOVERY="sleep 2-3 s after the full-reset BEFORE issuing arm (let residual gate-3 collision "
                  "geometry clear); wired in partA_cycle.run_cycle. NOT induced here (needs a gate-3 "
                  "hard collision; violates the zero-contact validity rule).")


def demo_zombie() -> None:
    print("\n=== DEMO: ZOMBIE dual-instance (detect + kill-all + relaunch) — TEARDOWN ===")
    before = n_sim_procs()
    _log("zombie", dcgame_procs_before=before)
    # induce: launch a SECOND FlightSim launcher
    subprocess.Popen([SIM_EXE])
    time.sleep(25.0)                 # let a 2nd DCGame spawn (GPU-bound)
    during = n_sim_procs()
    _log("zombie", dcgame_procs_after_2nd_launch=during,
         ZOMBIE_DETECTED=(during is not None and during > 1))
    # RECOVERY: kill ALL sim processes (killing one kills both anyway) + relaunch ONE fresh
    subprocess.run(["powershell.exe", "-NoProfile", "-Command",
                    "Get-Process -Name 'DCGame-Win64-Shipping','FlightSim' "
                    "-ErrorAction SilentlyContinue | Stop-Process -Force"],
                   timeout=20)
    time.sleep(4.0)
    killed = n_sim_procs()
    _log("zombie", dcgame_procs_after_killall=killed)
    subprocess.Popen([SIM_EXE])
    # wait for the fresh instance to emit telemetry
    healthy, waited = False, 0.0
    while waited < 70.0:
        time.sleep(5.0)
        waited += 5.0
        if n_sim_procs() == 1 and probe(2.0)["heartbeat"]:
            healthy = True
            break
    _log("zombie", relaunched_dcgame_procs=n_sim_procs(), heartbeat_after_s=waited,
         RECOVERED=healthy,
         RECOVERY="kill ALL (DCGame+FlightSim) then relaunch ONE FlightSim.exe -> single clean "
                  "instance on 14550. A live worker must verify exactly 1 DCGame before flying.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", default="pause,stale,artefact",
                    help="comma list: pause,stale,zombie,artefact")
    ap.add_argument("--out", default="handoff/simops-mastery-2026-06-13/logs/partA_recovery.json")
    args = ap.parse_args()
    demos = {"pause": demo_pause, "stale": demo_stale, "zombie": demo_zombie,
             "artefact": demo_artefact}
    for name in [d.strip() for d in args.demo.split(",") if d.strip()]:
        if name in demos:
            demos[name]()
    Path(args.out).write_text(json.dumps(RESULTS, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
