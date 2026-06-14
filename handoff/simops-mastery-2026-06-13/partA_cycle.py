"""partA_cycle.py — PART A: probe-VERIFIED autonomous sim-ops cycle loop (SIMOPS-MASTERY).

Demonstrate N consecutive clean  launch -> waiting-room -> race-GO -> inc7 flight -> finish/abort
-> full-reset  cycles with ZERO manual intervention, logging every state transition (probe-confirmed,
not blind). Each cycle runs `rl/fly_rl.py --flights 1` (faithful inc7 passive launch path) and frees
the 14550 endpoint on exit so we can PROBE the true sim state between every step.

Why not `fly_rl --flights N`: that loop resets BLIND (it holds 14550, can't probe), so a missed key
send silently NO_GO's the next flight. Here every transition is verified by a passive MAVLink probe,
and each known failure mode has an explicit, logged recovery.

Sim state machine (Fengyou-confirmed, shadowpc-vision-cal 2026-06-13):
  NO_HEARTBEAT (login/title)  --Enter--> HOME (hb, pos=None)
  HOME                        --Enter--> WAITING (started=False, drone@origin)
  WAITING                     --Enter--> ACTIVE race GO (started=True, countdown -> moving, video)
  ACTIVE/FINISHED             --ESC,Down x3,Enter--> HOME (fresh restart; clock resets)
  STALE started (idle mins)   : started=True, to_go very -ve, NO video -> full chain to recover

Usage:
  .venv\\Scripts\\python handoff/simops-mastery-2026-06-13/partA_cycle.py --cycles 10 \
      --label simopsA --out handoff/simops-mastery-2026-06-13/logs/partA_timeline.json
"""
from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

_VK = {"enter": 0x0D, "esc": 0x1B, "down": 0x28, "up": 0x26}
_KEYUP = 0x0002
ENDPOINT = "udp:127.0.0.1:14550"
SIM_EXE = r"C:\Users\Shadow\Downloads\AI-GP Simulator v1.0.3364\AIGP_3364\FlightSim.exe"


# ----------------------- Win32 mechanics (proven; simops.py / fly_rl.py) -----------------------
def find_sim_window():
    user32 = ctypes.windll.user32
    found = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def _cb(hwnd, _lp):
        if user32.IsWindowVisible(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            if n:
                buf = ctypes.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(hwnd, buf, n + 1)
                t = buf.value.lower()
                if "ai-gp" in t or "flightsim" in t or "ai grand prix" in t:
                    found.append(hwnd)
        return True

    user32.EnumWindows(_cb, None)
    return found[0] if found else None


def force_foreground(hwnd, attempts=4):
    user32 = ctypes.windll.user32
    for _ in range(attempts):
        user32.ShowWindow(hwnd, 9)
        time.sleep(0.3)
        user32.keybd_event(0x12, 0, 0, 0)
        user32.SetForegroundWindow(hwnd)
        user32.keybd_event(0x12, 0, _KEYUP, 0)
        time.sleep(0.4)
        if user32.GetForegroundWindow() == hwnd:
            return True
    return False


def send_keys(tokens: str) -> bool:
    """Verified-foreground key driver. tokens = comma list of enter|esc|down[:settle_s]."""
    hwnd = find_sim_window()
    if hwnd is None:
        print("  [keys] sim window NOT found -> cannot send", file=sys.stderr)
        return False
    if not force_foreground(hwnd):
        print("  [keys] could NOT verify sim foreground -> refusing (would hit wrong window)",
              file=sys.stderr)
        return False
    user32 = ctypes.windll.user32
    for tok in tokens.split(","):
        tok = tok.strip()
        if not tok:
            continue
        name, _, settle = tok.partition(":")
        vk = _VK.get(name.lower())
        if vk is None:
            print(f"  [keys] unknown token {name!r}", file=sys.stderr)
            return False
        if user32.GetForegroundWindow() != hwnd and not force_foreground(hwnd):
            print("  [keys] lost foreground mid-sequence -> abort", file=sys.stderr)
            return False
        user32.keybd_event(vk, 0, 0, 0)
        time.sleep(0.06)
        user32.keybd_event(vk, 0, _KEYUP, 0)
        time.sleep(float(settle) if settle else 0.4)
    return True


# ----------------------- passive probe -----------------------
def probe(seconds: float = 2.0) -> dict:
    """Passive MAVLink read (NO heartbeat -> never flips the sim to ACRO). Binds 14550, so call
    only when fly_rl is NOT running. Returns a structured state dict."""
    from racer.mavlink_client import MavlinkClient
    c = MavlinkClient(ENDPOINT)
    c.send_heartbeats = False
    c.send_timesync = False
    out = {"heartbeat": False, "started": None, "finished": None, "to_go_s": None,
           "active_gate": None, "pos_off_m": None, "sim_time_ns": None, "armed": None,
           "n_gates": 0, "msg_types": {}}
    try:
        c.connect(wait_heartbeat=True, timeout_s=8.0)
    except Exception:
        return out
    out["heartbeat"] = True
    types: dict = {}
    if c.conn is not None:
        orig = c.conn.recv_match

        def tap(*a, **k):
            m = orig(*a, **k)
            if m is not None:
                t = m.get_type()
                types[t] = types.get(t, 0) + 1
            return m
        c.conn.recv_match = tap
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        c.pump()
        time.sleep(0.005)
    s, rs = c.state, c.race_status
    out["msg_types"] = types
    out["sim_time_ns"] = int(s.sim_time_ns)
    out["armed"] = bool(s.armed)
    out["n_gates"] = len(c.track_gates) if c.track_gates else 0
    if s.position_ned is not None:
        out["pos_off_m"] = round(float(np.linalg.norm(s.position_ned)), 2)
    if rs:
        out["started"] = bool(rs["started"])
        out["finished"] = bool(rs["finished"])
        out["to_go_s"] = round((rs["race_start_boot_time_ms"] - rs["sim_boot_time_ms"]) / 1000.0, 2)
        out["active_gate"] = rs.get("active_gate_index")
    try:
        if c.conn is not None:
            c.conn.close()
    except Exception:
        pass
    return out


def classify(p: dict) -> str:
    if not p["heartbeat"]:
        return "NO_HEARTBEAT"
    if p["started"] and p["finished"]:
        return "FINISHED"
    if p["started"] and p.get("video_active"):
        return "ACTIVE"
    if p["started"]:
        # started but not finished: fresh countdown/active, or a STALE parked race
        if p["to_go_s"] is not None and p["to_go_s"] < -120 and (p["pos_off_m"] or 0) < 2:
            return "STALE"
        return "ACTIVE_OR_COUNTDOWN"
    if p["pos_off_m"] is not None:
        return "WAITING"
    return "HOME"


def n_sim_procs() -> int:
    """Zombie dual-instance guard: count the DCGame RENDER instance only (the one that binds
    14550). Normal = exactly 1 DCGame (+ 1 windowless FlightSim launcher, not counted). TWO
    DCGame instances == the zombie dual-instance that arms-but-deaf (memory)."""
    try:
        out = subprocess.check_output(
            ["powershell.exe", "-NoProfile", "-Command",
             "@(Get-Process -Name 'DCGame-Win64-Shipping' -ErrorAction SilentlyContinue).Count"],
            stderr=subprocess.DEVNULL, timeout=15).decode().strip()
        return int(out or "0")
    except Exception:
        return -1


# ----------------------- drive helpers -----------------------
def drive_to_waiting(log, max_attempts=4) -> dict:
    """Get the sim into the WAITING room (started=False, drone@origin), verified by probe.
    Handles login/home/finished/stale entry states with explicit recovery."""
    for attempt in range(1, max_attempts + 1):
        p = probe(2.0)
        st = classify(p)
        log.append({"t": time.time(), "step": f"drive_to_waiting probe#{attempt}", "state": st,
                    "probe": p})
        if st == "WAITING":
            return p
        if st == "NO_HEARTBEAT":
            send_keys("enter:2.0,enter:2.0")          # login/title -> HOME
        elif st == "HOME":
            send_keys("enter:2.0")                     # HOME -> waiting room
        elif st in ("FINISHED", "ACTIVE", "ACTIVE_OR_COUNTDOWN", "STALE"):
            # full chain to HOME then ONE Enter to the waiting room (stop short of race GO)
            send_keys("esc:1.0,down:0.4,down:0.4,down:0.4,enter:2.5,enter:1.8")
        time.sleep(1.0)
    return probe(2.0)


def run_cycle(i: int, args, log: list) -> dict:
    """One verified cycle: ensure waiting -> launch fly_rl --flights 1 -> GO -> finish -> reset."""
    cyc = {"cycle": i, "transitions": [], "result": None, "collisions": None,
           "gate_index": None, "clean": False, "t_start": time.time()}

    def mark(step, **kw):
        e = {"t": round(time.time(), 3), "step": step, **kw}
        cyc["transitions"].append(e)
        print(f"  [c{i}] {step}: {kw}")

    # 1) ensure WAITING room
    p = drive_to_waiting(log)
    mark("waiting_room", state=classify(p), started=p["started"], pos_off=p["pos_off_m"],
         n_gates=p["n_gates"])
    if classify(p) != "WAITING":
        mark("FAILED_to_reach_waiting", probe=p)
        return cyc

    # 2) launch fly_rl --flights 1 (binds 14550); it waits passively for the GO
    label = f"{args.label}_c{i:02d}"
    log_path = Path(args.logdir) / f"flyrl_{label}.log"
    cmd = [str(ROOT / ".venv/Scripts/python.exe"), str(ROOT / "rl/fly_rl.py"),
           "--flights", "1", "--label", label, "--no-bridge", "--no-auto-reset",
           "--checkpoint", str(ROOT / "rl/checkpoints/stage1_inc7_actor.pth"),
           "--max-seconds", str(args.max_seconds), "--wait-seconds", "60"]
    if args.debug_obs:
        cmd.append("--debug-obs")
    mark("launch_flyrl", cmd_label=label)
    fl = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=fl, stderr=subprocess.STDOUT, cwd=str(ROOT))

    # 3) wait until fly_rl is connected + waiting, then send the GO Enter (organizer race start)
    waited, go_sent = 0.0, False
    while waited < 30.0:
        time.sleep(1.0)
        waited += 1.0
        txt = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""
        # prefer fly_rl's passive-wait banner; fixed fallback only if it never appears
        if "Waiting PASSIVELY" in txt or "GO!" in txt or waited >= 14.0:
            send_keys("enter:1.5")                    # WAITING -> race GO
            mark("GO_sent", after_s=waited, via=("banner" if "Waiting PASSIVELY" in txt else "timeout"))
            go_sent = True
            break
    if not go_sent:
        send_keys("enter:1.5")
        mark("GO_sent_fallback")

    # 4) wait for fly_rl to finish (it disarms + exits on finish/abort)
    try:
        proc.wait(timeout=args.max_seconds + 40)
    except subprocess.TimeoutExpired:
        proc.kill()
        mark("flyrl_TIMEOUT_killed")
    fl.close()
    rc = proc.returncode

    # 5) read the flight result from the session meta.json
    sess = sorted(Path("data/runs").glob(f"*_{label}_f1"), key=lambda d: d.stat().st_mtime)
    final_state, gate_index, collisions, n_frames = "UNKNOWN", None, None, None
    if sess:
        meta = json.loads((sess[-1] / "meta.json").read_text())
        final_state = meta.get("final_state")
        gate_index = meta.get("gate_index")
        collisions = meta.get("collisions")
        n_frames = meta.get("video_frames")
    mark("flyrl_exit", rc=rc, final_state=final_state, gate_index=gate_index,
         collisions=collisions, video_frames=n_frames, session=(sess[-1].name if sess else None))
    cyc.update(result=final_state, gate_index=gate_index, collisions=collisions)

    # 6) detect spawn-artefact 0-tick crash (obs header only / 0 usable ticks)
    if n_frames is not None and n_frames < 10 and final_state != "FINISHED":
        mark("SPAWN_ARTEFACT_suspected", recovery="sleep 3s before next arm")
        time.sleep(3.0)

    # 7) probe to confirm post-flight state
    p2 = probe(2.0)
    mark("post_flight_probe", state=classify(p2), finished=p2["finished"],
         active_gate=p2["active_gate"], pos_off=p2["pos_off_m"])

    # 8) full reset to a fresh HOME for the next cycle (verified next cycle by drive_to_waiting)
    if i < args.cycles:
        ok = send_keys("esc:1.0,down:0.4,down:0.4,down:0.4,enter:2.5")   # -> HOME
        mark("full_reset_to_home", keys_ok=ok)

    cyc["clean"] = (final_state == "FINISHED" and (collisions or 0) == 0)
    cyc["t_end"] = time.time()
    cyc["dur_s"] = round(cyc["t_end"] - cyc["t_start"], 1)
    return cyc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cycles", type=int, default=10)
    ap.add_argument("--label", default="simopsA")
    ap.add_argument("--max-seconds", type=float, default=30.0)
    ap.add_argument("--out", default="handoff/simops-mastery-2026-06-13/logs/partA_timeline.json")
    ap.add_argument("--logdir", default="handoff/simops-mastery-2026-06-13/logs")
    ap.add_argument("--debug-obs", action="store_true",
                    help="pass --debug-obs to fly_rl (for the mandatory frame_residual_report check)")
    args = ap.parse_args()
    Path(args.logdir).mkdir(parents=True, exist_ok=True)

    print(f"\n#### PART A: {args.cycles} verified sim-ops cycles (inc7) ####")
    nproc = n_sim_procs()
    print(f"sim processes detected: {nproc}  "
          f"({'OK' if nproc == 1 else 'ZOMBIE RISK — expect exactly 1' if nproc > 1 else 'sim down?'})")

    log: list = []
    cycles: list = []
    for i in range(1, args.cycles + 1):
        print(f"\n================= CYCLE {i}/{args.cycles} =================")
        # zombie guard each cycle
        npc = n_sim_procs()
        if npc > 1:
            print(f"  ZOMBIE DUAL-INSTANCE: {npc} sim procs -> drone arms-but-deaf. "
                  f"Manual relaunch required (killing one kills both).")
            log.append({"t": time.time(), "step": "ZOMBIE_DETECTED", "n_procs": npc})
        cyc = run_cycle(i, args, log)
        cycles.append(cyc)
        Path(args.out).write_text(json.dumps({"cycles": cycles, "drive_log": log}, indent=2,
                                             default=float), encoding="utf-8")
        verdict = "CLEAN" if cyc["clean"] else f"NOT-CLEAN ({cyc['result']})"
        print(f"  ==> CYCLE {i}: {verdict}  gates={cyc['gate_index']} "
              f"coll={cyc['collisions']}  dur={cyc.get('dur_s')}s")

    n_clean = sum(c["clean"] for c in cycles)
    n_consec = 0
    best = 0
    for c in cycles:
        n_consec = n_consec + 1 if c["clean"] else 0
        best = max(best, n_consec)
    print(f"\n#### PART A SUMMARY: {n_clean}/{args.cycles} clean, "
          f"longest consecutive-clean streak = {best} ####")
    Path(args.out).write_text(json.dumps(
        {"summary": {"cycles": args.cycles, "clean": n_clean, "longest_streak": best},
         "cycles": cycles, "drive_log": log}, indent=2, default=float), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0 if best >= 10 or n_clean >= args.cycles else 1


if __name__ == "__main__":
    raise SystemExit(main())
