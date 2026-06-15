"""simops_helper.py — boresight mission helpers: probe + key-send + zombie-guard.

Derived from handoff/simops-mastery-2026-06-13/partA_cycle.py (proven chain).
NOT modifying src/racer, rl/, scripts/, or memory/. Disposable helper only.
"""
from __future__ import annotations
import argparse
import ctypes
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

_VK = {"enter": 0x0D, "esc": 0x1B, "down": 0x28, "up": 0x26}
_KEYUP = 0x0002
ENDPOINT = "udp:127.0.0.1:14550"


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
    hwnd = find_sim_window()
    if hwnd is None:
        print("  [keys] sim window NOT found", file=sys.stderr)
        return False
    if not force_foreground(hwnd):
        print("  [keys] could NOT verify sim foreground -> refusing", file=sys.stderr)
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
            print("  [keys] lost foreground mid-sequence", file=sys.stderr)
            return False
        user32.keybd_event(vk, 0, 0, 0)
        time.sleep(0.06)
        user32.keybd_event(vk, 0, _KEYUP, 0)
        time.sleep(float(settle) if settle else 0.4)
    return True


def n_sim_procs() -> int:
    try:
        out = subprocess.check_output(
            ["powershell.exe", "-NoProfile", "-Command",
             "@(Get-Process -Name 'DCGame-Win64-Shipping' -ErrorAction SilentlyContinue).Count"],
            stderr=subprocess.DEVNULL, timeout=15).decode().strip()
        return int(out or "0")
    except Exception:
        return -1


def probe(seconds: float = 2.0) -> dict:
    from racer.mavlink_client import MavlinkClient
    c = MavlinkClient(ENDPOINT)
    c.send_heartbeats = False
    c.send_timesync = False
    out = {"heartbeat": False, "started": None, "finished": None, "to_go_s": None,
           "active_gate": None, "pos_off_m": None, "sim_time_ns": None, "armed": None,
           "n_gates": 0}
    try:
        c.connect(wait_heartbeat=True, timeout_s=8.0)
    except Exception as e:
        print(f"  [probe] connect failed: {e}", file=sys.stderr)
        return out
    out["heartbeat"] = True
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        c.pump()
        time.sleep(0.005)
    s, rs = c.state, c.race_status
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
    if p["started"]:
        if p["to_go_s"] is not None and p["to_go_s"] < -120 and (p["pos_off_m"] or 0) < 2:
            return "STALE"
        return "ACTIVE_OR_COUNTDOWN"
    if p["pos_off_m"] is not None:
        return "WAITING"
    return "HOME"


def drive_to_waiting(max_attempts=4) -> dict:
    for attempt in range(1, max_attempts + 1):
        p = probe(2.0)
        st = classify(p)
        print(f"  [drive#{attempt}] state={st} pos_off={p.get('pos_off_m')} started={p.get('started')}")
        if st == "WAITING":
            return p
        if st in ("FINISHED", "ACTIVE_OR_COUNTDOWN", "STALE"):
            print("  -> sending esc,down:0.3,down:0.3,down:0.3,enter:1.5 to go HOME")
            send_keys("esc,down:0.3,down:0.3,down:0.3,enter:1.5")
            time.sleep(0.5)
            p2 = probe(2.0)
            st2 = classify(p2)
            print(f"    after reset: state={st2}")
            if st2 == "HOME":
                print("  -> sending enter:1.0 to go WAITING")
                send_keys("enter:1.0")
                time.sleep(0.5)
        elif st == "HOME":
            print("  -> sending enter:1.0 to go WAITING")
            send_keys("enter:1.0")
            time.sleep(0.5)
        elif st == "NO_HEARTBEAT":
            print("  -> sim not running or no heartbeat", file=sys.stderr)
            time.sleep(2.0)
    print("  [drive] FAILED to reach WAITING", file=sys.stderr)
    return probe(2.0)


def cmd_probe():
    n = n_sim_procs()
    print(f"DCGame instances: {n}")
    if n != 1:
        print(f"  WARNING: expected 1, got {n}", file=sys.stderr)
    p = probe(3.0)
    st = classify(p)
    print(json.dumps({"state": st, **p}, indent=2))


def cmd_drive():
    n = n_sim_procs()
    print(f"DCGame instances: {n}")
    p = drive_to_waiting()
    print(f"Final: {classify(p)} pos_off={p.get('pos_off_m')}")


def cmd_go():
    """Send Enter once to trigger GO from WAITING state."""
    print("Sending Enter (WAITING -> ACTIVE GO)...")
    ok = send_keys("enter:1.0")
    print(f"  key send: {'ok' if ok else 'FAILED'}")


def cmd_reset():
    """Full reset: any state -> HOME -> WAITING."""
    print("Full reset: esc + down×3 + enter -> WAITING...")
    send_keys("esc,down:0.3,down:0.3,down:0.3,enter:1.5")
    time.sleep(1.0)
    p = probe(2.0)
    st = classify(p)
    print(f"  after esc chain: state={st}")
    if st == "HOME":
        send_keys("enter:1.5")
        time.sleep(0.5)
        p = probe(2.0)
        st = classify(p)
        print(f"  after enter: state={st} pos_off={p.get('pos_off_m')}")
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["probe", "drive", "go", "reset"])
    args = ap.parse_args()
    {"probe": cmd_probe, "drive": cmd_drive, "go": cmd_go, "reset": cmd_reset}[args.cmd]()
