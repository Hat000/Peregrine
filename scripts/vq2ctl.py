"""vq2ctl.py - one-stop CLI to drive the AI-GP simulator (v1.0.3379) into/around VQ2 TRAINING.

Built for AI sessions: every command is idempotent-ish, verifies window foreground before
sending any key (the fullscreen sim auto-minimizes on focus loss), and can verify sim
state over passive MAVLink. Run `vq2ctl.py guide` for the full cheat sheet.

Quick start (from ANY sim state, ends in a running VQ2 TRAINING race):

    C:\\Users\\Shadow\\Peregrine\\.venv\\Scripts\\python.exe scripts\\vq2ctl.py fly

Menu map (screenshot-verified LIVE on build 3379, 2026-07-01):
  cold launch -> [Enter] login page -> [Enter, cached] MAIN MENU
  main menu ACTIVE EVENTS order: R1 (=VQ1, default) / R2-SUBMISSION (=VQ2 race) /
  R2-TRAINING (=VQ2 train, LAST item). Selection PINS at the bottom (no wrap), so extra
  Downs are safe and guarantee R2-TRAINING. NOTE the default highlight is NOT reliable:
  R1 on cold boot, but returning from a race can preselect another event (observed
  R2-SUBMISSION) - never trust position, always pin with Downs. [Enter] -> waiting room ("PLEASE ENSURE YOUR
  AUTOMATED PILOT IS READY") -> [Enter] -> 3 s countdown -> GO (race running).
  In-race pause menu is a GRID: action list (RESUME/RESTART/TOGGLE HUD/BACK TO MAIN
  MENU) + GRAPHICS/SOUND tiles to its right. LEFT x2 pins the action list (RIGHT moves
  ONTO the tiles, where Down is a NO-OP - verified by screenshot; the action list also
  pins at the bottom). So:
    Esc, Left x2, Down x4, Enter  = BACK TO MAIN MENU
    Esc, Left x2, Down x1, Enter  = RESTART RACE (immediate, no waiting room; the Enter
                                    confirm IS required - verified: highlight alone does
                                    nothing. --no-confirm omits it if you want to arm
                                    the selection without firing.)

Key-send mechanics proven in handoff/simops-mastery-2026-06-13/partA_cycle.py and
scripts/sim_focus.py (keybd_event + ALT-trick SetForegroundWindow + verify).
"""
from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SIM_EXE = r"C:\Users\Shadow\Downloads\AI-GP Simulator v1.0.3379\AIGP_3379\FlightSim.exe"
ENDPOINT = "udp:127.0.0.1:14550"

_VK = {"enter": 0x0D, "esc": 0x1B, "up": 0x26, "down": 0x28,
       "left": 0x25, "right": 0x27, "space": 0x20}
_KEYUP = 0x0002
SW_RESTORE = 9

# ------------------------- canned sequences (all live-verified 2026-07-01) -------------
SEQ_LOGIN = "enter:2.5,enter:2.5"      # app loaded -> login page -> (cached) main menu
SEQ_GO = "enter:1.0"                   # waiting room -> 3 s countdown -> GO
# pause menu: Left x2 pins the action list (NOT right - right lands on GRAPHICS/SOUND
# tiles where Down no-ops); both lists pin at the bottom, so the extra Down/Left are safe
SEQ_TO_MENU = "esc:1.0,left:0.4,left:0.4,down:0.4,down:0.4,down:0.4,down:0.4,enter:2.5"
SEQ_RESTART = "esc:1.0,left:0.4,left:0.4,down:0.4"     # + ",enter:1.5" unless --no-confirm
DEFAULT_DOWNS = 4                      # 2 reach R2-TRAINING; extras pin the bottom (safe)


def seq_waiting(downs: int) -> str:
    """Main menu -> select VQ2 TRAIN (Down xN, pins at bottom) -> Enter -> waiting room."""
    return ",".join(["down:0.4"] * downs) + ",enter:3.0"


# ------------------------- win32 window + keys (proven) -------------------------
def find_sim_windows():
    user32 = ctypes.windll.user32
    found = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def _cb(hwnd, _lp):
        n = user32.GetWindowTextLengthW(hwnd)
        if n:
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, buf, n + 1)
            t = buf.value.lower()
            if "ai-gp" in t or "flightsim" in t or "ai grand prix" in t:
                found.append((hwnd, buf.value, bool(user32.IsWindowVisible(hwnd)),
                              bool(user32.IsIconic(hwnd))))
        return True

    user32.EnumWindows(_cb, None)
    return found


def find_sim_window():
    wins = [w for w in find_sim_windows() if w[2]]  # visible only
    return wins[0][0] if wins else None


def force_foreground(hwnd, attempts=4) -> bool:
    user32 = ctypes.windll.user32
    for _ in range(attempts):
        user32.ShowWindow(hwnd, SW_RESTORE)
        time.sleep(0.3)
        # synthetic ALT press unlocks SetForegroundWindow from a background process
        user32.keybd_event(0x12, 0, 0, 0)
        user32.SetForegroundWindow(hwnd)
        user32.keybd_event(0x12, 0, _KEYUP, 0)
        time.sleep(0.4)
        if user32.GetForegroundWindow() == hwnd:
            return True
    return False


def send_tokens(tokens: str, dry: bool = False) -> bool:
    """Send a comma list of key[:settle_s] tokens with verified foreground between keys."""
    if dry:
        print(f"  [dry-run] would send: {tokens}")
        return True
    hwnd = find_sim_window()
    if hwnd is None:
        print("  [keys] sim window NOT found -> cannot send", file=sys.stderr)
        return False
    if not force_foreground(hwnd):
        print("  [keys] could NOT verify sim foreground -> refusing (keys would hit the "
              "wrong window)", file=sys.stderr)
        return False
    user32 = ctypes.windll.user32
    for tok in tokens.split(","):
        tok = tok.strip()
        if not tok:
            continue
        name, _, settle = tok.partition(":")
        vk = _VK.get(name.lower())
        if vk is None:
            print(f"  [keys] unknown key {name!r} (known: {', '.join(_VK)})", file=sys.stderr)
            return False
        if user32.GetForegroundWindow() != hwnd and not force_foreground(hwnd):
            print("  [keys] lost foreground mid-sequence -> abort", file=sys.stderr)
            return False
        user32.keybd_event(vk, 0, 0, 0)
        time.sleep(0.06)
        user32.keybd_event(vk, 0, _KEYUP, 0)
        time.sleep(float(settle) if settle else 0.4)
        print(f"  [keys] sent {name}")
    return True


# ------------------------- processes -------------------------
def n_procs(name: str) -> int:
    try:
        out = subprocess.check_output(
            ["powershell.exe", "-NoProfile", "-Command",
             f"@(Get-Process -Name '{name}' -ErrorAction SilentlyContinue).Count"],
            stderr=subprocess.DEVNULL, timeout=15).decode().strip()
        return int(out or "0")
    except Exception:
        return -1


def n_dcgame() -> int:
    # DCGame-Win64-Shipping is the actual game (binds 14550); FlightSim is the launcher.
    return n_procs("DCGame-Win64-Shipping")


# ------------------------- passive MAVLink probe -------------------------
def probe(seconds: float = 2.5) -> dict:
    """Passive read of udp:14550 (no heartbeats -> never flips the sim mode).
    WARNING: binds 14550 - do NOT call while fly_rl / any MAVLink client is running."""
    sys.path.insert(0, str(ROOT / "src"))
    out = {"heartbeat": False, "started": None, "finished": None, "to_go_s": None,
           "active_gate": None, "pos_off_m": None, "sim_time_ns": None, "armed": None,
           "n_gates": 0}
    try:
        from racer.mavlink_client import MavlinkClient
    except Exception as e:
        return {"error": f"probe unavailable ({e}) - run with the repo .venv python"}
    c = MavlinkClient(ENDPOINT)
    c.send_heartbeats = False
    c.send_timesync = False
    try:
        c.connect(wait_heartbeat=True, timeout_s=8.0)
    except Exception:
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
        try:
            out["pos_off_m"] = round(sum(float(v) ** 2 for v in s.position_ned) ** 0.5, 2)
        except Exception:
            pass
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
    """VQ2-aware (live-verified 2026-07-01 on 3379): position is wire-blocked on VQ2, so
    the discriminator is RACE_STATUS.started - the WAITING room RESETS it to False, while
    the post-race MAIN MENU keeps STREAMING the dead race's started=True (clock advancing)
    -> on the wire, main-menu-after-race is indistinguishable from a live race. fly()
    handles that ambiguity with a convergent recovery loop."""
    if "error" in p:
        return "UNPROBEABLE"
    if not p["heartbeat"]:
        return "NO_HEARTBEAT"      # cold login/title/never-raced menu, or sim down
    if p["started"] is None:
        return "HOME"              # heartbeat but no RACE_STATUS seen in the window
    if p["started"] is False:
        return "WAITING"           # waiting room (started reset; VQ2-verified)
    if p["finished"]:
        return "FINISHED"
    return "RACE_OR_MENU"          # live race, PAUSED race, or main-menu-after-race


# ------------------------- commands -------------------------
def cmd_status(args) -> int:
    out = {"flightsim_procs": n_procs("FlightSim"), "dcgame_procs": n_dcgame(),
           "windows": [{"hwnd": h, "title": t, "visible": v, "minimized": m,
                        "foreground": h == ctypes.windll.user32.GetForegroundWindow()}
                       for h, t, v, m in find_sim_windows()]}
    if args.probe:
        p = probe(2.5)
        out["probe"] = p
        out["state"] = classify(p)
    else:
        out["state"] = "UNPROBED (pass --probe; NOT while fly_rl runs - it binds 14550)"
    if out["dcgame_procs"] > 1:
        out["warning"] = "DUAL INSTANCE - armed-but-deaf drone; `vq2ctl.py kill` then relaunch"
    print(json.dumps(out, indent=2))
    return 0


def cmd_probe(args) -> int:
    p = probe(args.seconds)
    print(json.dumps({"state": classify(p), **p}, indent=2))
    return 0


def _launch(args) -> bool:
    exe = Path(args.exe)
    if not exe.exists():
        print(f"FAIL: sim exe not found: {exe}", file=sys.stderr)
        return False
    n = n_dcgame()
    if n > 0:
        print(f"FAIL: sim already running ({n} DCGame instance(s)). A second instance "
              f"splits the UDP stream (armed-but-deaf). Use `vq2ctl.py kill` first.",
              file=sys.stderr)
        return False
    print(f"launching {exe} ...")
    DETACHED = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    subprocess.Popen([str(exe)], cwd=str(exe.parent), creationflags=DETACHED,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     stdin=subprocess.DEVNULL, close_fds=True)
    t0 = time.monotonic()
    while time.monotonic() - t0 < args.window_wait:
        if find_sim_window() is not None:
            print(f"  window up after {time.monotonic() - t0:.0f}s")
            break
        time.sleep(2.0)
    else:
        print("FAIL: sim window never appeared", file=sys.stderr)
        return False
    if args.boot_wait > 0:
        print(f"  waiting {args.boot_wait:.0f}s for the app to finish loading ...")
        time.sleep(args.boot_wait)
    return True


def cmd_launch(args) -> int:
    return 0 if _launch(args) else 1


def cmd_kill(args) -> int:
    subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command",
         "Stop-Process -Name 'FlightSim','DCGame-Win64-Shipping' -Force "
         "-ErrorAction SilentlyContinue"], timeout=30)
    time.sleep(2.0)
    n = n_dcgame()
    print(f"DCGame instances after kill: {n}")
    return 0 if n == 0 else 1


def cmd_login(args) -> int:
    print("login: Enter (login page) + Enter (cached login) -> MAIN MENU")
    return 0 if send_tokens(SEQ_LOGIN, args.dry) else 1


def cmd_waiting(args) -> int:
    print(f"main menu -> VQ2 TRAIN (Down x{args.downs}, pins at bottom) -> waiting room")
    return 0 if send_tokens(seq_waiting(args.downs), args.dry) else 1


def cmd_go(args) -> int:
    print("waiting room -> GO (3 s countdown, then greenlight)")
    return 0 if send_tokens(SEQ_GO, args.dry) else 1


def cmd_to_menu(args) -> int:
    print("in-race -> pause -> BACK TO MAIN MENU (Esc, Left x2, Down x4, Enter)")
    return 0 if send_tokens(SEQ_TO_MENU, args.dry) else 1


def cmd_restart(args) -> int:
    seq = SEQ_RESTART if args.no_confirm else SEQ_RESTART + ",enter:1.5"
    extra = " (no confirm Enter - selection armed but NOT fired)" if args.no_confirm \
        else " + Enter confirm"
    print(f"in-race -> immediate RESTART (Esc, Left x2, Down x1{extra}); "
          f"no waiting room, 3 s countdown -> GO")
    print("  note: restart keeps the CURRENT track - only use when already in VQ2 TRAIN")
    return 0 if send_tokens(seq, args.dry) else 1


def cmd_keys(args) -> int:
    tokens = ",".join(args.tokens)
    return 0 if send_tokens(tokens, args.dry) else 1


def cmd_focus(args) -> int:
    hwnd = find_sim_window()
    if hwnd is None:
        print("no sim window found", file=sys.stderr)
        return 1
    ok = force_foreground(hwnd)
    print(f"foreground: {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


def cmd_shot(args) -> int:
    hwnd = find_sim_window()
    if hwnd is None:
        print("no sim window found", file=sys.stderr)
        return 1
    force_foreground(hwnd)  # restore first; a minimized fullscreen window grabs garbage
    import ctypes.wintypes as wt
    r = wt.RECT()
    ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(r))
    try:
        import PIL.ImageGrab as ig
    except Exception:
        print("PIL not available in this python - use the repo .venv", file=sys.stderr)
        return 1
    img = ig.grab(bbox=(r.left, r.top, r.right, r.bottom))
    img.save(args.out)
    print(f"saved {args.out}  rect=({r.left},{r.top},{r.right},{r.bottom})")
    return 0


def _verify(expect: str, seconds: float = 2.5) -> tuple[bool, dict]:
    p = probe(seconds)
    st = classify(p)
    if "error" in p:
        print(f"  [verify] {p['error']} - continuing UNVERIFIED")
        return True, p
    if expect == "RACING":
        # fresh race: started=True and race_start within the last ~30 s. (A stale menu
        # stream also has started=True but with an old race_start -> very negative to_go.)
        ok = st == "RACE_OR_MENU" and (p.get("to_go_s") is None or p["to_go_s"] > -30.0)
    else:
        ok = st == expect
    print(f"  [verify] state={st} (expected {expect}) "
          f"started={p.get('started')} to_go={p.get('to_go_s')}")
    return ok, p


def cmd_fly(args) -> int:
    """From ANY state to a freshly-GOne VQ2 TRAINING race."""
    steps: list[str] = []
    if args.dry:
        print("  [dry-run] fly is state-dependent; no keys/probe under --dry. Plans:\n"
              "    sim down        -> launch, login, waiting, go\n"
              "    WAITING         -> go\n"
              "    HOME            -> waiting, go\n"
              "    race OR menu    -> converge: to-menu, [waiting] (probe-checked), go\n"
              "    NO_HEARTBEAT    -> needs --assume login|menu")
        print(json.dumps({"result": "DRY_RUN", "steps": ["<state-dependent>"]}))
        return 0
    n = n_dcgame()
    if n > 1:
        print("FAIL: dual sim instance (armed-but-deaf). `vq2ctl.py kill`, then re-run fly.",
              file=sys.stderr)
        return 1

    waiting_confirmed = False
    if n <= 0:
        # cold path is blind (menus emit no MAVLink); verified at the WAITING checkpoint
        if not _launch(args):
            return 1
        steps.append("launch")
        for step, seq in (("login", SEQ_LOGIN), ("waiting", seq_waiting(args.downs))):
            steps.append(step)
            if not send_tokens(seq):
                return 1
    else:
        st = None
        if not args.no_verify:
            p = probe(2.5)
            st = classify(p)
            print(f"current state: {st}")
        if st == "WAITING":
            waiting_confirmed = True
        elif st == "HOME":
            steps.append("waiting")
            if not send_tokens(seq_waiting(args.downs)):
                return 1
        elif st in ("RACE_OR_MENU", "FINISHED"):
            if args.fast:
                # immediate in-race restart; ONLY correct if the loaded track is VQ2 TRAIN
                steps.append("restart")
                if not send_tokens(SEQ_RESTART + ",enter:1.5"):
                    return 1
            else:
                # RACE_OR_MENU is genuinely ambiguous on the wire (the post-race main menu
                # keeps streaming the dead race's started=True). Convergent loop, all
                # branches live-verified 2026-07-01:
                #   live race  --to-menu-->  main menu --waiting-chain--> WAITING
                #   main menu  --to-menu-->  WAITING   (esc/left no-op, downs pin, enter)
                #   paused race--to-menu-->  resumes (esc toggles) + no-ops -> next lap exits
                for attempt in range(1, 4):
                    steps.append("to_menu")
                    if not send_tokens(SEQ_TO_MENU):
                        return 1
                    ok, _ = _verify("WAITING")
                    if ok:
                        waiting_confirmed = True
                        break
                    steps.append("waiting")
                    if not send_tokens(seq_waiting(args.downs)):
                        return 1
                    ok, _ = _verify("WAITING")
                    if ok:
                        waiting_confirmed = True
                        break
                    print(f"  [fly] not in waiting room yet (attempt {attempt}/3)")
                if not waiting_confirmed:
                    print("FAIL: could not converge to the waiting room - `shot` and "
                          "recover manually (see `guide`).", file=sys.stderr)
                    return 1
        elif args.assume == "login":
            for step, seq in (("login", SEQ_LOGIN), ("waiting", seq_waiting(args.downs))):
                steps.append(step)
                if not send_tokens(seq):
                    return 1
        elif args.assume == "menu":
            steps.append("waiting")
            if not send_tokens(seq_waiting(args.downs)):
                return 1
        else:
            print(
                "FAIL: sim is running but the screen state is unknown (cold menus emit no "
                "MAVLink,\nor --no-verify was set). Disambiguate: `vq2ctl.py shot check.png`"
                " and look at it, then\n"
                "  at login/title screen -> re-run:  fly --assume login\n"
                "  at main menu          -> re-run:  fly --assume menu\n"
                "  in waiting room       -> just run: go\n"
                "  in a race             -> run: restart   (or: to-menu, then fly)",
                file=sys.stderr)
            return 1

    if "restart" not in steps:
        if not waiting_confirmed and not args.no_verify:
            ok, _p = _verify("WAITING")
            if not ok:
                print("FAIL: not in the waiting room after navigation - take `shot` and "
                      "recover manually (see `guide`).", file=sys.stderr)
                return 1
        steps.append("go")
        if not send_tokens(SEQ_GO):
            return 1

    verified = False
    if not args.no_verify:
        time.sleep(2.0)
        ok, p = _verify("RACING")
        verified = ok and "error" not in p
        if not ok:
            print("WARNING: race start not confirmed by probe - check with `status --probe` "
                  "or `shot`.", file=sys.stderr)
    print(json.dumps({"result": "RACING", "verified": verified, "steps": steps}))
    return 0


GUIDE = """\
=============================================================================
vq2ctl.py - AI-session cheat sheet for the AI-GP sim (v1.0.3379, VQ2 TRAINING)
=============================================================================
Python:   C:\\Users\\Shadow\\Peregrine\\.venv\\Scripts\\python.exe   (repo .venv; system
          python works for everything except probe/verify + shot)
Sim exe:  C:\\Users\\Shadow\\Downloads\\AI-GP Simulator v1.0.3379\\AIGP_3379\\FlightSim.exe

THE ONE COMMAND you usually want (any state -> running VQ2 TRAINING race):
    python scripts\\vq2ctl.py fly

COMMANDS
  status [--probe]   processes + window + (opt) MAVLink state. Safe, read-only w/o --probe.
  launch             start FlightSim.exe (refuses if already running), wait for window+load
  kill               kill FlightSim + DCGame (killing either kills both)
  login              Enter, Enter          (app loaded -> login -> cached -> MAIN MENU)
  waiting [--downs N] Down xN + Enter      (main menu -> R2-TRAINING -> waiting room)
  go                 Enter                 (waiting room -> 3 s countdown -> GO)
  restart [--no-confirm]  Esc,Left x2,Down x1,Enter  (in-race -> IMMEDIATE fresh race,
                     no waiting room. Keeps the CURRENT track!)
  to-menu            Esc,Left x2,Down x4,Enter          (in-race -> MAIN MENU)
  fly [--fast] [--assume login|menu] [--no-verify]      full orchestrator (see below)
  probe [--seconds S] passive MAVLink state (binds 14550 - NOT while fly_rl runs)
  keys <tok> [...]   raw keys, e.g.: keys esc down:0.5 enter   (enter/esc/up/down/left/right)
  focus              force the sim window to verified foreground
  shot [out.png]     screenshot the sim window (read the PNG to disambiguate menus)
  Every key command accepts --dry to print the sequence without sending.

SIM STATE MACHINE (probe classification; VQ2 blocks position, so started is the key)
  NO_HEARTBEAT   sim down, or COLD title/login/menu (no MAVLink before the first race)
  HOME           heartbeat but no RACE_STATUS in the window (rare)
  WAITING        waiting room - RACE_STATUS streams with started=False (it RESETS here)
  RACE_OR_MENU   started=True: live race, PAUSED race, OR main-menu-after-race - the menu
                 KEEPS STREAMING the dead race (clock advancing) => wire can't tell them
                 apart. fly's convergence loop handles it; `shot` if you need certainty.
  FINISHED       started=True + finished=True

`fly` LOGIC: not running -> launch + login + select R2-TRAINING + waiting + GO (verified
  at the WAITING checkpoint + fresh-GO check). WAITING -> GO. RACE_OR_MENU -> convergent
  loop: to-menu chain, probe; if not WAITING yet, waiting chain, probe; repeat (<=3).
  Works from live race, paused race, or main menu because every stray key in the chains
  is a verified no-op in the other screens. --fast swaps that for an in-race restart
  (quicker, but keeps whatever track is loaded). Cold-but-running (NO_HEARTBEAT) is
  unprobeable -> `shot`, then re-run with --assume login|menu.

FOOTGUNS (each cost a past session)
  1. ONE sim instance ever. Two DCGame procs = UDP split = armed-but-deaf. status warns;
     kill + relaunch (killing one kills both).
  2. Never send keys without verified foreground - the fullscreen sim AUTO-MINIMIZES on
     focus loss and blind keys land in another window. This tool always verifies; if you
     bypass it, don't.
  3. probe/verify binds udp:14550. NEVER probe while rl/fly_rl.py (or any MAVLink client)
     is attached - you'd steal/split its stream. Use `fly --no-verify`, `go`, `restart`
     (pure key-sends) around a live fly_rl.
  4. The post-race MAIN MENU keeps streaming the dead race's RACE_STATUS (started=True,
     clock advancing) - "telemetry alive" does NOT mean "race running". Only
     started=False (waiting room) and a fresh to_go are trustworthy.
  5. COLD menus emit no MAVLink - "no telemetry" right after launch is normal. The FPV
     video port shows the drone camera only; menus never render there. Disambiguate
     screens with `shot`.
  6. In menus, arrows PIN at list ends (no wrap) - extra Downs/Lefts are SAFE and are how
     the chains guarantee their landing spot. The main-menu default highlight is NOT
     reliable (R1 on cold boot, but returning from a race can preselect another event) -
     always pin, never count on position. Do not improvise extra ENTERS in menus - one
     Enter too many starts the wrong thing. (In-race and waiting-room stray Enter/Down
     were verified harmless.)
  7. In the PAUSE menu, Right moves onto the GRAPHICS/SOUND tiles where Down no-ops and
     Enter opens settings - always pin with LEFT.
  8. `restart` restarts the CURRENT track. If you're not sure it's VQ2 TRAIN, use
     to-menu + fly.
  9. VQ2 visual check: lit warehouse + glowing RED gates (VQ1 = wireframe).

TYPICAL RECIPES
  fresh race, cold machine:        fly
  fly_rl is waiting for GO:        go                (pure key-send, no probe)
  crashed, want another attempt:   restart           (immediate, same track)
  done for the session:            to-menu           (leave exactly one sim instance)
  wedged / unsure:                 status --probe    then shot check.png and look

Full doc: docs/vq2ctl.md.  Wire/telemetry details: docs/sim_ops.md.
"""


def cmd_guide(_args) -> int:
    print(GUIDE)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, aliases=(), **kw):
        p = sub.add_parser(name, aliases=list(aliases), help=kw.pop("help", None))
        p.set_defaults(fn=fn)
        return p

    p = add("status", cmd_status, help="processes + window + optional --probe state")
    p.add_argument("--probe", action="store_true",
                   help="also probe MAVLink (binds 14550 - not while fly_rl runs)")

    p = add("probe", cmd_probe, help="passive MAVLink state probe (binds 14550)")
    p.add_argument("--seconds", type=float, default=2.5)

    p = add("launch", cmd_launch, help="start FlightSim.exe and wait for load")
    p.add_argument("--exe", default=SIM_EXE)
    p.add_argument("--window-wait", type=float, default=120.0)
    p.add_argument("--boot-wait", type=float, default=30.0,
                   help="seconds to wait after the window appears (app load)")

    add("kill", cmd_kill, help="kill FlightSim + DCGame")

    for name, fn, help_ in [
            ("login", cmd_login, "Enter,Enter: loaded app -> login -> main menu"),
            ("go", cmd_go, "Enter: waiting room -> countdown -> race"),
            ("to-menu", cmd_to_menu, "in-race -> main menu (Esc,Right x2,Down x4,Enter)"),
            ("focus", cmd_focus, "force sim window to verified foreground")]:
        p = add(name, fn, aliases=(["menu"] if name == "to-menu" else []), help=help_)
        if fn is not cmd_focus:
            p.add_argument("--dry", action="store_true")

    p = add("waiting", cmd_waiting, aliases=["select-vq2"],
            help="main menu -> VQ2 TRAIN -> waiting room")
    p.add_argument("--downs", type=int, default=DEFAULT_DOWNS,
                   help=f"Down presses (2 reach VQ2 TRAIN, extras pin; default {DEFAULT_DOWNS})")
    p.add_argument("--dry", action="store_true")

    p = add("restart", cmd_restart, help="in-race -> immediate restart (same track)")
    p.add_argument("--no-confirm", action="store_true",
                   help="send the literal Esc,Right x2,Down x1 without the trailing Enter")
    p.add_argument("--dry", action="store_true")

    p = add("keys", cmd_keys, help="raw key tokens, e.g.: keys esc down:0.5 enter")
    p.add_argument("tokens", nargs="+")
    p.add_argument("--dry", action="store_true")

    p = add("shot", cmd_shot, help="screenshot the sim window")
    p.add_argument("out", nargs="?", default="sim_window.png")

    p = add("fly", cmd_fly, aliases=["race"],
            help="ANY state -> running VQ2 TRAINING race (the one-stop command)")
    p.add_argument("--exe", default=SIM_EXE)
    p.add_argument("--window-wait", type=float, default=120.0)
    p.add_argument("--boot-wait", type=float, default=30.0)
    p.add_argument("--downs", type=int, default=DEFAULT_DOWNS)
    p.add_argument("--fast", action="store_true",
                   help="if already in a race, use in-race restart (keeps current track)")
    p.add_argument("--assume", choices=["login", "menu"],
                   help="screen to assume when running-but-unprobeable")
    p.add_argument("--no-verify", action="store_true",
                   help="never touch MAVLink (REQUIRED if fly_rl or any client is attached)")
    p.add_argument("--dry", action="store_true")

    add("guide", cmd_guide, help="print the AI-session cheat sheet")

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
