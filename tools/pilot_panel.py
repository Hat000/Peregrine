#!/usr/bin/env python3
"""
VQ2 Pilot Control Panel  --  a localhost web app to launch fly_rl pilots in the
background with EVERY setting exposed, record logs locally, and pick which logs
to commit to git.

Run it:
    python tools/pilot_panel.py            # serves http://127.0.0.1:8787
    python tools/pilot_panel.py --port 9000

Dependency-free (Python 3.8+ stdlib only). It launches flights with the tensorrt
venv (VENV_PY below) from the ego-flight repo root, exactly like the CLI:

    PYTHONPATH=src <VENV_PY> rl/fly_rl.py <flags...>   >  tools/pilot_panel_logs/<id>.log

Nothing is sent anywhere until you click "Commit to git" (and push is an explicit,
off-by-default checkbox). Flights are launched detached, so they keep running even
if you stop this server.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# --------------------------------------------------------------------------- #
# Paths / environment  (edit here if the machine layout changes)
# --------------------------------------------------------------------------- #
REPO      = Path(r"C:/Users/Shadow/Peregrine-ego-flight")
VENV_PY   = Path(r"C:/Users/Shadow/Peregrine/.venv/Scripts/python.exe")   # tensorrt venv
MODELS    = Path(r"C:/Users/Shadow/Peregrine/models")                     # detector engines/.pt
SCRIPT    = "rl/fly_rl.py"
RUNS_DIR  = REPO / "data" / "runs"
LOG_DIR   = REPO / "tools" / "pilot_panel_logs"
STATE_F   = LOG_DIR / "pilots.json"
DEFAULT_PORT = 8787

IS_WIN = os.name == "nt"
# detached so a server restart does not kill in-flight pilots; no console popup
_CREATE = 0
if IS_WIN:
    _CREATE = 0x00000008 | 0x00000200 | 0x08000000  # DETACHED|NEW_GROUP|NO_WINDOW

# --------------------------------------------------------------------------- #
# Asset discovery
# --------------------------------------------------------------------------- #
def _ls(globs, roots, rel_to=None):
    out = []
    for root in roots:
        root = Path(root)
        for g in globs:
            for p in sorted(root.glob(g)):
                if rel_to is not None:
                    try:
                        out.append(os.path.relpath(p, rel_to).replace("\\", "/")); continue
                    except Exception:
                        pass
                out.append(str(p).replace("\\", "/"))
    # de-dup, keep order
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p); uniq.append(p)
    return uniq

def discover():
    # Repo-internal assets are returned REPO-RELATIVE so they match the relative
    # defaults in SCHEMA (ckpts/..., configs/...); detectors stay absolute (other repo).
    return {
        "detectors": _ls(["*.engine", "*.pt"], [MODELS]),
        "ego_ckpts": _ls(["*.pth"], [REPO / "ckpts"], rel_to=REPO),
        "base_ckpts": _ls(["*.pth"], [REPO / "rl" / "checkpoints"], rel_to=REPO),
        "coarse_maps": _ls(["*.json"], [REPO / "configs"], rel_to=REPO),
    }

# --------------------------------------------------------------------------- #
# Setting schema  --  the single source of truth for BOTH the form and the CLI.
#   action: "value"    -> emit [flag, value]  when value != ""
#           "flag"     -> emit [flag]         when truthy   (store_true)
#           "boolopt"  -> emit [flag] or [--no-<rest>]      (BooleanOptionalAction)
#   ui:     select | number | text | bool
#   opts:   discovery key for a select (detectors / ego_ckpts / base_ckpts / coarse_maps)
# Defaults reproduce the known-good 4-gate "record" config.
# --------------------------------------------------------------------------- #
NEG_RE = re.compile(r"^--")
def _neg(flag):  # --ego-slot1 -> --no-ego-slot1 ; --virtual-flip -> --no-virtual-flip
    return NEG_RE.sub("--no-", flag, count=1)

SCHEMA = [
    # ---- Flight stack -----------------------------------------------------
    dict(key="ego_ckpt", flag="--ego-ckpt", action="value", ui="select", opts="ego_ckpts",
         group="Flight stack", default="ckpts/vpeffs0_actor.pth", allow_blank=True,
         help="RL policy actor (.pth). This IS the flight stack. Blank = classical CTBR (non-ego)."),
    dict(key="bridge", flag="--bridge", action="boolopt", ui="bool",
         group="Flight stack", default=False,
         help="Fly proven CTBR to ~3 m before gate 0 then hand off to the policy. Off = standing start."),
    dict(key="base_ckpt", flag="--checkpoint", action="value", ui="select", opts="base_ckpts",
         group="Flight stack", default="", allow_blank=True,
         help="Legacy/base inc actor. Only used on the NON-ego path; ignored when ego-ckpt is set."),

    # ---- Vision / detector ------------------------------------------------
    dict(key="seeker_detector", flag="--seeker-detector", action="value", ui="select",
         group="Vision", default="yolo", choices=["yolo", "red_glow"],
         help="Detector backend. 'yolo' = neural (needs seeker-weights). 'red_glow' = classical, no-GPU."),
    dict(key="seeker_weights", flag="--seeker-weights", action="value", ui="select", opts="detectors",
         group="Vision", default="C:/Users/Shadow/Peregrine/models/vq2_darkred_negreal42_2026-07-05_fp16_384x640.engine",
         allow_blank=True,
         help="Vision model: a TensorRT .engine (fast) or ultralytics .pt. Required for yolo."),
    dict(key="video_dedup_fastpath", flag="--video-dedup-fastpath", action="boolopt", ui="bool",
         group="Vision", default=True,
         help="Fast-drop the sim's ~33x UDP video re-send flood in the receiver (4-byte frame_id, no "
              "unpack) so it doesn't GIL-starve nav_update. ON = smooth (loop 17->30 Hz); OFF = pre-fix (A/B)."),
    # (--video-async-detect toggle REMOVED 2026-07-14: fly-tested net-negative -- no loop-tail fix
    #  and LOWER gates on every model; the p99 tail is a ~56ms non-inference nav GPU-sync, not the
    #  detect() call, and the tail doesn't correlate with gates anyway. The flag still exists in
    #  fly_rl.py, defaults OFF; the infer_ms instrument stays.)

    # ---- Ego perception ---------------------------------------------------
    dict(key="ego_sector_mode", flag="--ego-sector-mode", action="value", ui="select",
         group="Ego perception", default="map", choices=["auto", "zero", "map"],
         help="Coarse next-gate turn prior source. 'map' = the hand-authored coarse map below."),
    dict(key="ego_coarse_map", flag="--ego-coarse-map", action="value", ui="select", opts="coarse_maps",
         group="Ego perception", default="configs/vq2_coarse_map.json", allow_blank=True,
         help="Per-gate [horiz,vert] turn buckets, used only when sector-mode = map."),
    dict(key="ego_slot1", flag="--ego-slot1", action="flag", ui="bool",
         group="Ego perception", default=True,
         help="Activate the WINDOW=2 next-gate obs slot (multi-gate _pef champions)."),
    dict(key="ego_max_valid_range", flag="--ego-max-valid-range", action="value", ui="number",
         group="Ego perception", default=30.0, step=1,
         help="GATE DISTANCE CAP (m): drop detections beyond this (billboard-FP guard). 30 = record."),
    dict(key="ego_gate_z_bias", flag="--ego-gate-z-bias", action="value", ui="number",
         group="Ego perception", default=0.0, step=0.1,
         help="Perceived-gate VERTICAL bias (m) added to EVERY vision emission (both slots). "
              "+ lowers the gate so the drone aims/passes LOWER through it. 0 = off. Try 0.3."),
    dict(key="ego_det_hold", flag="--ego-det-hold", action="value", ui="number",
         group="Ego perception", default=0.2, step=0.05,
         help="Seconds a lost gate is still treated as 'detected' before masking to zero."),
    dict(key="ego_stale_horizon", flag="--ego-stale-horizon", action="value", ui="number",
         group="Ego perception", default=0.5, step=0.05,
         help="Confidence staleness horizon (s); conf = clamp(1-age/horizon,0,1)."),
    dict(key="ego_obs_coast", flag="--ego-obs-coast", action="boolopt", ui="bool",
         group="Ego perception", default=False,
         help="Coast the rel_pos + decaying confidence through blackout (only for coast-trained ckpts)."),
    dict(key="ego_kp_persist", flag="--ego-kp-persist", action="value", ui="number",
         group="Ego perception", default=0, step=1,
         help="Keypoint-persistence debounce: N consecutive fresh frames before a fix transmits. 0/1 = off."),

    # ---- Ego control ------------------------------------------------------
    dict(key="ego_pitch_clamp", flag="--ego-pitch-clamp", action="value", ui="number",
         group="Ego control", default=10.0, step=1,
         help="Nose-down PITCH fence (deg, 0=off). Must exceed 17.8 resting tilt to never fence hover."),
    dict(key="ego_roll_clamp", flag="--ego-roll-clamp", action="value", ui="number",
         group="Ego control", default=0.0, step=1,
         help="SYMMETRIC roll fence (deg, 0=off). Blocks banking past +/-cap, allows return to level. "
              "Set ABOVE the ~18.5 turn bank (try 25); too tight starves banked turns."),
    dict(key="ego_floor_clamp", flag="--ego-floor-clamp", action="value", ui="number",
         group="Ego control", default="", step=0.05,
         help="Descent-arrest floor (m above pad, blank=off): below this height, if SINKING, force "
              "thrust over-hover to decelerate the dive. One-sided -- never touches attitude, never "
              "pins altitude (can't block a gate above the line). Fires off the vision altitude est "
              "(vertical bias B -> true trigger ~clamp+B), so sweep against what you SEE: set between "
              "the lowest gate and the ground, start 0.4, drop if it bounces you at a low gate; if it "
              "misses a fast dive raise ego_assist_thrust (shared w/ takeoff floor). NEEDS commit c851802."),
    dict(key="ego_speed_gov", flag="--ego-speed-gov", action="value", ui="text",
         group="Ego control", default="",
         help="Speed governor \"SOFT,HARD\" m/s -- or a single number = hard cap at that speed (blank or 0 = off). Caps the OVER-hover "
              "thrust as speed runs SOFT->HARD (hard-clamps to hover at/above HARD); altitude-neutral -- "
              "only ever caps TOWARD hover, never forces a sink. Universal ~5 m/s cap on ANY model, no "
              "retrain: try 5,6.5 (smooth ramp) or 5 (abrupt hard cap). Watch gov / gov_engaged in ego_obs.jsonl (is it braking? altitude "
              "hold through the band?). NEEDS commit 293ffee."),
    dict(key="ego_yaw_clamp", flag="--ego-yaw-clamp", action="value", ui="number",
         group="Ego control", default=0.7, step=0.05,
         help="Hard yaw-rate command clip (rad/s, 0=off). MANDATORY 0.7 for despin ckpts."),
    dict(key="ego_rate_scale", flag="--ego-rate-scale", action="value", ui="number",
         group="Ego control", default=1.0, step=0.1,
         help="Uplink cmd_rate_scale the ego path forces (plant sysid'd at 1.0)."),
    dict(key="ego_takeoff_assist", flag="--ego-takeoff-assist", action="boolopt", ui="bool",
         group="Ego control", default=True,
         help="Autonomous ground-unstick thrust floor until airborne, then disarms permanently."),
    dict(key="ego_assist_thrust", flag="--ego-assist-thrust", action="value", ui="number",
         group="Ego control", default=1.10, step=0.05,
         help="Takeoff-assist thrust floor in g (hover=1.0)."),
    dict(key="ego_assist_max_s", flag="--ego-assist-max-s", action="value", ui="number",
         group="Ego control", default=1.5, step=0.1,
         help="Hard time cap (s) after which takeoff-assist disarms unconditionally."),
    dict(key="virtual_flip", flag="--virtual-flip", action="boolopt", ui="bool",
         group="Ego control", default=True,
         help="Run the policy in the tail-first body frame (required for gate passes)."),
    dict(key="yaw_scale", flag="--yaw-scale", action="value", ui="number",
         group="Ego control", default=1.0, step=0.1,
         help="Scale the policy yaw-rate command (0 = drop yaw)."),
    dict(key="max_rate", flag="--max-rate", action="value", ui="number",
         group="Ego control", default=0.0, step=0.5,
         help="Clip ALL 3 body-rate axes (rad/s, 0=off). NOT for despin ckpts -- use yaw-clamp."),

    # ---- Run / timing -----------------------------------------------------
    dict(key="label", flag="--label", action="value", ui="text",
         group="Run", default="panel_run",
         help="Session label -> data/runs/<timestamp>_<label>_f1. Spaces become underscores."),
    dict(key="rate", flag="--rate", action="value", ui="number",
         group="Run", default=30.0, step=1,
         help="Control loop Hz (training dt = 30)."),
    dict(key="max_seconds", flag="--max-seconds", action="value", ui="number",
         group="Run", default=120.0, step=5,
         help="Hard flight time cap (s)."),
    dict(key="wait_seconds", flag="--wait-seconds", action="value", ui="number",
         group="Run", default=180.0, step=10,
         help="How long the pilot waits passively for the race GO before giving up."),
    dict(key="flights", flag="--flights", action="value", ui="number",
         group="Run", default=1, step=1, danger_over=1,
         help="Back-to-back attempts. KEEP AT 1 -- >1 turns on --full-reset and auto-resets the sim."),
    dict(key="endpoint", flag="--endpoint", action="value", ui="text",
         group="Run", default="udp:127.0.0.1:14550",
         help="MAVLink endpoint."),

    # ---- Advanced / safety ------------------------------------------------
    dict(key="dev_auto_reset", flag="--dev-auto-reset", action="flag", ui="bool",
         group="Advanced", default=False, danger=True,
         help="DEV ONLY: permit the pilot to emit sim-control (SIM_RESET). SUBMISSION-UNSAFE -- leave OFF."),
    dict(key="full_reset", flag="--full-reset", action="boolopt", ui="bool",
         group="Advanced", default=True,
         help="Reset sim state between flights (only matters when flights>1)."),
    dict(key="deploy_profile", flag="--deploy-profile", action="value", ui="text",
         group="Advanced", default="vq2_case_c",
         help="Seeker deploy profile."),
    dict(key="spin_rate_abort", flag="--spin-rate-abort", action="value", ui="number",
         group="Advanced", default=6.0, step=0.5,
         help="Abort if body-rate magnitude exceeds this (rad/s)."),
    dict(key="spin_time_abort", flag="--spin-time-abort", action="value", ui="number",
         group="Advanced", default=2.0, step=0.5,
         help="Sustained-spin abort window (s)."),
    dict(key="debug_obs", flag="--debug-obs", action="boolopt", ui="bool",
         group="Advanced", default=False,
         help="Verbose per-tick obs debug printing."),
]

GROUP_ORDER = ["Flight stack", "Vision", "Ego perception", "Ego control", "Run", "Advanced"]
BY_KEY = {s["key"]: s for s in SCHEMA}

# Per-model deploy DEFAULTS -- auto-applied in the panel when a checkpoint is picked (keyed by
# .pth basename). The yaw clamp is TRAINING-matched and per-RELEASE: 0.7 for everything EXCEPT
# the ego-ckpts-tracka-2026-07-13 arm (w0 + m8) at 0.35. MIXING THE TWO = OOD (Fengyou 2026-07-13).
# NOTE: only the yaw clamp is encoded so far -- whether the vtrackA lineage also needs the vpef
# recipe (slot1 / sector=map / tight pitch) is UNCONFIRMED; extend per model once known.
MODEL_DEFAULTS = {
    # ego-ckpts-vpef-2026-07-12  -- yaw 0.7
    "vpefwh2_actor.pth":    {"ego_yaw_clamp": 0.7},
    "vpeffs0_actor.pth":    {"ego_yaw_clamp": 0.7},
    # ego-ckpts-tracka-yaw07-2026-07-13  -- yaw 0.7
    "vtrackAm8b_actor.pth": {"ego_yaw_clamp": 0.7},
    "vtrackArs0_actor.pth": {"ego_yaw_clamp": 0.7},
    "vtrackAw1_actor.pth":  {"ego_yaw_clamp": 0.7},
    # ego-ckpts-tracka-2026-07-13  -- yaw 0.35 (DO NOT mix with the 0.7 lineage)
    "vtrackAw0_actor.pth":  {"ego_yaw_clamp": 0.35},
    "vtrackAm8_actor.pth":  {"ego_yaw_clamp": 0.35},
}

# --------------------------------------------------------------------------- #
# Command building
# --------------------------------------------------------------------------- #
def _truthy(v):
    return v is True or str(v).lower() in ("1", "true", "on", "yes")

def build_cmd(values: dict):
    """Return (argv_list, warnings) for a launch given posted form values."""
    warn = []
    argv = [str(VENV_PY), SCRIPT]
    for s in SCHEMA:
        v = values.get(s["key"], s["default"])
        act = s["action"]
        if act == "value":
            if v is None or str(v).strip() == "":
                continue
            if s["key"] == "label":
                v = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(v).strip()) or "panel_run"
            argv += [s["flag"], str(v)]
        elif act == "flag":
            if _truthy(v):
                argv += [s["flag"]]
        elif act == "boolopt":
            argv += [s["flag"] if _truthy(v) else _neg(s["flag"])]
    # free-form extra flags
    extra = (values.get("extra_flags") or "").strip()
    if extra:
        try:
            argv += shlex.split(extra, posix=False)
        except ValueError as e:
            warn.append(f"could not parse extra flags: {e}")
    # sanity warnings (do not block -- the user asked for every setting)
    try:
        if float(values.get("flights", 1)) > 1:
            warn.append("flights > 1 enables --full-reset auto-resets between flights (SINGLE-FLIGHT rule).")
    except Exception:
        pass
    if _truthy(values.get("dev_auto_reset")):
        warn.append("dev-auto-reset ON: the pilot may emit sim-control commands (submission-unsafe).")
    if values.get("seeker_detector") == "yolo" and not (values.get("seeker_weights") or "").strip():
        warn.append("yolo detector selected but no seeker-weights -- fly_rl will refuse.")
    return argv, warn

# --------------------------------------------------------------------------- #
# Pilot registry
# --------------------------------------------------------------------------- #
_LOCK = threading.Lock()
PILOTS: dict = {}          # id -> {id,label,cmd,log,pid,started,session,...}
_procs: dict = {}          # id -> Popen (only for pilots we launched this run)

def _load_state():
    if STATE_F.exists():
        try:
            for p in json.loads(STATE_F.read_text()):
                PILOTS[p["id"]] = p
        except Exception:
            pass

def _save_state():
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        STATE_F.write_text(json.dumps(list(PILOTS.values()), indent=1))
    except Exception:
        pass

_counter = [0]
def _new_id():
    _counter[0] += 1
    return f"p{int(time.time())}_{_counter[0]}"

def launch(values: dict):
    argv, warn = build_cmd(values)
    pid_key = _new_id()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logpath = LOG_DIR / f"{pid_key}.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    env["PYTHONUNBUFFERED"] = "1"
    logf = open(logpath, "wb")
    try:
        p = subprocess.Popen(argv, cwd=str(REPO), env=env,
                             stdout=logf, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, creationflags=_CREATE)
    except Exception as e:
        logf.close()
        return {"ok": False, "error": str(e), "cmd": argv}
    rec = dict(id=pid_key, label=values.get("label", "panel_run"),
               cmd=argv, cmd_str=_pretty(argv), log=str(logpath), pid=p.pid,
               started=time.strftime("%H:%M:%S"), started_ts=time.time(),
               session=None, state="LAUNCHING", gates=None, warnings=warn,
               config=dict(values))   # exact form values -> one-click reload per flight
    with _LOCK:
        PILOTS[pid_key] = rec
        _procs[pid_key] = p
        _save_state()
    return {"ok": True, "id": pid_key, "cmd": argv, "warnings": warn}

def _pretty(argv):
    # PowerShell-compatible one-liner (this box runs PowerShell). NOTE: the Launch
    # button sets PYTHONPATH via the subprocess env itself -- you do NOT need to paste
    # this; it is shown only so you can see/repro the exact command.
    py, rest = argv[0], argv[1:]
    out = [f'"{a}"' if (" " in a or not a) else a for a in rest]
    return f'$env:PYTHONPATH="src"; & "{py}" ' + " ".join(out)

_REC_RE   = re.compile(r"recording ->\s*(\S+)")
_RES_RE   = re.compile(r"flight\s+\d+:\s+([A-Z_]+)\s+gates=(\d+)")

def _parse_log(logpath):
    session, state, gates = None, None, None
    try:
        txt = Path(logpath).read_text(errors="ignore")
    except Exception:
        return session, state, gates
    m = _REC_RE.search(txt)
    if m:
        session = m.group(1).replace("\\", "/").split("data/runs/")[-1]
    for m in _RES_RE.finditer(txt):
        state, gates = m.group(1), int(m.group(2))
    if state is None:
        if "Waiting PASSIVELY" in txt:
            state = "WAITING"
        elif "Traceback" in txt or "Error" in txt:
            state = "ERROR"
    return session, state, gates

def _loop_hz(session):
    if not session:
        return None
    tf = RUNS_DIR / session / "ego_timing.jsonl"
    if not tf.exists():
        return None
    try:
        w = []
        for line in tf.read_text().splitlines()[1:]:
            if line.strip():
                w.append(json.loads(line).get("work_ms", 0))
        if not w:
            return None
        w = sorted(1000.0 / x for x in w if x > 0)
        return round(w[len(w) // 2], 1)   # median hz
    except Exception:
        return None

def refresh_pilots():
    with _LOCK:
        for pid_key, rec in PILOTS.items():
            proc = _procs.get(pid_key)
            running = proc is not None and proc.poll() is None
            sess, state, gates = _parse_log(rec["log"])
            if sess:
                rec["session"] = sess
            if gates is not None:
                rec["gates"] = gates
            if state:
                rec["state"] = state
            if not running and proc is not None and rec["state"] in ("LAUNCHING", "WAITING", "FLYING"):
                # process exited but log had no result line
                rec["state"] = rec["state"] if rec["state"] not in ("LAUNCHING",) else "EXITED"
            rec["running"] = running
            rec["hz"] = _loop_hz(rec.get("session"))
        _save_state()
        return list(PILOTS.values())

def stop_pilot(pid_key):
    proc = _procs.get(pid_key)
    if proc and proc.poll() is None:
        try:
            if IS_WIN:
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            proc.terminate()
        except Exception:
            try: proc.kill()
            except Exception: pass
        return True
    return False

def _sys_flyrl_pids():
    """Every python process on the machine running fly_rl.py -- catches orphans the
    panel did NOT launch (stale CLI runs / prior sessions) that choke the loop rate."""
    pids = []
    try:
        if IS_WIN:
            ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe' OR Name='pythonw.exe'\" | "
                  "Where-Object { $_.CommandLine -like '*fly_rl.py*' } | ForEach-Object { $_.ProcessId }")
            r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                               capture_output=True, text=True, timeout=10)
            pids = [int(x) for x in r.stdout.split() if x.strip().isdigit()]
        else:
            r = subprocess.run(["pgrep", "-f", "fly_rl.py"], capture_output=True, text=True, timeout=10)
            pids = [int(x) for x in r.stdout.split() if x.strip().isdigit()]
    except Exception:
        pass
    return pids

_proc_cache = {"t": 0.0, "n": 0}
def sys_flyrl_count():
    # While a tracked pilot is LIVE, do NOT spawn the PowerShell system scan: a periodic subprocess
    # spawn mid-flight steals a core/GIL for ~200 ms and adds a control-loop hitch (it can seed a
    # roll divergence). Report the tracked-running count instead; the full external-orphan scan
    # resumes only when idle (and kill_all always does a fresh scan on demand).
    with _LOCK:
        live = sum(1 for k in PILOTS if _procs.get(k) is not None and _procs[k].poll() is None)
    if live:
        return live
    now = time.time()
    if now - _proc_cache["t"] > 10.0:           # idle only: throttle the scan (was 3 s)
        _proc_cache["n"] = len(_sys_flyrl_pids())
        _proc_cache["t"] = now
    return _proc_cache["n"]

def kill_all():
    """Terminate every fly_rl pilot -- panel-tracked AND system orphans."""
    targets = _sys_flyrl_pids()                  # snapshot BEFORE killing (all fly_rl python)
    for _k, proc in list(_procs.items()):        # 1) tracked Popen handles
        try:
            if proc and proc.poll() is None:
                proc.kill()
        except Exception:
            pass
    for pid in targets:                          # 2) hard tree-kill each (children too)
        try:
            if IS_WIN:
                subprocess.run(["taskkill", "/PID", str(pid), "/F", "/T"],
                               capture_output=True, timeout=10)
            else:
                os.kill(pid, signal.SIGKILL)
        except Exception:
            pass
    with _LOCK:
        for rec in PILOTS.values():
            if rec.get("running"):
                rec["state"] = "KILLED"; rec["running"] = False
        _save_state()
    time.sleep(0.5)
    _proc_cache["t"] = 0.0                       # force a fresh count next poll
    remaining = _sys_flyrl_pids()
    return {"killed": targets, "remaining": remaining}

def clear_finished():
    """Drop non-running pilots from the list (does not touch live flights)."""
    with _LOCK:
        keep = {k: rec for k, rec in PILOTS.items()
                if _procs.get(k) is not None and _procs[k].poll() is None}
        PILOTS.clear(); PILOTS.update(keep)
        _save_state()
    return {"remaining": len(PILOTS)}

# --------------------------------------------------------------------------- #
# Sessions + git
# --------------------------------------------------------------------------- #
SMALL_FILES = ["meta.json", "ego_obs.jsonl", "ego_timing.jsonl", "video_index.jsonl"]
HEAVY_FILES = ["video.bin", "mavlink.tlog"]

def list_sessions(limit=60):
    out = []
    if not RUNS_DIR.exists():
        return out
    dirs = sorted([d for d in RUNS_DIR.iterdir() if d.is_dir()], reverse=True)[:limit]
    for d in dirs:
        meta = d / "meta.json"
        info = dict(name=d.name, state=None, gates=None, dur=None,
                    has_video=(d / "video.bin").exists())
        if meta.exists():
            try:
                m = json.loads(meta.read_text())
                info.update(state=m.get("final_state"), gates=m.get("gate_index"),
                            dur=round(m.get("duration_s", 0), 1), label=m.get("label"))
            except Exception:
                pass
        out.append(info)
    return out

def git_commit(sessions, include_heavy, do_push):
    if not sessions:
        return {"ok": False, "error": "no sessions selected"}
    files = []
    for s in sessions:
        d = RUNS_DIR / s
        if not d.is_dir():
            continue
        for f in SMALL_FILES + (HEAVY_FILES if include_heavy else []):
            fp = d / f
            if fp.exists():
                files.append(str(fp))
    # also grab the panel console logs whose session matches
    for rec in PILOTS.values():
        if rec.get("session") in sessions and Path(rec["log"]).exists():
            files.append(rec["log"])
    if not files:
        return {"ok": False, "error": "no matching files found for the selected sessions"}
    try:
        # the selected sessions live under the gitignored data/runs (+ the panel .log under the
        # gitignored pilot_panel_logs); the user is EXPLICITLY choosing them, so force-add the
        # specific files. -f + an explicit file list never pulls in pilots.json / other ignored cruft.
        subprocess.run(["git", "-C", str(REPO), "add", "-f", "--", *files],
                       check=True, capture_output=True, text=True)
        msg = f"logs(panel): {len(sessions)} session(s) -- " + ", ".join(sessions[:4]) + \
              (" ..." if len(sessions) > 4 else "")
        r = subprocess.run(["git", "-C", str(REPO), "commit", "-m", msg],
                           capture_output=True, text=True)
        if r.returncode != 0 and "nothing to commit" not in (r.stdout + r.stderr):
            return {"ok": False, "error": (r.stdout + r.stderr)[-800:]}
        commit_out = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "committed"
        pushed = None
        if do_push:
            rp = subprocess.run(["git", "-C", str(REPO), "push"], capture_output=True, text=True)
            pushed = "ok" if rp.returncode == 0 else (rp.stdout + rp.stderr)[-800:]
        return {"ok": True, "committed": commit_out, "files": len(files),
                "include_heavy": include_heavy, "pushed": pushed}
    except subprocess.CalledProcessError as e:
        return {"ok": False, "error": (e.stdout or "") + (e.stderr or "")}

# --------------------------------------------------------------------------- #
# Render (onboard video + real detector overlay) via tools/render_yolo.py
# --------------------------------------------------------------------------- #
_renders: dict = {}   # session -> Popen

def _render_out(session):
    return RUNS_DIR / session / "render.mp4"

def start_render(session, last_seconds=9.0, slowmo=3.0):
    d = RUNS_DIR / session
    if not d.is_dir() or not (d / "video.bin").exists():
        return {"ok": False, "error": "no video.bin for this session"}
    proc = _renders.get(session)
    if proc is not None and proc.poll() is None:
        return {"ok": True, "state": "rendering"}
    if _render_out(session).exists():
        return {"ok": True, "state": "ready"}
    env = os.environ.copy(); env["PYTHONPATH"] = "src"
    try:
        logf = open(LOG_DIR / f"render_{session}.log", "wb")
        proc = subprocess.Popen(
            [str(VENV_PY), "tools/render_yolo.py", str(d), str(_render_out(session)),
             "--last-seconds", str(last_seconds), "--slowmo", str(slowmo)],
            cwd=str(REPO), env=env, stdout=logf, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, creationflags=_CREATE)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    _renders[session] = proc
    return {"ok": True, "state": "rendering"}

def render_status(session):
    proc = _renders.get(session)
    if proc is not None and proc.poll() is None:
        return {"state": "rendering"}
    out = _render_out(session)
    if out.exists():
        return {"state": "ready", "size": out.stat().st_size}
    return {"state": "none"}


# --------------------------------------------------------------------------- #
# Coarse-map editor  --  read/write a configs/*.json sector map from the panel.
# One [horiz, vert] per gate; row g answers "arriving at gate g, where is gate
# g+1?" in the drone's gravity-leveled approach frame. SPATIAL notation (the
# arrow points AT the next gate):  UP-LEFT UP UP-RIGHT / LEFT . RIGHT / DOWN...
#   horiz: RIGHT = -1, LEFT = +1, straight = 0   (fly-check: banks wrong -> flip)
#   vert : UP = +1, DOWN = -1, level = 0
# --------------------------------------------------------------------------- #
def _map_path(rel):
    """Resolve a map path SAFELY -- repo-relative, under configs/, .json only."""
    rel = (rel or "configs/vq2_coarse_map.json").replace("\\", "/")
    p = (REPO / rel).resolve()
    if p.suffix != ".json" or (REPO / "configs").resolve() not in p.parents:
        raise ValueError("map must be a .json under configs/")
    return p

def read_coarse_map(rel):
    p = _map_path(rel)
    m = json.loads(p.read_text())
    return {"path": os.path.relpath(p, REPO).replace("\\", "/"),
            "sector": m.get("sector", []), "doc": m.get("_doc", "")}

def write_coarse_map(rel, sector):
    p = _map_path(rel)
    sec = []
    for row in sector:
        if not (isinstance(row, (list, tuple)) and len(row) == 2):
            raise ValueError("each gate must be [horiz, vert]")
        h, v = int(row[0]), int(row[1])
        if h not in (-1, 0, 1) or v not in (-1, 0, 1):
            raise ValueError("horiz/vert must each be -1, 0, or 1")
        sec.append([h, v])
    m = json.loads(p.read_text()) if p.exists() else {}
    m["sector"] = sec                                    # preserves _doc + key order
    m["_provenance"] = f"Edited via panel map editor ({len(sec)} gates)."
    txt = re.sub(r"\[\s+(-?\d),\s+(-?\d)\s+\]", r"[\1, \2]", json.dumps(m, indent=2))
    p.write_text(txt)
    return {"ok": True, "path": os.path.relpath(p, REPO).replace("\\", "/"), "rows": len(sec)}

# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # quiet
        pass

    def _serve_video(self, session):
        out = _render_out(session)
        if not session or not out.exists():
            return self._send(404, {"error": "no render for this session"})
        size = out.stat().st_size
        rng = self.headers.get("Range")
        start, end, partial = 0, size - 1, False
        if rng:
            mm = re.match(r"bytes=(\d+)-(\d*)", rng)
            if mm:
                start = int(mm.group(1))
                end = int(mm.group(2)) if mm.group(2) else size - 1
                end = min(end, size - 1); partial = True
        with open(out, "rb") as f:
            f.seek(start); data = f.read(end - start + 1)
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/":
            return self._send(200, HTML, "text/html; charset=utf-8")
        if u.path == "/api/schema":
            return self._send(200, {"schema": SCHEMA, "groups": GROUP_ORDER,
                                    "assets": discover(), "model_defaults": MODEL_DEFAULTS})
        if u.path == "/api/pilots":
            return self._send(200, {"pilots": refresh_pilots(), "sys_flyrl": sys_flyrl_count()})
        if u.path == "/api/sessions":
            return self._send(200, {"sessions": list_sessions()})
        if u.path == "/api/log":
            pid_key = q.get("id", [""])[0]
            rec = PILOTS.get(pid_key)
            if not rec:
                return self._send(404, {"error": "no such pilot"})
            try:
                txt = Path(rec["log"]).read_text(errors="ignore")
            except Exception:
                txt = "(log not readable yet)"
            return self._send(200, {"log": txt[-8000:]})
        if u.path == "/api/render_status":
            return self._send(200, render_status(q.get("session", [""])[0]))
        if u.path == "/api/video":
            return self._serve_video(q.get("session", [""])[0])
        if u.path == "/api/coarse_map":
            try:
                return self._send(200, read_coarse_map(q.get("path", [""])[0]))
            except Exception as e:
                return self._send(400, {"error": str(e)})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        ln = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(ln) if ln else b"{}"
        try:
            data = json.loads(body or b"{}")
        except Exception:
            data = {}
        if u.path == "/api/preview":
            argv, warn = build_cmd(data)
            return self._send(200, {"cmd": _pretty(argv), "warnings": warn})
        if u.path == "/api/launch":
            return self._send(200, launch(data))
        if u.path == "/api/stop":
            return self._send(200, {"stopped": stop_pilot(data.get("id"))})
        if u.path == "/api/git":
            return self._send(200, git_commit(data.get("sessions", []),
                                              bool(data.get("include_heavy")),
                                              bool(data.get("push"))))
        if u.path == "/api/kill_all":
            return self._send(200, kill_all())
        if u.path == "/api/clear":
            return self._send(200, clear_finished())
        if u.path == "/api/render":
            return self._send(200, start_render(data.get("session", "")))
        if u.path == "/api/coarse_map":
            try:
                return self._send(200, write_coarse_map(data.get("path", ""), data.get("sector", [])))
            except Exception as e:
                return self._send(400, {"error": str(e)})
        return self._send(404, {"error": "not found"})

# --------------------------------------------------------------------------- #
# Front-end (single page; fetches /api/schema and renders)
# --------------------------------------------------------------------------- #
HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<title>VQ2 Pilot Control Panel</title>
<style>
:root{--bg:#0e1116;--panel:#171b22;--line:#2a313c;--fg:#d7dde5;--mut:#8b95a3;--acc:#4a9eff;--ok:#3fb950;--warn:#d29922;--bad:#f85149}
*{box-sizing:border-box}body{margin:0;font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:var(--bg);color:var(--fg)}
h1{font-size:15px;margin:0;padding:10px 14px;border-bottom:1px solid var(--line);background:var(--panel);display:flex;gap:12px;align-items:center}
h1 .sub{color:var(--mut);font-weight:400;font-size:12px}
.wrap{display:grid;grid-template-columns:1fr 1fr;gap:0;height:calc(100vh - 42px)}
.col{overflow:auto;padding:12px 14px}
.col.left{border-right:1px solid var(--line)}
.grp{border:1px solid var(--line);border-radius:6px;margin-bottom:10px;background:var(--panel)}
.grp>summary{cursor:pointer;padding:7px 10px;font-weight:600;color:var(--acc);user-select:none}
.grp[data-danger] > summary{color:var(--warn)}
.row{display:grid;grid-template-columns:180px 1fr;gap:8px;align-items:center;padding:4px 10px}
.row label{color:var(--fg)}
.row .hint{grid-column:2;color:var(--mut);font-size:11px;margin:-2px 0 4px}
input,select{width:100%;background:#0b0e13;border:1px solid var(--line);color:var(--fg);border-radius:4px;padding:5px 7px;font:inherit}
input[type=checkbox]{width:auto}
.danger input,.danger select{border-color:var(--warn)}
button{background:var(--acc);color:#04121f;border:0;border-radius:5px;padding:8px 14px;font:inherit;font-weight:700;cursor:pointer}
button.sec{background:#232a34;color:var(--fg)}
button.stop{background:var(--bad);color:#fff;padding:3px 8px;font-weight:600}
button:disabled{opacity:.5;cursor:not-allowed}
.bar{position:sticky;top:0;background:var(--bg);padding:6px 0 10px;z-index:5;display:flex;gap:8px;flex-wrap:wrap;align-items:center}
pre.cmd{white-space:pre-wrap;word-break:break-all;background:#0b0e13;border:1px solid var(--line);border-radius:5px;padding:8px;color:#9fb3c8;font-size:11px;margin:6px 0}
.warn{color:var(--warn)}.ok{color:var(--ok)}.bad{color:var(--bad)}.mut{color:var(--mut)}
table{width:100%;border-collapse:collapse;font-size:12px}
th,td{text-align:left;padding:4px 6px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--mut);font-weight:600}
.pill{padding:1px 7px;border-radius:9px;font-size:11px;font-weight:700}
.s-WAITING{background:#20303f;color:#9cd}.s-FLYING,.s-LAUNCHING{background:#20303f;color:#7cf}
.s-CRASH,.s-ERROR,.s-EXITED,.s-STALLED{background:#3a1d1d;color:#f99}
.s-FINISH,.s-PASSED{background:#16351f;color:#8f8}
.tabs{display:flex;gap:6px;margin-bottom:8px}
.tabs button{background:#232a34;color:var(--fg);font-weight:600;padding:5px 12px}
.tabs button.on{background:var(--acc);color:#04121f}
.logbox{white-space:pre-wrap;background:#080a0e;border:1px solid var(--line);border-radius:5px;padding:8px;height:300px;overflow:auto;font-size:11px;color:#b9c4d0}
.sess{display:flex;gap:8px;align-items:center;padding:3px 4px;border-bottom:1px solid var(--line)}
.sess input{width:auto}
small.k{color:var(--mut)}
.maplegend{background:#0b0e13;border:1px solid var(--line);border-radius:5px;padding:8px 10px;color:var(--mut);font-size:11px;margin-bottom:10px;line-height:1.7}
.maplegend b{color:var(--fg)}
.mapgate{display:flex;gap:12px;align-items:center;padding:5px 2px;border-bottom:1px solid var(--line)}
.mapgate .lbl{width:92px;color:var(--fg);font-size:12px}
.grid3{display:grid;grid-template-columns:repeat(3,30px);grid-template-rows:repeat(3,26px);gap:2px;flex:none}
.gc{background:#0b0e13;border:1px solid var(--line);color:var(--mut);border-radius:3px;cursor:pointer;font-size:14px;padding:0;line-height:1}
.gc:hover{border-color:var(--acc);color:var(--fg)}
.gc.on{background:var(--acc);color:#04121f;border-color:var(--acc);font-weight:700}
</style></head><body>
<h1>🚁 VQ2 Pilot Control Panel <span class="sub" id="sub">launch pilots in the background · logs local · commit to git on demand</span></h1>
<div class="wrap">
  <div class="col left">
    <div class="bar">
      <button onclick="doLaunch()">▶ Launch pilot</button>
      <button class="sec" onclick="loadRecord()">Load record config</button>
      <button class="sec" onclick="preview()">Refresh command</button>
      <span id="lwarn" class="warn"></span>
    </div>
    <pre class="cmd" id="cmd">building…</pre>
    <div class="row"><label>Extra raw flags</label><input id="extra_flags" placeholder="--any --flag value" oninput="preview()"></div>
    <div id="form"></div>
  </div>
  <div class="col right">
    <div class="tabs">
      <button id="tab-pilots" class="on" onclick="tab('pilots')">Pilots</button>
      <button id="tab-git" onclick="tab('git')">Logs → git</button>
      <button id="tab-log" onclick="tab('log')">Log viewer</button>
      <button id="tab-video" onclick="tab('video')">🎬 Video</button>
      <button id="tab-map" onclick="tab('map')">🗺 Map</button>
    </div>
    <div id="view-pilots">
      <div class="bar">
        <button class="stop" style="padding:8px 14px" onclick="killAll()">&#9940; Kill ALL pilots</button>
        <button class="sec" onclick="clearFinished()">Clear finished</button>
        <span id="procbadge" class="pill" style="background:#232a34">fly_rl procs: ?</span>
        <span class="mut">stale pilots choke the loop rate</span>
      </div>
      <table><thead><tr><th>label</th><th>state</th><th>gates</th><th>Hz</th><th>session</th><th></th></tr></thead>
      <tbody id="ptbody"></tbody></table>
    </div>
    <div id="view-git" style="display:none">
      <div class="bar">
        <button onclick="doGit(false)">Commit selected</button>
        <label class="mut"><input type="checkbox" id="incl_heavy"> include video.bin/tlog (heavy)</label>
        <label class="mut"><input type="checkbox" id="do_push"> push to origin</label>
        <button class="sec" onclick="loadSessions()">↻</button>
      </div>
      <div id="gitmsg" class="mut"></div>
      <div id="sesslist"></div>
    </div>
    <div id="view-log" style="display:none">
      <div class="bar"><select id="logsel" onchange="showLog()"></select><button class="sec" onclick="showLog()">↻</button></div>
      <div class="logbox" id="logbox">select a pilot…</div>
    </div>
    <div id="view-video" style="display:none">
      <div class="bar"><span id="vidstatus" class="mut">pick a flight's 🎬 (Pilots or Logs→git) to render its onboard video + detector overlay</span></div>
      <video id="player" controls playsinline style="width:100%;max-height:72vh;background:#000;border:1px solid var(--line);border-radius:6px"></video>
    </div>
    <div id="view-map" style="display:none">
      <div class="bar">
        <button onclick="saveMap()">💾 Save map</button>
        <button class="sec" onclick="loadMap()">↻ Reload</button>
        <button class="sec" onclick="addGate()">+ gate</button>
        <button class="sec" onclick="delGate()">− gate</button>
        <span class="mut">editing <b id="mappath">—</b></span>
        <span id="mapmsg"></span>
      </div>
      <div class="maplegend" id="maplegend"></div>
      <div id="mapbody"></div>
    </div>
  </div>
</div>
<script>
let SCHEMA=[], GROUPS=[], ASSETS={}, curLog=null, PBID={}, PBSESS={}, MODEL_DEFAULTS={}, MAP=null;
const $=id=>document.getElementById(id);
async function boot(){
  const r=await (await fetch('/api/schema')).json();
  SCHEMA=r.schema; GROUPS=r.groups; ASSETS=r.assets; MODEL_DEFAULTS=r.model_defaults||{};
  renderForm(); preview(); loadPilots(); loadSessions();
  setInterval(loadPilots,2000);
}
function optList(key){return (ASSETS[key]||[])}
function renderForm(){
  const byG={}; GROUPS.forEach(g=>byG[g]=[]);
  SCHEMA.forEach(s=>{(byG[s.group]=byG[s.group]||[]).push(s)});
  let html='';
  GROUPS.forEach(g=>{
    const open=(g!=='Advanced')?'open':'';
    const danger=(g==='Advanced')?'data-danger':'';
    html+=`<details class="grp" ${open} ${danger}><summary>${g}</summary>`;
    (byG[g]||[]).forEach(s=>{
      const dcls=s.danger?'danger':'';
      html+=`<div class="row ${dcls}"><label title="${s.key}">${s.key.replace(/_/g,' ')}</label>`;
      if(s.ui==='bool'){
        html+=`<input type="checkbox" id="f_${s.key}" ${s.default?'checked':''} onchange="preview()">`;
      }else if(s.ui==='select'){
        let opts=(s.choices?s.choices:optList(s.opts)).slice();
        // ensure the default is present in THIS select's own option list (no global replace)
        if(s.default && !s.choices && !opts.includes(s.default)) opts.unshift(s.default);
        html+=`<select id="f_${s.key}" onchange="${s.key==='ego_ckpt'?'onCkptChange()':'preview()'}">`;
        if(s.allow_blank) html+=`<option value="">(none)</option>`;
        opts.forEach(o=>{const label=o.split('/').pop();html+=`<option value="${o}" ${o==s.default?'selected':''}>${label}</option>`});
        html+=`</select>`;
      }else{
        const t=s.ui==='number'?'number':'text';
        const st=s.step?`step="${s.step}"`:'';
        html+=`<input type="${t}" ${st} id="f_${s.key}" value="${s.default}" oninput="preview()">`;
      }
      html+=`<div class="hint">${s.help||''}</div></div>`;
    });
    html+=`</details>`;
  });
  $('form').innerHTML=html;
}
function collect(){
  const v={};
  SCHEMA.forEach(s=>{
    const el=$('f_'+s.key); if(!el)return;
    v[s.key]= s.ui==='bool'? el.checked : el.value;
  });
  v.extra_flags=$('extra_flags').value;
  return v;
}
async function preview(){
  const r=await (await fetch('/api/preview',{method:'POST',body:JSON.stringify(collect())})).json();
  $('cmd').textContent=r.cmd;
  $('lwarn').textContent=(r.warnings||[]).join('  •  ');
}
async function onCkptChange(){
  // apply the picked model's per-release defaults (yaw clamp is training-matched -- mixing = OOD)
  const el=$('f_ego_ckpt');
  const base=el?(el.value||'').split('/').pop():'';
  const d=MODEL_DEFAULTS[base]; const applied=[];
  if(d){for(const k in d){const f=$('f_'+k); if(!f)continue;
    if(f.type==='checkbox')f.checked=!!d[k]; else f.value=d[k];
    applied.push(k.replace(/^ego_/,'').replace(/_/g,' ')+'='+d[k]);}}
  await preview();
  if(applied.length){const w=$('lwarn').textContent;
    $('lwarn').textContent='✓ '+base+' defaults ('+applied.join(', ')+')'+(w?'  •  '+w:'');}
}
function loadRecord(){
  // the known-good 4-gate config
  const rec={ego_ckpt:'ckpts/vpeffs0_actor.pth',seeker_detector:'yolo',
    seeker_weights:'C:/Users/Shadow/Peregrine/models/vq2_darkred_negreal42_2026-07-05_fp16_384x640.engine',
    ego_sector_mode:'map',ego_coarse_map:'configs/vq2_coarse_map.json',ego_slot1:true,
    ego_max_valid_range:30,ego_pitch_clamp:10,ego_yaw_clamp:0.7,flights:1,label:'record_cfg'};
  for(const k in rec){const el=$('f_'+k);if(!el)continue; if(el.type==='checkbox')el.checked=rec[k]; else el.value=rec[k];}
  preview();
}
function negFlag(f){return f.replace(/^--/,'--no-');}   // mirror server _neg
function valuesFromCmd(cmd){
  // reconstruct form values from a stored launch argv (inverse of build_cmd), so EVERY past
  // flight can reload -- not just ones launched after the config field was added.
  if(!cmd||!cmd.length)return null;
  const v={};
  SCHEMA.forEach(s=>{
    const i=cmd.indexOf(s.flag);
    if(s.action==='value'){ if(i>=0 && i+1<cmd.length) v[s.key]=cmd[i+1]; }
    else if(s.action==='flag'){ v[s.key]= i>=0; }
    else if(s.action==='boolopt'){ v[s.key]= i>=0 ? true : (cmd.indexOf(negFlag(s.flag))>=0 ? false : s.default); }
  });
  return v;
}
function cfgOf(p){ return p ? (p.config || valuesFromCmd(p.cmd)) : null; }
function applyConfig(cfg){
  // one-click reload of a past flight's settings into the launch form
  if(!cfg){alert('no config recoverable for this flight');return;}
  SCHEMA.forEach(s=>{const el=$('f_'+s.key); if(!el||!(s.key in cfg))return;
    if(el.type==='checkbox')el.checked=!!cfg[s.key]; else el.value=cfg[s.key];});
  if(('extra_flags' in cfg) && $('extra_flags')) $('extra_flags').value=cfg.extra_flags||'';
  preview();
  const c=$('cmd'); c.style.transition='box-shadow .05s'; c.style.boxShadow='0 0 0 2px var(--ok)';
  setTimeout(()=>{c.style.transition='box-shadow .6s'; c.style.boxShadow='';},500);
}
function loadConfig(id){applyConfig(cfgOf(PBID[id]));}
function loadConfigSess(name){applyConfig(cfgOf(PBSESS[name]));}
async function doLaunch(){
  const r=await (await fetch('/api/launch',{method:'POST',body:JSON.stringify(collect())})).json();
  if(!r.ok){alert('launch failed: '+(r.error||'?'));return;}
  if(r.warnings&&r.warnings.length) $('lwarn').textContent='launched • '+r.warnings.join(' • ');
  loadPilots();
}
async function loadPilots(){
  const r=await (await fetch('/api/pilots')).json();
  const n=r.sys_flyrl||0, bdg=$('procbadge');
  if(bdg){bdg.textContent='fly_rl procs: '+n;
    bdg.style.background=n>0?'#3a2d16':'#16351f'; bdg.style.color=n>0?'#f0c674':'#8f8';}
  PBID={}; PBSESS={};
  const rows=(r.pilots||[]).slice().reverse().map(p=>{
    const st=p.state||'—';
    PBID[p.id]=p; if(p.session)PBSESS[p.session]=p;
    const stop=p.running?`<button class="stop" onclick="stopP('${p.id}')">stop</button>`:'';
    const cfgb=(p.config||(p.cmd&&p.cmd.length))?` <button class="sec" style="padding:2px 6px" title="load this flight's config into the launch form" onclick="loadConfig('${p.id}')">⚙</button>`:'';
    return `<tr><td>${p.label||''}<br><small class="k">${p.started||''}</small></td>
      <td><span class="pill s-${st}">${st}</span>${p.running?' <small class="k">live</small>':''}</td>
      <td>${p.gates==null?'—':p.gates}</td><td>${p.hz==null?'—':p.hz}</td>
      <td><small class="k">${p.session||'—'}</small></td>
      <td>${stop} <button class="sec" style="padding:2px 7px" onclick="viewLog('${p.id}')">log</button>${cfgb} <button class="sec" style="padding:2px 6px" title="render onboard video" onclick="renderVid('${p.session||''}')">🎬</button></td></tr>`;
  }).join('');
  $('ptbody').innerHTML=rows||'<tr><td colspan=6 class="mut">no pilots launched yet</td></tr>';
  // refresh log selector
  const sel=$('logsel'); const cur=sel.value;
  sel.innerHTML=(r.pilots||[]).map(p=>`<option value="${p.id}">${p.label} · ${p.state||''}</option>`).join('');
  if(cur) sel.value=cur;
}
async function stopP(id){await fetch('/api/stop',{method:'POST',body:JSON.stringify({id})});loadPilots();}
async function killAll(){
  if(!confirm('Kill ALL fly_rl pilots on this machine (including orphans NOT launched here)?'))return;
  const b=$('procbadge'); if(b)b.textContent='killing…';
  const r=await (await fetch('/api/kill_all',{method:'POST',body:'{}'})).json();
  alert('killed '+(r.killed||[]).length+' process(es); remaining fly_rl: '+(r.remaining||[]).length);
  loadPilots();
}
async function clearFinished(){await fetch('/api/clear',{method:'POST',body:'{}'});loadPilots();}
function tab(t){['pilots','git','log','video','map'].forEach(x=>{$('view-'+x).style.display=x==t?'':'none';$('tab-'+x).className=x==t?'on':'';});if(t=='git')loadSessions();if(t=='map')loadMap();}
async function renderVid(session){
  if(!session||session=='—'){alert('no recorded session for this flight yet');return;}
  tab('video'); const st=$('vidstatus'); $('player').removeAttribute('src'); $('player').load();
  st.innerHTML='rendering <b>'+session+'</b> … (detector runs on every frame, ~20-40s)';
  await fetch('/api/render',{method:'POST',body:JSON.stringify({session})});
  const poll=async()=>{
    const s=await (await fetch('/api/render_status?session='+encodeURIComponent(session))).json();
    if(s.state=='ready'){
      st.innerHTML='<span class="ok">▶ '+session+'</span> &nbsp;<small class="k">'+Math.round((s.size||0)/1e6)+' MB · 3× slow-mo</small>';
      const p=$('player'); p.src='/api/video?session='+encodeURIComponent(session)+'&t='+Date.now(); p.load(); p.play().catch(()=>{});
    } else if(s.state=='rendering'){ setTimeout(poll,1500); }
    else { st.innerHTML='<span class="bad">render produced no file — check the session has video.bin</span>'; }
  };
  setTimeout(poll,1200);
}
function viewLog(id){tab('log');$('logsel').value=id;showLog();}
async function showLog(){
  const id=$('logsel').value; if(!id)return;
  const r=await (await fetch('/api/log?id='+id)).json();
  const b=$('logbox'); b.textContent=r.log||'(empty)'; b.scrollTop=b.scrollHeight;
}
async function loadSessions(){
  const r=await (await fetch('/api/sessions')).json();
  $('sesslist').innerHTML=(r.sessions||[]).map(s=>{
    const st=s.state||'—';
    const scfg=PBSESS[s.name]?`<button class="sec" style="padding:1px 7px;margin-left:auto" title="load this flight's config into the launch form" onclick="loadConfigSess('${s.name}')">⚙ cfg</button>`:'';
    const svid=s.has_video?`<button class="sec" style="padding:1px 7px;${scfg?'':'margin-left:auto'}" onclick="renderVid('${s.name}')">🎬 render</button>`:'';
    return `<div class="sess"><input type="checkbox" class="sc" value="${s.name}">
      <span class="pill s-${st}">${st}</span>
      <span>${s.name}</span>
      <small class="k">gates=${s.gates==null?'—':s.gates} · ${s.dur||'?'}s</small>
      ${scfg}${svid}</div>`;
  }).join('')||'<div class="mut">no sessions in data/runs</div>';
}
async function doGit(){
  const sel=[...document.querySelectorAll('.sc:checked')].map(c=>c.value);
  if(!sel.length){alert('select at least one session');return;}
  $('gitmsg').textContent='committing…';
  const r=await (await fetch('/api/git',{method:'POST',body:JSON.stringify({
    sessions:sel,include_heavy:$('incl_heavy').checked,push:$('do_push').checked})})).json();
  if(r.ok){$('gitmsg').innerHTML=`<span class="ok">✔ ${r.committed} (${r.files} files${r.include_heavy?' +heavy':''})`+
      (r.pushed?(r.pushed=='ok'?' · pushed':' · <span class=bad>push: '+r.pushed+'</span>'):'')+`</span>`;}
  else{$('gitmsg').innerHTML='<span class="bad">&#10007;  '+(r.error||'failed')+'</span>';}
}
// ---- coarse-map editor: a 3x3 spatial grid per gate (the arrow points AT the next gate) ----
const MAP_NAME={'0,1':'up','-1,1':'upper-right','1,1':'upper-left','1,0':'left','0,0':'straight / level','-1,0':'right','1,-1':'lower-left','0,-1':'down','-1,-1':'lower-right'};
const MAP_GLYPH={'0,1':'↑','-1,1':'↗','1,1':'↖','1,0':'←','0,0':'•','-1,0':'→','1,-1':'↙','0,-1':'↓','-1,-1':'↘'};
const MAP_LEGEND=`<b>Row 0 = gate 0</b> (where the start gate is from spawn). Each next row is the leg <b>gate g-1 → gate g</b>: click where that gate is as you arrive (gravity-leveled approach frame) — the arrow points AT it.<br>`+
  `↖ upper-left · ↑ up · ↗ upper-right &nbsp;/&nbsp; ← left · • straight+level · → right &nbsp;/&nbsp; ↙ lower-left · ↓ down · ↘ lower-right<br>`+
  `encodes <b>[horiz, vert]</b> — horiz: <b>RIGHT=-1</b>, <b>LEFT=+1</b>, straight=0 &nbsp;·&nbsp; vert: <b>UP=+1</b>, <b>DOWN=-1</b>, level=0.&nbsp; 1-bit fly-check: if the drone banks the WRONG way, flip left↔right.`;
async function loadMap(){
  const path=($('f_ego_coarse_map')&&$('f_ego_coarse_map').value)||'configs/vq2_coarse_map.json';
  $('maplegend').innerHTML=MAP_LEGEND;
  const r=await (await fetch('/api/coarse_map?path='+encodeURIComponent(path))).json();
  if(r.error){$('mapbody').innerHTML='<span class="bad">'+r.error+'</span>';MAP=null;return;}
  MAP={path:r.path, sector:(r.sector||[]).map(x=>[x[0]|0,x[1]|0])};
  $('mappath').textContent=r.path; $('mapmsg').textContent='';
  renderMap();
}
function renderMap(){
  if(!MAP){$('mapbody').innerHTML='<span class="mut">(no map loaded)</span>';return;}
  let html='';
  MAP.sector.forEach((hv,gi)=>{
    const lbl = gi===0 ? 'gate 0' : `gate ${gi-1} → ${gi}`;   // row 0 = current gate; rest = legs
    html+=`<div class="mapgate"><span class="lbl">${lbl}</span><div class="grid3">`;
    for(let r=0;r<3;r++)for(let c=0;c<3;c++){
      const h=1-c,v=1-r,k=h+','+v,on=(hv[0]===h&&hv[1]===v);
      html+=`<button class="gc${on?' on':''}" title="${MAP_NAME[k]}  [${h}, ${v}]" onclick="setCell(${gi},${h},${v})">${MAP_GLYPH[k]}</button>`;
    }
    html+=`</div><span class="mut" style="font-size:11px">[${hv[0]}, ${hv[1]}] &nbsp;${MAP_NAME[hv[0]+','+hv[1]]||'?'}</span></div>`;
  });
  $('mapbody').innerHTML=html;
}
function setCell(gi,h,v){if(MAP){MAP.sector[gi]=[h,v];renderMap();}}
function addGate(){if(MAP){MAP.sector.push([0,0]);renderMap();}}
function delGate(){if(MAP&&MAP.sector.length>1){MAP.sector.pop();renderMap();}}
async function saveMap(){
  if(!MAP)return;
  const r=await (await fetch('/api/coarse_map',{method:'POST',body:JSON.stringify({path:MAP.path,sector:MAP.sector})})).json();
  $('mapmsg').innerHTML=r.ok?`<span class="ok">✔ saved ${r.path} (${r.rows} gates)</span>`:`<span class="bad">${r.error||'save failed'}</span>`;
}
boot();
</script></body></html>"""

# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _load_state()
    for probe, label in [(VENV_PY, "venv python"), (REPO / SCRIPT, "fly_rl.py"), (MODELS, "models dir")]:
        if not Path(probe).exists():
            print(f"  [warn] {label} not found at {probe}")
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    url = f"http://{a.host}:{a.port}"
    print(f"\n  VQ2 Pilot Control Panel  ->  {url}")
    print(f"  repo   : {REPO}")
    print(f"  python : {VENV_PY}")
    print(f"  logs   : {LOG_DIR}")
    print("  Ctrl-C to stop the panel (launched pilots keep running).\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  panel stopped.")

if __name__ == "__main__":
    main()
