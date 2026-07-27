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

# This file's own mtime, served with /api/schema. The browser caches MODEL_DEFAULTS at page load, so
# after a panel restart an OPEN TAB still applies the OLD recipe on a model-pick -- silently, because
# the form looks entirely normal. That has now cost flights twice: eight runs on a 2-generation-stale
# actor (2026-07-23), then a batch flown on M with the merge off while the server had M+1 pinned
# (2026-07-24). The page compares this against its own copy and tells the pilot to reload.
_BUILD_ID = str(int(os.path.getmtime(__file__)))

SCHEMA = [
    # ---- Flight stack -----------------------------------------------------
    dict(key="ego_ckpt", flag="--ego-ckpt", action="value", ui="select", opts="ego_ckpts",
         group="Flight stack", default="ckpts/v19Ws0_actor.pth", allow_blank=True,
         help="RL policy actor (.pth). This IS the flight stack. Blank = classical CTBR (non-ego). "
              "DEFAULT TRACKS THE DEPLOY LEAD: on 2026-07-23 eight flights were launched on the old "
              "vpeffs0 default (2 generations stale) + negreal42 + pitch clamp 10 after a panel "
              "restart reset the form -- the config looked plausible and nobody noticed. Keep this "
              "pointed at the current lead so an accidental default-launch flies something current."),
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
         group="Vision", default="C:/Users/Shadow/Peregrine/models/vq2_partial_m_2026-07-06_fp16_384x640.engine",
         allow_blank=True,
         help="Vision model: a TensorRT .engine (fast) or ultralytics .pt. Required for yolo."),
    # Gate-emission port (2026-07-22). Both shipped engines are 8-KEYPOINT pose models (kpt_shape
    # [8,3]: 4 inner corners of the 1.5 m opening + 4 outer corners of the 2.72 m frame). Until this
    # port the flight path sliced the outer 4 off at the model boundary and threw them away, and
    # dropped any detection with <3 confident inner corners -- the cropped close-range frames right
    # before a pass. Defaults ON; these are A/B switches back to the old behaviour.
    dict(key="seeker_no_outer", flag="--seeker-no-outer", action="flag", ui="bool",
         group="Vision", default=False,
         help="OFF (default) = fuse the model's 4 OUTER corners into the pose (task2 centre error "
              "0.111 m vs 0.149 m inner-only). ON = discard them, the pre-port behaviour."),
    dict(key="seeker_no_rescue", flag="--seeker-no-rescue", action="flag", ui="bool",
         group="Vision", default=False,
         help="OFF (default) = rescue cropped gates: any >=4 usable keypoints fit the gate-plane "
              "homography and reconstruct the full inner square (recovers 24.8% of dropped "
              "detections at 0.060 m median). Rescued fixes report n_corners=3 so their covariance "
              "is inflated. ON = drop them, the pre-port behaviour."),
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
         help="CAP 1 of 2 -- VALID range (m): drop detections beyond this from the pool "
              "(billboard-FP guard). 30 = record. A pose must ALSO pass the acquire cap to be locked."),
    dict(key="ego_max_acquire_range", flag="--ego-max-acquire-range", action="value", ui="number",
         group="Ego perception", default=22.0, step=1,
         help="CAP 2 of 2 -- ACQUIRE range (m): a gate may only be LOCKED (cold start / re-acquire) "
              "within this. Was HARD-CODED at 22 with no knob until 2026-07-21, so a gate at 25 m was "
              "valid-but-unlockable -- a blind window on any leg longer than 22 m. 22 = the 9-gate "
              "record. Raise toward the valid cap to close a long-leg window; too high re-admits the "
              "far off-axis downrange gate this exists to reject."),
    dict(key="seeker_emit", flag="--seeker-emit", action="value", ui="select",
         group="Ego perception", default="pnp", choices=["pnp", "centre"],
         help="Where the gate's 3-D position comes from. 'pnp' = today's IPPE/P3P translation. "
              "'centre' = M+1's directly-regressed centre (bearing) x apparent corner size (range) -- "
              "both ambiguity-free, so it also deletes the ~1.2 m task2 error, and it keeps emitting "
              "when corners crop (92%->100% of observations on 300 real frames; depth within 0.425 m "
              "median of PnP inside 30 m). ⚠ 'centre' REQUIRES the 5-kpt M+1 engine in Vision->seeker "
              "weights (vq2_m1_darkred_*): with an 8-kpt model every candidate is dropped and the "
              "flight is BLIND. fly_rl hard-fails on the pad if they disagree."),
    dict(key="ego_area_src", flag="--ego-area-src", action="value", ui="select",
         group="Ego perception", default="pnp", choices=["pnp", "corners"],
         help="Source of the visible_area (approach-angle) cue. 'pnp' re-derives it from the pose "
              "rotation -- MEASURED at median 0.990 / max 1.000 over 184 real ticks, i.e. it reports "
              "head-on on ~90%% of ticks while banking 50-61 deg, so it tells the policy nothing "
              "about angle, and it correlates 0.44 with distance despite being range-free. "
              "'corners' uses the measured corner-quad ratio (no PnP, no range coupling). Scales "
              "agree to <=0.073 over 0-60 deg tilt. Needs 4 corners; below that the cue is masked."),
    dict(key="ego_track_ema_alpha", flag="--ego-track-ema-alpha", action="value", ui="number",
         group="Ego perception", default=0.5, step=0.05,
         help="Gate-TRACK smoothing, both slots: new = (1-a)*old + a*measurement. 1.0 = SNAP to the "
              "latest measurement (no smoothing), small = heavy/laggy. Was HARD-CODED at 0.5 with no "
              "knob until 2026-07-22; 0.5 = every flight to date incl. the 9-gate record. RAISE when "
              "the track LAGS a fast-closing gate (prediction trails, honest candidates start failing "
              "the jump gates); LOWER if noisy depth makes it jitter. The jump gates test against the "
              "PREDICTED value, so this also changes what counts as a consistent candidate."),
    dict(key="ego_dup_merge_bearing", flag="--ego-dup-merge-bearing", action="value", ui="number",
         group="Ego perception", default=0.0, step=0.02,
         help="Duplicate-gate MERGE, radians. A gate bigger than the frame is found by several "
              "anchors on different fragments that NMS cannot merge (IoU 0.287); M+1 emits 1.00 such "
              "duplicate pairs/frame vs M's 0.10. Folds poses agreeing in bearing (within this) AND "
              "range (within 25%) into one, keeping the fullest quad. 0.0 = OFF. 0.10 (6 deg) is the "
              "calibrated value: 0.000 residual dups, distinct gates sit at 0.33 rad so none merge. "
              "RECALL-SAFE (only removes a pose with a near-twin) -- unlike training it out with hard "
              "negatives, which collapsed small-gate recall 98% -> 38%."),
    dict(key="ego_gate_z_bias", flag="--ego-gate-z-bias", action="value", ui="number",
         group="Ego perception", default=0.0, step=0.1,
         help="Perceived-gate VERTICAL bias (m) added to EVERY vision emission (both slots). "
              "+ lowers the gate so the drone aims/passes LOWER through it. 0 = off. Try 0.3."),
    dict(key="ego_aim_offsets", flag="--ego-aim-offsets", action="value", ui="text",
         group="Ego perception", default="",
         help="PER-GATE aim offset \"gate:lateral,vertical;...\" in metres, 0-based gate_index -- "
              "e.g. \"4:0,10;5:0,10\". Shifts the PERCEIVED gate the policy chases, in BODY FLU, "
              "only while that gate is active and only on slot0. Blank = OFF. SIGNS: +lateral = "
              "RIGHT, +vertical = UP (aims/passes HIGHER) -- the OPPOSITE vertical sign to "
              "z-bias above, which is camera +Y = DOWN, and a different frame (camera-frame "
              "vertical also moves perceived RANGE by 0.342x; body-frame does not). FOR: the two "
              "invisible obstacles ~14.5 m short of gate 4 (14/31 deaths there) and gate 5 "
              "(9/21). Pilot flew \"5:0,10\" x5; among flights that REACHED gate 5: band deaths "
              "0/5 vs 9/23 (p=0.118), at-gate deaths 3/5 vs 3/23 (p=0.050), pass 2/5 vs 5/23 "
              "(p=0.367). CORRECTED 2026-07-27 (pilot + geometry): the at-gate deaths are NOT "
              "caused by the dodge -- it releases at 12.2-21.8 m and those flights die at "
              "1.9-3.8 m, i.e. 10-20 m (1.7-3.3 s) of recovery after the offset is gone, and "
              "the SHORTEST dodge (0.47 s, released furthest out, most recovery) still died at "
              "the gate. They are ordinary at-gate deaths, the mode that kills 27-34% of every "
              "approach. What the dodge does buy is 0/5 obstacle-band deaths vs 9/23. Duration "
              "was never a fixed rule either -- the pilot tuned it 2.00/1.97/0.94/0.70/0.47 s "
              "across one session. Gate 4 has NEVER been probed. Fly it as an A/B."),
    dict(key="ego_aim_release", flag="--ego-aim-release", action="value", ui="number",
         group="Ego perception", default=12.0, step=0.5,
         help="Range (m) at/below which the per-gate aim offset is RELEASED, measured on the held "
              "belief excluding its own offset. 12.0 = median of the pilot's five hand-timed "
              "releases (10.6-17.2 m): below the 11-16 m obstacle band (so it is still up across "
              "the obstacle) and clear of the gate (so the last 12 m are threaded honestly). His "
              "own rule was a ~2.0 s wall-clock hold, which does not transfer across speeds."),
    dict(key="ego_aim_fade", flag="--ego-aim-fade", action="value", ui="number",
         group="Ego perception", default=0.0, step=0.5,
         help="Linear fade width (m) above the release range: offset scales 1->0 over "
              "[release, release+fade]. 0 = HARD STEP = exactly what flew. >0 has NEVER FLOWN."),
    dict(key="ego_fix_gain", flag="--ego-fix-gain", action="value", ui="number",
         group="Ego perception", default=1.0, step=0.05,
         help="Obs-builder FIX GAIN K, both slots: rel_new = (1-K)*propagated_held + K*fix. 1.0 = SNAP "
              "to every fix -- what every flight to date flew. Inside 1-2 m of a gate 96% of fixes "
              "carry <4 corners and 23% fall back to bbox range, so the snap puts p99 half-metre "
              "tick-to-tick jumps straight into the vertical target. TRAINING low-passed at K = 1/N_eff "
              "with N_eff ~ U[4,9] (~0.154), i.e. the policy learned on a belief averaging ~6 fixes. "
              "Lower = smoother/laggier; the held lever is ego-propagated between fixes, so a low K is "
              "dead-reckoning corrected by vision, not a frozen target. K snaps back to 1.0 on "
              "RE-ACQUISITION (nothing held, or held past the stale horizon)."),
    dict(key="seeker_propagate_range", flag="--seeker-propagate-range", action="flag", ui="bool",
         group="Ego perception", default=False,
         help="Dead-reckon the tracked gate RANGE between detector fixes (r -= (v.u_hat)*dt along the "
              "line of sight, floored at 0.3 m), instead of freezing it on its EMA. The bearing is "
              "already gyro-propagated; the range is not, so at race speed it is stale by the whole "
              "detection gap (~0.1 s at 10 Hz). Both slots. Changes what the continuity/jump gates, the "
              "pass-drop rule and the slot1 promote check compare against -- it does NOT loosen them. "
              "OFF = byte-identical to every flight to date."),
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
         group="Ego control", default=20.0, step=1,
         help="Nose-down PITCH fence (deg, 0=off). BIMODAL, not a dial: resting obs[4] = -17.80 deg "
              "and the fence tests obs[4] <= -clamp, so <=17.8 is FENCE-ON AT REST and >=17.9 (or 0) "
              "is OFF -- there is no in-between. Default was 10 (fence ON, the OLD regime) and eight "
              "flights on 2026-07-23 silently flew it after a panel restart reset the form. 20 = the "
              "9-gate record and the v16/v17/v18 line. Blocks nose-DOWN only; nose-up always passes."),
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
              "only ever caps TOWARD hover, never forces a sink. "
              "\U0001F6D1 DO NOT ARM THIS AT 5,6.5 -- the old help said 'try 5,6.5' and that value is "
              "HARMFUL. Adjudicated 2026-07-27 over n=1910 per-gate approaches in 92 "
              "(checkpoint x rate x gate) strata: per-gate survival vs speed is an INVERTED U peaking "
              "at 7.0-8.0 m/s (p=0.839, speed^2 z=-3.86) and the fleet already flies 6.58 m/s median -- "
              "BELOW the peak. A 5,6.5 governor engages on 99.6% of approaches and drags the fleet "
              "0.839 -> 0.731. SLOWING IS THE WRONG SIGN: the blind run-in is a FOV-FIXED ~2.0 m of "
              "DISTANCE (last-sighted range flat 1.86-2.08 m across speed quintiles 2.56-16.52 m/s, "
              "corr(speed, last-sighted) = +0.006), so slowing only stretches blind TIME -- and blind "
              "time kills independently (b=-2.31, z=-2.67). 8.0 -> 6.0 m/s = +34% open-loop flight. "
              "Never flown in 680 recorded flights. If armed at all, only a TAIL-CLIP 8.5,9.5 that "
              "cannot touch the 6-8 m/s operating point. Watch gov / gov_engaged in ego_obs.jsonl. "
              "NEEDS commit 293ffee."),
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

    # ---- Ego arrestor (ratchet) -------------------------------------------
    dict(key="ego_arrestor", flag="--ego-arrestor", action="flag", ui="bool",
         group="Ego arrestor", default=False,
         help="Post-gate STARE-BRAKE takeover: after banking a listed gate, take the wire, bleed "
              "speed while staring at the next gate, hand back to the policy slow+level+in-view. "
              "Fail-open (gate-lost/timeout/drift aborts return control to the policy). Applies at "
              "LAUNCH; engages automatically in-flight. NEEDS branch ratchet-arrestor-2026-07-18."),
    dict(key="ego_arrest_after_gates", flag="--ego-arrest-after-gates", action="value", ui="text",
         group="Ego arrestor", default="1",
         help="Comma list of gate indices whose PASS engages the arrestor (once each). '1' = brake "
              "right after banking gate 1, before the gate-2 descend wall. Blank = no gate trigger."),
    dict(key="ego_arrest_speed_hi", flag="--ego-arrest-speed-hi", action="value", ui="number",
         group="Ego arrestor", default=0.0, step=0.5,
         help="Runaway catcher: engage whenever KF horizontal speed exceeds this (m/s, 0=off). "
              "~2 s refractory after each handback so it can't chatter. Probe recipe: 10."),
    dict(key="ego_arrest_max_s", flag="--ego-arrest-max-s", action="value", ui="number",
         group="Ego arrestor", default=4.0, step=0.5,
         help="Hard arrest timeout (s) -> fail-open handback to the policy."),

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

GROUP_ORDER = ["Flight stack", "Vision", "Ego perception", "Ego control", "Ego arrestor", "Run", "Advanced"]
BY_KEY = {s["key"]: s for s in SCHEMA}

# Per-model deploy DEFAULTS -- auto-applied in the panel when a checkpoint is picked (keyed by
# .pth basename). The yaw clamp is TRAINING-matched and per-RELEASE: 0.7 for everything EXCEPT
# the ego-ckpts-tracka-2026-07-13 arm (w0 + m8) at 0.35. MIXING THE TWO = OOD (Fengyou 2026-07-13).
# NOTE: only the yaw clamp is encoded so far -- whether the vtrackA lineage also needs the vpef
# recipe (slot1 / sector=map / tight pitch) is UNCONFIRMED; extend per model once known.
# ego-ckpts-v1-2026-07-18 (v1 PICK release): full REPORT4 champion recipe, auto-applied on pick.
# TRT M-engine detector, assist 1.3 g, the champion LOG-ECHO map (frozen side file -- NOT the
# live survey map), tight pitch 1.0, det-hold 0.2, rate 40 @ uplink scale 1.2, yaw 0.7 MANDATORY
# (fly_rl default is 0.0). A = sweep pick (4.65 gates @1200-step); R = stability fallback
# (recovery-trained, best roll of portfolio). Settled GO >=3 s per run; labels v1pick_*.
_V1_RECIPE = {
    "ego_yaw_clamp": 0.7,
    # WP5 (Patch-1): explicitly PIN the two knobs that silently rode across model switches on
    # 2026-07-19 (ego_yaw_clamp 0.35 + ego_gate_z_bias 0.25 corrupted a v16 batch). Pinned here at
    # the _V1 base so v1/v15/v16 all RESULT in yaw 0.7 + z-bias 0.0 on a model-pick. z-bias LOWERS
    # every emitted gate point (gate_seeker _valid_poses); the champion value is 0 (0.25 landed 1:1
    # in a true-crossing miss). Combined with the WP6a model-pick reset, no stale knob can ride in.
    "ego_gate_z_bias": 0.0,
    # 2026-07-27 (commander): the per-gate AIM OFFSET is pinned here at the _V1 base to the INERT
    # value for the same reason as the two above -- ``_recipe_managed_keys()`` is the UNION of recipe
    # pins and is exactly the set a model-pick RESETS, so a knob that appears in NO recipe is a knob
    # that silently RIDES across a model switch. This one moves where the drone believes the gate is,
    # so a stale ride-in is a wrong-target flight. Pinned EMPTY (never active): a recipe may not ARM
    # an aim offset, and picking any model clears one the pilot set for a previous model.
    "ego_aim_offsets": "",
    "seeker_detector": "yolo",
    "seeker_weights": "C:/Users/Shadow/Peregrine/models/vq2_partial_m_2026-07-06_fp16_384x640.engine",
    "ego_assist_thrust": 1.3,
    "ego_sector_mode": "map",
    # Fengyou 2026-07-19: fly the HAND-MADE panel map only -- picking a model no longer swaps to
    # any frozen/canonical map. This points at the panel map-editor's own file; edit it via the
    # 🗺 tab and it takes effect on the next launch. (Champion log-echo + commander canonical are
    # IGNORED here by directive; the frozen side file still exists for the commander's reference.)
    "ego_coarse_map": "configs/vq2_coarse_map.json",
    "ego_slot1": True,
    "ego_pitch_clamp": 1.0,
    "ego_det_hold": 0.2,
    "ego_stale_horizon": 0.5,
    "ego_rate_scale": 1.2,
    # 2026-07-27 (commander): 40.0 -> 30.0. THE EFFECT IS LARGE AND MEASURED -- but read the
    # mechanism carefully, because the OBVIOUS one is FALSE and I had it backwards for half a day.
    #
    # MEASURED, all 670 recorded flights:  rate 30  n=189  mean max gate 2.460
    #                                      rate 40  n=481  mean max gate 1.769
    # plus a same-night same-checkpoint pair whose meta.json differ on ``rate_hz`` ALONE (1.14 ->
    # 3.78 mean gates, n=7 vs 9 -- a small-cohort figure; the corpus-wide within-checkpoint effect
    # is nearer 1.4x, so do not re-quote 3.78 as the cost of a single re-arm).
    #
    # 🛑 NOT "this matches the training dt". THE LOOP NEVER ACHIEVES ITS COMMANDED RATE -- it is
    # COMPUTE-BOUND (per-tick work p50 29.5 ms, p90 62.5). Measured achieved rate:
    #       commanded 30 -> 21.72 Hz median   (|err| from the 30.03 Hz training dt = 8.28)
    #       commanded 40 -> 25.34 Hz median   (|err| = 4.66)
    # So commanding 30 lands FURTHER from the training cadence and still wins decisively.
    #
    # ✅ THE REAL MECHANISM IS VISION FRESHNESS. A slower commanded loop leaves the perception
    # pipeline time to deliver a NEW fix per tick instead of the policy re-reading a stale one:
    #       fresh-fix fraction  commanded 30 -> 94.9%   commanded 40 -> 75.6%   (median, same n)
    # ⇒ 🚩 A TESTABLE CONSEQUENCE THE WRONG MECHANISM WOULD HAVE HIDDEN: if freshness is what pays,
    # commanding SLOWER STILL (25, or 20) may be better again -- the "matches training" story said
    # 30 was the optimum and there was nothing below it to look for. Worth one cohort.
    #
    # Why it needed fixing at all: this knob's schema default is 30.0 and its help reads "Control
    # loop Hz (training dt = 30)"; the pin was the only thing saying 40, and because ``rate`` is
    # inside ``_recipe_managed_keys()`` EVERY model-pick silently re-applied it.
    # 🛑 DO NOT DELETE THIS KEY. A knob that appears in NO recipe is never reset by a model-pick and
    # rides across model switches at whatever was last typed (the WP5 rationale above). Pinning it
    # CORRECTLY is the fix; removing it re-opens a different hole.
    "rate": 30.0,
    "virtual_flip": True,
}

# ego-ckpts-v15-2026-07-19 (v15 PICK): champion recipe with TWO training-faithful deltas --
# PITCH FREE (clamp 0 = off; this lineage trained unclamped and priced the runaway in training:
# overspeed abort 12 m/s, census max 11.97) + yaw 0.7 MANDATORY. Do NOT re-add the pitch clamp
# mid-session even if a flight looks divey -- land and relay. Qs1 = fly first; Ws0 = quiet fallback.
_V15_RECIPE = {**_V1_RECIPE, "ego_pitch_clamp": 0.0}

# ego-ckpts-v16-2026-07-19 (v1.6 Q PICK): v15 recipe (PITCH FREE, this IS the A/B vs a clamped
# baseline) with ONE deliberate delta -- ego_rate_scale 1.2 -> 1.0 (commander 2026-07-19). Unchanged:
# yaw 0.7, det-hold 0.2, slot1, editor map (configs/vq2_coarse_map.json), M-engine, assist 1.3.
# Settled GO >=3 s. Re-fly any flight with max|gyro| > 2.5 rad/s in the first 0.75 s. NO mixing
# with v15 ckpts. Target: beat Ws0's clamp-15 ledger {5,5,5,4,4,3,3} UNCLAMPED + CONTACT-FREE.
_V16_RECIPE = {**_V15_RECIPE, "ego_rate_scale": 1.0}

# ego-ckpts-v17-2026-07-21 -- ⚠ NOT A DEPLOY LEAD. Shipped as a FALSIFICATION TEST; the deploy lead
# stays v16Qs1. v1.7's M1 ("handoff spawn realism") assumed the wire hands over LEVEL with the gate
# ~20 deg BELOW the optical axis; measured over 161 flights it hands over at the PAD TILT (-0.3107 rad
# on 63%, 0/161 level) with the gate DEAD-CENTRE (+2.42 deg median) -- so M1 INTRODUCED the mismatch it
# was built to remove. Replay of 388 flights predicts these actors pitch down HARDER in the first 4
# ticks after assist release (median worst nose-down rate cmd: parent v16Qs1 -0.856 vs v17Qs0 -1.386
# = +62%, worse on 386/388) and therefore LOSE GATE 0 MORE OFTEN. We fly to try to FALSIFY that.
#
# Recipe = the 9-GATE RECORD CONFIG WITH ONLY THE ACTOR SWAPPED, because the release's binding
# instruction is "fly at the same clamp as your v16Qs1 baseline -- the comparison is only meaningful
# single-knob", and the record (+ its 11-flight batch) is that baseline. So pitch clamp 20, z-bias 0.4,
# stale 0.6, assist 1.1 are inherited from the BASELINE, not from _V16_RECIPE. Release-pinned and
# unchanged from v1.6: rate_scale 1.0, yaw 0.7, det-hold 0.2, slot1, settled GO >=3 s. No cross-release
# mixing. Labels are v17test_* (NOT "pick") so these never read as deploy-lead flights.
#
# THE CLAMP IS BIMODAL, NOT A DIAL (v1.7 notes, supersedes the v1.6 "pitch FREE" rule): resting obs[4]
# = -0.3107 (-17.80 deg) and the fence is obs[4] <= -clamp_rad, so clamp <=17.8 = FENCE ON AT REST,
# >=17.9 (or 0) = FENCE OFF. Clamp 20 and clamp 0 are the SAME REGIME at rest -- the record's
# "pitch-clamp drift" vs _V16_RECIPE's 0.0 is therefore cosmetic, not a regime change.
_V17_RECIPE = {
    **_V16_RECIPE,
    "ego_pitch_clamp": 20.0,      # baseline-matched; fence OFF at rest, same regime as the recipe's 0
    "ego_gate_z_bias": 0.4,       # baseline-matched (the record flew it; _V16_RECIPE pins 0.0)
    # 2026-07-27 (commander): 0.6 -> 0.5, matching TRAINING. The v1.7 note that put 0.6 here cited the
    # v1.6 recipe, but _V16_RECIPE pins 0.5 and so does training (``ego_estimator.py`` EgoEstimatorCfg
    # stale_horizon_s = 0.5); the deploy docstring at ``src/racer/ego_obs.py`` already CLAIMS 0.5.
    # obs[14] = clamp(1 - age/stale_horizon, 0, 1), so flying 0.6 against a 0.5-trained policy reports
    # ~20% MORE confidence for the same staleness -- with det_hold 0.2 the wire's obs[14] floor moves
    # 0.667 -> 0.600, onto training's coasted-confidence value at the gate plane (~0.52-0.60).
    # Small (<=0.07 on one channel) but free, and the same class of silent drift as the ``rate`` pin.
    "ego_stale_horizon": 0.5,
    "ego_assist_thrust": 1.1,     # baseline-matched (_V16_RECIPE pins 1.3)
}

# ego-ckpts-v18-2026-07-22 -- ★ THE DEPLOY LEAD (supersedes v16Qs1). v1.8 = v1.7 with ONE knob
# changed: M1 OFF (handoff_spawn_frac 0.33 -> 0.0). M1 was the SOLE cause of v1.7's release-pitch dive
# and removing it returned the dive to PARENT LEVEL, verified on 425 real flights (worst nose-down
# rate cmd in the 4 ticks after assist release: parent v16Qs1 mean -0.805 / max -2.369; v17 pooled
# -1.225 / dove harder than parent on ~423/425; v18Qs1 -0.801 / 248/425; v18Qs0 -0.633 / only 33/425).
# Deploy pick = v18Qs1 (training lead: gates 5.75, center_pen 0.163, frame_factor 0.841, dive == parent).
# v18Qs0 = designated ALTERNATE, gentlest start in the fleet by a wide margin (33/425) -- take it if the
# launch window looks hot, costs ~0.4 gates of reward. v18Ws0 = W arm, third seed, for completeness.
#
# Recipe = the SAME BASELINE the user has been flying all session (the 9-gate-record config = _V17_RECIPE
# with only the actor swapped: pitch clamp 20, z-bias 0.4, stale 0.6, assist 1.1), so v18-vs-v17 is the
# single-knob (M1) comparison the commander built -- fly them back to back on identical deploy knobs.
# The release pins ONLY yaw 0.7 + virtual-flip + bimodal-per-release clamp + the re-fly/GO rules; z-bias
# and assist are Fengyou's deploy knobs, unmentioned, so they carry the baseline. If you'd rather fly the
# CLEAN canonical recipe (z-bias 0, assist 1.3, stale 0.5 = _V16_RECIPE) say so -- one-line change.
# ⚠ Do NOT re-add M1, do NOT retune the training action space, do NOT mix a clamp across lineages (OOD).
_V18_RECIPE = {
    **_V17_RECIPE,   # inherits the 9-gate baseline; labels below mark these as deploy PICKS
    # ---- VISION SHIPPED 2026-07-24: M+1 centre-emit + the duplicate MERGE ----------------------
    # Scoped to the DEPLOY LEAD only. Every older checkpoint keeps M + pnp + merge OFF, so nothing
    # already flown changes and no stale model meets a detector it was never flown with.
    #
    # M+1 (5 kpt: inner 4 + regressed CENTRE) replaces M as the deploy detector. It emits for the
    # 55.7% of real gate views whose centre is in frame while a corner is cropped -- the population
    # M is structurally blind to -- and in flight it lifted the 2-4 m fix rate 47.2% -> 90.1% while
    # halving loop time (46.0 -> 23.9 ms, i.e. 21.8 -> 41.8 Hz).
    #
    # Its ONE regression was duplicate detections: a gate larger than the frame is found by several
    # anchors on different fragments whose boxes overlap too little for NMS (measured IoU 0.287), so
    # M+1 emitted 4.53 detections/frame against 2.62 real gates and 1.00 duplicate PAIRS/frame (M:
    # 0.10). Fixed HERE, in the seeker, not in training: dup_merge folds candidates agreeing in
    # bearing (<=0.10 rad) AND range (<=25%) into one, measured residual 0.000 pairs/frame with gates
    # >=60 px preserved at 98-99%. It is recall-safe by construction -- it only ever drops a pose that
    # has a near-twin, and genuinely distinct gates sit at 0.33 rad, far outside the threshold.
    #
    # TRAINING THE DUPLICATES OUT WAS TRIED TWICE AND REJECTED (2026-07-23/24). Hard negatives at
    # yolo11s killed them (1.00 -> 0.02) but COLLAPSED distant-gate recall in the acquisition band:
    # 30-60 px (~15-29 m) fell 99% -> 50%, verified by eye as REAL gates missed (177/400 frames), not
    # a metric artifact. Confuser size was not the lever -- large-solid negatives suppressed just as
    # hard. If confident FPs on non-gate red ever prove to matter, retrain at yolo11m (M's size, which
    # carries negatives fine) -- NOT more negatives at this capacity.
    "seeker_weights": "C:/Users/Shadow/Peregrine/models/vq2_m1_darkred_2026-07-23_fp16_384x640.engine",
    "seeker_emit": "centre",          # REQUIRES the 5-kpt model above; centre+8-kpt aborts at launch
    "ego_dup_merge_bearing": 0.10,
    # Fengyou 2026-07-24, flying the M+1 stack:
    #   TAKEOFF ASSIST OFF. The thrust floor was a ground-unstick crutch; with the assist armed the
    #   release is thrust-floored for up to 1.5 s, which fights the policy's own launch behaviour on
    #   this lineage. ego_assist_thrust stays pinned only so a stale form cannot ride a value in --
    #   it is inert while the toggle is off.
    "ego_takeoff_assist": False,
    #   Z-BIAS 0.4 -> 0.25. Lowers every emitted gate by 0.25 m instead of 0.4 (the 9-gate record's
    #   value), i.e. aim less far under the gate now that the centre emit is supplying the position.
    "ego_gate_z_bias": 0.25,
}

# CURRENT-GENERATION checkpoints. Anything outside this set gets a loud launch warning: on
# 2026-07-23 a panel restart reset the form to schema defaults and EIGHT flights went out on
# vpeffs0 (two generations stale) + negreal42 + pitch clamp 10 before anyone noticed -- the config
# looked entirely plausible, and the only after-the-fact signal was reading the stored argv. A
# ckpt merely HAVING a MODEL_DEFAULTS entry is not enough (vpeffs0 has one, for its yaw clamp);
# membership here is the deliberate "still flown" statement. Add new releases as they ship.
CURRENT_CKPTS = {
    "v19Ws0_actor.pth",
    "v18Qs1_actor.pth", "v18Qs0_actor.pth", "v18Ws0_actor.pth",
    "v17Qs0_actor.pth", "v17Qs1_actor.pth", "v17Ws0_actor.pth",
    "v16Qs0_final_actor.pth", "v16Qs1_final_actor.pth",
}

# ego-ckpts-v19-2026-07-24 -- ★ DEPLOY LEAD per the release. v1.9 = v1.8's W arm + roll-rate
# penalties (rw_roll_jerk / rw_roll_duty on ch1), which DAMP the close-in roll limit cycle -- the
# exact thrash diagnosed on the wire 2026-07-24 (roll cmd saturating to +-3.14 with sign-flips ~3/s
# while valid_poses_empty, then tumble). Release measures it strictly better than v1.8 on every axis:
# roll signflips/s 2.595->2.255, roll cmd_absmean 0.306->0.289, gates 5.72->6.03, AND the release
# dive IMPROVED (worst -1.561 vs v1.8 -1.885, dove harder on only 1/451 replayed flights).
# LINEAGE: this is the W arm; you have been flying the Q arm (v18Qs1). The Q v19 arms were REJECTED
# and are NOT downloaded -- v19Qs1 dove harder than v1.8 on 450/451 flights (floor-planting), v19Qs0
# over-damped (roll magnitude UP, gates DOWN). Only Ws0 shipped.
#
# Recipe = the release's "fly it EXACTLY like the best v1.8 config, swap only the checkpoint": M+1
# darkred + emit centre + merge inherited from _V18_RECIPE, with the ONE listed delta z_bias 0.30
# (the v1.8 wire sweet spot, 11/12 gate-1; _V18 currently holds Fengyou's 0.25). pitch 20, yaw 0.7,
# ema 0.5 (schema default, unpinned), stale 0.6, det-hold 0.2, coast off, map -- all already match.
# TAKEOFF ASSIST: inherited OFF from _V18 (Fengyou's 2026-07-24 setting). The release does not mention
# assist, so this is the one knob not pinned by it; if the vision session's 11/12 config flew assist
# ON, that is the first thing to try. FLAGGED to Fengyou, not silently chosen.
_V19_RECIPE = {**_V18_RECIPE, "ego_gate_z_bias": 0.30}

# ego-ckpts-v20-2026-07-26 (v2.0) -- 2026-07-27 (commander): v20 HAD NO RECIPE ENTRY AT ALL. Its six
# flown sessions therefore fell through to whatever the panel last held: all 6 ran rate 40 (the old
# _V1_RECIPE pin) and ``ego_pitch_clamp`` drifted 20/20/20/0/0/30 ACROSS THE COHORT -- so the shipped
# v2.0 lead has never been flown at the correct loop rate, and its six flights are not one cohort.
# A checkpoint with no MODEL_DEFAULTS entry gets no reset and no recipe pins, which is exactly the
# hole ``_recipe_managed_keys()`` exists to close. Inherits v19 (its training lineage) unchanged;
# the ONE thing v2.0 changed was the training course structure, which no deploy knob mirrors.
_V20_RECIPE = {**_V19_RECIPE}

MODEL_DEFAULTS = {
    # ego-ckpts-v20-2026-07-26 -- v2.0. Vert env 5.884 vs v19Ws0 4.893 (+20%), flat 5.973 vs 5.880
    # => v19 course-FRAGILE, v20 course-ROBUST. NOT yet the deploy lead: every wire flight it has is
    # contaminated by the rate-40 pin above, so it must be RE-FLOWN at rate 30 before any comparison
    # against v19Ws0 means anything.
    "v20Vs0_actor.pth": {**_V20_RECIPE, "label": "v20pick_Vs0"},
    "v20Vs1_actor.pth": {**_V20_RECIPE, "label": "v20pick_Vs1"},
    # ego-ckpts-v19-2026-07-24  -- ★ DEPLOY LEAD. See _V19_RECIPE above. W lineage; the Q arms were
    # rejected (dive / over-damp) and not shipped. Re-fly rule + settled GO + contact=INVALID as v18.
    "v19Ws0_actor.pth": {**_V19_RECIPE, "label": "v19pick_Ws0"},
    # ego-ckpts-v18-2026-07-22  -- prior lead. See _V18_RECIPE above. Qs1 = the pick; Qs0 = gentlest
    # alternate (hot launch window); Ws0 = third seed. Re-fly rule: ||gyro|| > 2.5 for >=3 CONSECUTIVE
    # ticks (NOT the old 0.75 s rule -- that was a clamp detector). Settled GO >= 3 s. Contact = INVALID.
    "v18Qs1_actor.pth": {**_V18_RECIPE, "label": "v18pick_Qs1"},
    "v18Qs0_actor.pth": {**_V18_RECIPE, "label": "v18pick_Qs0"},
    "v18Ws0_actor.pth": {**_V18_RECIPE, "label": "v18pick_Ws0"},
    # ego-ckpts-v17-2026-07-21  -- see _V17_RECIPE above. FALSIFICATION TEST, not a deploy lead.
    # Qs0 = the census lead AND the worst predicted diver (-1.386) = the sharpest test. Ws0 =
    # heavier-damped arm. Prediction under test: more gate-0 losses than the v16Qs1 baseline.
    "v17Qs0_actor.pth": {**_V17_RECIPE, "label": "v17test_Qs0"},
    "v17Qs1_actor.pth": {**_V17_RECIPE, "label": "v17test_Qs1"},
    "v17Ws0_actor.pth": {**_V17_RECIPE, "label": "v17test_Ws0"},
    # ego-ckpts-v16-2026-07-19  -- see _V16_RECIPE above (PITCH FREE + rate scale 1.0)
    "v16Qs0_final_actor.pth": {**_V16_RECIPE, "label": "v16pick_Qs0"},
    "v16Qs1_final_actor.pth": {**_V16_RECIPE, "label": "v16pick_Qs1"},
    # ego-ckpts-v15-2026-07-19  -- see _V15_RECIPE above (PITCH FREE)
    "v15Qs1_final_actor.pth":         {**_V15_RECIPE, "label": "v15pick_Qs1"},
    "v15Ws0_periodic_prev_actor.pth": {**_V15_RECIPE, "label": "v15pick_Ws0"},
    # ego-ckpts-v1-2026-07-18  -- see _V1_RECIPE above
    "v1As0_upd17500_actor.pth": {**_V1_RECIPE, "label": "v1pick_A"},
    "v1Rs0_upd17500_actor.pth": {**_V1_RECIPE, "label": "v1pick_R"},
    # ego-ckpts-vpef-2026-07-12  -- yaw 0.7
    "vpefwh2_actor.pth":    {"ego_yaw_clamp": 0.7},
    "vpeffs0_actor.pth":    {"ego_yaw_clamp": 0.7},
    # ego-ckpts-vpef8gate-2026-07-14 -- 8-gate champ (vpeffs0-lineage, natural ~9.5 m/s, no vel cap); yaw 0.7
    "vpef8nc_actor.pth":    {"ego_yaw_clamp": 0.7},
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

# --------------------------------------------------------------------------- #
# Recipe guards (Patch-1 WP6) — pure, unit-testable
# --------------------------------------------------------------------------- #
def _schema_default(key):
    """The SCHEMA default for a knob key (None if the key is not a schema knob)."""
    for s in SCHEMA:
        if s["key"] == key:
            return s.get("default")
    return None

def _recipe_managed_keys(model_defaults: dict | None = None) -> set:
    """Every knob key ANY model recipe pins, minus the per-model ``label``. These are the knobs the
    recipe system governs -> the set a model-pick RESETS to its schema default before applying the
    picked recipe, so a knob edited for a PRIOR model never RIDES into the next one."""
    md = MODEL_DEFAULTS if model_defaults is None else model_defaults
    keys: set = set()
    for r in md.values():
        keys |= set(r.keys())
    keys.discard("label")
    return keys

def _knob_equal(a, b) -> bool:
    """Compare a form value (often a string) to a recipe pin (native type) tolerantly: boolean when the
    pin is bool, numeric when both parse as float, else string-equal."""
    if isinstance(b, bool):
        return _truthy(a) == b
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return str(a) == str(b)

def repin_recipe(current: dict, recipe: dict, model_defaults: dict | None = None) -> dict:
    """WP6a: the knob state a model-pick should PRODUCE. RESET every recipe-managed knob to its schema
    default (clearing any value that rode from a prior model), then overlay the picked recipe's pins.
    Non-managed fields (endpoint, flights, label unless the recipe sets it, ...) pass through untouched.
    Pure (no UI, no side effects) so the model-pick behaviour is unit-testable; the browser
    ``onCkptChange`` mirrors this exact logic on the live form."""
    out = dict(current)
    for k in _recipe_managed_keys(model_defaults):
        out[k] = _schema_default(k)
    out.update(recipe)
    return out

def compute_recipe_drift(values: dict, model_defaults: dict | None = None) -> list:
    """WP6b/c: the recipe-pinned knob keys whose FLOWN value differs from the picked model's recipe pin
    (empty when the flown config matches the recipe = clean). Looks the recipe up by the selected
    ``ego_ckpt`` basename; an unknown model -> no drift. ``label`` is excluded (per-model, not a knob).
    Pure + unit-testable; ``build_cmd`` uses it to warn + pass ``--recipe-drift`` to fly_rl."""
    md = MODEL_DEFAULTS if model_defaults is None else model_defaults
    base = str(values.get("ego_ckpt", "") or "").replace("\\", "/").split("/")[-1]
    recipe = md.get(base)
    if not recipe:
        return []
    drift = []
    for k, pinned in recipe.items():
        if k == "label":
            continue
        if not _knob_equal(values.get(k, _schema_default(k)), pinned):
            drift.append(k)
    return sorted(drift)

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
    # WP6: recipe-drift guard. Surface (UI warning) + pass to fly_rl (--recipe-drift) any recipe-pinned
    # knob whose flown value != the picked model's pin, so it lands in meta.json + a flight-start console
    # warning (the 2026-07-19 stale-knob ride-in guard). Empty when clean.
    drift = compute_recipe_drift(values)
    if drift:
        argv += ["--recipe-drift", ",".join(drift)]
        warn.append("recipe drift: " + ", ".join(drift)
                    + " differ from the picked model's recipe (flown values override the pins).")
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
    # UNRECOGNISED ACTOR: the 2026-07-23 silent-revert guard. A panel restart (or any page reload)
    # resets the form to schema defaults, and eight flights were launched that way on a 2-generation
    # -stale actor before anyone noticed -- the config looked perfectly plausible. Any ckpt with no
    # recipe in MODEL_DEFAULTS is either obsolete or brand-new; say so LOUDLY at launch, because the
    # only other signal is reading the argv after the fact.
    _base = str(values.get("ego_ckpt", "") or "").replace("\\", "/").split("/")[-1]
    if _base and _base not in CURRENT_CKPTS:
        warn.append(f"STALE ACTOR '{_base}' -- not a current-generation checkpoint. The 2026-07-23 "
                    f"silent revert flew vpeffs0 eight times this way. If you did not pick it "
                    f"deliberately, the form has reverted to defaults: re-pick your model.")
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
    # mtime+size cache: refresh_pilots re-parses EVERY tracked pilot every 2 s, and pilots
    # accumulate (1061 of them on this box), so re-reading finished logs dominated the poll --
    # 1.44 s per refresh against a 2 s interval, and it is all under _LOCK, which launch() also
    # needs. That is what made "Launch" take ~30 s after a few flights. Keying on the file's own
    # mtime+size stays correct for a LIVE pilot (its log keeps growing, so the cache misses and we
    # re-parse) including the panel-restarted-mid-flight case, where the Popen handle is gone but
    # the detached pilot is still writing. [2026-07-22]
    try:
        st = os.stat(logpath)
        key = (st.st_mtime_ns, st.st_size)
    except Exception:
        return session, state, gates
    hit = _LOG_CACHE.get(logpath)
    if hit is not None and hit[0] == key:
        return hit[1]
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
    _LOG_CACHE[logpath] = (key, (session, state, gates))
    return session, state, gates

# Poll caches, both keyed on the source file's own (mtime_ns, size) so a LIVE pilot always misses
# and a finished one is read exactly once. Unbounded by design: one small entry per pilot/session,
# and the panel already keeps every pilot in memory anyway.
_LOG_CACHE: dict = {}    # logpath -> ((mtime_ns, size), (session, state, gates))
_HZ_CACHE: dict = {}     # session -> ((mtime_ns, size), hz)


_LOOPRATE_RE = re.compile(r"\[loop-rate\]\s+([\d.]+)\s+Hz over\s+(\d+)\s+ticks")


def _achieved_hz_from_log(session):
    """The pilot's OWN measured loop rate for ``session`` (ticks / elapsed), or None.

    fly_rl prints '[loop-rate] X Hz over N ticks (target T)' at the end of every flight -- the
    authoritative achieved rate. Scans this session's pilot log for it. Ignores runs shorter than
    ~40 ticks, where the figure is dominated by startup."""
    try:
        for lp in sorted(LOG_DIR.glob("p*.log"), key=lambda p: -p.stat().st_mtime):
            txt = lp.read_text(errors="ignore")
            if session not in txt:
                continue
            hits = _LOOPRATE_RE.findall(txt)
            for hz, ticks in reversed(hits):
                if int(ticks) >= 40:
                    return round(float(hz), 1)
            return None
    except Exception:
        return None
    return None


def _loop_hz(session):
    if not session:
        return None
    tf = RUNS_DIR / session / "ego_timing.jsonl"
    try:
        st = tf.stat()
        key = (st.st_mtime_ns, st.st_size)
    except Exception:
        return None
    # Same mtime+size cache as _parse_log, and this one mattered MORE: it json.loads one line per
    # CONTROL TICK, so a finished flight was re-parsed from scratch every 2 s forever (1.00 s of
    # the 1.44 s refresh, over 17.6 MB of timing files).
    hit = _HZ_CACHE.get(session)
    if hit is not None and hit[0] == key:
        return hit[1]
    try:
        w = []
        for line in tf.read_text().splitlines()[1:]:
            if line.strip():
                w.append(json.loads(line).get("work_ms", 0))
        if not w:
            _HZ_CACHE[session] = (key, None)
            return None
        # ACHIEVED loop rate, not a capability figure. This used to report
        # median(1000 / work_ms) -- the rate the loop COULD hit if per-tick work were the only
        # cost -- which ignores pacing entirely and reported 40.9 median / 980.4 max across 1190
        # flights while the pilot's own [loop-rate] line said 22-26 Hz for every one of them. That
        # gap read as "the loop used to do 40+ and now does 30" when the achieved rate had never
        # moved (24.3 pre-M+1 vs 24.1 with it); what actually changed was median work_ms 24.5 ->
        # 27.0, i.e. 1000/24.5=40.8 -> 1000/27.0=37.0. A number whose max is 980 Hz cannot be a
        # loop rate. fly_rl already computes the real one, so take THAT and fall back to the old
        # estimate only when the line is absent (a live flight that has not printed it yet).
        hz = _achieved_hz_from_log(session)
        if hz is None:
            w = sorted(1000.0 / x for x in w if x > 0)
            hz = round(w[len(w) // 2], 1)   # capability estimate, flagged in the UI as such
    except Exception:
        return None
    _HZ_CACHE[session] = (key, hz)
    return hz

# How many recently-started pilots to re-examine per poll, on top of every LIVE one. A finished
# flight's log and timing file are immutable, so re-statting all 1062 of them every 2 s bought
# nothing; this bounds the poll's cost no matter how far the history grows. A pilot that survived a
# panel restart is by definition recent, so it stays in the window.
_REFRESH_RECENT = 60


def _refresh_keys():
    """Pilot keys worth re-examining: every live one, plus the most recently started."""
    live = [k for k in PILOTS
            if _procs.get(k) is not None and _procs[k].poll() is None]
    recent = sorted(PILOTS, key=lambda k: PILOTS[k].get("started_ts") or 0.0,
                    reverse=True)[:_REFRESH_RECENT]
    return set(live) | set(recent)


def refresh_pilots():
    with _LOCK:
        dirty = False
        todo = _refresh_keys()
        for pid_key, rec in PILOTS.items():
            if pid_key not in todo:
                continue
            before = (rec.get("session"), rec.get("state"), rec.get("gates"),
                      rec.get("running"), rec.get("hz"))
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
            if (rec.get("session"), rec.get("state"), rec.get("gates"),
                    rec.get("running"), rec.get("hz")) != before:
                dirty = True
        # Only persist when something actually CHANGED. This wrote a 4.16 MB pilots.json every 2 s
        # regardless -- pure disk churn during a flight, on the same box as the sim.
        if dirty:
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
SMALL_FILES = ["meta.json", "ego_obs.jsonl", "ego_timing.jsonl", "video_index.jsonl", "seeker.jsonl",
               # post-impact recorder (--ego-post-terminal-s): the aftermath ticks + contact events.
               # Small, CRASH-only, and absent on clean flights -- but it is the only record of the
               # strike itself, so it has to ride the panel sync or it never reaches the analysis box.
               "ego_postimpact.jsonl"]
HEAVY_FILES = ["video.bin", "mavlink.tlog"]

def list_sessions(limit=200):
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
        # specific files. -f + an explicit file list never pulls in pilots.json / other ignored
        # cruft. CHUNKED 40 paths/spawn: Windows caps a CreateProcess command line at ~32K chars,
        # and a big batch (or include_heavy) blows past it -> WinError 206 mid-"committing".
        for i in range(0, len(files), 40):
            subprocess.run(["git", "-C", str(REPO), "add", "-f", "--", *files[i:i + 40]],
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
    except OSError as e:
        # e.g. WinError 206 (command line too long): surface it instead of hanging the UI.
        return {"ok": False, "error": f"git spawn failed: {e}"}

# --------------------------------------------------------------------------- #
# Render (onboard video + real detector overlay) via tools/render_yolo.py
# --------------------------------------------------------------------------- #
_renders: dict = {}   # session -> Popen

def _render_out(session):
    return RUNS_DIR / session / "render.mp4"

_RENDER_TOOL = REPO / "tools" / "render_yolo.py"


def _render_is_stale(session) -> bool:
    """True when an existing render.mp4 was produced by an OLDER render_yolo.py than the one on disk.

    Without this the cache is permanent: 'file exists -> ready' meant a session rendered before a
    render-tool fix kept serving the OLD video forever, so a fix to the overlay simply never appeared
    on any flight already rendered. That is indistinguishable from the tool being broken -- e.g.
    renders made before 2026-07-19 still show the stale hardcoded engine and no fed-point overlay."""
    out = _render_out(session)
    try:
        return out.stat().st_mtime < _RENDER_TOOL.stat().st_mtime
    except OSError:
        return False


def start_render(session, last_seconds=9.0, slowmo=3.0, force=False):
    d = RUNS_DIR / session
    if not d.is_dir() or not (d / "video.bin").exists():
        return {"ok": False, "error": "no video.bin for this session"}
    proc = _renders.get(session)
    if proc is not None and proc.poll() is None:
        return {"ok": True, "state": "rendering"}
    if _render_out(session).exists():
        if not (force or _render_is_stale(session)):
            return {"ok": True, "state": "ready"}
        # re-render: drop the stale file first so render_status() cannot report the OLD one ready
        try:
            _render_out(session).unlink()
        except OSError as e:
            return {"ok": False, "error": f"could not remove stale render: {e}"}
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
                                    "assets": discover(), "model_defaults": MODEL_DEFAULTS,
                                    "build": _BUILD_ID})
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
            return self._send(200, start_render(data.get("session", ""),
                                                force=bool(data.get("force"))))
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
<div id="stalebar" style="display:none;background:#7a1520;color:#fff;padding:10px 14px;font-weight:600;border-bottom:2px solid #ff4d5e"></div>
<h1>🚁 VQ2 Pilot Control Panel <span class="sub" id="sub">launch pilots in the background · logs local · commit to git on demand</span></h1>
<div class="wrap">
  <div class="col left">
    <div class="bar">
      <button onclick="doLaunch()">▶ Launch pilot</button>
      <button class="sec" onclick="loadRecord()" title="20260721_040555_v16pick_Qs1_f1 -- gate 9, 2 collisions. Drifted v16: z-bias 0.4, pitch clamp 20, stale 0.6, assist 1.1.">Load 9-gate record</button>
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
      <table><thead><tr><th>label</th><th>state</th><th>gates</th><th title="ACHIEVED loop rate (ticks/elapsed), taken from the pilot's own [loop-rate] line. Was previously median(1000/work_ms) -- a capability figure that read ~41 while the real rate was ~24.">Hz</th><th>session</th><th></th></tr></thead>
      <tbody id="ptbody"></tbody></table>
    </div>
    <div id="view-git" style="display:none">
      <div class="bar">
        <button onclick="doGit(false)">Commit selected</button>
        <label class="mut"><input type="checkbox" id="incl_heavy"> include video.bin/tlog (heavy)</label>
        <label class="mut"><input type="checkbox" id="do_push"> push to origin</label>
        <label class="mut" style="margin-left:auto">sort <select id="sess_sort" onchange="renderSessions()">
          <option value="time">newest</option>
          <option value="gates">gates &#9660;</option>
          <option value="gates_asc">gates &#9650;</option>
          <option value="dur">longest</option>
          <option value="state">state</option>
        </select></label>
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
      <div class="bar"><span id="vidstatus" class="mut">pick a flight's 🎬 (Pilots or Logs→git) to render its onboard video + detector overlay</span>
        <button class="sec" id="rerender" style="margin-left:auto;display:none" title="Force a fresh render of this session, discarding the cached mp4. Renders are cached, and auto-refresh only when render_yolo.py itself is newer than the file." onclick="renderVid(CUR_VID,true)">↻ re-render</button></div>
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
let BUILD=null;
const $=id=>document.getElementById(id);
async function boot(){
  const r=await (await fetch('/api/schema')).json();
  SCHEMA=r.schema; GROUPS=r.groups; ASSETS=r.assets; MODEL_DEFAULTS=r.model_defaults||{};
  BUILD=r.build||null;
  renderForm();
  // APPLY THE SELECTED MODEL'S RECIPE ON LOAD. renderForm() paints SCHEMA defaults, and the recipe
  // only ever fired from the dropdown's 'change' event -- so once ego_ckpt's schema default became
  // the deploy lead (v18Qs1), picking it was a NO-OP and the recipe never applied. A fresh page then
  // showed v18Qs1 sitting next to seeker_weights=partial_m (M), emit=pnp and merge=0: a plausible-
  // looking form that is NOT the shipped recipe. That is exactly "reloaded but it did not
  // automatically load the correct vision engine". A page load now produces the same state as a
  // deliberate pick.
  await onCkptChange();
  preview(); loadPilots(); loadSessions();
  setInterval(loadPilots,2000);
  setInterval(checkBuild,5000);
}
// STALE-PAGE GUARD. MODEL_DEFAULTS is captured at page load, so an open tab keeps applying the OLD
// recipe after a panel restart -- silently, because the form looks normal. That shipped two bad
// batches (a 2-generation-stale actor, then M-with-merge-off while the server had M+1 pinned).
// Poll the build id and refuse to be quiet about it.
async function checkBuild(){
  try{
    const r=await (await fetch('/api/schema')).json();
    if(BUILD && r.build && r.build!==BUILD){
      const b=$('stalebar');
      b.style.display='';
      b.innerHTML='⚠ THIS PAGE IS STALE — the panel restarted with a different recipe. '
        +'Model-picks here apply the OLD settings. <button onclick="location.reload(true)" '
        +'style="margin-left:8px">Reload now</button>';
    }
  }catch(e){}
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
  if(d){
    // WP6a: RESET every recipe-managed knob to its schema default FIRST, so a knob edited for a PRIOR
    // model can't RIDE into this one (the 2026-07-19 z-bias/yaw ride-in that corrupted a v16 batch).
    // Then apply the picked recipe. Mirrors the server repin_recipe() pure function. (label excluded.)
    const managed=new Set();
    for(const m in MODEL_DEFAULTS){for(const k in MODEL_DEFAULTS[m])if(k!=='label')managed.add(k);}
    SCHEMA.forEach(s=>{ if(!managed.has(s.key))return; const f=$('f_'+s.key); if(!f)return;
      if(f.type==='checkbox')f.checked=!!s.default; else f.value=(s.default==null?'':s.default);});
    for(const k in d){const f=$('f_'+k); if(!f)continue;
      if(f.type==='checkbox')f.checked=!!d[k]; else f.value=d[k];
      applied.push(k.replace(/^ego_/,'').replace(/_/g,' ')+'='+d[k]);}
  }
  await preview();
  if(applied.length){const w=$('lwarn').textContent;
    $('lwarn').textContent='✓ '+base+' defaults ('+applied.join(', ')+')'+(w?'  •  '+w:'');}
}
function loadRecord(){
  // THE RECORD: flight 20260721_040555_v16pick_Qs1_f1 -- gate_index 9, 2 collisions, CRASH at 20.6 s.
  // Transcribed from that flight's own stored launch config (tools/pilot_panel_logs/pilots.json), not
  // from a recipe -- it is a DRIFTED v16 run and the drift is the point: z-bias 0.4, pitch clamp 20,
  // stale 0.6, assist 1.1 all differ from _V16_RECIPE, so loading this WILL raise the recipe-drift
  // warning (it named those same four keys in the flight's meta.json). Superseded the old 4-gate
  // vpeffs0/clamp-10 config on 2026-07-21; clamp 10 sits BELOW the -17.8 deg pad tilt, the opposite
  // fence regime from this flight's clamp 20, and it carried the older negreal42 engine.
  const rec={ego_ckpt:'ckpts/v16Qs1_final_actor.pth',seeker_detector:'yolo',
    seeker_weights:'C:/Users/Shadow/Peregrine/models/vq2_partial_m_2026-07-06_fp16_384x640.engine',
    video_dedup_fastpath:true,
    ego_sector_mode:'map',ego_coarse_map:'configs/vq2_coarse_map.json',ego_slot1:true,
    // BOTH range caps pinned: 22 is what this flight actually flew, back when it was hard-coded.
    ego_max_valid_range:30,ego_max_acquire_range:22,
    ego_gate_z_bias:0.4,ego_det_hold:0.2,ego_stale_horizon:0.6,ego_obs_coast:false,ego_kp_persist:0,
    ego_pitch_clamp:20,ego_roll_clamp:0,ego_floor_clamp:'',ego_speed_gov:'',ego_yaw_clamp:0.7,
    ego_rate_scale:1,ego_takeoff_assist:true,ego_assist_thrust:1.1,ego_assist_max_s:1,
    ego_arrestor:false,ego_arrest_after_gates:1,ego_arrest_speed_hi:0,ego_arrest_max_s:4,
    virtual_flip:true,yaw_scale:1,max_rate:0,rate:40,max_seconds:120,flights:1,
    deploy_profile:'vq2_case_c',spin_rate_abort:6,spin_time_abort:2,label:'record9_cfg'};
  // Same anti-ride-in reset as a model pick (WP6a): clear every recipe-managed knob to its schema
  // default first, so a knob this dict does not name cannot survive from the previous selection.
  const managed=new Set();
  for(const m in MODEL_DEFAULTS){for(const k in MODEL_DEFAULTS[m])if(k!=='label')managed.add(k);}
  SCHEMA.forEach(s=>{ if(!managed.has(s.key)||(s.key in rec))return; const f=$('f_'+s.key); if(!f)return;
    if(f.type==='checkbox')f.checked=!!s.default; else f.value=(s.default==null?'':s.default);});
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
var CUR_VID='';
async function renderVid(session,force){
  if(!session||session=='—'){alert('no recorded session for this flight yet');return;}
  CUR_VID=session; $('rerender').style.display='';
  tab('video'); const st=$('vidstatus'); $('player').removeAttribute('src'); $('player').load();
  st.innerHTML=(force?'re-rendering ':'rendering ')+'<b>'+session+'</b> … (detector runs on every frame, ~20-40s)';
  await fetch('/api/render',{method:'POST',body:JSON.stringify({session,force:!!force})});
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
let SESSIONS=[];
async function loadSessions(){
  const r=await (await fetch('/api/sessions')).json();
  SESSIONS=r.sessions||[];
  renderSessions();
}
function renderSessions(){
  const mode=($('sess_sort')||{}).value||'time';
  const checked=new Set([...document.querySelectorAll('.sc:checked')].map(c=>c.value));
  const g=s=>(s.gates==null?-1:s.gates), nm=(a,b)=>(a.name<b.name?1:-1);  // name desc = newest first
  const arr=SESSIONS.slice();
  if(mode==='gates')          arr.sort((a,b)=> g(b)-g(a) || nm(a,b));
  else if(mode==='gates_asc') arr.sort((a,b)=> g(a)-g(b) || nm(a,b));
  else if(mode==='dur')       arr.sort((a,b)=> (b.dur||0)-(a.dur||0) || nm(a,b));
  else if(mode==='state')     arr.sort((a,b)=> String(a.state).localeCompare(String(b.state)) || nm(a,b));
  else                        arr.sort(nm);
  $('sesslist').innerHTML=arr.map(s=>{
    const st=s.state||'—';
    const scfg=PBSESS[s.name]?`<button class="sec" style="padding:1px 7px;margin-left:auto" title="load this flight's config into the launch form" onclick="loadConfigSess('${s.name}')">⚙ cfg</button>`:'';
    const svid=s.has_video?`<button class="sec" style="padding:1px 7px;${scfg?'':'margin-left:auto'}" onclick="renderVid('${s.name}')">🎬 render</button>`:'';
    return `<div class="sess"><input type="checkbox" class="sc" value="${s.name}">
      <span class="pill s-${st}">${st}</span>
      <span>${s.name}</span>
      <small class="k">gates=${s.gates==null?'—':s.gates} · ${s.dur||'?'}s</small>
      ${scfg}${svid}</div>`;
  }).join('')||'<div class="mut">no sessions in data/runs</div>';
  document.querySelectorAll('.sc').forEach(c=>{if(checked.has(c.value))c.checked=true;});
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
