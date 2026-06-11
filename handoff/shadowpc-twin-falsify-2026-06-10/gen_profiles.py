"""Generate the committed probe profiles for the TWIN-FALSIFY campaign (rate_sysid --mode profile).

Geometry (track_map_20260602_114630): gates X in [-159,-23], Y ~= 0, at/below start altitude;
start pose = origin, yaw 180 deg (FACING the gates at -X), reported resting pitch -17.8 deg.
Safe corridor: +X (behind the start) and +-Y, flown 6-10 m above start altitude.

Sign conventions (validated by the sweep's vel-damp mapping, reported frame):
  pitch_des +theta -> body -x accel; facing -X that is WORLD +X  (the "backward" drag runs --
                      no yaw maneuver needed, and they double as the body-frame asymmetry probe)
  after a -180 deg yaw ramp (facing +X), pitch_des -theta -> WORLD +X (true forward runs)
  roll_des  +theta -> body -y accel; facing -X that is WORLD +Y  (lateral runs)

Run:  .venv\\Scripts\\python.exe handoff\\shadowpc-twin-falsify-2026-06-10\\gen_profiles.py
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).parent / "profiles"
OUT.mkdir(exist_ok=True)

HOVER = 0.2656


def ph(name, dur, **kw):
    return {"name": name, "dur": dur, **kw}


def init(alt=0.0, dur=3.5):
    # 3.5 s: the -17.8 deg resting tilt kicks ~2.6 m/s of -X drift (smoke run fix100b2 reached
    # x=-8 with a 2 s init); the longer hold lets vel-damp + pos-pull re-center before climbing.
    return ph("init", dur, vel_damp=True, alt_target_m=alt)


def climb(alt, dur):
    return ph("climb", dur, vel_damp=True, alt_target_m=alt)


PROFILES: dict[str, list[dict]] = {}

# ---- P1 drag: backward runs (world +X, no yaw maneuver; body -x) --------------------
for theta, t_acc, t_coast in ((8, 4.0, 6.0), (17, 4.0, 6.0), (25, 3.5, 5.5), (32, 3.0, 5.0)):
    PROFILES[f"drag_back{theta:02d}"] = [
        init(), climb(-6.0, 5.0),
        ph("accel", t_acc, pitch_deg=+theta, alt_target_m=-6.0, tilt_ff=True),
        ph("coast", t_coast, alt_target_m=-6.0),
    ]

# ---- P1 drag: true forward runs (yaw ramp to face +X first; body +x) ----------------
for theta, t_acc, t_coast in ((17, 4.0, 6.0), (25, 3.5, 5.5)):
    # ramp 6 s + settle 2 s: the hold's yaw authority (~50 deg/s realized) lagged the original
    # 4 s/60 dps ramp by ~100 deg -- fwd17's accel phase had a rotating heading (data still
    # usable, world-frame fit; fwd25 starts the accel actually pointed +X).
    PROFILES[f"drag_fwd{theta:02d}"] = [
        init(), climb(-6.0, 5.0),
        ph("yawramp", 6.0, yaw_ramp_deg=-180.0, yaw_rate_dps=45.0, vel_damp=True, alt_target_m=-6.0),
        ph("settle", 2.0, vel_damp=True, alt_target_m=-6.0),
        ph("accel", t_acc, pitch_deg=-theta, alt_target_m=-6.0, tilt_ff=True),
        ph("coast", t_coast, alt_target_m=-6.0),
    ]

# ---- P1 drag: lateral runs (body +-y; world +-Y) ------------------------------------
for theta, sgn, t_acc, t_coast in ((17, +1, 4.0, 6.0), (17, -1, 4.0, 6.0), (25, +1, 3.5, 5.5)):
    tag = f"drag_lat{theta:02d}{'p' if sgn > 0 else 'n'}"
    PROFILES[tag] = [
        init(), climb(-6.0, 5.0),
        ph("accel", t_acc, roll_deg=sgn * theta, alt_target_m=-6.0, tilt_ff=True),
        ph("coast", t_coast, alt_target_m=-6.0),
    ]

# ---- P1 recon: cautious first lateral run (12 deg, short) ---------------------------
PROFILES["recon_lat12"] = [
    init(), climb(-6.0, 5.0),
    ph("accel", 3.0, roll_deg=12.0, alt_target_m=-6.0, tilt_ff=True),
    ph("coast", 5.0, alt_target_m=-6.0),
]

# ---- P1/P3 vertical: climb + descent a(vz) curves (CLI: --max-up-m 45 --max-vz 17) --
PROFILES["vert_mid"] = [
    init(), ph("hold", 1.0, vel_damp=True, alt_target_m=-2.0),
    ph("climb40", 2.5, thrust=0.40, vel_damp=True),
    ph("coast_v", 1.0, thrust=HOVER, vel_damp=True),
    ph("desc15", 4.0, thrust=0.15, vel_damp=True),
    ph("catch", 1.5, thrust=0.40, vel_damp=True),
]
PROFILES["vert_high"] = [
    init(), ph("hold", 1.0, vel_damp=True, alt_target_m=-2.0),
    ph("climb55", 1.4, thrust=0.55, vel_damp=True),
    ph("coast_v", 1.0, thrust=HOVER, vel_damp=True),
    ph("desc05", 2.5, thrust=0.05, vel_damp=True),
    ph("catch", 2.0, thrust=0.45, vel_damp=True),
]

# ---- P3 collective map at hover (from 10 m up; micro-steps + re-hover) --------------
_coll = [init(), climb(-10.0, 6.0)]
for lvl, dur in ((0.0, 0.5), (0.10, 0.6), (0.20, 0.8), (0.32, 0.8), (0.45, 0.5),
                 (0.60, 0.4), (0.80, 0.35), (1.00, 0.3)):
    _coll.append(ph(f"c{int(round(lvl*100)):03d}", dur, thrust=lvl, vel_damp=True))
    _coll.append(ph("rehov", 2.5, vel_damp=True, alt_target_m=-10.0))
PROFILES["coll_hover"] = _coll

# ---- P3 collective at forward speed (the cross-coupling probe) ----------------------
PROFILES["coll_speed"] = [
    init(), climb(-8.0, 6.0),
    ph("accel", 3.5, pitch_deg=+20.0, alt_target_m=-8.0, tilt_ff=True),
    ph("c034", 0.6, thrust=0.34), ph("chov", 0.8, thrust=HOVER),
    ph("c018", 0.6, thrust=0.18), ph("chov2", 0.8, thrust=HOVER),
    ph("c045", 0.5, thrust=0.45), ph("chov3", 1.0, thrust=HOVER),
]

# ---- P2 rate loop at airspeed (small/mid: held steps; 3.14: open-loop astep) --------
for ax, ai in (("roll", 0), ("pitch", 1)):
    for mag, dur in ((0.3, 0.8), (1.0, 0.8)):
        PROFILES[f"rspd_{ax[0]}{int(mag*10):02d}"] = [
            init(), climb(-8.0, 6.0),
            ph("accel", 3.5, pitch_deg=+25.0, alt_target_m=-8.0, tilt_ff=True),
            ph("relevel", 0.4, alt_target_m=-8.0),
            ph(f"{ax}{mag:+g}", dur, kind="step", axis=ax, value=mag, alt_target_m=-8.0),
            ph("catch", 1.5, vel_damp=True, alt_target_m=-8.0),
        ]
    vec = [0.0, 0.0, 0.0]
    vec[ai] = 3.14
    PROFILES[f"rspd_{ax[0]}314"] = [          # CLI: --max-tilt-deg 9999 (it will tumble)
        init(), climb(-10.0, 6.0),
        ph("accel", 3.5, pitch_deg=+25.0, alt_target_m=-10.0, tilt_ff=True),
        ph("relevel", 0.4, alt_target_m=-10.0),
        ph(f"a{ax[0]}314", 1.0, kind="astep", value=vec, thrust=HOVER),
    ]

# ---- P7 mixed-axis coordinated turn (2 turns, banked 20 deg) ------------------------
PROFILES["turn20"] = [
    init(), climb(-8.0, 6.0),
    ph("turn", 22.0, roll_deg=20.0, yaw_ramp_deg=-720.0, yaw_rate_dps=34.0,
       alt_target_m=-8.0, tilt_ff=True),
    ph("recover", 2.0, vel_damp=True, alt_target_m=-8.0),
]

# ---- P4 long-duration drift (CLI: --ki-alt 0.004 --max-seconds 380 --thrust 0.2656) -
PROFILES["drift320"] = [
    init(), ph("hover_drift", 320.0, vel_damp=True, alt_target_m=-5.0),
]

# ---- P5/P6 fixed maneuver for determinism (x5 @100 Hz) + rate sensitivity (50/200) --
PROFILES["fixmnvr"] = [
    init(), climb(-6.0, 4.0),
    ph("accel", 2.5, pitch_deg=+17.0, alt_target_m=-6.0, tilt_ff=True),
    ph("relevel", 0.5, alt_target_m=-6.0),
    # 0.4 s: realized ~2.77 rad/s -> ~63 deg excursion, just under the 70 deg abort
    # (0.8 s aborted the smoke run at 72 deg mid-step)
    ph("roll+1", 0.4, kind="step", axis="roll", value=1.0, alt_target_m=-6.0),
    ph("catch", 1.5, vel_damp=True, alt_target_m=-6.0),
]

for name, phases in PROFILES.items():
    (OUT / f"{name}.json").write_text(json.dumps({"phases": phases}, indent=1), encoding="utf-8")
print(f"wrote {len(PROFILES)} profiles -> {OUT}")
for n in sorted(PROFILES):
    print(" ", n)
