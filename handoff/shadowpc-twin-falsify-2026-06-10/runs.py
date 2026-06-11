"""Shared loader + run registry for the TWIN-FALSIFY campaign analysis (2026-06-10/11).

Loads rate_sysid commands.jsonl (the 100 Hz control-loop log: cmd, thrust, reported rpy,
world pos/vel, EMA'd quat-FD body rate, per-phase labels) into per-run numpy bundles,
deduped to ODOMETRY updates (rows repeat the latest state between odo messages -- finite
differencing the raw rows would divide by zero sim-time steps).

Conventions: world NED, reported attitude frame (telemetry conventions). The PHYSICAL
attitude = reported * odo_att_report_sign = (-roll, pitch, yaw) -- use `phys_rpy` for any
thrust-vector geometry (twin_fit._ODO_ATT_REPORT_SIGN, the Gate-0 saga).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = _REPO / "data" / "runs"

# label -> run dir (all 2026-06-11; sim build 1.0.3364, single clean instance after the
# dual-instance UDP collision was found + killed -- see WRITEUP "sim ops" section)
RUNS = {
    # P6 determinism + P5 rate-sensitivity baseline (fixmnvr @100 Hz)
    "det1": "20260611_042919_det1", "det2": "20260611_042959_det2",
    "det3": "20260611_043026_det3", "det4": "20260611_043052_det4",
    "det5": "20260611_043118_det5",
    # P5 rate sensitivity (same profile, 50/200 Hz)
    "fix50a": "20260611_065602_fix50a", "fix50b": "20260611_065628_fix50b",
    "fix200a": "20260611_065655_fix200a", "fix200b": "20260611_065721_fix200b",
    # P1 drag matrix (recon + tilt runs; back = body -x, fwd = body +x, lat = body +-y)
    "recon_lat12": "20260611_043203_recon_lat12",
    "drag_back08": "20260611_043234_drag_back08",
    "drag_back17": "20260611_064109_drag_back17",
    "drag_back25": "20260611_064548_drag_back25",
    "drag_back32": "20260611_064739_drag_back32",
    "drag_fwd17": "20260611_064213_drag_fwd17",
    "drag_fwd25": "20260611_064659_drag_fwd25",
    "drag_lat17p": "20260611_064443_drag_lat17p",
    "drag_lat17n": "20260611_064515_drag_lat17n",
    "drag_lat25p": "20260611_064619_drag_lat25p",
    # P1/P3 vertical
    "vert_mid": "20260611_064809_vert_mid",
    "vert_high": "20260611_064849_vert_high",   # aborted at the vz-17 guard (climb data valid)
    # P3 collective
    "coll_hover": "20260611_064946_coll_hover",
    "coll_speed": "20260611_065112_coll_speed2",
    # P2 rate at airspeed
    "rspd_r03": "20260611_065153_rspd_r03", "rspd_r10": "20260611_065223_rspd_r10",
    "rspd_p03": "20260611_065252_rspd_p03", "rspd_p10": "20260611_065321_rspd_p10",
    "rspd_r314": "20260611_065405_rspd_r314", "rspd_p314": "20260611_065433_rspd_p314",
    # P7 mixed
    "turn20": "20260611_065502_turn20",
    # P4 drift (filled in after the background run lands)
    "drift320": "20260611_065756_drift320",
}

G = 9.80665
ATT_SIGN = np.array([-1.0, 1.0, 1.0])     # reported -> physical attitude (twin_fit saga value)


@dataclass
class Run:
    label: str
    t: np.ndarray          # (N,) sim time s, strictly increasing (odo-deduped)
    pos: np.ndarray        # (N,3) world NED
    vel: np.ndarray        # (N,3) world NED
    rpy: np.ndarray        # (N,3) REPORTED attitude
    cmd: np.ndarray        # (N,3) commanded wire body rate
    thr: np.ndarray        # (N,) commanded collective
    rate: np.ndarray       # (N,3) EMA'd quat-FD body rate -- WARNING: magnitude biased HIGH
                           # by LPN/ODO sim_time stagger (dt under-measured), loop-rate
                           # dependent (nh50/nh100/nh200 probes). Use odo_rate for DC reads.
    odo_rate: np.ndarray   # (N,3) RAW ODOMETRY rate (sign-inverted roll+pitch; magnitude GOOD
                           # -- matches the euler-slope ground truth at all loop rates)
    phase: list            # (N,) phase label
    kind: list             # (N,)
    alt_i: np.ndarray      # (N,) alt-hold integrator
    meta: dict

    @property
    def phys_rpy(self) -> np.ndarray:
        return self.rpy * ATT_SIGN

    def seg(self, name: str) -> np.ndarray:
        """Boolean mask for a phase (prefix match: 'coast' hits 'coast', 'c034' etc. exact)."""
        return np.array([p == name for p in self.phase])


def load(label: str) -> Run:
    d = RUNS_DIR / RUNS[label]
    rows = [json.loads(l) for l in (d / "commands.jsonl").read_text().splitlines()]
    meta = json.loads((d / "meta.json").read_text())
    # dedupe on sim_time (keep the LAST row per stamp: it carries the freshest command context)
    by_t: dict[int, dict] = {}
    for r in rows:
        by_t[r["sim_time_ns"]] = r
    rs = [by_t[k] for k in sorted(by_t)]
    return Run(
        label=label,
        t=np.array([r["sim_time_ns"] for r in rs], dtype=np.float64) * 1e-9,
        pos=np.array([r["pos"] for r in rs], dtype=np.float64),
        vel=np.array([r["vel"] for r in rs], dtype=np.float64),
        rpy=np.array([r["rpy"] for r in rs], dtype=np.float64),
        cmd=np.array([r["cmd"] for r in rs], dtype=np.float64),
        thr=np.array([r["thrust"] for r in rs], dtype=np.float64),
        rate=np.array([r["true_rate"] for r in rs], dtype=np.float64),
        odo_rate=np.array([r["meas_rate"] for r in rs], dtype=np.float64),
        phase=[r["phase"] for r in rs],
        kind=[r["kind"] for r in rs],
        alt_i=np.array([r.get("alt_i", 0.0) for r in rs], dtype=np.float64),
        meta=meta,
    )


def accel(run: Run, smooth_n: int = 9) -> np.ndarray:
    """(N,3) world acceleration: central finite-diff of velocity over sim time, boxcar-smoothed
    (default 9 odo samples ~ 90-120 ms). Ends use one-sided diffs."""
    v, t = run.vel, run.t
    a = np.gradient(v, t, axis=0)
    if smooth_n > 1:
        k = np.ones(smooth_n) / smooth_n
        a = np.column_stack([np.convolve(a[:, i], k, mode="same") for i in range(3)])
    return a


def thrust_world(run: Run, hover: float = 0.2656) -> np.ndarray:
    """(N,3) world-frame specific force of the rotor thrust under the TWIN's collective map
    a_up = g*thr/hover, directed along the PHYSICAL body -z. The drag fit subtracts this and
    gravity; whatever remains is 'aero'. (The collective map itself is probed separately --
    the vertical fits return the map empirically; horizontal projections only need tilt,
    which is small * the map error -> second-order.)"""
    from scipy.spatial.transform import Rotation
    pr = run.phys_rpy
    R = Rotation.from_euler("ZYX", pr[:, [2, 1, 0]]).as_matrix()    # body->world
    a_up = G * run.thr / hover
    return np.einsum("nij,nj->ni", R, np.column_stack([np.zeros_like(a_up),
                                                       np.zeros_like(a_up), -a_up]))


def aero_resid(run: Run, hover: float = 0.2656, smooth_n: int = 9) -> np.ndarray:
    """(N,3) residual acceleration = measured - thrust(model) - gravity = aero force/mass."""
    g_vec = np.array([0.0, 0.0, G])
    return accel(run, smooth_n) - thrust_world(run, hover) - g_vec
