"""twin_fidelity_probe.py -- FALLBACK-TIER twin-fidelity probe for the VQ2 self-localizing wire.

Answers: "Is our diffaero twin (racer.rl_plant) faithful enough to VQ2's REAL dynamics to
train an RL controller against?"  T1 of the 2026-07-04 RL-controller dive.

CONTEXT (why this is the FALLBACK tier, not BEST):
  The VQ2 competitive wire (verified from the run pile) carries ONLY HIGHRES_IMU (accel+gyro,
  ~185 Hz) + ACTUATOR_OUTPUT_STATUS (all-zero, useless) + COLLISION.  There is NO
  ATTITUDE / ODOMETRY / LOCAL_POSITION_NED -> no ground-truth world position, velocity,
  OR attitude.  So the BEST-tier "commanded-input replay vs GT velocity" is impossible.
  We fall back to two ATTITUDE-FREE body-frame residuals that need no world-frame GT:

  (A) GYRO-RESPONSE RESIDUAL  (validates the inner rate loop -- the transfer-critical axis;
      the 2026-06-11 VQ1 transfer failed on unmodeled MIXER rate<->thrust coupling):
        The twin's rate loop maps a commanded body rate -> realized body rate via
        `gain * rate_sign * cmd_rate`, first-order lag tau, super-rate map, slew clip.
        On the wire we KNOW the realized body rate: it is the HIGHRES_IMU gyro (post the
        deploy-profile gyro_sign correction) == the true FRD omega.  We seed the twin's
        omega at tick k from the measured omega, drive the twin one control step with the
        ACTUAL sent command (body_rate_sent = 0.4 * body_rate_frd), and compare the twin's
        predicted omega at k+1 to the measured omega at k+1.  This is a ONE-STEP-AHEAD
        residual -- it never integrates attitude, so no AHRS contamination.  It directly
        tests rate_gain (the ~2.5x realization), rate_tau, and (with the map on) super-rate.

  (B) SPECIFIC-FORCE-MAGNITUDE RESIDUAL  (validates the collective->accel thrust map + hover):
        HIGHRES_IMU specific force |a_body| is attitude-INVARIANT (a rotation preserves norm).
        The twin's thrust model predicts an upward specific force a_up along body -Z from the
        commanded collective (interp on COLL_MAP_* or g*coll/hover).  At low speed drag is
        negligible, so |a_body| ~= a_up.  Comparing |a_body|_IMU to a_up(coll) tests the
        thrust map + hover point WITHOUT any attitude estimate.  (At speed, quad-drag adds a
        body-frame term we cannot separate attitude-free, so we bin by |cmd_rate| and restrict
        the headline check to near-hover / low-rate ticks.)

WIRE CONVENTIONS (from src/racer/deploy_profile.py + mavlink_client.py, live-confirmed 3379):
  * body_rate_sent = cmd_rate_scale * body_rate_frd,  cmd_rate_scale = 0.4  (in commands.jsonl)
  * realized_omega_FRD = -1 * raw_wire_gyro   (gyro_sign = (-1,-1,-1) applied before AHRS;
    the code's FRD omega is the sign-corrected gyro).  We reconstruct realized omega from the
    RAW HIGHRES_IMU gyro by applying gyro_sign.
  * accel_body (HIGHRES_IMU xacc/yacc/zacc) = specific force in body FRD (m/s^2); at a level
    hover it reads ~ (0, 0, -g) (thrust up = body -Z).

These sign/scale conventions live in the WIRE layer (mavlink_client / deploy_profile), OUTSIDE
the plant.  The plant is telemetry-free physics.  See the spec doc for where they belong in the
RL actuation path.

USAGE:
  .venv\\Scripts\\python.exe scripts\\vq2_loadday\\twin_fidelity_probe.py <run_dir> [<run_dir> ...]
  .venv\\Scripts\\python.exe scripts\\vq2_loadday\\twin_fidelity_probe.py --glob "data/runs/*_rl_s1_f1"
  add --json out.json to dump the machine-readable verdict.

Author: VQ2 RL-controller dive, 2026-07-04.  FALLBACK tier only (competitive wire).
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from racer.rl_plant import (  # noqa: E402
    ALPHA_MAX_RPS2_MEASURED, COLL_MAP_ACCEL_MEASURED, COLL_MAP_THR_MEASURED,
    MIXER_IDLE_MEASURED, MIXER_KAPPA_ERR_MEASURED, MIXER_KAPPA_HOLD_MEASURED,
    MIXER_ZETA_YAW_MEASURED, PlantParams, PlantState, SUPER_RATE_S_MEASURED,
    step as plant_step,
)

_G = 9.80665
_DT_CTRL = 1.0 / 30.0

# The deploy-profile wire conventions (vq2_case_c). These live OUTSIDE the plant.
CMD_RATE_SCALE = 0.4
GYRO_SIGN = np.array([-1.0, -1.0, -1.0])

# GO/MARGINAL/NO-GO thresholds (from twin_fidelity_probe_DESIGN.md §4, fallback tier).
GYRO_GO = 0.5          # rad/s p90 one-step gyro residual -> GO
GYRO_NOGO = 1.5        # > this -> NO-GO
SF_GO = 0.5            # m/s^2 median |a_body| - a_up residual (near-hover) -> GO
SF_NOGO = 1.5          # > this -> NO-GO


def _measured_plant() -> PlantParams:
    """The full sim-faithful measured plant (map ON): super-rate + quad-drag + convex coll map
    + mixer.  rate_sign = (1,1,1) = the TRUE live command sign (FRAME-AUDIT 2026-06-12); the
    trained-world (1,1,-1) is a telemetry artifact and must NOT be used against real wire data."""
    return PlantParams(
        rate_sign=np.array([1.0, 1.0, 1.0]),
        super_rate_s=SUPER_RATE_S_MEASURED,
        alpha_max_rps2=ALPHA_MAX_RPS2_MEASURED.copy(),
        linear_drag=0.0,
        quad_drag_c2=None,   # not needed for gyro loop or near-hover SF; body-frame drag can't be
                             # separated attitude-free at speed, so we bin it out instead.
        coll_map_thr=COLL_MAP_THR_MEASURED.copy(),
        coll_map_accel=COLL_MAP_ACCEL_MEASURED.copy(),
        mixer_idle=MIXER_IDLE_MEASURED, mixer_kappa_err=MIXER_KAPPA_ERR_MEASURED,
        mixer_kappa_hold=MIXER_KAPPA_HOLD_MEASURED, mixer_zeta_yaw=MIXER_ZETA_YAW_MEASURED,
    )


def _flat_plant() -> PlantParams:
    """The LEGACY flat-gain plant (map OFF): the config every shipped inc7/inc8 checkpoint was
    actually trained against (rate_gain flat 2.5, g*coll/hover thrust, no super-rate/mixer).
    We report BOTH so the spec can see whether the *trained-against* plant matches the wire,
    not just the best-available measured plant."""
    return PlantParams(rate_sign=np.array([1.0, 1.0, 1.0]))


def load_run(run: Path) -> dict | None:
    """Load commands.jsonl + the HIGHRES_IMU stream from mavlink.tlog, aligned on sim_time_ns.

    Returns per-control-tick arrays:
      t_ns        (N,)   sim_time_ns of the command tick
      rate_sent   (N,3)  body_rate_sent (FRD, post cmd_rate_scale) -- the plant's rate command
      coll        (N,)   commanded collective [0,1]
      omega_meas  (N,3)  realized body rate FRD = gyro_sign * raw_wire_gyro, sampled at t_ns
      sf_mag      (N,)   |a_body| specific-force magnitude (m/s^2), sampled at t_ns
      cmd_rate_scale (float)
    """
    cmd_f = run / "commands.jsonl"
    tlog_f = run / "mavlink.tlog"
    if not cmd_f.exists() or not tlog_f.exists():
        print(f"  !! missing commands.jsonl or mavlink.tlog in {run}", file=sys.stderr)
        return None

    rows = [json.loads(l) for l in open(cmd_f, encoding="utf-8")]
    rows = [r for r in rows if r.get("mode") == "BODY_RATE" and r.get("body_rate_sent") is not None]
    if len(rows) < 30:
        print(f"  !! <30 BODY_RATE commands in {run}", file=sys.stderr)
        return None
    t_ns = np.array([r["sim_time_ns"] for r in rows], dtype=np.int64)
    rate_sent = np.array([r["body_rate_sent"] for r in rows], dtype=np.float64)
    coll = np.array([r["thrust"] for r in rows], dtype=np.float64)
    crs = float(rows[0].get("cmd_rate_scale", CMD_RATE_SCALE))

    # -- parse HIGHRES_IMU (accel+gyro, ~185 Hz) from the tlog --
    from pymavlink import mavutil
    m = mavutil.mavlink_connection(str(tlog_f))
    imu_t, imu_gyro, imu_sf = [], [], []
    while True:
        msg = m.recv_match(type="HIGHRES_IMU", blocking=False)
        if msg is None:
            break
        if not hasattr(msg, "xgyro"):
            continue
        imu_t.append(int(msg.time_usec) * 1000)
        imu_gyro.append([msg.xgyro, msg.ygyro, msg.zgyro])
        a = np.array([msg.xacc, msg.yacc, msg.zacc], dtype=np.float64)
        imu_sf.append(float(np.linalg.norm(a)))
    if len(imu_t) < 100:
        print(f"  !! <100 HIGHRES_IMU samples in {run}", file=sys.stderr)
        return None
    imu_t = np.array(imu_t, dtype=np.int64)
    imu_gyro = np.array(imu_gyro, dtype=np.float64)
    imu_sf = np.array(imu_sf, dtype=np.float64)
    order = np.argsort(imu_t)
    imu_t, imu_gyro, imu_sf = imu_t[order], imu_gyro[order], imu_sf[order]

    # realized omega FRD = gyro_sign * raw wire gyro (the sign-corrected value the AHRS consumes)
    omega_wire = imu_gyro * GYRO_SIGN[None, :]

    # nearest-IMU-sample lookup for each control tick (both clocks are HIGHRES_IMU time_usec-based)
    idx = np.searchsorted(imu_t, t_ns)
    idx = np.clip(idx, 0, len(imu_t) - 1)
    # snap to the closer of idx-1 / idx
    lo = np.clip(idx - 1, 0, len(imu_t) - 1)
    take_lo = np.abs(imu_t[lo] - t_ns) < np.abs(imu_t[idx] - t_ns)
    nn = np.where(take_lo, lo, idx)
    align_err_ms = np.abs(imu_t[nn] - t_ns) / 1e6

    return dict(
        run=run.name,
        t_ns=t_ns,
        rate_sent=rate_sent,
        coll=coll,
        omega_meas=omega_wire[nn],
        sf_mag=imu_sf[nn],
        cmd_rate_scale=crs,
        align_err_ms=align_err_ms,
        n_imu=len(imu_t),
    )


def _pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def probe_gyro(d: dict, params: PlantParams) -> dict:
    """One-step-ahead gyro-response residual: seed twin omega = measured omega at k, drive the
    twin one control step with rate_sent[k], compare twin.omega to measured omega at k+1.

    The twin's rate loop is agnostic to attitude/pos/vel for the omega update (it only reads
    omega, cmd_rate, thrust for the mixer).  We seed a hover attitude (identity) -- irrelevant
    to the omega channel except via the mixer's collective mean, which we feed the real coll."""
    N = len(d["t_ns"])
    res = []
    keys_rate = []
    for k in range(N - 1):
        dt_ns = d["t_ns"][k + 1] - d["t_ns"][k]
        dt = dt_ns / 1e9
        if not (0.02 < dt < 0.10):      # a plausible single control tick
            continue
        om0 = d["omega_meas"][k]
        st = PlantState(
            pos=np.zeros(3), vel=np.zeros(3),
            quat=np.array([1.0, 0.0, 0.0, 0.0]),
            omega=om0.copy(), thrust=np.float64(d["coll"][k]),
        )
        act = np.concatenate([d["rate_sent"][k], [d["coll"][k]]])
        st1 = plant_step(st, act, dt, params)
        r = st1.omega - d["omega_meas"][k + 1]
        if np.all(np.isfinite(r)) and np.max(np.abs(r)) < 30:
            res.append(r)
            keys_rate.append(np.max(np.abs(d["rate_sent"][k])))
    res = np.array(res)
    keys_rate = np.array(keys_rate)
    if len(res) == 0:
        return {"n": 0}
    norm = np.linalg.norm(res, axis=1)
    out = {
        "n": int(len(res)),
        "per_axis_median_abs": [float(np.median(np.abs(res[:, i]))) for i in range(3)],
        "per_axis_p90_abs": [_pct(np.abs(res[:, i]), 90) for i in range(3)],
        "norm_p50": _pct(norm, 50),
        "norm_p90": _pct(norm, 90),
        "norm_p99": _pct(norm, 99),
    }
    # split by command activity: near-zero cmd (hover/settle) vs active maneuver
    active = keys_rate > 0.15
    if active.sum() >= 15:
        out["active_norm_p90"] = _pct(norm[active], 90)
        out["active_n"] = int(active.sum())
    if (~active).sum() >= 15:
        out["quiet_norm_p90"] = _pct(norm[~active], 90)
    return out


def probe_specific_force(d: dict, params: PlantParams) -> dict:
    """Specific-force-magnitude residual (attitude-free): |a_body|_IMU vs twin a_up(coll).

    a_up is the twin's upward specific force from the commanded collective (the thrust map).
    At low speed / near hover, |a_body| ~= a_up (drag negligible).  We report the residual on
    NEAR-HOVER, LOW-RATE ticks (where the attitude-free approximation holds) as the headline,
    and the full distribution for context."""
    from racer.rl_plant import _interp1d
    coll = d["coll"]
    if params.coll_map_thr is not None:
        a_up = _interp1d(coll, params.coll_map_thr, params.coll_map_accel)
    else:
        a_up = params.g * coll / params.hover_thrust
    resid = d["sf_mag"] - a_up          # measured minus modeled specific-force magnitude
    rate_mag = np.max(np.abs(d["rate_sent"]), axis=1)
    # NOTE (2026-07-04 finding): a low-body-rate tick does NOT imply near-level.  A drone in
    # steady forward flight (pursuit/cruise) has low body rate but ~18-30 deg tilt AND non-trivial
    # speed, so |a_body| there mixes thrust + drag + a tilted specific-force direction that we
    # cannot separate attitude-free.  Above the hover band this residual is CONFOUNDED (it is not a
    # clean twin thrust-map error).  The ONLY attitude-free-valid SF test is the HOVER BAND, where
    # the collective is ~hover, tilt is small, and |a_body| ~= a_up ~= g.  That is the headline.
    clean = rate_mag < 0.10
    out = {
        "n": int(len(resid)),
        "all_median": float(np.median(resid)),
        "all_p90_abs": _pct(np.abs(resid), 90),
    }
    if clean.sum() >= 15:
        rc = resid[clean]
        out["clean_n"] = int(clean.sum())
        out["clean_median"] = float(np.median(rc))
        out["clean_median_abs"] = float(np.median(np.abs(rc)))   # CONFOUNDED above hover -- context only
        out["clean_p90_abs"] = _pct(np.abs(rc), 90)
    # HOVER-BAND ticks (coll near hover_thrust AND low rate) -- the attitude-free-VALID test of the
    # hover point + thrust map at hover.  This is the SF metric the verdict uses.
    hover = (np.abs(coll - 0.2656) < 0.02) & (rate_mag < 0.10)
    if hover.sum() >= 10:
        out["hover_n"] = int(hover.sum())
        out["hover_sf_median"] = float(np.median(d["sf_mag"][hover]))
        out["hover_resid_median"] = float(np.median(resid[hover]))
        out["hover_resid_median_abs"] = float(np.median(np.abs(resid[hover])))
    return out


def verdict(gyro_p90: float, sf_med_abs: float) -> str:
    if not np.isfinite(gyro_p90) or not np.isfinite(sf_med_abs):
        return "INCONCLUSIVE"
    g = "GO" if gyro_p90 <= GYRO_GO else ("NO_GO" if gyro_p90 > GYRO_NOGO else "MARGINAL")
    s = "GO" if sf_med_abs <= SF_GO else ("NO_GO" if sf_med_abs > SF_NOGO else "MARGINAL")
    order = {"NO_GO": 0, "MARGINAL": 1, "GO": 2, "INCONCLUSIVE": 1}
    return min([g, s], key=lambda x: order[x])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run dirs (each with commands.jsonl + mavlink.tlog)")
    ap.add_argument("--glob", default=None, help="glob pattern for run dirs")
    ap.add_argument("--json", default=None, help="dump the aggregate verdict to this JSON file")
    ap.add_argument("--plant", choices=["measured", "flat", "both"], default="both")
    args = ap.parse_args()

    run_paths = [Path(r) for r in args.runs]
    if args.glob:
        run_paths += [Path(p) for p in sorted(globmod.glob(args.glob))]
    run_paths = [p for p in run_paths if p.is_dir()]
    if not run_paths:
        print("no run dirs given", file=sys.stderr)
        return 2

    plants = {}
    if args.plant in ("measured", "both"):
        plants["measured"] = _measured_plant()
    if args.plant in ("flat", "both"):
        plants["flat"] = _flat_plant()

    per_run = []
    agg = {pn: {"gyro_res": [], "gyro_rate_key": [], "sf_resid_clean": []} for pn in plants}

    for rp in run_paths:
        d = load_run(rp)
        if d is None:
            continue
        print(f"\n==== {d['run']}  ({len(d['t_ns'])} cmds, {d['n_imu']} IMU, "
              f"align p90 {_pct(d['align_err_ms'], 90):.1f} ms, crs={d['cmd_rate_scale']}) ====")
        rec = {"run": d["run"], "n_cmds": len(d["t_ns"])}
        for pn, P in plants.items():
            g = probe_gyro(d, P)
            s = probe_specific_force(d, P)
            v = verdict(g.get("norm_p90", float("nan")),
                        s.get("clean_median_abs", s.get("all_p90_abs", float("nan"))))
            rec[pn] = {"gyro": g, "sf": s, "verdict": v}
            gp90 = g.get("norm_p90", float("nan"))
            gact = g.get("active_norm_p90", float("nan"))
            print(f"  [{pn:8s}] gyro one-step resid: norm p50={g.get('norm_p50', float('nan')):.3f} "
                  f"p90={gp90:.3f} p99={g.get('norm_p99', float('nan')):.3f} rad/s"
                  f"  (active p90={gact:.3f}, n={g.get('n', 0)})")
            print(f"             per-axis p90 |resid| = "
                  f"{[round(x,3) for x in g.get('per_axis_p90_abs',[float('nan')]*3)]} rad/s")
            print(f"             SF |a_body|-a_up: clean median|.|="
                  f"{s.get('clean_median_abs', float('nan')):.3f} "
                  f"p90={s.get('clean_p90_abs', float('nan')):.3f} m/s^2 "
                  f"(clean n={s.get('clean_n',0)}); hover SF median="
                  f"{s.get('hover_sf_median', float('nan')):.3f} "
                  f"(resid {s.get('hover_resid_median', float('nan')):+.3f})")
            print(f"             VERDICT [{pn}]: {v}")
            # collect for aggregate (re-run to gather raw arrays cheaply from percentile-free path)
        per_run.append(rec)

    # aggregate: pool the per-run p90s (report the distribution of run-level p90s)
    print("\n\n================ AGGREGATE ================")
    aggout = {}
    for pn in plants:
        gp90s = [r[pn]["gyro"].get("norm_p90") for r in per_run if r[pn]["gyro"].get("n")]
        gacts = [r[pn]["gyro"].get("active_norm_p90") for r in per_run
                 if r[pn]["gyro"].get("active_norm_p90") is not None
                 and np.isfinite(r[pn]["gyro"].get("active_norm_p90", float("nan")))]
        sfs = [r[pn]["sf"].get("clean_median_abs") for r in per_run
               if r[pn]["sf"].get("clean_median_abs") is not None]
        hov = [r[pn]["sf"].get("hover_sf_median") for r in per_run
               if r[pn]["sf"].get("hover_sf_median") is not None]
        gp90s = [x for x in gp90s if x is not None and np.isfinite(x)]
        sfs = [x for x in sfs if x is not None and np.isfinite(x)]
        hov = [x for x in hov if x is not None and np.isfinite(x)]
        med_gyro_p90 = float(np.median(gp90s)) if gp90s else float("nan")
        med_active_p90 = float(np.median(gacts)) if gacts else float("nan")
        med_sf = float(np.median(sfs)) if sfs else float("nan")
        med_hover_sf = float(np.median(hov)) if hov else float("nan")
        v = verdict(med_gyro_p90, med_sf)
        aggout[pn] = {
            "n_runs": len(gp90s),
            "median_gyro_norm_p90_rps": med_gyro_p90,
            "median_active_gyro_p90_rps": med_active_p90,
            "median_sf_clean_resid_abs_ms2": med_sf,
            "median_hover_sf_ms2": med_hover_sf,
            "verdict": v,
        }
        print(f"[{pn:8s}]  runs={len(gp90s)}  "
              f"median run gyro-resid-norm-p90 = {med_gyro_p90:.3f} rad/s "
              f"(active {med_active_p90:.3f})  |  "
              f"median SF clean-resid |.| = {med_sf:.3f} m/s^2  "
              f"(hover |a_body| median {med_hover_sf:.3f}, g={_G})  ->  {v}")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"aggregate": aggout, "per_run": per_run,
             "thresholds": {"gyro_go": GYRO_GO, "gyro_nogo": GYRO_NOGO,
                            "sf_go": SF_GO, "sf_nogo": SF_NOGO}},
            indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
