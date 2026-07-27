"""THE CLASSIFIER.  One primary failure mode + confidence per session, from the VG instrument.

Thresholds are CALIBRATED against 1151 confirmed gate PASSES (true |miss| < 0.75 m by construction
-- gate contact invalidates a run), so every threshold carries a measured false-positive rate:

  tier gate rms_r<=0.8 & s_end<=3.0   (62.7% of crash events, 51% of passes)
      |y_v| > 0.75   -> FP  6.3%     VERTICAL call
      y_h   > 1.5    -> FP  2.9%     LATERAL call, CONSERVATIVE
      y_h   > 1.0    -> FP 14.8%     LATERAL call, NOMINAL      (reported as a bracket)

Everything below reads vision + gyro only.  obs[0:3] is never used as a motion reference.
"""
import os
import pickle
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import terminal
import vg

# ---- calibrated constants ----
GATE_HALF = 0.75
YV_STRIKE = 0.75          # FP 6.3% on confirmed passes
YH_CONS = 1.50            # FP 2.9%
YH_NOM = 1.00             # FP 14.8%
FAR_M = 3.5               # died this far (or more) from the gate at the log end -> mid-leg death
LAUNCH_T = 2.5
OBSTACLE_GATE = 5         # the leg after log-gate-4; confirmed by the pilot's own aim_off usage

MODES = ["launch_dive", "vertical_undershoot", "vertical_overshoot", "lateral_diverge",
         "corner_both", "gate_blackout", "obstacle_leg", "at_gate_axis_undetermined",
         "other", "no_crash"]


def analyse(r):
    """Full per-session analysis: returns the feature dict used by classify()."""
    recs, seek, timing, meta = vg.load_session(r["path"])
    d = vg.build(recs, seek, meta)
    t = d["t"]
    n = len(t)
    ao = d["aim_off"]
    aim = (np.abs(ao[:, 0]) > 1e-6) | (np.abs(ao[:, 1]) > 1e-6)
    F = dict(run=r["run"], lineage=r["lineage"], ckpt=r["ckpt"], label=r["label"],
             final_state=r["final_state"], collisions=r["collisions"],
             t_end=float(t[-1]), gi_end=int(d["gi"][-1]), max_gate=int(d["gi"].max()),
             n_ticks=n, aim_ticks=int(aim.sum()), aim_gates=sorted({int(g) for g in d["gi"][aim]}),
             has_seeker=bool(seek), zbias=d["zbias"])
    g = int(d["gi"][-1])
    sel = terminal.approach_sel(d, g, n - 1)
    tg = None
    if len(sel) >= 5:
        try:
            tg = terminal.terminal_geometry(d, sel, n - 1)
        except Exception:
            tg = None
    F["fit_ok"] = int(tg is not None)
    if tg is not None:
        for k in ("s_end", "d_end", "y_h", "y_v", "y_h_now", "y_v_now", "ydot_h", "ydot_v",
                  "speed", "closure", "rms_r", "rms_t", "t_plane", "t_used", "capped",
                  "dt_blind", "extrap_m", "last_rho", "n_near", "n_axis"):
            F[k] = tg[k]
        # distance from the gate AT THE LOG END, from range + robust closure (most robust form)
        F["d_death"] = float(tg["last_rho"] - tg["closure"] * tg["dt_blind"])
        F["s_death"] = float(tg["s_end"] - tg["closure"] * tg["dt_blind"])
        # vertical velocity of the drone (VG), leveled frame
        F["vz"] = float(tg["v_end"][2])
    # ---- supply / thrash in the last second ----
    w1 = t >= t[-1] - 1.0
    F["blind_frac_1s"] = float((~d["seen"][w1]).mean())
    F["freshfrac_1s"] = float(d["fresh"][w1].mean())
    if seek:
        F["vempty_frac_1s"] = float((d["reason"][w1] == "valid_poses_empty").mean())
        F["creject_frac_1s"] = float((d["reason"][w1] == "continuity_reject").mean())
    else:
        F["vempty_frac_1s"] = F["creject_frac_1s"] = np.nan
    gap, best = 0.0, 0.0
    for i in np.where(t >= t[-1] - 2.0)[0]:
        if d["fresh"][i]:
            gap = 0.0
        else:
            gap += (t[i] - t[i - 1]) if i > 0 else 0.0
            best = max(best, gap)
    F["max_fix_gap"] = float(best)
    rr = d["rate_frd"][w1, 0]
    rr = rr[np.isfinite(rr)]
    F["roll_flips_1s"] = int(np.sum(np.diff(np.sign(rr[np.abs(rr) > 0.15])) != 0)) if rr.size > 2 else 0
    F["roll_max_1s"] = float(np.max(np.abs(rr))) if rr.size else np.nan
    gy = d["w_flu"][w1]
    F["gyro_roll_p90"] = float(np.percentile(np.abs(gy[:, 0]), 90)) if gy.size else np.nan
    F["gyro_roll_flips"] = int(np.sum(np.diff(np.sign(gy[np.abs(gy[:, 0]) > 1.0, 0])) != 0)) if gy.shape[0] > 2 else 0
    F["roll_deg_max_1s"] = float(np.nanmax(np.abs(np.degrees(d["roll"][w1]))))
    # ---- launch window ----
    e = t <= 1.2
    F["early_pitchcmd_min"] = float(np.nanmin(d["rate_frd"][e, 1])) if e.sum() else np.nan
    F["early_pitch_fenced"] = float(np.mean(np.abs(d["rate_frd"][e, 1]) < 1e-6)) if e.sum() else np.nan
    F["early_thrust_med"] = (float(np.nanmedian(d["collective"][e]))
                             if np.isfinite(d["collective"][e]).any() else np.nan)
    # VG vertical motion over the launch window (fresh fixes only)
    fe = np.where(d["fresh"] & e & ~aim)[0]
    F["launch_vz"] = np.nan
    if len(fe) >= 6:
        try:
            f = vg.vg_fit(d, fe)
            if f is not None:
                F["launch_vz"] = float(f["v_end"][2])
                F["launch_dz"] = float(f["p_end"][2] - f["p_raw"][0][2])
        except Exception:
            pass
    return F


def tier_of(F):
    if not F.get("fit_ok"):
        return "none"
    if F["rms_r"] <= 0.35 and F["s_end"] <= 2.0 and F["extrap_m"] <= 0.6:
        return "high"
    if F["rms_r"] <= 0.8 and F["s_end"] <= 3.0:
        return "med"
    return "low"


BLIND_FRAC = 0.70    # 0.0% of 291 confirmed gate passes reach this (healthy p99 = 0.61)
FIX_GAP = 0.60       # 0.0% of 291 confirmed gate passes reach this (healthy p99 = 0.54 s)


def classify(F):
    """Assign exactly ONE primary mode + confidence. Returns (mode, confidence, why, alt)."""
    if F["final_state"] != "CRASH":
        return "no_crash", "n/a", f"final_state={F['final_state']}", ""
    # ---- 0. pilot aim_off probe on the fatal gate: the geometry is deliberately offset ----
    if F.get("aim_ticks", 0) > 0 and F["gi_end"] in (F.get("aim_gates") or []):
        return ("excluded_aim_probe", "n/a",
                f"pilot aim_off applied on the fatal leg (gate {F['gi_end']}, "
                f"{F['aim_ticks']} ticks) -- geometry deliberately offset, excluded from the census", "")
    # ---- 1. launch / release window ----
    if F["t_end"] <= LAUNCH_T and F["max_gate"] == 0:
        c = "high" if F["t_end"] <= 2.2 else "med"
        return "launch_dive", c, f"died t={F['t_end']:.2f}s having never passed gate 0", ""
    tier = tier_of(F)
    dd = F.get("d_death", np.nan)
    sd = F.get("s_death", np.nan)
    far = (np.isfinite(dd) and dd > FAR_M) or (np.isfinite(sd) and sd > FAR_M)
    blind = F.get("blind_frac_1s", np.nan)
    gapb = F.get("max_fix_gap", np.nan)
    blackout = (np.isfinite(blind) and blind >= BLIND_FRAC) or (np.isfinite(gapb) and gapb >= FIX_GAP)
    if not F.get("fit_ok"):
        if blackout:
            return ("gate_blackout", "med",
                    "no fittable fresh fix in the final approach; "
                    f"blind {100*blind:.0f}% of the last second, max fix gap {gapb:.2f} s", "")
        return "other", "low", "no fittable terminal vision fix", ""
    # ---- 2. died mid-leg ----
    if far:
        if F["gi_end"] == OBSTACLE_GATE and not blackout:
            return ("obstacle_leg", "med",
                    f"died {dd:.1f} m short of gate 5 on the post-gate-4 leg with vision healthy "
                    f"(blind {100*blind:.0f}%, max gap {gapb:.2f} s) -- the known invisible low obstacle", "other")
        if blackout:
            return ("gate_blackout", "high" if (blind >= 0.85 or gapb >= 1.0) else "med",
                    f"died {dd:.1f} m from gate {F['gi_end']}; blind {100*blind:.0f}% of the last second, "
                    f"max fix gap {gapb:.2f} s (0% of healthy passes reach either threshold)",
                    "obstacle_leg" if F["gi_end"] == OBSTACLE_GATE else "")
        return ("other", "low",
                f"died {dd:.1f} m from gate {F['gi_end']} with vision HEALTHY "
                f"(blind {100*blind:.0f}%, max gap {gapb:.2f} s) -- mid-leg collision, not a blackout "
                f"and not a gate strike", "obstacle_leg" if F["gi_end"] == OBSTACLE_GATE else "")
    # ---- 3. at-gate ----
    if tier in ("low", "none"):
        return ("at_gate_axis_undetermined", "low",
                f"at-gate death, terminal geometry unresolvable "
                f"(range-channel rms {F['rms_r']:.2f} m, log ends {F['s_end']:.2f} m short of the plane)", "")
    yv, yh = F["y_v"], F["y_h"]
    vert = abs(yv) > YV_STRIKE
    lat_c = yh > YH_CONS
    lat_n = yh > YH_NOM
    # ORDER MATTERS.  The vertical channel is the better-calibrated one (6.3% FP on confirmed
    # passes at 0.75 m vs 14.8% for the lateral channel at 1.0 m), so a vertical call is only
    # blocked by the CONSERVATIVE lateral threshold, never by the nominal one.
    if vert and not lat_c:
        m = "vertical_undershoot" if yv < 0 else "vertical_overshoot"
        return (m, "high" if (tier == "high" and abs(yv) > 1.0) else "med",
                f"ballistic plane crossing {abs(yv):.2f} m {'BELOW' if yv < 0 else 'ABOVE'} centre "
                f"(half-opening 0.75 m); lateral only {yh:.2f} m", "")
    if lat_c and not vert:
        return ("lateral_diverge", "high" if tier == "high" else "med",
                f"ballistic plane crossing {yh:.2f} m off the line (conservative threshold 1.50 m, "
                f"2.9% FP on confirmed passes); vertical only {yv:+.2f} m", "")
    if lat_c and vert:
        return ("corner_both", "med",
                f"both axes out: lateral {yh:.2f} m, vertical {yv:+.2f} m", "")
    # 1.0 < yh <= 1.5 is the FALSE-PRECISION band: 14.8% of CONFIRMED PASSES read this high.
    # Refuse to name a mode there; carry it as an explicit bracket instead.
    if lat_n:
        return ("at_gate_axis_undetermined", "low",
                f"lateral reads {yh:.2f} m -- inside the 1.0-1.5 m band where 14.8% of confirmed "
                f"PASSES also read; not separable from a clean crossing with this instrument",
                "lateral_diverge")
    return ("at_gate_axis_undetermined", "med" if tier == "high" else "low",
            f"reads INSIDE the opening (lateral {yh:.2f}, vertical {yv:+.2f}) yet the run crashed -- "
            f"the kill is inside the {F['s_end']:.2f} m / {F['s_end']/max(F['closure'],1e-6):.2f} s "
            f"the log never observes", "")


def main():
    rows = [r for r in pickle.load(open(os.path.join(HERE, "sessions.pkl"), "rb"))
            if (r.get("n_ticks") or 0) > 8 and not str(r.get("note", "")).startswith("ERR")]
    out = []
    for i, r in enumerate(rows):
        try:
            F = analyse(r)
        except Exception as ex:
            F = dict(run=r["run"], lineage=r["lineage"], final_state=r.get("final_state"),
                     fit_ok=0, note=f"ERR:{ex}", t_end=np.nan, max_gate=-1, gi_end=-1,
                     blind_frac_1s=np.nan, collisions=0, ckpt=r.get("ckpt"), label=r.get("label"))
        try:
            m, c, why, alt = classify(F)
        except Exception as ex:
            m, c, why, alt = "other", "low", f"classify error {ex}", ""
        F.update(mode=m, conf=c, why=why, alt=alt, tier=tier_of(F))
        out.append(F)
        if i % 120 == 0:
            print(f"  [{i}/{len(rows)}]", flush=True)
    pickle.dump(out, open(os.path.join(HERE, "census.pkl"), "wb"))
    print(f"classified {len(out)}")


if __name__ == "__main__":
    main()
