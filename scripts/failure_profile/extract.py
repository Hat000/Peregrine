"""Corpus-wide feature extraction for the Peregrine failure-mode census.

One row per session.  All motion quantities come from the VG (vision+gyro) instrument -- NEVER
from obs[0:3] (measured slope -0.110 / corr -0.057 against vision-truth lateral motion).
"""
import os
import pickle
import sys
import traceback
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vg

SCR = (r"C:\Users\Fengy\AppData\Local\Temp\claude"
       r"\C--Users-Fengy-Downloads-Projects-Anduril--claude-worktrees-peregrine-rl-commander-74b0dd"
       r"\b85130f9-ac88-4db2-8c68-0e28b966cf80\scratchpad")
OUT = os.path.join(SCR, "profile")
ROOTS = [(r"C:\Users\Fengy\Downloads\Projects\wt-arrest\data\runs", "main"),
         (os.path.join(SCR, "new25", "data", "runs"), "new25"),
         (os.path.join(SCR, "v2f", "data", "runs"), "v2f"),
         (os.path.join(SCR, "case", "data", "runs"), "case"),
         (os.path.join(SCR, "gate1", "data", "runs"), "gate1")]

LINEAGE = [("v20", ("v20",)), ("v19", ("v19",)), ("v18", ("v18",)), ("v16", ("v16", "vn16")),
           ("v15", ("v15",)), ("vtrackA", ("vtrackA",)), ("vpef", ("vpef",)), ("v1", ("v1A", "v1R"))]


def lineage_of(ck):
    b = os.path.basename(str(ck or ""))
    for name, pats in LINEAGE:
        for p in pats:
            if b.startswith(p):
                return name
    return "other"


def pct(a, qq, default=np.nan):
    a = np.asarray([x for x in np.ravel(a) if np.isfinite(x)], dtype=float)
    return float(np.percentile(a, qq)) if a.size else default


def nanmin(a):
    a = np.asarray([x for x in np.ravel(a) if np.isfinite(x)], dtype=float)
    return float(a.min()) if a.size else np.nan


def nanmax(a):
    a = np.asarray([x for x in np.ravel(a) if np.isfinite(x)], dtype=float)
    return float(a.max()) if a.size else np.nan


def process(rd, corpus):
    import json
    recs, seek, timing, meta = vg.load_session(rd)
    run = os.path.basename(rd)
    row = dict(run=run, corpus=corpus, path=rd,
               ckpt=meta.get("ego_ckpt"), lineage=lineage_of(meta.get("ego_ckpt")),
               label=meta.get("label"), final_state=meta.get("final_state"),
               collisions=int(meta.get("collisions") or 0),
               duration_s=float(meta.get("duration_s") or np.nan),
               zbias=float(meta.get("ego_gate_z_bias") or 0.0),
               sector_mode=meta.get("ego_sector_mode"),
               obs_coast=bool(meta.get("ego_obs_coast")),
               pitch_clamp=float(meta.get("ego_pitch_clamp") or 0.0),
               has_seeker=bool(seek), n_ticks=len(recs), note="")
    rs = meta.get("race_status") or {}
    row["rs_gate"] = rs.get("active_gate_index")
    if len(recs) < 8:
        row["note"] = "too_short"
        return row
    d = vg.build(recs, seek, meta)
    row["rel_src"] = d.get("rel_src")
    t = d["t"]
    n = len(t)
    row["t_end"] = float(t[-1])
    row["gi_end"] = int(d["gi"][-1])
    row["max_gate"] = int(d["gi"].max())
    dts = np.diff(t)
    dts = dts[dts > 0]
    row["hz_med"] = float(1.0 / np.median(dts)) if dts.size else np.nan
    # ---- aim_off (manual pilot offset; EXCLUDE those ticks from motion analysis) ----
    ao = d["aim_off"]
    aim_mask = (np.abs(ao[:, 0]) > 1e-6) | (np.abs(ao[:, 1]) > 1e-6)
    row["aim_ticks"] = int(aim_mask.sum())
    row["aim_max_lat"] = float(np.max(np.abs(ao[:, 0]))) if n else 0.0
    row["aim_max_vert"] = float(np.max(np.abs(ao[:, 1]))) if n else 0.0
    row["aim_gates"] = sorted({int(gg) for gg in d["gi"][aim_mask]}) if aim_mask.any() else []
    row["aim_at_end"] = bool(aim_mask[-min(20, n):].any())

    fresh = d["fresh"].copy()
    fresh[aim_mask] = False
    row["n_fresh"] = int(fresh.sum())
    row["n_fresh_all"] = int(d["fresh"].sum())
    fi = np.where(fresh)[0]
    last_fresh = int(fi[-1]) if fi.size else -1
    row["dt_since_fresh"] = float(t[-1] - t[last_fresh]) if last_fresh >= 0 else np.nan

    # ---- terminal supply / thrash (seeker-free where seeker.jsonl is absent) ----
    w1 = t >= t[-1] - 1.0
    w05 = t >= t[-1] - 0.5
    row["blind_frac_1s"] = float((~d["seen"][w1]).mean())
    row["freshfrac_1s"] = float(d["fresh"][w1].mean())
    row["blind_frac_05s"] = float((~d["seen"][w05]).mean())
    if seek:
        row["vempty_frac_1s"] = float((d["reason"][w1] == "valid_poses_empty").mean())
        row["creject_frac_1s"] = float((d["reason"][w1] == "continuity_reject").mean())
    else:
        row["vempty_frac_1s"] = np.nan
        row["creject_frac_1s"] = np.nan
    # longest continuous no-fix gap inside the last 1.5 s
    w15 = np.where(t >= t[-1] - 1.5)[0]
    gap, best = 0.0, 0.0
    for i in w15:
        if d["fresh"][i]:
            gap = 0.0
        else:
            gap += (t[i] - t[i - 1]) if i > 0 else 0.0
            best = max(best, gap)
    row["max_fix_gap_15s"] = float(best)
    rr = d["rate_frd"][w1, 0]
    rr = rr[np.isfinite(rr)]
    row["roll_flips_1s"] = int(np.sum(np.diff(np.sign(rr[np.abs(rr) > 0.15])) != 0)) if rr.size > 2 else 0
    row["roll_absmean_1s"] = float(np.mean(np.abs(rr))) if rr.size else np.nan
    row["roll_max_1s"] = float(np.max(np.abs(rr))) if rr.size else np.nan
    gy = d["w_flu"][w1]
    grr = np.abs(gy[:, 0])
    row["gyro_roll_flips_1s"] = int(np.sum(np.diff(np.sign(gy[np.abs(gy[:, 0]) > 0.5, 0])) != 0)) if gy.shape[0] > 2 else 0
    row["gyro_roll_p90"] = pct(grr, 90)
    row["gyro_norm_max"] = nanmax(np.linalg.norm(gy, axis=1))
    row["thrust_med_1s"] = float(np.nanmedian(d["collective"][w1])) if np.isfinite(d["collective"][w1]).any() else np.nan
    row["roll_deg_max_1s"] = float(np.nanmax(np.abs(np.degrees(d["roll"][w1]))))
    pf = d["rate_frd"][w05, 1]
    row["pitch_fenced_frac"] = float(np.mean(np.abs(pf) < 1e-6)) if pf.size else np.nan
    row["pitchcmd_min_1s"] = nanmin(d["rate_frd"][w1, 1])
    # ---- launch window (first 1.5 s) ----
    e15 = t <= 1.5
    row["early_pitch_min"] = nanmin(d["rate_frd"][e15, 1])
    row["early_pitch_fenced"] = float(np.mean(np.abs(d["rate_frd"][e15, 1]) < 1e-6)) if e15.sum() else np.nan
    row["early_thrust_med"] = float(np.nanmedian(d["collective"][e15])) if np.isfinite(d["collective"][e15]).any() else np.nan

    # ---- VG terminal fit ----
    row["fit_ok"] = 0
    f = None
    for win in (0.55, 0.85, 1.3, 2.0):
        sel = np.where(fresh & (t >= t[last_fresh] - win))[0] if last_fresh >= 0 else np.array([], int)
        if len(sel) >= 5:
            f = vg.vg_fit(d, sel)
            if f is not None and np.isfinite(f["p_end"]).all() and np.isfinite(f["v_end"]).all():
                break
            f = None
    if f is not None:
        sp = float(np.linalg.norm(f["v_end"]))
        tstar, miss, mm = vg.closest_approach(f["p_end"], f["v_end"])
        vh = f["v_end"] / max(sp, 1e-9)
        perp = f["p_end"] - (f["p_end"] @ vh) * vh
        row.update(fit_ok=1, fit_n=len(f["idx"]), fit_win=win,
                   rms_radial=f["rms_radial"], rms_trans=f["rms_trans"],
                   p_end=f["p_end"].tolist(), v_end=f["v_end"].tolist(), speed_end=sp,
                   d_end=float(np.linalg.norm(f["p_end"])),
                   t_star=float(tstar), miss_vec=miss.tolist(),
                   miss_h=float(np.hypot(miss[0], miss[1])), miss_v=float(miss[2]),
                   miss_abs=float(mm),
                   s_end=float(-(f["p_end"] @ vh)),
                   obs_perp_h=float(np.hypot(perp[0], perp[1])), obs_perp_v=float(perp[2]),
                   raw_rho_flap_max=(float(np.max(np.abs(np.diff(f["rho"])))) if len(f["rho"]) > 1 else np.nan),
                   raw_rho_flap_p90=pct(np.abs(np.diff(f["rho"])), 90))
        row["blind_travel_m"] = float(sp * row["dt_since_fresh"]) if np.isfinite(row["dt_since_fresh"]) else np.nan
        # honest terminal-blindness flag: does the LOG reach the plane?
        row["reached_plane"] = int(row["s_end"] <= 0.35)
    # ---- VG smoothed shape over the FINAL approach segment (drone motion, not estimate) ----
    seg = np.where((d["gi"] == d["gi"][-1]))[0]
    segf = np.array([i for i in seg if fresh[i]], dtype=int)
    if len(segf) >= 8 and f is not None:
        sm = vg.vg_smooth(d, segf)
        if sm is not None:
            good = np.isfinite(sm["p_sm"]).all(axis=1)
            axis = np.asarray(row["v_end"], float)
            st_raw = vg.shape_stats(sm["p_raw"][good], axis)
            st_sm = vg.shape_stats(sm["p_sm"][good], axis)
            for kk, vv in st_raw.items():
                row[f"raw_{kk}"] = vv
            for kk, vv in st_sm.items():
                row[f"vg_{kk}"] = vv
            row["sm_rms_radial"] = float(np.sqrt(np.nanmean(sm["res_radial"][good] ** 2)))
            row["sm_rms_trans"] = float(np.sqrt(np.nanmean(sm["res_trans"][good] ** 2)))
            # outward transverse velocity at the end, against the terminal approach axis
            a = axis / max(float(np.linalg.norm(axis)), 1e-9)
            ps = sm["p_sm"][good]
            vs = sm["v_sm"][good]
            y = ps - (ps @ a)[:, None] * a
            ny = np.linalg.norm(y, axis=1)
            vt = np.einsum("ni,ni->n", vs, y / np.maximum(ny, 1e-9)[:, None])
            row["v_outward_end"] = float(vt[-1]) if len(vt) else np.nan
            row["v_outward_max"] = nanmax(vt[-min(8, len(vt)):])
            row["seg_lat_min"] = float(np.nanmin(ny))
            row["seg_lat_end"] = float(ny[-1])
    # ---- raw segment range floor ----
    rho_seg = d["rho"][seg][fresh[seg]]
    row["seg_rho_min"] = nanmin(rho_seg)
    row["seg_ticks"] = int(len(seg))
    row["seg_t"] = float(t[-1] - t[seg[0]]) if len(seg) else np.nan
    if last_fresh >= 0:
        i = last_fresh
        cr, sr = np.cos(d["roll"][i]), np.sin(d["roll"][i])
        cp, sp_ = np.cos(d["pitch"][i]), np.sin(d["pitch"][i])
        Rl = (np.array([[cp, 0, sp_], [0, 1, 0], [-sp_, 0, cp]])
              @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))
        rl = Rl @ d["rel"][i]
        rl[2] += d["zbias"] * vg.ZBIAS_COS
        row.update(last_rho=float(d["rho"][i]), last_fwd_lev=float(rl[0]),
                   last_lat_lev=float(rl[1]), last_up_lev=float(rl[2]),
                   last_ncorn=float(d["ncorn"][i]), last_rsrc=str(d["rsrc"][i]),
                   last_roll_deg=float(np.degrees(d["roll"][i])),
                   last_pitch_deg=float(np.degrees(d["pitch"][i])))
    return row


def main():
    rows = []
    seen = set()
    todo = []
    for R, corpus in ROOTS:
        if not os.path.isdir(R):
            continue
        for nme in sorted(os.listdir(R)):
            p = os.path.join(R, nme)
            if (os.path.isdir(p) and os.path.exists(os.path.join(p, "meta.json"))
                    and os.path.exists(os.path.join(p, "ego_obs.jsonl")) and nme not in seen):
                seen.add(nme)
                todo.append((p, corpus))
    print(f"{len(todo)} sessions", flush=True)
    for i, (p, corpus) in enumerate(todo):
        try:
            row = process(p, corpus)
        except Exception as e:
            row = dict(run=os.path.basename(p), corpus=corpus, path=p,
                       note=f"ERR:{type(e).__name__}:{e}")
            if i < 3:
                traceback.print_exc()
        rows.append(row)
        if i % 50 == 0:
            print(f"  [{i}/{len(todo)}] {os.path.basename(p)}", flush=True)
    with open(os.path.join(OUT, "sessions.pkl"), "wb") as fh:
        pickle.dump(rows, fh)
    errs = [r for r in rows if str(r.get("note", "")).startswith("ERR")]
    print(f"done: {len(rows)} rows, {len(errs)} errors")
    for r in errs[:10]:
        print("  ", r["run"], r["note"])


if __name__ == "__main__":
    main()
