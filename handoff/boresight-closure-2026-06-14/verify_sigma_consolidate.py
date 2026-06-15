"""Consolidate the ADVERSARIAL verify_sigma evidence into verify_sigma_results.json:
  - Attack 1 CI   <- verify_sigma_cluster_boot_results.json (cluster vs iid bootstrap of the per-fix SD)
  - Attack 1/2    <- verify_sigma_ckpt.json (A1 upper-CI margin cells; A2 at-speed sweep + ceiling)
  - Attack 3      <- margin_rerun_results.json (published, IDENTICAL engine): aniso[.19/.10] vs iso0.10
                     post-bake att0 random3d ladder -> p99 over-optimism + first-closing fr gap.
[verify_sigma adversarial, 2026-06-14]
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
M030 = 0.235
M038 = 0.155
L3_LAT = 0.1914
L3_VERT = 0.1008


def a2_ceiling(a2):
    out = {}
    cfg = {}
    for c in a2:
        cfg.setdefault(f"{c['extra']}|v{int(c['v'])}|{c['bias_mode']}", []).append(c)
    for k, rows in cfg.items():
        rows = sorted(rows, key=lambda x: x["sigma_lat"])
        cl = [r["sigma_lat"] for r in rows if r["close030"]]
        op = [r["sigma_lat"] for r in rows if not r["close030"]]
        lc = max(cl) if cl else None
        fo = min([x for x in op if (lc is None or x > lc)], default=None)
        mid = round((lc + fo) / 2, 3) if (lc is not None and fo is not None) else None
        out[k] = {"bracket_close_open": [lc, fo], "ceiling_mid_sigma_lat": mid,
                  "ratio_over_L3_0.191": round(mid / L3_LAT, 2) if mid else None,
                  "headroom_m_over_L3": round(mid - L3_LAT, 3) if mid else None}
    return out


def main():
    cb = json.loads((HERE / "verify_sigma_cluster_boot_results.json").read_text())
    ck = json.loads((HERE / "verify_sigma_ckpt.json").read_text())["cells"]
    mr = json.loads((HERE / "margin_rerun_results.json").read_text())["cells_A"]

    a1 = sorted([c for c in ck if c["tag"] == "A1"], key=lambda x: (str(x["extra"]), x["v"], x["fr"]))
    a2 = sorted([c for c in ck if c["tag"] == "A2"], key=lambda x: (x["extra"], x["v"], x["sigma_lat"]))

    # ---- Attack 3 from published margin_rerun (post-bake, att0, random3d) ----
    def find(v, sig, fr):
        for c in mr:
            if (abs(c["v"] - v) < 1e-6 and c["sigma"] == sig and abs(c["fix_rate"] - fr) < 1e-6
                    and c["bias_mode"] == "random3d" and abs(c["att_bias_deg"]) < 1e-6 and c["bake"] == "post"):
                return c
        return None
    a3rows = []
    for v in (30.0, 37.0):
        for fr in (0.07, 0.15, 0.25, 0.35, 0.50):
            a = find(v, "aniso0.19/0.10", fr)
            i = find(v, "iso0.10", fr)
            if not a or not i:
                continue
            ac = bool(a["p90"] < M030 and a["p99"] < M030)
            ic = bool(i["p90"] < M030 and i["p99"] < M030)
            a3rows.append(dict(v=v, fr=fr,
                               honest_aniso_p90=round(a["p90"], 4), honest_aniso_p99=round(a["p99"], 4), honest_close030=ac,
                               naive_iso010_p90=round(i["p90"], 4), naive_iso010_p99=round(i["p99"], 4), naive_close030=ic,
                               p99_optimism_m=round(a["p99"] - i["p99"], 4),
                               p99_optimism_pct=round(100 * (a["p99"] - i["p99"]) / a["p99"], 1)))

    def first_fr(key, v):
        frs = [r["fr"] for r in a3rows if r["v"] == v and r[key]]
        return min(frs) if frs else None

    iso_hi90 = cb["pessimistic_upper_edge"]["iso_equiv_upper_ci90"]
    out = {
        "_meta": {"role": "ADVERSARIAL verifier -- refute sigma honesty + r=0.30 closure robustness",
                  "claim_under_test": "L3-measured gate-4 sigma (aniso [lat0.19,vert0.10]) is honest, and "
                                      "r=0.30 closure at fix-rate 0.35-0.50 is robust to the sigma uncertainty",
                  "engine": "margin_driver_v2 == ME.fly_lap to 1e-9 in iso/zerobias limit (verified this session)",
                  "margin_030": M030, "margin_038": M038, "L3_aniso_sigma": [L3_LAT, L3_VERT],
                  "regime": "COLD case-C, bias=0, latency 15ms, POST-bake, random3d"},

        "attack1_effectiveN_clusterCI": {
            "verdict": "claim SURVIVES (the within-batch per-fix sigma CI is genuinely tight; ICC~0)",
            "N": cb["_meta"]["N"], "n_windows": cb["_meta"]["n_windows"],
            "vert_point_sd": cb["vert"]["point_sd"], "vert_cluster_ci90": cb["vert"]["cluster_ci90"],
            "vert_iid_ci90": cb["vert"]["iid_ci90"], "vert_icc": cb["vert"]["icc_oneway"]["icc"],
            "lat_point_sd": cb["lat"]["point_sd"], "lat_cluster_ci90": cb["lat"]["cluster_ci90"],
            "lat_iid_ci90": cb["lat"]["iid_ci90"], "lat_icc": cb["lat"]["icc_oneway"]["icc"],
            "cluster_over_iid_width_vert": cb["vert"]["cluster_over_iid_width90"],
            "cluster_over_iid_width_lat": cb["lat"]["cluster_over_iid_width90"],
            "pessimistic_upper_edge": cb["pessimistic_upper_edge"],
            "note": "cluster bootstrap (resample 13 whole sessions) gives a CI ~= iid CI (ICC vert -0.03, "
                    "lat +0.03) -> the 81 fixes are ~independent; the published 'cluster narrower than iid' "
                    "is REAL (near-zero within-window correlation), NOT a bug. BUT 13 clusters is a small "
                    "cluster count and this CI only bounds the WITHIN-BATCH per-fix scatter at 18 m/s; it "
                    "does NOT bound the at-speed/regime extrapolation (Attack 2).",
            "margin_at_upperCI_sigma_r030": a1,
            "margin_finding": "at the cluster-upper-CI sigma, r=0.30 still CLOSES at fr=0.50 (all 4 "
                              "upper-CI sigma sets, both speeds). At fr=0.35 the upper-CI sigma sets FAIL "
                              "(honest aniso 0.210/0.117 and 0.214/0.120 both speeds; iso0.170 fails v30 "
                              "p99 0.238). => closure at fr=0.50 is robust to the within-batch sigma CI; "
                              "the fr=0.35 corner is NOT (it needs the central sigma)."},

        "attack2_atspeed_sigma_growth": {
            "verdict": "claim REFUTED for the at-speed extrapolation -- HEADROOM IS SMALL",
            "sweep": a2,
            "ceiling": a2_ceiling(a2),
            "ceiling_summary": {"held_vert0.10": "sigma_lat ceiling ~0.245 (x1.28 L3, +0.054 m) both 30 & 37 m/s",
                                "scaled_L3ratio": "sigma_lat ceiling ~0.23 (x1.20 L3, +0.039 m) both 30 & 37 m/s"},
            "blur_extrapolation_linear": {"L3_speed_mps": 18.0, "L3_sigma_lat": L3_LAT,
                                          "v30_pred": round(L3_LAT * 30 / 18, 3),
                                          "v37_pred": round(L3_LAT * 37 / 18, 3),
                                          "model": "pixel-blur ~ speed*exposure/range at fixed ~22m"},
            "finding": "r=0.30 closure (fr=0.50) breaks once per-fix sigma_lat exceeds ~0.23-0.245 m -- only "
                       "+0.04-0.05 m (x1.2-1.28) above the L3-measured 0.191. The L3 sigma was measured at "
                       "17-18 m/s; the binding gate-4 is 30 (doctrine) / 37 m/s. Under a LINEAR motion-blur "
                       "model the at-speed sigma_lat would be ~0.32 (30 m/s) / ~0.39 (37 m/s) -- FAR above "
                       "the ceiling. Even a modest x1.3 at-speed inflation breaks closure. The at-speed "
                       "sigma is UNMEASURED (memory's own standing caveat). => sigma_lat 0.191 is NOT a safe "
                       "floor for the deployment speed; it is a 17-18 m/s LOWER BOUND."},

        "attack3_lateral_dominance": {
            "verdict": "CONFIRMED -- the in-plane miss is LATERAL(0.19)-dominated; using iso-0.10 is grossly optimistic",
            "ladder": a3rows,
            "first_fr_close030_honest_aniso": {f"v{int(v)}": first_fr("honest_close030", v) for v in (30.0, 37.0)},
            "first_fr_close030_naive_iso010": {f"v{int(v)}": first_fr("naive_close030", v) for v in (30.0, 37.0)},
            "over_optimism_iso_equiv_for_engine": iso_hi90,
            "finding": "Honest aniso [0.19,0.10] first closes r=0.30 at fr=0.50 (v30) / fr=0.35 (v37). Naive "
                       "iso-0.10 (using only the clean vertical) first closes at fr=0.15 (BOTH speeds) -- a "
                       "2.3-3.3x fix-rate over-claim, plus a uniform +30-41% p99 understatement. The bake "
                       "fixes the vertical BIAS, not the lateral SCATTER; closure rides on the 0.19 lateral "
                       "+ fix-rate, exactly as the recal's own honesty_flag states. The engine's correct "
                       "single-sigma surrogate is the in-plane-radial-equiv ~0.153 (upper-CI ~0.170), NOT 0.10."},
    }
    (HERE / "verify_sigma_results.json").write_text(json.dumps(out, indent=2))
    print("WROTE verify_sigma_results.json")
    # quick console digest
    print("\nA1 r=0.30 @ upper-CI sigma: closes at fr=0.50 (all sets); fails at fr=0.35 (aniso upper-CI + iso0.170@v30)")
    print("A2 ceiling: held ~0.245 (x1.28), scaled ~0.23 (x1.20); linear-blur pred 0.32(v30)/0.39(v37) >> ceiling")
    print("A3 first-close fr: honest aniso v30=%s v37=%s | naive iso0.10 v30=%s v37=%s" % (
        first_fr("honest_close030", 30.0), first_fr("honest_close030", 37.0),
        first_fr("naive_close030", 30.0), first_fr("naive_close030", 37.0)))


if __name__ == "__main__":
    main()
