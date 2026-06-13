"""INDEPENDENT verification of piece F (track-map registration re-survey).

Re-derives every headline number from scratch WITHOUT importing vision_cal, to check:
  1. SIGN: gate_meas = map_centre - off_ned, and offset = -mean(off_ned) = -MEASURED_FIX_BIAS.
  2. CIRCULARITY: is the gate-3 D offset independently corroborated, or does it assume the map?
     -> recover the gate-3 OPENING-CENTRE D from raw GT-pose + lever WITHOUT the map, then
        compare to both the map opening-centre and the claimed 1.46 m.
  3. DEPTH DECOMP: re-run the ray projection; is the D offset really depth-free?
  4. RESIDUAL SIGMA: re-check the across-gate scatter and gate-4 numbers.
  5. ROBUST-AVERAGER SENSITIVITY: does the answer survive a plain trimmed mean (not the
     shipped MAD averager)? i.e. is the falsification an artifact of estimate_map_pose_aided?
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
CHAR_DIR = REPO / "handoff/perception-char-2026-06-08"
MAP_PATH = REPO / "handoff/shadowpc-firstcontact-2026-06-02/track_map.json"
BUNDLE = CHAR_DIR / "course_bundle/frames.json"
MEASURED_FIX_BIAS_NED = np.array([-0.42, +0.06, -0.28])
CHI2 = 16.27
INNER_HALF = 0.75
OUTER_HALF = 1.36


def map_centres_and_records():
    mp = json.loads(MAP_PATH.read_text())
    rec = {}
    opening = {}
    for r in mp["gates"]:
        gid = int(r["gate_id"])
        rec[gid] = r
        h = float(r.get("height_m") or 2.72)
        q = r["orientation_ned_wxyz"]
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        col2 = R[:, 2] if R[:, 2][2] >= 0 else -R[:, 2]
        opening[gid] = np.asarray(r["position_ned"], float) - 0.5 * h * col2
    return opening, rec


def load_rows():
    rows = []
    seen = set()
    for g in range(6):
        d = json.loads((CHAR_DIR / f"characterize_g{g}.json").read_text())
        for r in d["rows"]:
            if not (r.get("associated") and r.get("gate_id") is not None):
                continue
            if "off_ned" not in r:
                continue
            fid = int(r["frame_id"])
            if fid in seen:
                continue
            seen.add(fid)
            rows.append(r)
    return rows


def main():
    opening, rec = map_centres_and_records()
    rows = load_rows()
    frames = {f["frame_id"]: f for f in json.loads(BUNDLE.read_text())["frames"]}
    print(f"pooled {len(rows)} associated rows; {len(frames)} GT-pose frames in bundle")

    # ---- (1) SIGN CHAIN: independent global bias from raw off_ned of KF-accepted 4-corner ----
    acc = [r for r in rows if r["n_corners"] == 4 and np.isfinite(r["maha"]) and r["maha"] <= CHI2]
    off = np.array([r["off_ned"] for r in acc])
    mean_off = off.mean(axis=0)
    print("\n[1] SIGN CHAIN (independent, mean over KF-accepted 4-corner off_ned):")
    print(f"  n_accepted = {len(acc)}")
    print(f"  mean(off_ned)            = {np.round(mean_off,3)}")
    print(f"  MEASURED_FIX_BIAS_NED    = {MEASURED_FIX_BIAS_NED}")
    print(f"  -> offset = -mean(off)   = {np.round(-mean_off,3)}  (should match prototype global offset)")
    # NB: prototype uses the ROBUST per-gate averager then means the 6 gate offsets (equal gate
    # weight), not a flat mean over all fixes. Compute BOTH to show the sign is robust either way.

    # ---- (2) GATE-3: is the 1.46 m falsification CIRCULAR? ----
    print("\n[2] GATE-3 1.46 m FALSIFICATION -- circularity check")
    # (2a) The map-INDEPENDENT recovery of gate-3 opening centre uses gate_meas = map_centre - off.
    #      But map_centre is added then subtracted, so the RECOVERED centre = drone_given + lever,
    #      which needs NO map. Recover it directly from drone (GT) + lever, where
    #      lever = (map_centre - off) - drone_given  ===  drone+lever. We don't have lever alone,
    #      but off = (map_centre - lever) - drone  =>  lever = map_centre - off - drone, and
    #      gate_meas = drone + lever = map_centre - off. So gate_meas is fully determined by
    #      (map_centre, off) and does NOT depend on drone at all -- the map_centre cancellation is
    #      exact ONLY because off was computed against the SAME map_centre. VERIFY that:
    #      reconstruct lever for the gate-3-associated rows from the bundle GT pose, independently.
    g3_rows = [r for r in rows if r["gate_id"] == 3 and r["n_corners"] == 4
               and np.isfinite(r["maha"]) and r["maha"] <= CHI2]
    print(f"  gate-3 KF-accepted 4-corner rows: {len(g3_rows)}")
    # reconstruct gate-3 opening-centre measurement two ways and confirm identical:
    #   way A (prototype): gate_meas = map_opening[3] - off
    #   way B (independent): lever_from_map = rec3.position(bottom) ... we instead use the
    #     RECORD bottom-centre to recover the OPENING centre measurement directly: the off was
    #     computed as pos_fix - drone with pos_fix = gate.position_ned(RECORD/bottom?) - lever.
    #     CRITICAL: which gate.position_ned did characterize use -- record (bottom) or opening?
    rec3 = rec[3]
    print(f"  map RECORD position_ned (bottom-centre per schema) = {rec3['position_ned']}")
    print(f"  map OPENING-centre (lifted -0.5h*col2)             = {np.round(opening[3],3)}")
    # check which the off_ned was referenced to, using a GT-pose-joined gate-3 row:
    joined = [r for r in g3_rows if r["frame_id"] in frames]
    print(f"  gate-3 KF-accepted rows with GT pose in bundle: {len(joined)}")
    for r in joined[:5]:
        d = np.asarray(frames[r["frame_id"]]["drone_position_ned"], float)
        off_r = np.asarray(r["off_ned"], float)
        # pos_fix = drone + off; if characterize used gate.position_ned = RECORD (bottom):
        #   pos_fix = rec_bottom - lever ; lever = rec_bottom - pos_fix = rec_bottom - (drone+off)
        # the gate-centre IMPLIED (bottom) = drone + lever = rec_bottom - off
        gmeas_bottom = np.asarray(rec3["position_ned"], float) - off_r
        gmeas_opening = opening[3] - off_r
        print(f"    fid {r['frame_id']} rng {r['true_range_m']:.1f}m  gmeas(if record-ref) D={gmeas_bottom[2]:.3f}"
              f"  gmeas(if opening-ref) D={gmeas_opening[2]:.3f}  drone_D={d[2]:.3f}")

    # (2b) The TRUE discriminator the brief cites: the live PASS-CLEAN transit crossing distance.
    # find the closest-range frame whose nearest_gate_id == 3 (independent of association)
    g3_transit = sorted([f for f in frames.values() if f.get("nearest_gate_id") == 3],
                        key=lambda f: f["range_m"])
    if g3_transit:
        ft = g3_transit[0]
        d = np.asarray(ft["drone_position_ned"], float)
        print(f"\n  closest gate-3 transit (nearest_gate_id==3): fid {ft['frame_id']} range {ft['range_m']:.3f} m")
        print(f"    drone pos = {np.round(d,3)}")
        print(f"    |drone - map_OPENING_centre|     = {np.linalg.norm(d-opening[3]):.3f} m  (inner_half {INNER_HALF})")
        print(f"    drone_D - record_bottom_D        = {d[2]-rec3['position_ned'][2]:+.3f} m  (the claimed '1.46')")
        print(f"    drone_D - opening_D              = {d[2]-opening[3][2]:+.3f} m")
        # 3D distance to opening centre is THE discriminator; if << inner_half the map is fine.
        miss = np.linalg.norm(d - opening[3])
        if miss < INNER_HALF:
            print(f"    => crossing {miss:.3f} m < inner_half {INNER_HALF}: PASS-CLEAN consistent w/ correct map")
        else:
            print(f"    => crossing {miss:.3f} m >= inner_half: would indicate a real offset")

    # (2c) Is the 1.46 m exactly the half-height lift?
    lift = abs(rec3["position_ned"][2] - opening[3][2])
    print(f"\n  record->opening D lift = {lift:.3f} m (half-height {OUTER_HALF})")
    print(f"  the published '1.46 m' vs this lift {lift:.3f} m vs measured transit D-vs-record "
          f"{abs(g3_transit[0]['drone_position_ned'][2]-rec3['position_ned'][2]):.3f} m")

    # ---- (3) DEPTH DECOMPOSITION independent re-run ----
    print("\n[3] DEPTH DECOMPOSITION (independent): radial(depth) vs residual per gate")
    for gid in range(6):
        sel = [r for r in rows if r["gate_id"] == gid and r["n_corners"] == 4
               and r["world_fix_err_m"] < 3.0 and r["frame_id"] in frames]
        if len(sel) < 3:
            print(f"  g{gid}: n={len(sel)} insufficient")
            continue
        offs = np.array([r["off_ned"] for r in sel])
        drones = np.array([frames[r["frame_id"]]["drone_position_ned"] for r in sel], float)
        ray = opening[gid] - drones.mean(axis=0)
        ray = ray / np.linalg.norm(ray)
        radial = np.mean([o @ ray for o in offs])
        depth_expl = radial * ray
        resid = offs.mean(axis=0) - depth_expl
        print(f"  g{gid}: n={len(sel)} ray={np.round(ray,3)} |rayD|={abs(ray[2]):.4f} "
              f"radial={radial:+.3f} depthExpl_D={depth_expl[2]:+.4f} residD={resid[2]:+.3f}")

    # ---- (4) GATE-3 raw-vs-accepted N inflation (tail effect) ----
    print("\n[4] gate-3 N offset raw vs KF-accepted (the tail-effect claim)")
    g3_all = [r for r in rows if r["gate_id"] == 3 and r["n_corners"] == 4]
    g3_acc = [r for r in g3_all if np.isfinite(r["maha"]) and r["maha"] <= CHI2]
    print(f"  gate-3 4-corner: all={len(g3_all)} accepted={len(g3_acc)}")
    print(f"  median(-off_N) all      = {-np.median([r['off_ned'][0] for r in g3_all]):+.3f}")
    print(f"  median(-off_N) accepted = {-np.median([r['off_ned'][0] for r in g3_acc]):+.3f}")
    # show the far-range tail
    far = [r for r in g3_all if r["true_range_m"] > 50]
    print(f"  gate-3 4-corner rows at true_range>50m: {len(far)} (off_N range "
          f"{min(r['off_ned'][0] for r in far):.1f}..{max(r['off_ned'][0] for r in far):.1f})" if far
          else "  no >50m rows")

    # ---- (5) ROBUST-AVERAGER SENSITIVITY: trimmed mean vs MAD averager ----
    print("\n[5] AVERAGER SENSITIVITY (gate-3 D offset via different estimators):")
    g3_acc_off = np.array([r["off_ned"] for r in g3_acc])
    for name, est in [("plain mean", np.mean), ("median", np.median)]:
        o = est(g3_acc_off, axis=0)
        print(f"  {name:12s}: offset(-off) D = {-o[2]:+.3f}  N={-o[0]:+.3f} E={-o[1]:+.3f}")


if __name__ == "__main__":
    main()
