"""Calibrate the VG instrument on the fully-adjudicated case flight (side strike, gate 3)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vg

SCR = (r"C:\Users\Fengy\AppData\Local\Temp\claude"
       r"\C--Users-Fengy-Downloads-Projects-Anduril--claude-worktrees-peregrine-rl-commander-74b0dd"
       r"\b85130f9-ac88-4db2-8c68-0e28b966cf80\scratchpad")
CASE = os.path.join(SCR, "case", "data", "runs", "20260726_181345_v19pick_Ws0_f1")

recs, seek, timing, meta = vg.load_session(CASE)
d = vg.build(recs, seek, meta)
n = len(d["t"])
print(f"ticks={n} dur={d['t'][-1]:.3f}s zbias={d['zbias']} final={meta['final_state']} "
      f"gi_final={meta['gate_index']} coll={meta['collisions']}")
print("gate_index segments:", [(int(g), int((d['gi'] == g).sum())) for g in np.unique(d["gi"])])
print("fresh fixes:", int(d["fresh"].sum()), "of", n)

seg = np.where(d["gi"] == 3)[0]
print(f"\ngate-3 segment ticks {seg[0]}..{seg[-1]}  t {d['t'][seg[0]]:.3f}..{d['t'][seg[-1]]:.3f}")

fr = seg[d["fresh"][seg]]
print(f"fresh in segment: {len(fr)}")
print("\n k     t      rho   rel_flu(f,l,u)        lat_lev  up_lev  ncorn rsrc     reason")
for i in seg:
    r = d["rel"][i]
    cr, sr = np.cos(d["roll"][i]), np.sin(d["roll"][i])
    cp, sp = np.cos(d["pitch"][i]), np.sin(d["pitch"][i])
    Rl = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]) @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    rl = Rl @ r
    print(f"{d['k'][i]:4d} {d['t'][i]:6.3f} {d['rho'][i]:6.2f} "
          f"[{r[0]:6.2f},{r[1]:6.2f},{r[2]:6.2f}] {rl[1]:7.2f} {rl[2]:7.2f} "
          f"{d['ncorn'][i]:4.0f} {str(d['rsrc'][i]):8s} {str(d['reason'][i]):18s} "
          f"{'FRESH' if d['fresh'][i] else '  -  '} roll={np.degrees(d['roll'][i]):+6.1f}")

# ---- VG fit over the terminal window ----
for win_s in (0.35, 0.5, 0.8):
    sel = fr[d["t"][fr] >= d["t"][fr[-1]] - win_s]
    f = vg.vg_fit(d, sel)
    if f is None:
        print(f"\nwin={win_s}: too few fresh fixes ({len(sel)})")
        continue
    tstar, miss, mm = vg.closest_approach(f["p_end"], f["v_end"])
    print(f"\n=== VG fit  window={win_s}s  n_fresh={len(sel)} ===")
    print(f"  p_end (drone rel gate, leveled) = [{f['p_end'][0]:+.2f},{f['p_end'][1]:+.2f},{f['p_end'][2]:+.2f}]  |p|={np.linalg.norm(f['p_end']):.2f}")
    print(f"  v_end = [{f['v_end'][0]:+.2f},{f['v_end'][1]:+.2f},{f['v_end'][2]:+.2f}] |v|={np.linalg.norm(f['v_end']):.2f} m/s")
    print(f"  rms residual radial={f['rms_radial']:.3f} m   transverse={f['rms_trans']:.3f} m")
    print(f"  closest approach t*={tstar:.3f}s  miss=[{miss[0]:+.2f},{miss[1]:+.2f},{miss[2]:+.2f}] |miss|={mm:.2f}")
    hz = float(np.hypot(miss[0], miss[1]))
    print(f"  -> horizontal miss={hz:.2f} m  vertical miss={miss[2]:+.2f} m   (gate half={vg.GATE_HALF})")

# ---- Cross-check: gyro vs AHRS attitude divergence over the approach ----
i0, i1 = int(fr[0]), int(fr[-1])
Rg = vg.gyro_attitude(d, i0, i1, i0)
Ra = vg.ahrs_attitude(d, i0, i1, i0)
ang = []
for j in range(i1 - i0 + 1):
    M = Rg[j].T @ Ra[j]
    ang.append(np.degrees(np.arccos(np.clip((np.trace(M) - 1) / 2, -1, 1))))
print(f"\ngyro-vs-AHRS attitude divergence over the gate-3 approach: "
      f"max={max(ang):.2f} deg  end={ang[-1]:.2f} deg")

# ---- The OLD vision instrument on this approach: raw estimate lateral vs FITTED ----
sel = fr[d["t"][fr] >= d["t"][fr[-1]] - 1.4]
f = vg.vg_fit(d, sel)
print("\n=== estimate motion vs drone motion (last 1.4 s of fresh fixes) ===")
print("  k    rho   p_raw(x,y,z)              p_fit(x,y,z)              res_rad  res_tra")
for j, i in enumerate(f["idx"]):
    pr, pf = f["p_raw"][j], f["p_fit"][j]
    print(f"{d['k'][i]:4d} {f['rho'][j]:6.2f} [{pr[0]:+6.2f},{pr[1]:+6.2f},{pr[2]:+6.2f}]  "
          f"[{pf[0]:+6.2f},{pf[1]:+6.2f},{pf[2]:+6.2f}]  {f['res_radial'][j]:+7.3f} {f['res_trans'][j]:7.3f}  w={f['w'][j]:.2f}")
