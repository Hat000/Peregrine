#!/usr/bin/env python
"""G2 DESCEND-WALL forensics (analysis-only) -- ratchet-P0.3.

Reproduces every number in G2-FORENSICS.md from the committed ego_obs.jsonl logs of the
N=5 settled-GO champion-recipe policy flights + the champion tape source. No shipping-code,
config, test, or data edits -- pure read of data/runs/*/ego_obs.jsonl.

Per-tick log schema (rl/fly_rl.py _ego_log.append, ~line 2394) + 21-dim obs contract
(src/racer/ego_obs.py module docstring):
  obs[0:3] velocity BODY-FLU (fwd,left,up), virtual-flipped   obs[3]=roll obs[4]=pitch (rad)
  obs[5:8] body rates   obs[8] prev normed_thrust(g)          obs[9]=sector_horiz obs[10]=sector_vert
  obs[11:14] slot0 rel to ACTIVE gate (obs[13]=UP-offset,+=above)  obs[14]=conf obs[15]=area
  obs[16:19] slot1 rel to NEXT gate  (obs[18]=UP-offset)           obs[19]=conf1 obs[20]=area1
  slotN rel/conf/area MASK TO 0 once age>=det_hold(0.2s) (obs_coast OFF = champion default).
  act_raw=[a_thrust,a_roll,a_pitch,a_yaw] rescaled policy action (g / rad-s), PRE deploy clamps.
  rate_frd = wire body-rate FRD after yaw/pitch/roll clamps; rate_frd[1] = +a_pitch (nose-up>0).
  normed_thrust = EMITTED thrust g after assist/floor (feeds obs[8]); collective = wire [0,1].
  assist/floor = did takeoff-assist / EgoFloorClamp override this tick. kf_pos_ned = KF NED (D=down+).
Floor semantics (rl/fly_rl.py): takeoff-assist floors emitted collective to --ego-assist-thrust
  (1.3g) ONLY until a PERMANENT handover (|gyro|>1 rad/s OR climb>0.5m OR timeout); EgoFloorClamp
  is OFF unless --ego-floor-clamp>0 (absent from the champion recipe). Both reuse the 1.3g ref.
"""
import json, os
import numpy as np

RUNS = {
    'r1': '20260718_023752_ratchet_p03_r1_f1',   # 2 gates, STALL/dive hunting g2
    'r2': '20260718_023849_ratchet_p03_r2_f1',   # 2, crash g2 leg
    'r3': '20260718_024314_ratchet_p03_r3_f1',   # 1, died on g1 leg (never reached g2)
    'r4': '20260718_024533_ratchet_p03_r4_f1',   # 2, died just after g1
    'r5': '20260718_024621_ratchet_p03_r5_f1',   # 2, died at/after g1
    'champ': '20260714_202317_panel_run_f1',     # tape source: passed g0-g4
}
BASE = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'runs')
HOVER_G = 1.0


def load(dirname):
    rows = [json.loads(l) for l in open(os.path.join(BASE, dirname, 'ego_obs.jsonl')) if l.strip()]
    t0 = rows[0]['sim_time_ns']
    for r in rows:
        r['t'] = (r['sim_time_ns'] - t0) / 1e9
    return rows


def leg2(rows):
    """The g1->g2 descend leg = ticks with active gate index == 2 (g1 passed, targeting g2)."""
    return [r for r in rows if r['gate_index'] == 2]


def gpass(rows, gi):
    return next((r for r in rows if r['gate_index'] == gi), None)


def vvel_down(rows, i):
    """Robust world-vertical velocity (m/s, down+) at tick i from kf D via a +-3 tick window."""
    t = np.array([r['t'] for r in rows]); D = np.array([r['kf_pos_ned'][2] for r in rows])
    j0, j1 = max(0, i - 3), min(len(rows) - 1, i + 3)
    return (D[j1] - D[j0]) / (t[j1] - t[j0]) if (t[j1] - t[j0]) > 0.05 else float('nan')


def last_g2_fix(rows):
    """(src, t, lateral, up) of the last FRESH gate-2 detection: slot1 while gi==1, slot0 while gi==2."""
    latest = None
    for r in rows:
        if r['gate_index'] == 1 and r.get('pose_seen1'):
            latest = ('slot1', r['t'], r['obs'][17], r['obs'][18])
        if r['gate_index'] == 2 and r.get('pose_seen'):
            latest = ('slot0', r['t'], r['obs'][12], r['obs'][13])
    return latest


def main():
    D = {n: load(d) for n, d in RUNS.items()}
    order = ['r1', 'r2', 'r4', 'r5', 'champ']  # r3 never reached the g2 leg

    print("=== recipe floor constants: assist floor 1.3g -> collective 0.345; hover 1.0g -> 0.266;",
          "EgoFloorClamp OFF (no --ego-floor-clamp) ===\n")

    print("### Q1  FLOOR HYPOTHESIS on the g1->g2 leg (does a collective floor bind?) ###")
    print(f"{'run':>5} {'nTk':>4} {'assistF':>7} {'floorF':>6} {'coll_min':>8} {'norm_min':>8} "
          f"{'polThr_min':>10} {'polThr_med':>10} {'%pol<hover':>10} {'%floorBinds':>11}")
    for n in order:
        L = leg2(D[n])
        norm = np.array([r['normed_thrust'] for r in L]); pol = np.array([r['act_raw'][0] for r in L])
        coll = np.array([r['collective'] for r in L])
        binds = np.mean(norm > pol + 0.02)  # emitted exceeds policy-desired => something floored it up
        print(f"{n:>5} {len(L):>4} {np.mean([r['assist'] for r in L]):>7.2f} "
              f"{np.mean([r['floor'] for r in L]):>6.2f} {coll.min():>8.3f} {norm.min():>8.3f} "
              f"{pol.min():>10.3f} {np.median(pol):>10.3f} {np.mean(pol < HOVER_G):>10.2f} {binds:>11.2f}")
    print(" takeoff-assist handover (last assist=True tick) vs g0 pass -- proves the 1.3g floor is takeoff-only:")
    for n in order:
        at = [r for r in D[n] if r['assist']]
        print(f"   {n:>5}: last assist=True t={at[-1]['t']:.3f}s  (g0 pass ~{gpass(D[n],1)['t']:.2f}s, g1 leg ~5.4s)")

    print("\n### Q3  STATE AT G1 EXIT + descend-leg ranges (alt=-kfD m; speed=|obs[0:3]|) ###")
    print(f"{'run':>5} {'speed':>6} {'vhoriz':>6} {'alt':>6} {'vvel(dn+)':>9} | {'leg_alt[min..max]':>18} {'leg_spd[min..max]':>17}")
    for n in order:
        rows = D[n]; i = next(k for k, r in enumerate(rows) if r['gate_index'] == 2); r = rows[i]
        v = np.array(r['obs'][0:3]); L = leg2(rows)
        alt = np.array([-x['kf_pos_ned'][2] for x in L]); sp = np.array([np.linalg.norm(x['obs'][0:3]) for x in L])
        print(f"{n:>5} {np.linalg.norm(v):>6.2f} {np.hypot(v[0],v[1]):>6.2f} {-r['kf_pos_ned'][2]:>6.2f} "
              f"{vvel_down(rows,i):>9.2f} | [{alt.min():+5.2f}..{alt.max():+5.2f}] [{sp.min():>4.1f}..{sp.max():>4.1f}]")
    print(" champion kf-altitude at each gate pass (is g2 a descend?):")
    for gi in [1, 2, 3, 4, 5]:
        r = gpass(D['champ'], gi)
        if r:
            print(f"   pass g{gi-1} t={r['t']:.3f}: alt={-r['kf_pos_ned'][2]:+.2f}m  sector(g{gi})=[{r['obs'][9]:.0f},{r['obs'][10]:.0f}]")

    print("\n### Q4  PITCH FENCE + COMMAND-vs-ACHIEVED on the g1->g2 leg (pitch clamp = 1.0 DEG) ###")
    print(f"{'run':>5} {'pitch_deg':>9} {'%fence_on':>9} {'a_pitch_mean':>12} {'%nose-down floored':>18} "
          f"{'%thr<hover':>10} {'leg_alt_drop_m':>14}")
    for n in order:
        L = leg2(D[n])
        o4 = np.array([r['obs'][4] for r in L]) * 180 / np.pi
        ap = np.array([r['act_raw'][2] for r in L]); rf1 = np.array([r['rate_frd'][1] for r in L])
        pol = np.array([r['act_raw'][0] for r in L]); alt = np.array([-r['kf_pos_ned'][2] for r in L])
        floored = np.mean((ap < -1e-3) & (np.abs(rf1) < 1e-6))  # policy asked nose-down, fence zeroed it
        print(f"{n:>5} {o4.mean():>+9.1f} {np.mean(o4 <= -1.0):>9.2f} {ap.mean():>+12.2f} {floored:>18.2f} "
              f"{np.mean(pol < HOVER_G):>10.2f} {alt[0]-alt[-1]:>+14.2f}")

    print("\n### Q2  GATE-2 PERCEPTION: last fresh fix, det->death gap, drift ###")
    print(f"{'run':>5} {'lastG2fix_t':>11} {'src':>6} {'death_t':>8} {'gap_s':>6} {'up_off':>7} "
          f"{'lateral':>7} {'altDrift_det->death':>19}")
    for n in order:
        rows = D[n]; src, tt, lat, up = last_g2_fix(rows); death = rows[-1]['t']
        i = next(k for k, r in enumerate(rows) if abs(r['t'] - tt) < 1e-6)
        drift = (-rows[-1]['kf_pos_ned'][2]) - (-rows[i]['kf_pos_ned'][2])  # +=climb
        print(f"{n:>5} {tt:>11.3f} {src:>6} {death:>8.3f} {death-tt:>6.3f} {up:>+7.2f} {lat:>+7.2f} {drift:>+19.2f}")
    print(" gate-2 up-offset trend while approaching (last 6 slot1 fixes; >0 = g2 above nose):")
    for n in order:
        seq = [round(r['obs'][18], 1) for r in D[n] if r['gate_index'] == 1 and r.get('pose_seen1')]
        print(f"   {n:>5}: {seq[-6:]}")


if __name__ == '__main__':
    main()
