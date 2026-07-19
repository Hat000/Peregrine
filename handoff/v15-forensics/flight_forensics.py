#!/usr/bin/env python
"""DEPLOY FLIGHT FORENSICS (analysis-only) -- v1 pick-flights vs champion-ckpt population vs champion.

Quantifies (Q1) velocity-runaway / pitch-down, (Q2) discriminates the yaw-oscillation mechanism
among (a) scanning, (b) latency limit-cycle, (c) self-induced perception-loss feedback,
(Q3) the last-1.5s failure signature, and (Q4) a verdict table. Pure read of ego_obs.jsonl;
no shipping-code / config / data edits.

Field schema (rl/fly_rl.py _ego_log; 21-dim obs, src/racer/ego_obs.py):
  obs[0:3] velocity BODY-FLU (fwd,left,up) virtual-flipped; obs[3]=roll obs[4]=pitch (rad, leveled)
  obs[11:14] slot0 rel to ACTIVE gate; obs[16:19] slot1 rel to NEXT gate (obs neg-forward,neg-left vs FLU)
  act_raw=[a_thrust,a_roll,a_pitch,a_yaw] policy action pre-clamp; rate_frd=[roll,pitch,yaw] wire FRD post-clamp
  rate_frd[1]=+a_pitch (a_pitch<0 = nose-DOWN); rate_frd[2]=yaw cmd (clamp +/-0.7)
  rel_flu1=[fwd,left,up] next-gate body-FLU; kf_pos_ned D=down+ (vertical WEAK, no baro)
CONVENTION (empirical, champ): toward-gate yaw sign = -sign(bearing_FLU_left); gate-right => +yaw.
"""
import json, os
import numpy as np
from scipy.signal import savgol_filter

AR = 'C:/Users/Fengy/Downloads/Projects/wt-arrest/data/runs'
RA = 'C:/Users/Fengy/Downloads/Projects/wt-ratchet/data/runs'
RUNS = {  # label: (base, dirname, group)
    'v1_A':   (AR, '20260718_160138_v1pick_A_f1',        'v1'),
    'v1_R1':  (AR, '20260718_160210_v1pick_R_f1',        'v1'),
    'v1_R2':  (AR, '20260718_160247_v1pick_R_f1',        'v1'),
    'v1_R3':  (AR, '20260718_160308_v1pick_R_f1',        'v1'),
    'v1_R4':  (AR, '20260718_160335_v1pick_R_f1',        'v1'),
    'champ':  (RA, '20260714_202317_panel_run_f1',       'champ'),
    'p03_r1': (RA, '20260718_023752_ratchet_p03_r1_f1',  'p03'),
    'p03_r2': (RA, '20260718_023849_ratchet_p03_r2_f1',  'p03'),
    'p03_r3': (RA, '20260718_024314_ratchet_p03_r3_f1',  'p03'),
    'p03_r4': (RA, '20260718_024533_ratchet_p03_r4_f1',  'p03'),
    'p03_r5': (RA, '20260718_024621_ratchet_p03_r5_f1',  'p03'),
}
ORDER = list(RUNS)
YAWCLAMP = 0.7


def load(base, d):
    rows = [json.loads(l) for l in open(os.path.join(base, d, 'ego_obs.jsonl')) if l.strip()]
    t0 = rows[0]['sim_time_ns']
    for r in rows:
        r['t'] = (r['sim_time_ns'] - t0) / 1e9
    return rows


def _vel_component(t, x, span=0.06):
    """Per-row central-difference derivative robust to duplicate/jittery timestamps.
    Expands the +/- window until the spanned time exceeds `span` s (g2_forensics convention)."""
    n = len(t); out = np.full(n, np.nan)
    for i in range(n):
        j0, j1 = i, i
        while j0 > 0 and (t[i] - t[j0]) < span / 2: j0 -= 1
        while j1 < n - 1 and (t[j1] - t[i]) < span / 2: j1 += 1
        if (t[j1] - t[j0]) > 0.02:
            out[i] = (x[j1] - x[j0]) / (t[j1] - t[j0])
    # fill any residual nans by nearest
    good = ~np.isnan(out)
    if good.any():
        out = np.interp(np.arange(n), np.where(good)[0], out[good])
    return out


def horiz_speed(rows):
    """Horizontal ground speed (m/s) from kf N,E via robust windowed diff (kf-horizontal trustworthy)."""
    t = np.array([r['t'] for r in rows])
    N = np.array([r['kf_pos_ned'][0] for r in rows]); E = np.array([r['kf_pos_ned'][1] for r in rows])
    vN = _vel_component(t, N); vE = _vel_component(t, E)
    return np.hypot(vN, vE)


def vspeed_down(rows):
    t = np.array([r['t'] for r in rows]); D = np.array([r['kf_pos_ned'][2] for r in rows])
    return _vel_component(t, D)  # down+ ; +=descending


def accel(rows, v):
    """d|v|/dt (m/s^2) row-aligned, robust to timestamp jitter."""
    t = np.array([r['t'] for r in rows])
    return _vel_component(t, v, span=0.12)


def bearing0(r):  # active gate, FLU +left (rad)
    return np.arctan2(-r['obs'][12], -r['obs'][11])

def bearing1(r):  # next gate from rel_flu1=[fwd,left,up]
    v = r.get('rel_flu1')
    if v is None: return np.nan
    return np.arctan2(v[1], v[0])

def toward_sign(b):  # yaw sign that turns toward a gate at FLU-left bearing b
    return -np.sign(b)


def num(x, default=np.nan):
    return default if x is None else float(x)


# ----------------------------------------------------------------------------- Q1
def q1(D):
    print("\n" + "=" * 100)
    print("Q1  VELOCITY RUNAWAY + PITCH-DOWN  (horiz speed from kf N,E diff; pitch=obs[4])")
    print("=" * 100)
    print(f"{'run':>7} {'T_s':>5} {'gates':>5} {'v_end':>6} {'v_peak':>6} {'v@peak_t':>8} "
          f"{'ratchet?':>9} {'pitch_med':>9} {'pitch_p10':>9} {'pitch_min':>9} {'%t<-20':>7} {'%t<-30':>7}")
    for n in ORDER:
        rows = D[n]; v = horiz_speed(rows); t = np.array([r['t'] for r in rows])
        pitch = np.array([r['obs'][4] for r in rows]) * 180 / np.pi
        gates = max(r['gate_index'] for r in rows)
        vpk = v.max(); tpk = t[v.argmax()]
        # ratchet = fraction of post-takeoff (t>1s) ticks where speed is non-decreasing (smoothed)
        w = min(11, len(v) if len(v) % 2 else len(v) - 1)
        vs = savgol_filter(v, w, 2) if w >= 5 else v
        pt = t[1:] > 1.0
        mono = np.mean((np.diff(vs) >= -0.05)[pt]) if pt.any() else float('nan')
        print(f"{n:>7} {t[-1]:5.2f} {gates:>5} {v[-1]:6.2f} {vpk:6.2f} {tpk:8.2f} "
              f"{mono:9.2f} {np.median(pitch):+9.1f} {np.percentile(pitch,10):+9.1f} {pitch.min():+9.1f} "
              f"{np.mean(pitch<-20):7.2f} {np.mean(pitch<-30):7.2f}")
    print("\n  Accel-phase pitch: pitch distribution WHILE accelerating (d|v|/dt > +1 m/s^2) vs decelerating")
    print(f"{'run':>7} {'%t_accel':>8} {'%t_decel':>8} {'pitch|accel':>11} {'pitch|decel':>11} {'coll|accel':>10} {'coll|decel':>10}")
    for n in ORDER:
        rows = D[n]; v = horiz_speed(rows); t = np.array([r['t'] for r in rows])
        a = accel(rows, v)
        pitch = np.array([r['obs'][4] for r in rows]) * 180 / np.pi
        coll = np.array([r['collective'] for r in rows])
        acc = a > 1.0; dec = a < -1.0
        pa = pitch[acc].mean() if acc.any() else float('nan')
        pd = pitch[dec].mean() if dec.any() else float('nan')
        ca = coll[acc].mean() if acc.any() else float('nan')
        cd = coll[dec].mean() if dec.any() else float('nan')
        print(f"{n:>7} {np.mean(acc):8.2f} {np.mean(dec):8.2f} {pa:+11.1f} {pd:+11.1f} {ca:10.3f} {cd:10.3f}")
    print("\n  Braking phase = sustained (>=4 ticks ~0.11s) decel with nose-up (pitch rising) AND low collective (<hover 0.266):")
    for n in ORDER:
        rows = D[n]; v = horiz_speed(rows); t = np.array([r['t'] for r in rows])
        a = accel(rows, v); coll = np.array([r['collective'] for r in rows])
        apitch = np.array([r['act_raw'][2] for r in rows])  # <0 nose-down; >0 nose-up cmd
        brake = (a < -1.0) & (coll < 0.266) & (apitch > 0)
        # longest run of consecutive brake ticks
        best = cur = 0
        for b in brake:
            cur = cur + 1 if b else 0; best = max(best, cur)
        print(f"   {n:>7}: brake-ticks={brake.sum():>3} longest_streak={best} ({best*0.028:.2f}s)  "
              f"min speed after peak = {v[v.argmax():].min():.2f} m/s")
    print("\n  Champion speed at each gate pass (kf N,E diff):")
    ch = D['champ']; v = horiz_speed(ch)
    for gi in range(1, 6):
        idx = next((k for k, r in enumerate(ch) if r['gate_index'] == gi), None)
        if idx is not None:
            print(f"   pass g{gi-1} t={ch[idx]['t']:5.2f}s  v_horiz={v[idx]:5.2f} m/s")
    print(f"   champ final tick t={ch[-1]['t']:.2f}s v_horiz={v[-1]:.2f} m/s (peak {v.max():.2f})")


# ----------------------------------------------------------------------------- Q2
def flips(y):
    """indices where sign(y) changes (ignoring exact zeros)."""
    s = np.sign(y); s[s == 0] = 1
    return np.where(np.diff(s) != 0)[0] + 1


def q2(D):
    print("\n" + "=" * 100)
    print("Q2  YAW MECHANISM  (rate_frd[2] series; clamp +/-0.7)")
    print("=" * 100)
    print("  (A) MAGNITUDE + FREQUENCY STRUCTURE")
    print(f"{'run':>7} {'|yaw|med':>8} {'%satur':>7} {'flip/s':>7} {'zcross/s':>8} {'domHz':>6} "
          f"{'|dYaw|med':>9} {'a_yaw_p95':>9}")
    for n in ORDER:
        rows = D[n]; y = np.array([r['rate_frd'][2] for r in rows]); t = np.array([r['t'] for r in rows])
        T = t[-1] - t[0]
        satur = np.mean(np.abs(y) >= 0.699)
        fl = flips(y); flrate = len(fl) / T
        ym = y - y.mean(); zc = np.sum(np.diff(np.sign(ym)) != 0); zcr = zc / T
        # coarse spectrum on mean-removed yaw
        Y = np.abs(np.fft.rfft(ym * np.hanning(len(ym)))); fr = np.fft.rfftfreq(len(ym), np.median(np.diff(t)))
        domHz = fr[1 + np.argmax(Y[1:])] if len(Y) > 2 else float('nan')
        dY = np.abs(np.diff(y))
        ay = np.abs([r['act_raw'][3] for r in rows])
        print(f"{n:>7} {np.median(np.abs(y)):8.3f} {satur:7.2f} {flrate:7.2f} {zcr:8.2f} {domHz:6.1f} "
              f"{np.median(dY):9.3f} {np.percentile(ay,95):9.2f}")

    print("\n  (B) FLIP-CONDITIONAL SLOT STATE  (hyp-a scanning: flips coincide w/ slot1 staleness; hyp-b: independent)")
    print(f"{'run':>7} {'age_s@flip':>10} {'age_s@non':>10} {'age_s1@flip':>11} {'age_s1@non':>10} "
          f"{'%see1@flip':>10} {'%see1@non':>10} {'%see0@flip':>10}")
    for n in ORDER:
        rows = D[n]; y = np.array([r['rate_frd'][2] for r in rows])
        fl = set(flips(y)); idx = np.arange(len(rows))
        fmask = np.array([i in fl for i in idx]); nmask = ~fmask
        a0 = np.array([num(r['age_s']) for r in rows]); a1 = np.array([num(r['age_s1']) for r in rows])
        s1 = np.array([1.0 if r.get('pose_seen1') else 0.0 for r in rows])
        s0 = np.array([1.0 if r.get('pose_seen') else 0.0 for r in rows])
        def m(x, k): return np.nanmean(x[k]) if k.any() else float('nan')
        print(f"{n:>7} {m(a0,fmask):10.3f} {m(a0,nmask):10.3f} {m(a1,fmask):11.3f} {m(a1,nmask):10.3f} "
              f"{m(s1,fmask):10.2f} {m(s1,nmask):10.2f} {m(s0,fmask):10.2f}")

    print("\n  (C) SELF-INDUCED PERCEPTION LOSS  (causal lag: does |yaw| PRECEDE det degradation, or follow it?)")
    print("      d_age(t)=age_s(t)-age_s(t-1); +=staleness rising.  Compare fwd (yaw->loss) vs bwd (loss->yaw).")
    print(f"{'run':>7} {'corr|yaw|_t,dage_t+1':>20} {'corr|yaw|_t,dage_t+2':>20} {'corr dage_t,|yaw|_t+1':>21} "
          f"{'%see0 whole':>11}")
    for n in ORDER:
        rows = D[n]; y = np.abs([r['rate_frd'][2] for r in rows])
        age = np.array([num(r['age_s'], 0.0) for r in rows]); dage = np.diff(age, prepend=age[0])
        s0 = np.mean([1.0 if r.get('pose_seen') else 0.0 for r in rows])
        def lagcorr(x, z, lag):
            if lag > 0: a, b = x[:-lag], z[lag:]
            elif lag < 0: a, b = x[-lag:], z[:lag]
            else: a, b = x, z
            if len(a) < 5 or np.std(a) < 1e-9 or np.std(b) < 1e-9: return float('nan')
            return np.corrcoef(a, b)[0, 1]
        c1 = lagcorr(y, dage, 1); c2 = lagcorr(y, dage, 2)
        cb = lagcorr(dage, y, 1)
        print(f"{n:>7} {c1:>20.3f} {c2:>20.3f} {cb:>21.3f} {s0:>11.2f}")

    print("\n  (D) SWEEP DIRECTION MATCH  (each half-cycle: does its sign point toward slot0 or slot1 gate?)")
    print("      match0 = sweep sign == toward-slot0 (active); match1 = toward-slot1 (next). hyp-a=>match1 high.")
    print(f"{'run':>7} {'n_halfcyc':>9} {'%match_slot0':>12} {'%match_slot1':>12} {'|b0|med_deg':>11} {'|b1|med_deg':>11} {'|b1-b0|med':>10}")
    for n in ORDER:
        rows = D[n]; y = np.array([r['rate_frd'][2] for r in rows])
        fl = flips(y); segs = np.split(np.arange(len(rows)), fl)
        m0 = m1 = tot0 = tot1 = 0
        for seg in segs:
            if len(seg) == 0: continue
            sgn = np.sign(np.mean(y[seg]))
            if sgn == 0: continue
            b0 = np.nanmean([bearing0(rows[i]) for i in seg]); b1 = np.nanmean([bearing1(rows[i]) for i in seg])
            if np.isfinite(b0) and abs(b0) > 1e-6:
                tot0 += 1; m0 += (sgn == toward_sign(b0))
            if np.isfinite(b1) and abs(b1) > 1e-6:
                tot1 += 1; m1 += (sgn == toward_sign(b1))
        b0a = np.abs([np.degrees(bearing0(r)) for r in rows])
        b1a = np.array([np.degrees(bearing1(r)) for r in rows]); b1a = np.abs(b1a[np.isfinite(b1a)])
        sep = np.array([np.degrees(bearing1(r) - bearing0(r)) for r in rows]); sep = np.abs(sep[np.isfinite(sep)])
        print(f"{n:>7} {max(tot0,tot1):>9} {(m0/tot0 if tot0 else float('nan')):>12.2f} {(m1/tot1 if tot1 else float('nan')):>12.2f} "
              f"{np.median(b0a):>11.1f} {np.median(b1a) if len(b1a) else float('nan'):>11.1f} {np.median(sep) if len(sep) else float('nan'):>10.1f}")


# ----------------------------------------------------------------------------- Q3
def q3(D):
    print("\n" + "=" * 100)
    print("Q3  LAST ~1.5 s FAILURE SIGNATURE  (event chain into crash / final tick)")
    print("=" * 100)
    print(f"{'run':>7} {'final':>7} {'gates':>5} {'v_last':>6} {'v_1.5ago':>8} {'dV':>6} {'pitch_last':>10} "
          f"{'yawflip/s_1.5':>13} {'%satur_1.5':>10} {'age_end':>7} {'see0_end':>8} {'alt_dropLast':>12}")
    for n in ORDER:
        rows = D[n]; t = np.array([r['t'] for r in rows]); v = horiz_speed(rows)
        alt = np.array([-r['kf_pos_ned'][2] for r in rows])
        meta = json.load(open(os.path.join(RUNS[n][0], RUNS[n][1], 'meta.json')))
        last = rows[-1]; win = t >= (t[-1] - 1.5)
        y = np.array([r['rate_frd'][2] for r in rows])[win]
        fl = flips(y); Tw = t[win][-1] - t[win][0]
        i15 = np.argmax(win)  # first index in window
        pitch_last = last['obs'][4] * 180 / np.pi
        print(f"{n:>7} {meta.get('final_state','?'):>7} {max(r['gate_index'] for r in rows):>5} "
              f"{v[-1]:6.2f} {v[i15]:8.2f} {v[-1]-v[i15]:+6.2f} {pitch_last:+10.1f} "
              f"{len(fl)/Tw if Tw>0 else 0:13.2f} {np.mean(np.abs(y)>=0.699):10.2f} {num(last['age_s']):7.2f} "
              f"{str(last.get('pose_seen'))[0]:>8} {alt[i15]-alt[-1]:+12.2f}")
    print("\n  v1 short-run (g0) death chain -- fresh-detection loss vs geometry (last fresh slot0 fix -> death):")
    for n in ['v1_R2', 'v1_R3', 'v1_R4', 'v1_A', 'v1_R1']:
        rows = D[n]
        seen = [r for r in rows if r.get('pose_seen') and num(r['age_s'], 9) < 0.05]
        if not seen:
            print(f"   {n}: no fresh fixes"); continue
        lastfix = seen[-1]; death = rows[-1]
        print(f"   {n:>6}: lastfresh t={lastfix['t']:.2f} (b0={np.degrees(bearing0(lastfix)):+.0f}deg fwd0={-lastfix['obs'][11]:.1f}m) "
              f"death t={death['t']:.2f} gap={death['t']-lastfix['t']:.2f}s  conf_end={death['conf']:.2f} area_end={death['area']:.2f}")
    print("\n  Champion-pop (p03) g1-exit/g2 deaths already characterized in G2-FORENSICS: slower+lower, under-descend, det fresh to end.")
    print("  Cross-check det freshness at death for p03 crashers vs v1 crashers:")
    for n in ['p03_r2', 'p03_r4', 'p03_r5', 'v1_R2', 'v1_R3', 'v1_R4', 'v1_A']:
        rows = D[n]
        tail = rows[-6:]
        agemin = min(num(r['age_s'], 9) for r in tail)  # freshest in last 6 ticks
        print(f"   {n:>6}: freshest age_s in last 6 ticks = {agemin:.2f}s  (det fresh-to-end if ~0)  "
              f"yaw satur last6 = {np.mean([abs(r['rate_frd'][2])>=0.699 for r in tail]):.2f}")


# ----------------------------------------------------------------------------- Q2 deep confirmatory
def q2_deep(D):
    print("\n" + "=" * 100)
    print("Q2-DEEP  CONFIRMATORY: (1) yaw gain  (2) limit-cycle period  (3) perception-loss causality")
    print("=" * 100)
    print("  (1) YAW GAIN: |a_yaw_raw| vs |bearing0| on fresh ticks  (champ a_yaw stays < clamp; v1 ~15x)")
    print(f"{'run':>7} {'n':>4} {'a_yaw@10deg':>11} {'med|a_yaw|':>10} {'ratio_vs_champ':>14}")

    def ay10_of(n):
        fr = [r for r in D[n] if r.get('pose_seen') and num(r['age_s'], 9) < 0.05]
        if len(fr) < 8: return None
        b = np.abs([bearing0(r) for r in fr]); a = np.abs([r['act_raw'][3] for r in fr])
        slope, icpt = np.linalg.lstsq(np.vstack([b, np.ones_like(b)]).T, a, rcond=None)[0]
        return slope * np.radians(10) + icpt
    champ_ay10 = ay10_of('champ')
    for n in ORDER:
        fr = [r for r in D[n] if r.get('pose_seen') and num(r['age_s'], 9) < 0.05]
        if len(fr) < 8:
            print(f"{n:>7} {len(fr):>4} (few fresh)"); continue
        b = np.abs([bearing0(r) for r in fr]); a = np.abs([r['act_raw'][3] for r in fr])
        A = np.vstack([b, np.ones_like(b)]).T
        slope, icpt = np.linalg.lstsq(A, a, rcond=None)[0]
        ay10 = slope * np.radians(10) + icpt
        if n == 'champ': champ_ay10 = ay10
        ratio = ay10 / champ_ay10 if champ_ay10 else float('nan')
        print(f"{n:>7} {len(fr):>4} {ay10:>11.2f} {np.median(a):>10.2f} {ratio:>14.1f}")
    print("  (yaw clamp = 0.7 rad/s; champ commands < clamp so never saturates; v1 commands ~2 rad/s => bang-bang)")

    print("\n  (2) LIMIT-CYCLE PERIOD: autocorrelation of rate_frd[2]; first negative lag = half-period")
    print(f"{'run':>7} {'lag1_ac':>7} {'firstNegLag':>11} {'impliedHz':>9} {'satur%':>7}")
    for n in ORDER:
        y = np.array([r['rate_frd'][2] for r in D[n]]); dt = np.median(np.diff([r['t'] for r in D[n]])) or 0.028
        satur = np.mean(np.abs(y) >= 0.699); y = y - y.mean()
        if y.std() < 1e-9: print(f"{n:>7} flat"); continue
        ac = np.correlate(y, y, 'full')[len(y) - 1:] / (np.arange(len(y), 0, -1) * y.var())
        ac = ac[:min(20, len(ac))]
        fneg = next((k for k in range(1, len(ac)) if ac[k] < 0), float('nan'))
        hz = 1 / (2 * fneg * dt) if fneg == fneg else float('nan')
        print(f"{n:>7} {ac[1]:>7.2f} {fneg:>11} {hz:>9.1f} {satur:>7.2f}")

    print("\n  (3) PERCEPTION-LOSS EVENT STUDY: at pose_seen True->False onsets -- proximity vs prior yaw rotation")
    print("      netRot5 = |sum rate_frd[2] over prior 5 ticks| (achieved sweep). (c) needs netRot5@loss >> baseline")
    print(f"{'run':>7} {'nLoss':>5} {'fwd0@loss':>9} {'netRot5@loss':>12} {'netRot5@base':>12} {'ratio':>6} {'see0_whole':>10}")
    for n in ORDER:
        rows = D[n]; ps = [bool(r.get('pose_seen')) for r in rows]
        y = np.array([r['rate_frd'][2] for r in rows])
        onsets = [i for i in range(1, len(rows)) if ps[i - 1] and not ps[i]]
        see0 = np.mean(ps)
        if not onsets:
            print(f"{n:>7} {0:>5} {'-':>9} {'-':>12} {'-':>12} {'-':>6} {see0:>10.2f}"); continue
        fwd0 = np.median([-rows[i]['obs'][11] for i in onsets])
        nr = np.median([abs(y[max(0, i - 5):i].sum()) for i in onsets])
        nrb = np.mean([abs(y[max(0, i - 5):i].sum()) for i in range(5, len(rows))])
        print(f"{n:>7} {len(onsets):>5} {fwd0:>9.1f} {nr:>12.2f} {nrb:>12.2f} {nr/nrb if nrb else float('nan'):>6.2f} {see0:>10.2f}")


# ----------------------------------------------------------------------------- run
def main():
    D = {n: load(b, d) for n, (b, d, g) in RUNS.items()}
    q1(D); q2(D); q2_deep(D); q3(D)


if __name__ == '__main__':
    main()
