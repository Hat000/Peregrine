"""ADVERSARIAL VERIFIER -- terminal-fix-drought + bursty-clustering refutation of the
margin-closure CLAIM:

    "Post-bake, gate-4 r=0.30 CLOSES once the mean fix-rate reaches ~0.35-0.50
     (anisotropic sigma [lat 0.19, vert 0.10], bias<=0)."

The production engine (margin_envelope.fly_lap) and the boresight driver
(margin_driver_v2.fly_lap_v2) BOTH schedule fixes on a REGULAR cadence:

    if s_tgt[k] >= 0 and t >= next_fix_t:
        next_fix_t = t + fix_dt   # fix_dt = 1/(DETECTOR_HZ * fix_rate)
        <offer a fix>

i.e. evenly-spaced offered fixes whenever a gate is in the FIX_WINDOW_M=12 m band.

REAL gate-relative fixes CLUSTER, and the binding adversarial case is a TERMINAL FIX
DROUGHT: the camera loses the gate-4 centre in the last ~0.3-0.6 s (gate leaves FoV /
exits the accept band) so the COLD velocity coasts UN-corrected into the gate. The
banked memory calls terminal-drought / clustering "the dominant adversarial lens" and
says closure needs TERMINAL GATE-LOCK, not just a high pooled mean.

WHAT THIS FILE CHANGES vs the production stack: NOTHING in the truth/KF/cov/draw math.
It is a byte-faithful copy of margin_driver_v2.fly_lap_v2 with ONE added gate on WHICH
OFFERED FIXES ARE DROPPED near gate-4, plus an optional bursty (clustered) emission
schedule that preserves the SAME mean fix-rate. The filter (LinearKF + RewindKF), the
anisotropic draw, the cov, the accel-bias model, the latency queue, and the final
in-plane miss reduction onto u34 are all reused EXACTLY from ME via fly_lap_v2's body.

SCHEDULE MODELS (schedule= argument):
  'regular'  : the production cadence (next_fix_t = t + fix_dt). == fly_lap_v2 when
               terminal_gap_s == 0 and terminal_gap_m == 0  (REGRESSION baseline).
  'bursty'   : same mean rate, but fixes arrive in bursts of `burst_n` back-to-back
               detector frames separated by a long gap so the long-run average ==
               fix_rate. Tests whether burstiness ALONE (no terminal drought) matters.

TERMINAL DROUGHT (independent of schedule): an OFFERED fix is DROPPED (never enqueued,
never applied) if it occurs inside the drought zone, defined by EITHER
  - terminal_gap_s > 0 : nominal obs time t >= T - terminal_gap_s, OR
  - terminal_gap_m > 0 : range-to-gate-4 at step k <= terminal_gap_m.
Both default 0 (no drought). The drought models the camera losing terminal gate-lock;
the *mean fix-rate earlier in the lap is unchanged* (the cadence/burst clock keeps
running; only the near-gate emissions are suppressed). This is the conservative reading
of "high pooled mean, no terminal lock".

DROP-AT-EMISSION vs drop-at-apply: we drop at EMISSION (the offered-fix site), i.e. the
fix is never created. Because the latency queue would in any case apply a fix at
t + L_s (L_s = 15 ms), dropping at emission is the physically correct model of "no fix
was produced in the drought" (a fix produced just before the drought still applies,
slightly delayed -- which we KEEP; that is the optimistic-for-the-claim choice).

Run via run_verify_clustering.py. Seeds are derived from ME.SEED so a given cell is
reproducible and distinct droughts/schedules get distinct streams.
[BORESIGHT-CLOSURE ADVERSARIAL VERIFY 2026-06-14]
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ME_DIR = _HERE.parents[0] / "margin-closure-envelope-2026-06-14"
if str(_ME_DIR) not in sys.path:
    sys.path.insert(0, str(_ME_DIR))
import margin_envelope as ME  # noqa: E402  (production truth + KF wiring + constants)

# Reuse the EXACT production filter classes (same objects ME.fly_lap / fly_lap_v2 use).
from racer.state_estimator import LinearKF  # noqa: E402
from racer.kf_rewind import RewindKF        # noqa: E402

ACCEL_NOISE_STD = ME.ACCEL_NOISE_STD
RADIAL_SIGMA = ME.RADIAL_SIGMA
DETECTOR_HZ = ME.DETECTOR_HZ
FIX_WINDOW_M = ME.FIX_WINDOW_M


# =================================================================================================
# fly_lap_drought -- fly_lap_v2 verbatim (anisotropic sigma + vertical fix bias) with EXACTLY one
# added behaviour: a terminal-drought / bursty-schedule gate on WHICH offered fixes are emitted.
# Setting schedule='regular', terminal_gap_s=0, terminal_gap_m=0 reproduces fly_lap_v2 bit-for-bit
# given the same rng (verified by the regression in run_verify_clustering.py).
# =================================================================================================
def fly_lap_drought(rng, v_race, sigma_lat, sigma_vert, accel_bias_mag, bias_mode, fix_rate,
                    vel_mode, latency_ms, vert_fix_bias_m=0.0, horizon_s=0.5,
                    schedule="regular", burst_n=3, terminal_gap_s=0.0, terminal_gap_m=0.0):
    L_s = latency_ms / 1e3
    tr = ME.get_truth(v_race)
    u34 = tr["u34"]
    g4 = ME.GATES[4]

    # --- accel bias (UNCHANGED from fly_lap_v2 / ME.fly_lap) ---
    if accel_bias_mag <= 0.0:
        accel_bias_body = np.zeros(3)
    elif bias_mode == "inplane":
        e1, e2 = ME._inplane_basis(u34)
        ang = rng.uniform(0, 2 * np.pi)
        d_world = np.cos(ang) * e1 + np.sin(ang) * e2
        accel_bias_body = tr["R_wb_g4"].T @ (accel_bias_mag * d_world)
    else:  # random3d
        d = rng.normal(0, 1, 3)
        d /= (np.linalg.norm(d) + 1e-12)
        accel_bias_body = accel_bias_mag * d

    # --- COLD init (UNCHANGED) ---
    p_init = tr["p_init_true"] + rng.normal(0, 0.5, 3)
    if vel_mode == "warm":
        v_init = tr["v_init_true"].copy()
    else:
        v_init = tr["v_init_true"] + rng.normal(0, 1.5, 3)
    kf = LinearKF.initialize(p_init, v_init, pos_std=1.0, vel_std=1.5)
    rk = RewindKF(kf=kf, horizon_s=horizon_s)

    s_p, s_v, s_R, s_ab = tr["s_p"], tr["s_v"], tr["s_R"], tr["s_ab"]
    s_dt, s_ts, s_tns = tr["s_dt"], tr["s_ts"], tr["s_tns"]
    s_tgt, s_e1, s_e2, s_u = tr["s_tgt"], tr["s_e1"], tr["s_e2"], tr["s_u"]
    n_steps = tr["n_steps"]
    T_end = float(s_ts[n_steps - 1])
    t_drought_start = T_end - terminal_gap_s if terminal_gap_s > 0 else np.inf

    fix_apply_queue = []
    fix_dt = 1.0 / (DETECTOR_HZ * fix_rate)
    next_fix_t = 0.0
    last_fix_z = None
    last_fix_t = None
    warm = (vel_mode == "warm")
    weakvel = (vel_mode == "weakvel")
    sig2_lat = sigma_lat ** 2
    sig2_vert = sigma_vert ** 2
    sig2_rad = RADIAL_SIGMA ** 2

    # --- bursty schedule bookkeeping (only used when schedule=='bursty') ---
    # To preserve mean rate fr: a fix is offered on `burst_n` consecutive eligible detector frames,
    # then suppressed until the burst-period has elapsed. period = burst_n / fr detector-frames worth
    # of time => long-run accept fraction = fr. We track frames via a per-eligible-step detector clock.
    burst_count = 0
    next_burst_t = 0.0
    det_dt = 1.0 / DETECTOR_HZ
    burst_period = (burst_n / fix_rate) * det_dt  # wall-time between burst starts for mean rate fr

    n_emitted = 0
    n_dropped_drought = 0

    for k in range(n_steps):
        t = s_ts[k]
        t_ns = int(s_tns[k])
        accel_meas = s_ab[k] + accel_bias_body + rng.normal(0, ACCEL_NOISE_STD, 3)
        rk.predict(accel_meas, s_R[k], s_dt[k], t_ns)
        if warm:
            rk.update_velocity(s_v[k], (0.1 ** 2) * np.eye(3), sim_time_ns=t_ns)

        # ---- fix EMISSION decision ----
        offer = False
        if s_tgt[k] >= 0:
            if schedule == "bursty":
                # start a new burst when the burst clock ticks; emit on consecutive eligible frames
                if t >= next_burst_t:
                    burst_count = burst_n
                    next_burst_t = t + burst_period
                if burst_count > 0:
                    offer = True
                    burst_count -= 1
            else:  # 'regular' -- the production cadence, byte-identical to fly_lap_v2
                if t >= next_fix_t:
                    offer = True
                    next_fix_t = t + fix_dt

        if offer:
            # ---- TERMINAL DROUGHT gate: suppress this OFFERED fix if inside the drought zone ----
            rng_to_g4 = float(np.linalg.norm(s_p[k] - g4))
            in_drought = (t >= t_drought_start) or (terminal_gap_m > 0 and rng_to_g4 <= terminal_gap_m)
            if in_drought:
                n_dropped_drought += 1
            else:
                n_emitted += 1
                e1_raw, e2_raw, target_u = s_e1[k], s_e2[k], s_u[k]
                if abs(e1_raw[2]) > abs(e2_raw[2]):
                    e1, e2 = e2_raw, e1_raw
                else:
                    e1, e2 = e1_raw, e2_raw
                n_lat = (rng.normal(0, sigma_lat) * e1
                         + rng.normal(0, sigma_vert) * e2
                         + rng.normal(0, RADIAL_SIGMA) * target_u)
                z = s_p[k] + n_lat + vert_fix_bias_m * e2
                cov = (sig2_lat * np.outer(e1, e1)
                       + sig2_vert * np.outer(e2, e2)
                       + sig2_rad * np.outer(target_u, target_u))
                fix_apply_queue.append((t + L_s, t_ns, z.copy(), cov.copy()))

        if fix_apply_queue:
            fix_apply_queue.sort(key=lambda e: e[0])
            while fix_apply_queue and fix_apply_queue[0][0] <= t + 1e-12:
                _, cap_ns, z, cov = fix_apply_queue.pop(0)
                rk.update_position_at(cap_ns, z, cov)
                if weakvel:
                    if last_fix_z is not None and last_fix_t is not None:
                        dt_fix = (cap_ns - last_fix_t) / 1e9
                        if dt_fix > 1e-3:
                            v_meas = (z - last_fix_z) / dt_fix
                            sig_v = np.sqrt(2.0) * sigma_lat / dt_fix
                            rk.update_velocity(v_meas, (sig_v ** 2) * np.eye(3), sim_time_ns=t_ns)
                    last_fix_z = z.copy()
                    last_fix_t = cap_ns

    err = rk.position - tr["p_final_true"]
    along = float(np.dot(err, u34))
    inplane_vec = err - along * u34
    miss = float(np.linalg.norm(inplane_vec))
    return miss, n_emitted, n_dropped_drought


def _seed_for(s, v_race, att_bias_deg, sigma_lat, sigma_vert, fix_rate, vel_mode, bias_mode,
              latency_ms, vert_fix_bias_m, schedule, terminal_gap_s, terminal_gap_m, match_me):
    """Per-lap seed. With schedule=='regular', terminal_gap_s==0, terminal_gap_m==0 AND match_me=True,
    AND sigma_vert==sigma_lat AND vert_fix_bias_m==0, returns the IDENTICAL integer ME.run_cell uses
    (so the regression vs ME reproduces to ~0). Otherwise folds the verifier knobs in for distinct
    streams."""
    base = (ME.SEED + 101 * s + int(round(v_race)) * 13 + int(round(att_bias_deg * 100)) * 7
            + int(round(sigma_lat * 1000)) * 17 + int(round(fix_rate * 1000)) * 23
            + {"cold": 1, "warm": 2, "weakvel": 3}[vel_mode] * 1009
            + {"random3d": 0, "inplane": 1}[bias_mode] * 3001 + int(round(latency_ms)) * 53)
    if (match_me and abs(sigma_vert - sigma_lat) < 1e-12 and abs(vert_fix_bias_m) < 1e-12
            and schedule == "regular" and terminal_gap_s == 0 and terminal_gap_m == 0):
        return base
    extra = (int(round(sigma_vert * 1000)) * 131 + int(round(vert_fix_bias_m * 1000)) * 211
             + int(round(terminal_gap_s * 1000)) * 307 + int(round(terminal_gap_m * 1000)) * 401
             + {"regular": 0, "bursty": 1}[schedule] * 503)
    return base + extra


def run_cell_drought(v_race, sigma_lat, sigma_vert, att_bias_deg, bias_mode, fix_rate, vel_mode,
                     latency_ms=15.0, vert_fix_bias_m=0.0, schedule="regular", burst_n=3,
                     terminal_gap_s=0.0, terminal_gap_m=0.0, n_mc=2000, match_me_seeds=False):
    """Monte-Carlo one cell with the terminal-drought / bursty schedule + bootstrap CIs on p90/p99."""
    accel_bias_mag = ME.att_deg_to_accel_bias(att_bias_deg)
    misses = np.empty(n_mc)
    emit = np.empty(n_mc); drop = np.empty(n_mc)
    for s in range(n_mc):
        seed = _seed_for(s, v_race, att_bias_deg, sigma_lat, sigma_vert, fix_rate, vel_mode,
                         bias_mode, latency_ms, vert_fix_bias_m, schedule, terminal_gap_s,
                         terminal_gap_m, match_me_seeds)
        rng = np.random.default_rng(seed)
        m, ne, nd = fly_lap_drought(rng, v_race, sigma_lat, sigma_vert, accel_bias_mag, bias_mode,
                                    fix_rate, vel_mode, latency_ms, vert_fix_bias_m=vert_fix_bias_m,
                                    schedule=schedule, burst_n=burst_n,
                                    terminal_gap_s=terminal_gap_s, terminal_gap_m=terminal_gap_m)
        misses[s] = m; emit[s] = ne; drop[s] = nd
    m = misses
    p50 = float(np.percentile(m, 50))
    p90 = float(np.percentile(m, 90))
    p99 = float(np.percentile(m, 99))

    # bootstrap 90% CI on p90 and p99 (B resamples). Reproducible RNG keyed off the cell.
    B = 1000
    bs_rng = np.random.default_rng(1234567 + int(round(terminal_gap_s * 1000)) * 17
                                   + int(round(fix_rate * 1000)) * 7 + int(round(v_race)))
    idx = bs_rng.integers(0, n_mc, size=(B, n_mc))
    samp = m[idx]
    p90_bs = np.percentile(samp, 90, axis=1)
    p99_bs = np.percentile(samp, 99, axis=1)
    p90_ci = [float(np.percentile(p90_bs, 5)), float(np.percentile(p90_bs, 95))]
    p99_ci = [float(np.percentile(p99_bs, 5)), float(np.percentile(p99_bs, 95))]

    frac_over = {f"{r:.2f}": float(np.mean(m >= ME.margin_at(r))) for r in ME.RADIUS_BAND}
    # CI-honest closure: UPPER-CI of p99 < MARGIN (and upper-CI of p90 < MARGIN).
    clears_pt = {f"{r:.2f}": bool(p90 < ME.margin_at(r) and p99 < ME.margin_at(r))
                 for r in ME.RADIUS_BAND}
    clears_ci = {f"{r:.2f}": bool(p90_ci[1] < ME.margin_at(r) and p99_ci[1] < ME.margin_at(r))
                 for r in ME.RADIUS_BAND}
    return dict(
        v_race=v_race, sigma_lat=sigma_lat, sigma_vert=sigma_vert, att_bias_deg=att_bias_deg,
        bias_mode=bias_mode, fix_rate=fix_rate, vel_mode=vel_mode, latency_ms=latency_ms,
        vert_fix_bias_m=vert_fix_bias_m, schedule=schedule, burst_n=burst_n,
        terminal_gap_s=terminal_gap_s, terminal_gap_m=terminal_gap_m, n_mc=n_mc,
        inplane_rms=float(np.sqrt(np.mean(m ** 2))), inplane_p50=p50, inplane_p90=p90,
        inplane_p99=p99, inplane_max=float(m.max()),
        p90_ci90=p90_ci, p99_ci90=p99_ci,
        mean_emitted=float(emit.mean()), mean_dropped=float(drop.mean()),
        frac_over=frac_over, clears_point=clears_pt, clears_ci=clears_ci,
        margin_030=ME.margin_at(0.30), margin_038=ME.margin_at(0.38),
    )


if __name__ == "__main__":
    import json
    # quick eyeball: regular baseline vs a 0.3s terminal drought at the headline closing cell.
    base = run_cell_drought(37.0, 0.1914, 0.1008, 0.0, "inplane", 0.50, "cold",
                            schedule="regular", terminal_gap_s=0.0, n_mc=400)
    dro = run_cell_drought(37.0, 0.1914, 0.1008, 0.0, "inplane", 0.50, "cold",
                           schedule="regular", terminal_gap_s=0.30, n_mc=400)
    for tag, r in (("regular D=0", base), ("drought D=0.3s", dro)):
        print(tag, json.dumps({k: r[k] for k in ("inplane_p90", "inplane_p99", "p99_ci90",
              "mean_emitted", "mean_dropped", "clears_ci")}))
    print("MARGIN(0.30)=", base["margin_030"], " MARGIN(0.38)=", base["margin_038"])
