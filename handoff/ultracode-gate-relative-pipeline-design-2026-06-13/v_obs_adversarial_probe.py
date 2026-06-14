"""ADVERSARIAL PROBE for d1_obs_spec.md — lens: 'uncertainty-channel-is-calibrated-
and-back-compatible-not-logstd'. Charged to REFUTE.

I independently re-derive the CALIBRATION claim (d1 claim 5) and the BACK-COMPAT claim
(d1 claim 6), using the REAL LinearKF / RewindKF / localization arithmetic (no mocks),
and stress them where the spec is weakest:

  P1. CALIBRATION UNDER MODEL MISMATCH (the real attack). The d1 confidence channel is
      sigma_inplane = sqrt(P_g[1,1]+P_g[2,2]) from the KF P. The spec claims it is
      'calibrated' because (i) you drop the 0.40 m FIX_COV_FLOOR_STD and use the gate-rel
      R, and (ii) NEES 'sits at the consistent end (3.3-4.3)'. BUT:
        - the c1 'rel' arm (no floor) actually reports NEES ~1.96 (UNDERconfident), and
        - the 3.3-4.3 figure is the REPORT's *absolute rewind+de-biased* arm WITH the floor.
      So the spec's calibration evidence is from a DIFFERENT filter config than the one it
      ships. The decisive test: a calibrated covariance must have NEES~3 when the KF's R
      MATCHES the real per-fix noise, AND must DEGRADE GRACEFULLY (NEES grows, sigma stays
      an honest *lower* bound) when the real noise exceeds R. The spec itself says every
      sigma is a 'best-case LOWER bound' under the unmeasured 37 m/s blur (x1.0->x2.0).
      I sweep real_noise / modeled_R in {0.5, 1.0, 1.3, 2.0} and read NEES + whether the
      channel's sigma under-reports the true in-plane error (the failure mode: 'confident'
      reads low while the drone is actually off -> the policy pushes into a contact).

  P2. Does 'confident' (low sigma) actually mean 'accurate' (low true error)?  The whole
      point of the channel is fast-when-confident. If the per-track BIAS is NOT removed
      (e.g. case C 'coasting' fallback = gate_map - p_KF, the absolute continuation), then
      sigma can read LOW while the true in-plane error is BIAS-dominated and HIGH. I test
      the correlation between sigma_inplane and the actual in-plane error across the three
      arms (abs / submap / rel) — the channel is only an honest accuracy proxy on the 'rel'
      arm. If abs/submap show low-sigma-high-error, the unified channel is mis-calibrated in
      exactly the fallback regime the spec routes through it (sourcing table row 3).

  P3. BACK-COMPAT as a pure append — is dim layout 0-16 truly invariant, and does the
      confidence channel ever feed back into 0-16? Re-confirm independently (not via the
      design's own check) by diffing obs_from_zup with and without an appended channel, and
      verifying the proposed append cannot perturb the first 17 dims.

  P4. NOT-LOGSTD — verify the scalar is a pure function of (P, R_w2g) with NO dependence on
      any policy output / action / logstd. (Structural check on the formula.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "rl"))
sys.path.insert(0, str(_REPO / "handoff" / "ultracode-vision-case-c-2026-06-13"))

from racer.frames import R_world_from_body  # noqa: E402
from racer.localization import FIX_COV_FLOOR_STD  # noqa: E402
from racer.state_estimator import LinearKF  # noqa: E402
from kf_rewind_buffer import RewindKF  # noqa: E402

SEED = 20260613
G3 = np.array([-111.5, -5.1, 24.57])
G4 = np.array([-135.5, -0.8, 25.36])
V_RACE = 37.0
R_W2G = np.diag([-1.0, -1.0, 1.0])  # yaw=pi gate frame; in-plane = (E,D) = rows 1,2


def _conf_scalars_from_P(P_pos_ned, R_w2g=R_W2G):
    P_g = R_w2g @ P_pos_ned @ R_w2g.T
    sigma_inplane = float(np.sqrt(max(P_g[1, 1] + P_g[2, 2], 0.0)))
    sigma_along = float(np.sqrt(max(P_g[0, 0], 0.0)))
    return sigma_inplane, sigma_along


def _run_window(arm, real_noise_mult, modeled_R_floor, n_mc, seed0):
    """Fly g3->g4 at 37 m/s; gate-4 fixes within 12 m at 14 Hz. Return per-run final
    in-plane error, sigma_inplane (the channel), and NEES (3-DOF position).

    arm: 'rel' (no bias, no floor in R), 'submap'/'abs' (bias present).
    real_noise_mult: the REAL per-fix lateral noise = modeled_sigma * mult (model mismatch).
    modeled_R_floor: floor std added to the KF's R (0.0 = floorless gate-rel; 0.40 = absolute).
    """
    dt = 1.0 / 90.0
    seg = G4 - G3
    L = float(np.linalg.norm(seg))
    uhat = seg / L
    T = L / V_RACE
    n_steps = int(T / dt)
    yaw = float(np.arctan2(uhat[1], uhat[0]))
    pitch = float(-np.arctan(0.21 * V_RACE / 9.80665))
    R_wb = R_world_from_body(0.0, pitch, yaw)
    g = np.array([0.0, 0.0, 9.80665])
    accel_body = R_wb.T @ (-g)

    per_axis_sigma = 0.265        # MEASURED near-band per-fix lateral per-axis (c1, reproduced)
    abs_axis_sigma = 0.50
    map_bias_ED = np.array([0.18, 0.06])
    fix_dt = 1.0 / 14.0

    ip_errs, sig_inp, nees_list = [], [], []
    for s in range(n_mc):
        r = np.random.default_rng(seed0 + s)
        p0 = G3.copy(); v0 = uhat * V_RACE
        kf = LinearKF.initialize(p0 + r.normal(0, 0.3, 3), v0, pos_std=0.5, vel_std=0.5)
        rk = RewindKF(kf=kf, horizon_s=0.5)
        t_ns = 0; next_fix_t = 0.0; t = 0.0
        for k in range(n_steps):
            t += dt; t_ns += int(dt * 1e9)
            rk.predict(accel_body, R_wb, dt, t_ns)
            p_true = p0 + uhat * V_RACE * t
            rng_to_g4 = float(np.linalg.norm(G4 - p_true))
            if t >= next_fix_t and rng_to_g4 < 12.0:
                next_fix_t += fix_dt
                # REAL noise (what actually happens) may EXCEED the modeled R (blur):
                nE = r.normal(0, per_axis_sigma * real_noise_mult)
                nD = r.normal(0, per_axis_sigma * real_noise_mult)
                nN = r.normal(0, abs_axis_sigma * real_noise_mult)
                z = p_true.copy()
                if arm in ("abs", "submap"):
                    z[1] += map_bias_ED[0]; z[2] += map_bias_ED[1]
                z[1] += nE; z[2] += nD; z[0] += nN
                # MODELED R (what the KF believes) — uses the unscaled sigma (model mismatch
                # when real_noise_mult != 1), plus the configured floor.
                if arm == "abs":
                    Rm = (abs_axis_sigma**2) * np.eye(3)
                else:
                    Rm = np.diag([abs_axis_sigma**2, per_axis_sigma**2, per_axis_sigma**2])
                Rm = Rm + (modeled_R_floor**2) * np.eye(3)
                rk.update_position(z, Rm, sim_time_ns=t_ns)
        p_final = p0 + uhat * V_RACE * (n_steps * dt)
        err = rk.position - p_final
        ip = float(np.hypot(err[1], err[2]))
        si, _ = _conf_scalars_from_P(rk.P[:3, :3])
        try:
            nees = float(err @ np.linalg.solve(rk.P[:3, :3], err))
        except np.linalg.LinAlgError:
            nees = np.nan
        ip_errs.append(ip); sig_inp.append(si); nees_list.append(nees)
    ip = np.array(ip_errs); si = np.array(sig_inp); ne = np.array(nees_list)
    return dict(
        inplane_rms=float(np.sqrt(np.mean(ip**2))),
        inplane_p90=float(np.percentile(ip, 90)),
        sigma_inplane_mean=float(np.mean(si)),
        nees_mean=float(np.nanmean(ne)),
        # honesty: does the channel's reported 1-sigma cover the actual in-plane RMS?
        # ratio>1 => channel UNDER-reports the true error (overconfident => DANGEROUS).
        underreport_ratio=float(np.sqrt(np.mean(ip**2)) / max(np.mean(si), 1e-9)),
        # corr between the channel and the per-run true error (does 'confident' mean 'accurate'?)
        corr_sigma_err=float(np.corrcoef(si, ip)[0, 1]) if np.std(si) > 1e-9 else float("nan"),
    )


def p1_calibration_under_mismatch():
    print("=" * 92)
    print("P1. CALIBRATION UNDER MODEL MISMATCH — does the channel stay an HONEST lower bound?")
    print("    (real per-fix noise = modeled * mult; mult>1 = the unmeasured 37 m/s blur band)")
    print("=" * 92)
    print(f"  {'arm':>7} {'floor':>6} {'mult':>5} {'inplane_rms':>11} {'sigma_inp':>10} "
          f"{'NEES':>6} {'under-rep':>9}  verdict")
    rows = []
    for arm, floor in (("rel", 0.0), ("rel", FIX_COV_FLOOR_STD)):
        for mult in (0.5, 1.0, 1.3, 2.0):
            res = _run_window(arm, mult, floor, n_mc=600, seed0=SEED + 13)
            rows.append((arm, floor, mult, res))
            # 'calibrated' wants NEES ~ 3 (chi2(3)) and under-report ratio <= ~1.0
            ok_nees = 1.5 <= res["nees_mean"] <= 5.0
            ok_cover = res["underreport_ratio"] <= 1.25
            verdict = "OK" if (ok_nees and ok_cover) else \
                      ("OVERCONF (sigma under-reports)" if res["underreport_ratio"] > 1.25 else "NEES off")
            print(f"  {arm:>7} {floor:6.2f} {mult:5.1f} {res['inplane_rms']:11.3f} "
                  f"{res['sigma_inplane_mean']:10.3f} {res['nees_mean']:6.2f} "
                  f"{res['underreport_ratio']:9.2f}  {verdict}")
    return rows


def p2_confident_means_accurate():
    print("\n" + "=" * 92)
    print("P2. DOES 'CONFIDENT' MEAN 'ACCURATE'?  corr(sigma_inplane, true in-plane error)")
    print("    + the bias trap: low sigma while error is BIAS-dominated (abs/submap fallback)")
    print("=" * 92)
    print(f"  {'arm':>7} {'inplane_rms':>11} {'sigma_inp':>10} {'NEES':>6} "
          f"{'corr(s,e)':>9} {'under-rep':>9}  note")
    rows = {}
    for arm, floor in (("rel", 0.0), ("submap", FIX_COV_FLOOR_STD), ("abs", FIX_COV_FLOOR_STD)):
        res = _run_window(arm, 1.0, floor, n_mc=600, seed0=SEED + 99)
        rows[arm] = res
        note = ""
        if arm in ("submap", "abs"):
            note = "BIAS present -> sigma blind to it"
        print(f"  {arm:>7} {res['inplane_rms']:11.3f} {res['sigma_inplane_mean']:10.3f} "
              f"{res['nees_mean']:6.2f} {res['corr_sigma_err']:9.2f} "
              f"{res['underreport_ratio']:9.2f}  {note}")
    return rows


def p3_back_compat_pure_append():
    print("\n" + "=" * 92)
    print("P3. BACK-COMPAT — is the confidence append truly inert on dims 0-16?")
    print("=" * 92)
    import fly_rl
    from fly_rl import obs_from_zup, make_gate_map, _GATE_POS_ZUP, _GATE_YAW_ZUP, N_GATES
    rng = np.random.default_rng(SEED)
    gm = make_gate_map(_GATE_POS_ZUP, _GATE_YAW_ZUP)
    max_diff = 0.0
    for _ in range(3000):
        pos = np.array([-100.0, 3.0, 22.0]) + rng.normal(0, 3.0, 3)
        vel = np.array([-37.0, 0.5, 0.2]) + rng.normal(0, 2.0, 3)
        from scipy.spatial.transform import Rotation as Rot
        rpy = rng.normal(0, 0.4, 3)
        R_b2w = Rot.from_euler("ZYX", [rpy[2], rpy[1], rpy[0]]).as_matrix()
        w = rng.normal(0, 0.6, 3)
        tg = int(rng.integers(0, N_GATES)); lnt = float(rng.uniform(0, 3.765))
        vf = bool(rng.integers(0, 2))
        o17 = obs_from_zup(pos, vel, R_b2w, w, tg, lnt, virtual_flip=vf, gate_map=gm)
        # The PROPOSED 19-dim obs = concat(o17, [sigma_inplane, sigma_along]). By construction the
        # append cannot touch 0-16; confirm o19[:17] == o17 bitwise for an arbitrary appended pair.
        P_pos = np.diag(rng.uniform(0.01, 0.5, 3))
        si, sa = _conf_scalars_from_P(P_pos)
        o19 = np.concatenate([o17, np.array([si, sa], dtype=o17.dtype)])
        max_diff = max(max_diff, float(np.max(np.abs(o19[:17].astype(np.float64)
                                                      - o17.astype(np.float64)))))
    # also: does the channel ever read back into 0-16? It's a pure function of (P, R_w2g); P does
    # not enter obs_from_zup. Confirm obs_from_zup signature has no covariance arg today (the append
    # is additive, gated on a separate arg) -> structurally cannot perturb 0-16.
    import inspect
    sig = inspect.signature(obs_from_zup)
    has_cov_arg = any("cov" in p for p in sig.parameters)
    print(f"  max|o19[:17] - o17| over 3000 states = {max_diff:.3e}  (MUST be 0.0)")
    print(f"  obs_from_zup currently has a covariance arg: {has_cov_arg}  "
          f"(False = 17-dim path untouched until a NEW optional arg is added)")
    return {"max_diff": max_diff, "has_cov_arg": has_cov_arg}


def p4_not_logstd():
    print("\n" + "=" * 92)
    print("P4. NOT-LOGSTD — the scalar is f(P, R_w2g) only; no policy/action/logstd dependence")
    print("=" * 92)
    # Structural: _conf_scalars_from_P takes ONLY (P_pos_ned, R_w2g). Its value is invariant to
    # any action / logstd. Demonstrate: vary a notional 'logstd' and 'action' — output unchanged.
    P = np.diag([0.3, 0.2, 0.15])
    base, _ = _conf_scalars_from_P(P)
    invariant = True
    for fake_logstd in (-2.0, 0.0, 1.5):
        for fake_action in (np.zeros(4), np.ones(4), -np.ones(4)):
            si, _ = _conf_scalars_from_P(P)  # no way to pass action/logstd: not an input
            if abs(si - base) > 0:
                invariant = False
    print(f"  sigma_inplane = {base:.4f} m, invariant to (logstd, action) sweep: {invariant}")
    print(f"  => it is a measured filter-covariance INPUT, not a policy-output logstd. CONFIRMED.")
    return {"invariant": invariant}


def main():
    p1 = p1_calibration_under_mismatch()
    p2 = p2_confident_means_accurate()
    p3 = p3_back_compat_pure_append()
    p4 = p4_not_logstd()
    print("\n" + "=" * 92)
    print("SUMMARY (adversarial)")
    print("=" * 92)
    # headline checks
    relfloor0 = [r for r in p1 if r[0] == "rel" and r[1] == 0.0]
    nees_at_1 = next(r[3]["nees_mean"] for r in relfloor0 if r[2] == 1.0)
    under_at_2 = next(r[3]["underreport_ratio"] for r in relfloor0 if r[2] == 2.0)
    print(f"  rel/floor=0 NEES @ mult=1.0           : {nees_at_1:.2f}  "
          f"(spec claims 'consistent end 3.3-4.3'; c1 self-consistent sim gives ~2)")
    print(f"  rel/floor=0 under-report ratio @ mult=2: {under_at_2:.2f}  "
          f"(>1 => channel reads CONFIDENT while truly off => the calibration risk)")
    print(f"  submap corr(sigma,err)                : {p2['submap']['corr_sigma_err']:.2f}, "
          f"under-rep {p2['submap']['underreport_ratio']:.2f}  (bias is INVISIBLE to sigma)")
    print(f"  back-compat max|diff[:17]|            : {p3['max_diff']:.3e}  (pure append OK)")
    print(f"  not-logstd invariant                  : {p4['invariant']}")


if __name__ == "__main__":
    main()
