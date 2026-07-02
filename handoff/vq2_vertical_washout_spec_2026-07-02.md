# VQ2 vertical-velocity WASHOUT filter — implementation spec (Fable, 2026-07-02)

Replaces the A21 floor-pin vertical KF (`src/racer/vertical_estimator.py`). Floor-height is a
FALSE-PREMISE dead end (warehouse has no floor grid) → the whole floor-pin z-correction path is
ripped out. The vertical channel is gate-RELATIVE; absolute height is neither obtainable nor needed.

NED down-positive unless stated. `a_up` is UP-positive (verified `a_up_from_specific_force`);
`a_dn = −a_up`.

## R0 — kill the position term (LOAD-BEARING)
`make_seeker_controller` (gate_seeker.py:141) sets `kp_alt=2.0`; vq2_case_c does NOT override it, so
on the ff-owns path `kp_alt·(vert_z_est − z_target)` was LIVE on a diverging double-integrated z —
a second, LARGER poison than the vz term. **Add `"kp_alt": 0.0` to vq2_case_c.controller_overrides**
(deploy_profile.py:241-243). Effective law becomes `thrust = hover + Kd·(vz − vz_t)`, clamped. The
z_target-ramp machinery goes inert (only fed the kp_alt term) — leave it, unused.

## 1 — Filter: first-order washout (leaky integrator), NOT a KF
A 1-D KF with a bias state == a washout when there are no measurements (bias unobservable without an
external fix; the only vertical fix on this wire was the floor pins being deleted). Delete the KF.

Discrete recurrence, run at IMU rate inside the existing `predict()` site (navigator.py:643-644):
```
α  = exp(−dt/τ)                 # dt = navigator IMU dt (line 614); keep guards: skip if dt<=0 or dt>max_dt_s=0.2
vz ← α·vz + (a_dn − b̂)·dt
vz_export = clip(vz, −3.0, +3.0)   # structural cannot-diverge guarantee
```
- **Rate:** IMU (~140 Hz), NOT 30 Hz control loop (sub-sampling aliases motor/frame vibration into vz).
  Controller reads the latest export per control tick. `exp(−dt/τ) ≈ 1 − dt/τ` acceptable (err <1e-5 @7ms).
- **τ = 2.0 s** (`washout_tau_s`, band 1.5–3.0). Kills the bob by BOUNDEDNESS (old spanned 27× clamp;
  washout is bounded + tracks true vz above 0.5 rad/s). Outer gate-align loop stays well damped
  (ζ≈0.62 at τ=2, Kd=0.25); τ=1 gives ζ≈0.44 (marginal). Bias residual = τ·b (want τ smallish).
- **Steady climb:** vz washes to 0 over τ (a_dn≈0) → servo degrades to accel channel for SUSTAINED
  commands. OK BY DESIGN: outer gate-align loop owns position closure (offset_z shrinks → vz_t shrinks).
- **Cannot-diverge:** state bounded by τ·sup|a_dn−b̂|; a bias gives a CONSTANT offset τ·b, not a ramp;
  ±3 export clip makes it absolute regardless of input garbage.

## 2 — Fusion: NONE in v1 (washout-alone); v2 hook seam default OFF
Gate-offset-rate fusion adds only low-freq relative vz the outer loop already has; the measurement is
bad (~4 Hz, ~250 ms age, track flap → differentiating σ≈0.2 m PnP z over 0.25 s ≈ σ 1.1 m/s + step
spikes on track switches). v2 hook `use_gate_vz_correction=False`: on accepted same-gate-id pose pair
Δt∈(0.05,0.8]s, `vz_gate=−(lever_z_now−lever_z_prev)/Δt` (reject |·|>2.5 or point-blank<2m),
`vz ← vz + 0.15·(vz_gate − vz)`. Enable only if acceptance M4 shows sustained |bias|>0.3 m/s.

## 3 — Seeding / edge cases / bias
- Seed vz=0, b̂=0 at Navigator `_initialize` (on pad → true vz 0). Export valid from tick 1. No z state.
- **Pre-arm bias capture:** first 1.0 s after seed (grounded), running-mean a_dn → b̂, freeze; clip ±0.5 m/s².
  On sim reset (`reset_counter` nav:604) Navigator re-inits → recapture (resets are grounded respawns).
- Slow a_up bias rejected → constant vz_ss=τ·b (post-capture ~0.1–0.18 m/s @0.3–0.5° tilt err; worst 2°→0.68 m/s, shows in M4).
- Tilt: `a_up_from_specific_force` exact at any attitude; keep dt>0.2 s skip guard.
- Dropouts: no vision input in v1; seeker sends vz_t=0 on dropout → level coast (benign).
- **No anti-windup needed:** filter integrates MEASURED accel with a leak — no error integrator to wind up.
  When thrust rails the IMU measures the true resulting accel → state stays physics-consistent; no stored windup.

## 4 — Plug-in + controller changes
- Keep `vertical_estimator.py`, repurposed: replace 3-state KF internals with washout state `(vz, b̂)`;
  DELETE `update_z` (dies with floor-pin rip-out nav:783-809) and the z arg of seed.
- Navigator predict site unchanged: `predict(a_up_from_specific_force(ds.accel_body, R_wb), dt)` (nav:643-644, before vision — keep).
- Exports: `vert_vz_est = clip(vz, ±3)`; **`vert_z_est = NaN` permanently** (only functional consumer was
  controller.py:412; rest is packaging/tests/replay — safe to retire).
- **REQUIRED controller change (controller.py:410-415):** current gate needs BOTH exports finite; with
  vert_z_est=NaN it would silently fall to the fd-of-pos[2] path = unbounded dead-reckoning (worse).
  Change gate to **finite vz_est ALONE**; z_v stays pos[2] (harmless, kp_alt=0). NaN-vz fallback (pre-init
  only): **vz=0**, NEVER the fd path (`thrust=hover+Kd·(0−vz_t)` benign open-loop track).
- **`ff_vertical_kd_alt`: 0.5 → 0.25.** At Kd=0.5 the inner crossover G=c·Kd (c=g/hover=36.9) = 18.5 rad/s
  ≈ 2.9 Hz — SITTING ON the observed 2.5–3 Hz bob, ~37° PM at 50 ms delay. Kd=0.25 → 9.2 rad/s (~1.5 Hz,
  PM≈64°), doubles linear window to (vz−vz_t)∈[−0.86,+1.34] m/s, leaves outer damping ≈unchanged.
- **Controller LP (controller.py:290 bypass): re-enable lightly** — apply existing LP to vz_meas with
  `ff_vertical_vz_lp_alpha=0.8` (~19 Hz @30 Hz: <5° lag at 1.5 Hz crossover; catches single-tick glitches).
  Do NOT keep 0.5 (fc≈3 Hz, ~40° lag — destabilizing at Kd=0.5).

Param table: washout_tau_s=2.0 · max_dt_s=0.2 · export clip ±3.0 · bias-capture 1.0 s clip ±0.5 m/s² ·
ff_vertical_kd_alt=0.25 · ff_vertical_vz_lp_alpha=0.8 on vz_meas · kp_alt=0.0 on vq2_case_c ·
v2 hook k_g=0.15 Δt∈(0.05,0.8] |vz_gate|≤2.5 default OFF.

## 5 — Offline acceptance (commander runs; replay of a CLEAN flight — see COMMANDER NOTE)
Reconstruct a_up per IMU sample (recorded HIGHRES_IMU accel + logged ahrs_quat_wxyz via
a_up_from_specific_force); run recurrence; take recorded vz_t per control tick; synthesize
thrust = clip(0.2656 + 0.25·(vz − vz_t), 0.05, 0.60).
- M1 boundedness: max|vz| per 5-s window ≤ 2.0 m/s (baseline ±5–10). Hard.
- M2 rail occupancy: <10% of ticks at a rail (baseline ~75%), ZERO rail-to-rail flips within ≤3 ticks in any 10-s window.
- M3 spectrum: thrust PSD 2–4 Hz band down ≥20 dB vs flown log.
- M4 truth cross-check: over ≥1 s same-gate accepted-pose stretches, mean|vz_washout − (−Δlever_z/Δt)| ≤ 0.5 m/s + sign agree; on-pad |vz|≤0.15. Sustained >0.3 m/s consistent-sign ⇒ enable v2 hook.
- M5 linear window: ≥80% ticks |vz−vz_t| ≤ 1.0 m/s (baseline ≈0%).
Run M1–M3 at Kd=0.25 AND 0.5 to document margin; ship 0.25.

## COMMANDER ADDITIONS (Opus, 2026-07-02) — fold into implementation
- **A_UP INPUT OUTLIER CLAMP (new):** clamp `a_dn` to ±30 m/s² (±3g) BEFORE the washout update. A real
  slow-lap drone cannot sustain >3g vertical; this rejects contact-impact spikes AND sensor glitches at
  the SOURCE (not just the ±3 output clip). Empirically material on the contaminated replay: full-flight
  rail 81%→49%, in-band 19%→52%. Cheap, sound, defensive. Add `a_up_clamp_mps2=30.0`.
- **VALIDATION DATA CAVEAT:** run 20260702_171324 is a POOR benchmark — 1782 collisions from t=0.36s,
  flight-window |f| median 10.3 (sane) but 14.5% of samples >3g, max 200g (contact spikes). Offline
  replay on it is CONFOUNDED (and a_up is trajectory-dependent — the fixed closed loop flies different
  a_up). The offline M-tests need a CLEAN (few-collision) recording; the TRUE test is the next closed-loop
  flight. Offline replay already confirmed: estimator diverges (unbounded ±20), washout+clip bounds it (±3).
