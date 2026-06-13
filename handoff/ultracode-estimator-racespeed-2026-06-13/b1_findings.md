# b1 — ADVERSARIAL verification of a1's gate-4 race-speed estimator headline

Agent **b1** (estimator/perception, opus-4.8). ROLE: charged to **REFUTE** a1's headline. Offline
re-run only. Composes the SAME real stack a1 used (`racer.state_estimator.LinearKF` + `RewindKF` +
`racer.localization.gate_pose_to_world_position`), perturbs a1's assumptions, and re-measures.

Files: `b1_verify.py` (re-runs), `b1_verify_results.json` (13 cells). Reproducible: seed 20260613,
600 seeds/cell (>a1's 400). Numbers reproduce on re-run.

**a1 headline under attack:** "Case-C absolute world-fix CANNOT reach 0.05 m in-plane 1-σ at the
gate-4 37 m/s window; best filtered in-plane 1-σ ≈ 0.24 m/axis (in-plane RMS 0.33 m), DEBIASED, edge
L, 47% acceptance; NO cell clears (raw or debiased, any L/acceptance); binding term = VARIANCE; P is
honest in the deployed rewind+debiased path."

---

## VERDICT: **CONFIRMED** (on the <0.05 m claim, RAW and DEBIASED separately)

The headline **survives every adversarial perturbation**. I could not refute it; I made it *worse* in
every honest direction. RAW and DEBIASED both fail the 0.05 m in-plane bar by ≥5×, and the binding
term is indeed VARIANCE (with a per-gate BIAS floor making RAW and the *honest* debiased path fail
harder). One real correction: a1's reported DEBIASED 0.24 m is **optimistic** — the truly-deployable
(globally-de-biased, per-gate residual intact) figure is **~0.55 m in-plane RMS**, not 0.33 m. The
verdict direction is unchanged; the headline *number* should be corrected upward. Hence CONFIRMED, not
strengthened-only.

### Independent reproduction of a1's headline cell (sanity — the sim is sound)
| metric | a1 reported | b1 independent re-run | match |
|---|---|---|---|
| DEBIASED claimed 1-σ E/D (L6, 47%) | 0.239 / 0.239 | **0.239 / 0.239** | exact |
| DEBIASED in-plane RMS | 0.33 | **0.328** | exact |
| DEBIASED empirical std E/D | (≈claimed) | 0.244 / 0.219 | consistent |
| RAW in-plane RMS | 0.46 | **0.464** | exact |
| floor-probe: per-fix 0.50→0.30→0.10→0.05 filtered | 0.323/0.226/0.098/0.051 | 0.310/0.226/0.097/0.052 | match |
| per-fix σ near gate / @16 m | ~0.50 / 0.64 | **0.50 / 0.63** | exact |

a1's sim, fix-cov model, and floor-probe all reproduce independently. No arithmetic or API error found.

---

## Attack-by-attack results (all in-plane RMS, m, DEBIASED unless noted; bar = 0.05)

| # | attack | a1 cell | b1 perturbed | Δ | clears 0.05? |
|---|---|---|---|---|---|
| repro | a1 headline (L6, 47%, no blur, no leak) | 0.33 | **0.328** | 0 | NO |
| (1) | **A3 SHORT-shutter** (noise ×1.05, accept 0.45) | — | **0.346** | +0.02 | NO |
| (1) | **A3 LONG-exposure** (in-plane noise ×1.7, along ×1.3, accept 0.33, ~6.5 fixes) | — | **0.521** | +0.19 | NO |
| (2) | **leak injected** @0.0053 (a1 omitted) | — | **0.342** | +0.01 | NO |
| (2) | leak @5× stress (0.0265) | — | **0.340** | +0.01 | NO |
| (3) | **HONEST debias** (leaves a2 per-gate gate-4 residual) | — | **0.546** | +0.22 | NO |
| (4) | **COMBINED honest-worst** (A3 long + leak + per-gate residual) | — | **0.775** | +0.45 | NO |
| (5) | CPU L=112 ms (horizon check) | 0.35 | **0.374** | +0.02 | NO (no divergence) |
| (6) | idealized 100% accept (best variance crush) | ~0.25 | **0.252** | 0 | NO |

**Every cell fails by ≥5×.** No perturbation — favourable or adversarial — clears the bar. The single
direction that could refute (lower per-fix σ) requires per-fix ≈0.05 m (16× better), independently
re-confirmed.

---

## Answering the 6 charged attack questions

**(1) Is the fix stream realistic? Did a1 use the slow-speed stream where it should use A3-degraded?**
PARTLY a fair hit, but it does NOT save the headline. a1 fed the **5.35 m/s** resampled `off_ned`
pool at 47% accept with **NO motion-blur multiplier** → that is implicitly the A3 **SHORT-shutter**
regime (A3: noise ×1.0–1.1, accept ~0.45). a1 did NOT cross-check A3's LONG-exposure column. I re-ran
with both A3 degraded specs:
 - SHORT shutter → in-plane RMS **0.346** (vs a1 0.33): negligible, a1's implicit regime is fine.
 - LONG exposure (×1.7 in-plane noise, accept 0.33, cadence drops to **~6.5 effective fixes**) →
   in-plane RMS **0.521**. The fix COUNT is honest in a1 (≈9.4 fixes @47% over the 24.4 m straight,
   matching A3's ~7–8 — a1 did **not** use a dense 50-fix stream; that attack fails).
 **Delta: A3-degraded inputs move the headline from 0.33 → 0.35 (short) / 0.52 (long), all worse, all
 NO-GO.** a1's slow-speed assumption was the OPTIMISTIC end of the A3 band; using A3-degraded only
 hardens the verdict.

**(2) Is the motion model honest at 37 m/s (process noise / IMU / attitude)?** YES, and it's not the
binding term. a1 uses the real `LinearKF.predict` with `accel_noise_std=0.3`, the 1.4° attitude-skew
Q injection (`ATTITUDE_NOISE_STD_RAD`), drag-hold pitch 38.4°, 90 Hz IMU. I confirmed position
process-growth over a fix interval is ~mm — process noise is sub-resolution against the 0.24–0.55 m
variance wall. Perturbing attitude/accel changes nothing material (a1 already showed the 5°/+5 m/s²
variants move <0.02 m; consistent with my cells). **Not refutable here.**

**(3) Does the cov floor / 0.53% leak / chi2 gate degrade it — are leaks included at real rate?**
a1 **OMITTED** the catastrophic leak entirely (real omission). I injected chi²-gate-bounded leaks
(maha ≤ 16.27 vs the inflated cov → bounded offset ~√16.27·σ in a random direction, **not** the 139 m
gross outliers that the gate rejects) at the measured 0.0053 rate and at a 5× stress rate. Effect:
in-plane RMS 0.328 → **0.342** (1× rate) / **0.340** (5× rate). **Immaterial** — at ~9 fixes the
per-run leak count is ~0.05, and RewindKF + the 0.40 m bias-absorption floor keep a single bounded
leak sub-resolution against the variance wall. a1's omission did NOT flatter the result. The 0.40 m
floor is real and correctly applied (per-fix σ reproduces at ~0.50 m near gate). **Attack fails;
robustness confirmed.**

**(4) Is the rewind buffer used with horizon ≥ L (else ~21 m divergence)? Check the CPU cell.**
CONFIRMED safe. At CPU L=112 ms with horizon 0.5 s: mean dropped fixes **0.49** (only the very first
pre-buffer fix, exactly as a1/a4 stated — NOT a horizon-driven run-drop), in-plane RMS **0.374**, **no
~21 m divergence**. Horizon 0.5 s ≫ 112 ms with ~0.39 s margin. a1 and a4 are correct; this is not a
divergence trap at these L. (It WOULD diverge only if L spiked past 0.5 s — a detector stall — which
is a separate guard, not this regime.)

**(5) NEES — is the claimed P honest, or is a1 reading filter-P as truth while empirical is worse?**
a1 is **honest in the deployed (rewind + globally-debiased) path** — I reproduce DEBIASED NEES
**3.4–4.6** (near χ²(3)=3, marginally overconfident) and the claimed 1-σ (0.239) matches the empirical
std (0.244/0.219). a1 is NOT reading P-as-truth there. **BUT** the two arms a1 under-weighted are
over-confident: RAW NEES **7.5**, and the **honest-debias** (per-gate residual intact) arm NEES
**8.5** — the χ² flag that the *truly-deployable* path carries unmodeled bias the P does not see. So
a1's "P is honest" is true for the idealized debiased arm but **overstated for the realistic
deployed arm**, where P under-reports by ~2.8×.

**(6) Does a1 conflate VARIANCE with BIAS anywhere?** YES — one real conflation, and it is the only
substantive correction. a1's "DEBIASED" arm subtracts the **whole-course per-band mean**
(`[0,8)=[-0.376,-0.003,-0.255]`, `[8,16)=[-0.177,+0.017,-0.422]`) — a legitimate GLOBAL de-bias, NOT
gate-4-cheating (good). But a2 independently showed gate-4's **per-gate** in-plane bias *survives*
global de-bias at ≈[E −0.44, D −0.08] (in-plane ≈0.45 m), because gate-4's lateral error is genuinely
per-gate, not global. a1's debiased arm removed bias a real global de-bias cannot remove at gate-4
(its debiased filtered bias was ~[+0.01, 0, −0.03]). So **a1's 0.24 m DEBIASED 1-σ is the pure-noise
floor, not the deployable in-plane error.** Re-running with the a2 per-gate residual left in:
in-plane RMS **0.546 m** (NEES 8.5). a1 reported the variance term as if it were the total debiased
error; the honest deployable debiased number is ~0.55 m.

---

## Corrected numbers (what changed after the adversarial re-run)

| quantity | a1 reported | b1 corrected | why |
|---|---|---|---|
| best DEBIASED in-plane 1-σ (idealized, bias fully removed) | 0.24 m | 0.24 m (UNCHANGED — but it is a *variance-only* floor, not deployable) | reproduced exactly |
| **deployable** (global-debias, per-gate residual intact) in-plane RMS | (implied 0.33) | **~0.55 m** | a1 conflated variance with deployable total; a2 per-gate bias survives |
| A3-degraded LONG-exposure in-plane RMS | not run | **0.52 m** | a1 used optimistic short-shutter regime |
| combined honest-worst in-plane RMS | not run | **0.78 m** | A3-long + leak + per-gate residual |
| margin over 0.05 m bar (deployable) | ~5× | **~7–11×** | bias term restored |
| per-fix σ needed to clear @14 Hz | ~0.03–0.05 m | 0.05 m (16× better) | independently re-confirmed |
| NEES, realistic deployed path | "3.3–4.3, honest" | 3.4–4.6 (idealized) / **8.5 (honest-debias)** | P overconfident on the deployable arm |

**Headline as corrected:** Case-C absolute world-fix CANNOT reach 0.05 m in-plane 1-σ at the gate-4
37 m/s window. The pure-noise variance floor is ~0.24 m (5× over); the **deployable** in-plane error
(after a realistic global de-bias, with the speed-independent per-gate gate-4 bias intact) is
**~0.55 m** (≈11× over the bar, ≈3.5× over the 0.155 m contact margin), and the honest-worst with
A3 long-exposure + leak is **~0.78 m**. NO cell clears, RAW or DEBIASED, any L / acceptance / blur /
leak. The binding term is **VARIANCE** (per-fix ~0.50 m floor, KF averages only ~9 fixes since vel is
unobservable in case C); a per-gate **BIAS** floor (~0.45 m in-plane, un-filterable, survives global
de-bias) makes both RAW and the *honest* debiased path fail additionally. The estimator is the binding
VQ2 validity risk — confirmed and quantified upward.

---

## What this means (carries a1's recs, with two amendments)
1. **Gate-RELATIVE observation remains the only candidate fix** (sidesteps BOTH the per-gate absolute
   BIAS *and* converts the variance to close-range corner reprojection). a1 and a2 both name it; this
   re-run confirms absolute case-C has no clearing cell, so gate-relative is not optional.
2. **AMENDMENT to a1 rec-3 (de-bias):** a *global* VISION-CAL de-bias is necessary but NOT sufficient
   at gate-4 — it leaves ~0.45 m in-plane per-gate residual. Only a **per-gate** de-bias (needs a
   privileged-pose cal lap, case A/B) pulls it toward ~0.07 m (a2 §4), and a vision-only cal lap
   cannot reach the bar (chain-circularity). Do not bank the 0.24 m number as deployable.
3. **AMENDMENT to a1 rec on P-honesty:** P is honest only on the idealized debiased arm; the
   realistic deployed arm (per-gate residual intact) has NEES ~8.5 → the filter under-reports its
   error by ~2.8×. The chi² monitor should treat sustained NEES≫3 at gate approach as the live tell
   of un-removed registration bias.
4. CONDITIONAL on organizer Q① — if VQ2 streams LOCAL_POSITION_NED/ODOMETRY (case A/B), pose is
   pristine and this whole risk is moot. This is the case-C worst case.
5. Per-fix pool is ~5.35 m/s data; 37 m/s motion blur is MODELED not MEASURED — so even the ~0.55 m
   deployable figure is a **best-case lower bound** (A3 long-exposure 0.52–0.78 m is the honest band).
   The single highest-value resolver is a ShadowPC at-speed recording through gate-4 (A3 E1–E5).

---

## MEMORY-DELTA:
- b1 ADVERSARIAL re-run of a1: **VERDICT CONFIRMED** (could not refute; made it worse in every honest
  direction). Reproduced a1 headline EXACTLY (DEBIASED claimed E/D 0.239/0.239, in-plane RMS 0.328;
  RAW 0.464; floor-probe + per-fix σ ~0.50 m all match). a1's sim/cov/floor are sound.
- **ONE real correction (a1 conflated variance with deployable bias):** a1's 0.24 m DEBIASED 1-σ is a
  *pure-noise floor*, not the deployable error. a1's debias removed the gate-4 per-gate bias a real
  GLOBAL de-bias cannot remove (a2: ~[E−0.44,D−0.08] survives). HONEST deployable in-plane RMS =
  **~0.55 m** (≈11× over 0.05 m bar, ≈3.5× over 0.155 m margin); combined honest-worst (A3 long +
  leak + per-gate residual) = **~0.78 m**.
- A3-degraded inputs: short-shutter 0.35 (a1's implicit regime, fine), LONG-exposure 0.52 — a1 used
  the optimistic end. Fix COUNT honest (~9 @47%, matches A3 ~7–8). Catastrophic-leak omission
  IMMATERIAL (0.328→0.342 @ real rate, 0.340 @ 5×). CPU horizon 0.5 s ≥ 112 ms CONFIRMED (drop 0.49,
  ipRMS 0.374, no ~21 m divergence).
- NEES: a1 honest on idealized debiased arm (3.4–4.6) but OVERCONFIDENT on the realistic deployed arm
  (per-gate residual → NEES 8.5, P under-reports ~2.8×). BINDING TERM = VARIANCE confirmed; per-gate
  BIAS floor makes RAW + honest-debias fail additionally. Estimator IS the binding VQ2 risk.
