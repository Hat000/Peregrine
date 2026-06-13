# a3 — FIX-STREAM REALISM at the gate-4 ~37 m/s window

ROLE: bound how the vision world-fix **stream** degrades when the drone is at ~37 m/s on the
g3->g4 approach, so a1 (KF agent) and verifiers feed the estimator a REALISTIC input instead of
the ~5.35 m/s VQ1-recording profile. Every number tagged **MEASURED / MODELED / ASSUMED**.

Reproduce: `PYTHONPATH=src .venv/Scripts/python.exe handoff/ultracode-estimator-racespeed-2026-06-13/a3_realism.py`
(seed 20260613). Outputs `a3_realism_results.json`. FACTS bias/noise NED reproduced exactly.

---

## TL;DR — the degraded fix-stream spec (what to hand a1)

**The fix STREAM at 37 m/s is dominated by CADENCE collapse, NOT by per-fix accuracy loss — UNLESS
the eval camera has a long exposure, which we cannot rule out offline.** Two regimes:

| quantity | VQ1 baseline (5.35 m/s, MEASURED) | 37 m/s window — SHORT shutter (<=2 ms) | 37 m/s window — LONG exposure (8 ms) |
|---|---|---|---|
| per-fix noise std N,E,D (m) | [0.816, 0.577, 0.436] | **x1.0–1.1** -> ~[0.82, 0.61, 0.46] | **x1.3–2.0 in-plane** -> ~[1.1, 0.9, 0.7] |
| per-fix BIAS N,E,D (m) | [-0.285, +0.064, -0.346] | **UNCHANGED** (blur is zero-mean) | UNCHANGED |
| acceptance | 0.47 | **~0.45** | **~0.30–0.35** |
| fix spacing | 0.187 m/frame | **1.29 m/frame** | 1.29 m/frame |
| frames in accept window [3,24] m | 112 | **~16** | ~16 |
| eff. ACCEPTED fixes in window | ~53 | **~7–8** | **~5** |
| last-fix range before transit | ~0.2 m | ~3 m (geom floor 1.3 m) | ~3 m |

**Single most important number for a1:** the effective accepted-fix count over the entire gate-4
approach drops from ~50 to **~5–8**. Variance averaging gives only ~1/sqrt(N) ≈ 1/sqrt(7) ≈ **0.38x**
crush, NOT the ~1/sqrt(50) ≈ 0.14x the dense-stream intuition suggests. **This is the binding
realism penalty, and it is geometric/cadence — it holds regardless of exposure.**

---

## (1) MOTION BLUR — quantified, and it is SMALL from translation alone

Chain: `omega_LOS = v·d_eff/r²` (closing-bearing rate) -> `pix_vel = omega·f` -> `blur_px = pix_vel·exposure`
-> `sigma_px(B) = sqrt(1.5² + (0.40·B)²)` -> scale the calibrated c2 (depth, r²) and a1 (lateral, r¹)
PnP-noise laws linearly in sigma_px (floors 0.40/0.282 m do NOT scale).

- **g3->g4 is nearly straight** (24.0 m along -N, 4.3 m E, 0.79 m D = 4.37 m transverse extent, only
  0.8 m descent). So the **translational bearing-sweep blur is small**: omega_LOS = 0.28 rad/s at
  24 m rising to 6.5 rad/s at 5 m; pixel velocity 90 px/s (24 m) -> 2070 px/s (5 m). At nominal 2 ms
  exposure that is **0.18 px (24 m) -> 4.1 px (5 m)** of smear; the world-fix noise multiplier stays
  **≤1.1x** for r ≥ 8 m and only reaches ~2x lateral at 5 m + 8 ms. **MODELED.**
- **THE DOMINANT BLUR SOURCE IS BODY ANGULAR RATE, NOT TRANSLATION.** `blur_px = omega_body·f·exposure`.
  At 8 ms exposure, just **22 deg/s of body rate = 1 px**; 100 deg/s = 4.5 px (lat noise x1.34);
  200 deg/s = 8.9 px (lat noise x2.0). At short global shutter (≤0.5 ms), even 200 deg/s = 0.56 px
  (negligible). **The drone banks onto the line and yaws to acquire gate-5 during this window**, so
  body rates of 50–200 deg/s are plausible — but the actual rate is set by the *trained policy's*
  flight profile and is **NOT in any data we hold**. **MODELED + the exposure is ASSUMED.**

**Verdict on blur:** with a racing-appropriate short/global shutter it is a non-issue (≤10% noise
inflation). With a long auto-exposure it can double the in-plane noise AND cut acceptance. **Which
regime the eval uses is the pivotal unknown (see escape hatch).**

## (2) FIX CADENCE in metres — the real penalty

30 Hz (true 28.6 fps, MEASURED from firstcontact handoff) at 37 m/s = **1.29 m/frame** (vs 0.187 m
at 5.35 m/s). Over the [3, 24] m accept window (~21 m) that is **~16 frames**; at 47% acceptance,
**~7–8 accepted fixes** feed the KF over the whole approach. For variance averaging the **effective
independent-fix count ≈ 7–8** (frame-to-frame noise is ~independent; per-track bias is common and
does NOT average). **MEASURED cadence × MODELED window.**

## (3) RANGE DISTRIBUTION

- Inner-gate pixel span = f·s/r: 15 px @ 32 m, 20 px @ 24 m, 40 px @ 12 m. Below ~15 px (r > 32 m)
  4-corner pose is unreliable -> the `vision_max_range_m = 32 m` nav cap is well-placed. **MEASURED/config.**
- **MEASURED detection ceiling in the actual data is only 23.3 m** (taken at 5.35 m/s). So
  "first-accept at 24 m" is the **charitable** assumption; at 37 m/s blur could pull the first
  reliable accept inward, shrinking the already-thin window further. **MEASURED ceiling, MODELED at-speed.**
- **Last-fix range before transit ≈ 3 m** (gate exits the 58.7° VFoV at r ≈ 1.33 m; geometric
  one-frame floor 1.29 m). So the freshest fix entering the gate-4 plane is **~3 m / 0.08 s stale**
  along-track — the latency-staleness term (v·L) is **mostly along-track (N)**, NOT in the binding
  in-plane (E,D) miss. **MODELED.**

## (4) ACCEPTANCE drop

Blur widens the fix cov while leaving the bias fixed, so the Mahalanobis chi² gate (16.27) rejects
more, and the CNN misses more corners. MODELED multiplicative factor on the 0.47 base:
**short shutter ~0.45, long 8 ms ~0.35** (mid-window). Combined with the cadence collapse this is
the second-order hit; the first-order hit is the 7x fewer frames. **MODELED.**

---

## VARIANCE vs BIAS — the part a1 must not miss

- The blur degradation is **zero-mean** -> it inflates NOISE only; the KF averages it down.
- The **per-fix BIAS [-0.285, +0.064, -0.346] m N,E,D is UNCHANGED by speed** (it is a geometry/
  gate-size-model systematic, not a motion effect). The in-plane (E,D) bias is **+0.064 / -0.346 m**.
  The **D bias alone (-0.346 m) is 7x the 0.05 m variance bar and 2.2x the 0.155 m gate-4 margin**,
  and it does NOT average out within one track. VISION-CAL removes the GLOBAL offset; the per-track
  residual registration sigma (~[0.21,0.24,0.03] m) is the floor. **This bias — not blur — is the
  binding absolute-pose risk, and it is speed-INDEPENDENT.** a1 should treat the degraded NOISE spec
  above as the averaging input, but the GO/NO-GO on <0.05 m in-plane absolute hinges on de-bias, not
  on the fix stream. Gate-RELATIVE observation sidesteps it.

---

## ESCAPE HATCH — what CANNOT be honestly pinned offline, and what resolves each

| # | Cannot pin offline | Why | What resolves it |
|---|---|---|---|
| E1 | **Camera EXPOSURE / shutter type** (global vs rolling, integration time) | Not in any spec/data we hold; it is the multiplier between "blur negligible" and "blur 2–4x". Training uses `A.MotionBlur(3–15)` (appearance only, not localization-accuracy). | **Organizer spec** (camera model / exposure / shutter), OR a ShadowPC recording where exposure can be inferred from a known body-rate maneuver. |
| E2 | **Body angular rate of the trained policy on g3->g4** | Sets the DOMINANT blur term (item [5]). Depends on the policy that doesn't exist yet (inc8). | **ShadowPC recording at speed** of the actual flown policy: log body rates + frames, measure smear directly. |
| E3 | **Per-fix accuracy AT 37 m/s** (the multipliers) | All MEASURED data has speed ≤ 8.4 m/s (median 5.35); the speed↔reproj correlation is **0.004** (no dynamic range to see blur). The 1.0–2.0x multipliers are MODELED extrapolation 4–7x beyond data. | **ShadowPC recording at race speed** through gate-4 with track_map ground truth -> recompute off_ned/reproj/acceptance at speed. The single highest-value at-speed capture. |
| E4 | **At-speed detection CEILING / first-accept range** | MEASURED ceiling 23.3 m is at low speed; blur may pull it in. | Same at-speed recording: first accepted frame range. |
| E5 | **At-speed ACCEPTANCE / chi² leak** | Modeled 0.35–0.45; gate-4 leak risk (0.53% baseline) could rise with widened-but-biased fixes. | Same recording: accepted fraction + Mahalanobis distribution at speed. |

**Bottom line for a1/verifiers:** use the **SHORT-shutter column** as the nominal degraded input
(noise x1.0–1.1, accept ~0.45, **~7–8 effective fixes**), and the **LONG-exposure column** as the
worst-case stress (noise x1.3–2.0 in-plane, accept ~0.35, ~5 fixes). The cadence-driven drop to
~7 fixes is robust and the **dominant** realism penalty; the blur multiplier band is exposure-gated
and is the one number a single ShadowPC at-speed recording (E1–E5) would collapse. The absolute-pose
GO/NO-GO is governed by the **speed-independent per-track BIAS**, not by this fix stream.
