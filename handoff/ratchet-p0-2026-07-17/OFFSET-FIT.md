# Ratchet-P0.2 — systematic right-offset FIT (analysis-only): it is a LAUNCH-POSE / initial-state delta (~+11° heading ⇒ ~+2 m right at gate 0), NOT plant asymmetry and NOT a resample artifact

2026-07-18, laptop RL-commander side, branch `ratchet-tape-2026-07-17`. Fits the residual the
ShadowPC P0.2 probe left open (REPORT3): all settled replays fly a repeatable line RIGHT of every
gate. Inputs: the 3 settled replays (`20260718_02*_ratchet_p02_r{1,2,3}_f1`) + champion source
(`20260714_202317_panel_run_f1/ego_obs.jsonl`) + the tape. Script: `offset_fit.py` (reruns;
`offset_fit.png` alongside). No shipping-code / tape / test edits.

## Method + the one calibration that makes it trustworthy

Signals are ASYMMETRIC: the champion log has `kf_pos_ned` (estimator NED position track) + obs
attitude but **no IMU**; the replays have IMU (`gyro_*` raw wire, `accel_*` body) + `cmd_w*` but
**no position**. So the replay line is recovered by **strapdown integration** of its IMU (initial
attitude from the boot-row rest gravity vector — all 3 runs: pitch −17.80°, roll +0.01°, the 17°
block, zero lateral asymmetry) and compared to the champion's `kf_pos_ned`.

- **Gyro sign resolved EMPIRICALLY, not by convention** (raw `gyro_z` is negatively correlated
  with `cmd_wz`, corr −0.68 — the known self-consistent "tail-first" control alias; do not "fix").
  Physical FRD rate = **−gyro_raw**: with that sign the strapdown reproduces the champion's East
  sign (r1 @g0 East +2.45 m vs champion +0.47); the +raw sign gives the mirror image (−2.30 m).
- **Strapdown validated**: replay forward(N) @g0 = 10.83 m vs champion 11.05 (match to **−0.22 m**
  over 2.945 s) ⇒ frame + integration trustworthy in the horizontal plane. (Vertical is NOT: the
  strapdown climbs 0.85 m vs champion kf 4.69 m — champion kf-D is weakly observed, NO baro on the
  VQ2 wire; vertical is excluded from all conclusions. The right-miss is a horizontal phenomenon.)

## Q1 — CHARACTERIZE the offset

**East offset dE(t) = replay-mean East − champion East (champion NED frame), 3-run spread:**

| t (s) | 0.2 | 0.5 | 0.8 | 1.2 | 1.6 | 2.0 | 2.4 | **2.945 (g0)** | 3.5 | 4.2 | 5.17 (g1) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dE (m) | +0.00 | +0.01 | +0.05 | +0.22 | +0.49 | +0.85 | +1.29 | **+2.00** | +2.72 | +3.01 | +2.46 |
| spread | .00 | .00 | .00 | .01 | .04 | .09 | .16 | **.25** | .39 | .91 | 2.19 |

- **Onset shape: ≈0 for the first ~0.5 s, then super-linear `dE ~ t^2.8`** (n≈1 = translation,
  ≈2 = constant heading, ≈3 = rate-bias). Not linear ⇒ **not a pure lateral-translation offset**;
  the growth needs a persistent lateral *acceleration* difference (best-fit constant differential
  Δa_E = **0.49 m/s²**), i.e. a **heading/attitude** difference, not a velocity slip.
- **Equivalent heading bias ≈ +10.5–11°**, cross-checked two independent ways:
  1. replay course − champion course (from velocity vectors) grows **+8.7°(1.2 s) → +11.2°(2.4 s)
     then PLATEAUS** at ~+11°;
  2. chord geometry asin(2.0 / 11.06) = **10.4°**;
  3. rotation-null: rotating the replay frame **−10.5°** zeroes the g0 offset (consistency check).
- **Onset is heading, and it LOCKS**: the course offset accrues during the takeoff/climb (0–2.4 s)
  then stops growing (plateau), rather than climbing linearly forever.

**Estimated lateral miss at the champion's gate-pass times** (right = +):

| gate | t (s) | champ East | replay-mean East | **dE (miss)** | note |
|---|---|---|---|---|---|
| g0 | 2.945 | +0.47 | +2.47 ± 0.25 | **+2.00 m** | clean — systematic ≫ 3-run spread |
| g1 | 5.168 | +9.93 | +12.39 ± 2.19 | +2.46 m | spread already = signal (chaos onset) |
| g2 | 6.189 | +12.90 | +13.35 ± 3.37 | +0.45 m | indeterminate — spread ≫ signal |

Only **g0 (+2.0 ± ~0.5 m right)** is a clean systematic number; by g1+ the chaotic spread (Q2)
overtakes it. Uncertainty budget on g0: 3-run spread ±0.25 m + strapdown/yaw0 alignment ±~0.4 m.

## Q2 — chaotic divergence (pairwise replay residuals)

Matched wake-0 states (P0.2: accel Δ ≤ 0.001 m/s²) + bit-equivalent commands (±1-row straddles)
still diverge. Pairwise-mean residuals (3 pairs), interpolated to a common 25 ms grid:

| t (s) | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 | 3.5 | 4.0 | 4.5 |
|---|---|---|---|---|---|---|---|---|
| d\|P\| (m) | 0.04 | 0.09 | 0.18 | 0.30 | 0.44 | 0.74 | 1.32 | 2.20 |
| d\|V\| (m/s) | 0.08 | 0.15 | 0.23 | 0.27 | 0.32 | 0.86 | 1.48 | 2.02 |

- Position-residual growth over 1–4 s: **λ = 1.09 /s ⇒ e-fold 0.92 s, doubling 0.64 s.**
- Heading residual stays *bounded* (~0.7–0.9°) — the divergence is in *translation*, not attitude;
  gyro residual is near-zero except single-tick contact transients at 2.5–3.0 s.
- **Open-loop horizon at gate-aperture precision (d\|P\| = 0.5 m): t = 3.15 s.** Macroscopic
  outcome bifurcation (crash rows 199/271/297 = 5.0/6.8/7.4 s) is later. Both bracket the reported
  **3–5 s** — 0.5 m precision at the low end (~3 s), gross outcome at the high end (~5–7 s).

## Q3 — resample-bias check (ZOH 40 Hz vs raw timeline command integrals)

Net commanded attitude change ∫(3.14·a·1.2) dt, `tape_full.csv` (ZOH) vs `tape_full.raw.csv`:

| axis | g0 leg raw | g0 leg ZOH | **ZOH−raw** | full raw | full ZOH | ZOH−raw |
|---|---|---|---|---|---|---|
| yaw | 8.57° | 8.68° | **+0.10°** | −2.13° | −3.24° | −1.11° |
| roll | 16.04° | 15.32° | −0.72° | −10.50° | −13.59° | −3.09° |
| pitch | 0.00° | 0.00° | 0.00° | −23.94° | −22.79° | +1.15° |

Over the g0 leg the ZOH yaw bias is **+0.10°** — two orders below the ~11° needed to make the
miss. **The resample did NOT cause the offset.** (Full-tape residuals grow only because the hot
late segments accumulate edge quantization; irrelevant — chaos owns everything past g0.)

## Q4 — VERDICT

**(a) armed-silent LAUNCH-POSE / initial-state delta — BEST FIT.** By elimination + positive
onset evidence:
- **(c) resample artifact — RULED OUT** (HIGH conf): g0 ZOH bias 0.10° ≪ 11° needed (>100×).
- **(b) plant/wire constant rate bias — DISFAVORED** (MED-HIGH conf): the replay wire is
  **bit-identical** to the champion's (REPORT verified) and the sim is deterministic ⇒ identical
  plant response ⇒ a plant asymmetry would displace the CHAMPION the same way, but the champion
  passed all 5 gates. And the heading offset **saturates** (~11° plateau) instead of growing
  linearly; `dE ~ t^2.8`, not `t^3`. A persistent rate bias is not what the data shows.
- **(a) — CONSISTENT** (MED conf on the specific mechanism, HIGH conf it is an initial-condition
  effect): identical commands + identical wire + deterministic sim leave the **pre-row-0 state**
  as the only free variable. The champion sat **~2.8 s ARMED-SILENT** on the 17° block before its
  first command (REPORT3: a post-GO policy-warmup delay); the replays launched ~0.5 s in. That
  settle-regime difference seeds a slightly different attitude/velocity at row 0, which the
  identical takeoff commands amplify into an ~11° heading rotation accrued over the first ~2.4 s
  (dE ≈ 0 at t0, super-linear build, then locked) = the ~+2 m right at g0. The onset SHAPE
  (heading, accruing through takeoff, saturating) is exactly this signature, not (b)/(c).
- **Residual alternative not fully excludable**: subtle sim-build/reset lateral nondeterminism
  between the champion's recording sim and the replay sim (design risk (d)). Weighted LOW — the
  strapdown forward-match to −0.22 m shows the two physics tracks agree closely.

**Correction magnitude a segment-tape splice would need.** The offset is a **first-leg
(from-block) launch effect** of ~**+11° heading / +2 m lateral by gate 0** — it lives in the
block→gate-0 leg, not in the tape math. Options, in order of soundness:
1. **Settle-match the first-leg start (ops, zero code)** — the physically-correct fix. P0.2 proved
   pre-GO block time is controllable; the champion's *armed-silent* 2.8 s specifically is not
   ops-reproducible, so this needs a takeoff protocol that lands the same row-0 state (or a
   closed-loop takeoff instead of an open-loop first-leg tape).
2. A static ~−11° heading / −2 m lateral pre-bias on the first segment — **fragile**: the Q2
   horizon (e-fold 0.9 s, 0.5 m at 3.15 s) means open-loop cannot hold gate precision regardless.
3. **Strategic (confirms REPORT3):** full open-loop is dead; the ratchet needs the **arrestor +
   short (<3 s / one-leg) segment tapes with closed-loop correction between**. Segments spliced
   from arrestor-produced (calm, gate-facing) states do NOT inherit the block-launch offset — but
   each open-loop leg grows its OWN heading-divergence over ~2–3 s from any splice-state error, so
   **arrestor state-match quality + sub-3 s legs bound per-segment miss, not a static tape number.**
   The block→gate-0 leg is the one place the launch offset must be addressed directly.

**Confidence:** the ~+2 m right miss at g0 is real and pilot-confirmed (HIGH). Its character —
a heading/attitude rotation, not translation, not resample, not a plant-common rate bias — is
robust (HIGH). Attribution to the specific armed-silent settle is the best available explanation
but not directly testable ops-only (MEDIUM). The single hard limit: with no champion IMU I fit the
replay line against the champion's *estimator* course, so the ~11° cannot be split between "pure
constant launch-heading delta" and "fast heading divergence over takeoff" — but both are the same
mechanism (a), and both are equally addressed by fixing the first-leg start state.
