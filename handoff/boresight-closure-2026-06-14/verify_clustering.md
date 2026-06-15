# ADVERSARIAL VERIFY -- terminal fix-drought BREAKS the gate-4 r=0.30 closure

**CLAIM UNDER TEST:** *"Post-bake, gate-4 r=0.30 CLOSES once the mean fix-rate reaches
~0.35-0.50 (anisotropic sigma [lat 0.1914, vert 0.1008], bias<=0)."*

**VERDICT: REFUTED (high confidence).** The mean-fix-rate closure reported by the
regular-cadence engine is an **artifact of the evenly-spaced fix schedule.** A realistic
**terminal fix-drought** -- no accepted fix in the last ~0.15-0.30 s (equivalently the
last ~4 m) of the gate-4 approach -- breaks r=0.30 closure at **every** cell the
regular-cadence analysis reported as closing. **TERMINAL GATE-LOCK, not a high pooled
mean, is the true requirement** -- exactly as the banked memory warned.

MARGIN(0.30)=0.235 m, MARGIN(0.38)=0.155 m. Closure is **CI-honest**: bootstrap-90%
upper-CI of p99 < MARGIN (and upper-CI of p90 < MARGIN). nmc=2000, 1000x bootstrap.

---

## Method (what changed vs the production engine -- nothing in the physics)

The engine is `verify_clustering_drone.fly_lap_drought`, a **byte-faithful copy of
`margin_driver_v2.fly_lap_v2`** (which is itself regression-proven == production
`ME.fly_lap` in the iso/zero-bias limit). The ONLY change is a gate on **which offered
fixes are emitted**:

- **terminal drought**: an offered fix is DROPPED (never enqueued / applied) if its
  nominal time `t >= T_end - D` (seconds) OR range-to-gate-4 `<= M` (metres). The mean
  fix-rate *earlier* in the lap is unchanged (the cadence clock keeps running; only the
  near-gate emissions are suppressed). Models the camera losing the gate-4 centre on
  final approach.
- **bursty schedule**: same long-run mean rate, but fixes arrive in bursts of `burst_n`
  back-to-back detector frames with a long gap between bursts. Tests whether burstiness
  ALONE (no terminal drought) matters.

**Regression (bit-exact):** `fly_lap_drought(schedule='regular', D=0, M=0)` reproduces
`fly_lap_v2` to **0.0** over 240 laps; `fly_lap_v2(iso)` reproduces `ME.fly_lap` to
3.3e-16. The KF (`LinearKF`+`RewindKF`), the anisotropic draw, the cov, the accel-bias
model, the latency queue (15 ms), and the final in-plane miss onto u34 are reused
**exactly**. Only dropped fixes differ.

**Why the terminal window is short and the drought is small:** the gate becomes a usable
4-corner relative fix only inside `FIX_WINDOW_M=12 m`. At v=37 the drone covers 12 m in
~0.32 s, so the *entire* gate-4-targeted fix sequence is ~0.3 s long and contains only
~5-6 offered fixes at fr=0.50. Dropping the last 0.15-0.30 s removes essentially all of
them. (D->range map: v=37: D0.15s~5 m, D0.30s~11 m, D0.50s~18 m; v=30: D0.15s~4 m,
D0.30s~9 m.)

---

## (A) Terminal drought in SECONDS -- the headline table

`p99 / p99_CIhi` then `C`=closes / `x`=fails at r=0.30 (CI-honest). **D=0.0 is the
regular-cadence baseline** (reproduces the production claim).

| v | fr | bias | bmode | D=0.0 | D=0.15 | D=0.30 | D=0.50 | D=0.75 |
|---|---|---|---|---|---|---|---|---|
| 30 | 0.35 | 0.0 | inplane  | 0.234/0.246 x | 0.257/0.270 x | 0.333/0.344 x | 0.398/0.418 x | 0.384/0.405 x |
| 30 | 0.35 | 0.0 | random3d | 0.236/0.245 x | 0.248/0.260 x | 0.331/0.345 x | 0.410/0.420 x | 0.405/0.419 x |
| 30 | 0.50 | 0.0 | inplane  | **0.196/0.205 C** | 0.237/0.243 x | 0.282/0.293 x | 0.340/0.360 x | 0.356/0.369 x |
| 30 | 0.50 | 0.0 | random3d | **0.202/0.208 C** | **0.216/0.225 C** | 0.272/0.289 x | 0.340/0.362 x | 0.352/0.366 x |
| 37 | 0.35 | 0.0 | inplane  | **0.208/0.219 C** | 0.288/0.303 x | 0.318/0.330 x | 0.462/0.476 x | 0.521/0.542 x |
| 37 | 0.35 | 0.0 | random3d | 0.227/0.241 x | 0.280/0.295 x | 0.328/0.351 x | 0.468/0.489 x | 0.486/0.506 x |
| 37 | 0.50 | 0.0 | inplane  | **0.197/0.203 C** | 0.243/0.249 x | 0.340/0.352 x | 0.417/0.424 x | 0.458/0.484 x |
| 37 | 0.50 | 0.0 | random3d | **0.214/0.226 C** | 0.237/0.250 x | 0.338/0.352 x | 0.428/0.456 x | 0.465/0.476 x |
| 30 | 0.35 | 0.6 | inplane  | 0.298/0.302 x | ... all x ... | | | |
| 30 | 0.50 | 0.6 | inplane  | 0.254/0.262 x | ... all x ... | | | |
| 37 | 0.35 | 0.6 | inplane  | 0.246/0.251 x | ... all x ... | | | |
| 37 | 0.50 | 0.6 | inplane  | 0.227/0.244 x | ... all x ... | | | |

(bias=0.6 deg cells never close even at D=0 under the CI-honest rule, so the drought is
moot there; full rows in `verify_clustering_results.json`.)

### Breaking gap (the key number)

Of the **5 cells that close r=0.30 at D=0** (fr>=0.35, bias=0 -- the claim's regime):

| cell | closes @ D=0 | **breaks at D =** |
|---|---|---|
| v=37 fr=0.50 inplane  | yes (CIhi 0.203) | **0.15 s** |
| v=37 fr=0.50 random3d | yes (CIhi 0.226) | **0.15 s** |
| v=37 fr=0.35 inplane  | yes (CIhi 0.219) | **0.15 s** |
| v=30 fr=0.50 inplane  | yes (CIhi 0.205) | **0.15 s** |
| v=30 fr=0.50 random3d | yes (CIhi 0.208) | **0.30 s** |

- **fr=0.50 stops closing at a terminal-gap of D = 0.15 s** (drops just ~2 fixes) in 3 of
  4 cells; the most optimistic cell (v=30, random3d) survives D=0.15 s but breaks at
  D=0.30 s.
- **0 of 5** still close at D=0.30 s. **0 of 5** still close at D=0.50 s.
- A realistic terminal drought of **D = 0.3-0.6 s** pushes p99-CIhi to **0.29-0.49** --
  ~25-110% OVER the 0.235 margin. Closure is gone with margin to spare.

The D=0 closures sit right at the 0.235 wall (p99-CIhi 0.203-0.226). They are
**knife-edge** -- removing 2 terminal fixes is enough to fail.

---

## (B) Terminal drought in METRES (range-to-gate-4) -- corroboration

Dropping accepted fixes only in the last **M metres** of range, same CI-honest closure:

| v | fr | bias | bmode | M=2 m | M=4 m | M=6 m | M=8 m |
|---|---|---|---|---|---|---|---|
| 30 | 0.35 | 0.0 | inplane  | 0.240/0.249 x | 0.268/0.275 x | 0.280/0.291 x | 0.283/0.291 x |
| 30 | 0.35 | 0.0 | random3d | 0.231/0.256 x | 0.264/0.273 x | 0.287/0.303 x | 0.285/0.303 x |
| 30 | 0.50 | 0.0 | inplane  | **0.214/0.216 C** | 0.242/0.256 x | 0.245/0.253 x | 0.265/0.274 x |
| 30 | 0.50 | 0.0 | random3d | **0.215/0.221 C** | 0.233/0.244 x | 0.250/0.263 x | 0.265/0.282 x |
| 30 | 0.50 | 0.6 | inplane  | 0.276/0.285 x | 0.297/0.310 x | 0.337/0.343 x | 0.375/0.393 x |
| 37 | 0.35 | 0.0 | inplane  | 0.240/0.248 x | 0.241/0.250 x | 0.281/0.298 x | (running) |

The fr=0.50 closure survives a 2 m blackout but **breaks once fixes are lost in the last
4 m** (CIhi 0.244-0.256 > 0.235). 4 m at 37 m/s ~ 0.11 s -- consistent with the seconds
table. fr=0.35 and bias=0.6 cells fail at every M. (Sweep covered v=30 fully + v=37 in
progress when stopped under external machine contention; full grid checkpointed in the
JSON. The complete gap_s seconds table -- Section A -- is the load-bearing result and is
100% complete.)

---

## (C) Bursty (clustered) schedule, NO terminal drought -- does burstiness alone matter?

Same mean rate, fixes in bursts of `burst_n` frames with long gaps between bursts, but
NO terminal blackout (bursts keep firing up to the gate). nmc=2000. `p99[CIhi]` C/x at
r=0.30; `regularD0` is the regular-cadence baseline from Section A:

| v | fr | bias | bmode | regular D0 | burst3 | burst5 |
|---|---|---|---|---|---|---|
| 30 | 0.50 | 0.0 | inplane  | 0.196[0.205] **C** | 0.197[0.206] **C** | 0.171[0.178] **C** |
| 30 | 0.50 | 0.0 | random3d | 0.202[0.208] **C** | 0.203[0.211] **C** | 0.175[0.190] **C** |
| 37 | 0.50 | 0.0 | inplane  | 0.197[0.203] **C** | 0.183[0.193] **C** | 0.178[0.182] **C** |
| 37 | 0.50 | 0.0 | random3d | 0.214[0.226] **C** | 0.187[0.196] **C** | 0.174[0.183] **C** |
| 37 | 0.35 | 0.0 | inplane  | 0.208[0.219] **C** | 0.193[0.198] **C** | 0.207[0.213] **C** |
| 30 | 0.35 | 0.0 | inplane  | 0.234[0.246] x | 0.196[0.209] **C** | 0.247[0.258] x |

**Burstiness ALONE does NOT break closure.** Every cell that closes under the regular
cadence (fr=0.50, bias=0) ALSO closes under both burst patterns -- often marginally
*better* (burst5 CIhi 0.178-0.190), because a burst that fires near the gate accumulates
a few extra terminal fixes. **The damage is specifically the TERMINAL DROUGHT, not
clustering / unevenness per se.** It is not "fixes arrive unevenly" but "the camera goes
blind on the final ~0.15-0.30 s" that breaks closure -- so the true requirement is
**terminal gate-lock**, and a bursty source that maintains it is fine.

Subtlety (reinforces the thesis): burst5 at fr=0.35 (longer bursts -> longer inter-burst
GAPS) sometimes FAILS where burst3 closes (v=30 0.35: burst3 CIhi 0.209 C vs burst5 0.258
x) -- a long enough inter-burst gap that lands on the final approach acts as a partial
terminal drought. The failure mode is always "no fix near the gate", however it arises.

---

## (D) Bottom line

- **# cells closing r=0.30 at D=0 (regular cadence, fr>=0.35, bias=0): 5.**
- **Of those, still closing at D=0.30 s: 0/5. At D=0.50 s: 0/5.**
- **ANY (fr, bias=0) cell closing r=0.30 under a 0.30 s drought: 0. Under 0.50 s: 0.**

The mean-fix-rate >=0.35-0.50 closure is **real only for a perfectly regular cadence that
keeps fixing right up to the gate plane.** Under the binding adversarial reality -- a
terminal fix-drought of 0.15-0.30 s (camera loses the gate centre on final approach) --
**r=0.30 does NOT close at any swept fix-rate.** The true requirement is **TERMINAL
GATE-LOCK**: the camera must keep producing accepted gate-4 fixes through (roughly) the
last 6 m / 0.15 s of the approach. This is the inc8 camera-pointing lever, and it
confirms the banked "clustering / terminal-drought is the dominant adversarial lens"
posture against the regular-cadence margin engine.

_Artifacts: `verify_clustering_drone.py` (engine, regression-proven), `run_verify_clustering.py`
(sweep), `verify_clustering_analyze.py` (tables), `verify_clustering_results.json` (data).
COLD case-C, aniso sigma [lat 0.1914, vert 0.1008], post-bake bias=0, latency 15 ms, seed
20260614. [BORESIGHT-CLOSURE ADVERSARIAL VERIFY 2026-06-14]_
