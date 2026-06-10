# Verify two offline changes on ShadowPC — finish_hold + PnP cov-inflation (2026-06-09)

ShadowPC (sim host). Repo pulled to `origin/main @ 2e7df71`. venv = `.venv\Scripts\python.exe` (Python 3.13.2).
Verifying two changes that landed offline:
- **A) finish_hold** — `52f075d` *"fly_vq1: autonomous post-finish drain (finish_hold, --finish-hold-s 1.0)"* + new module `src/racer/finish_hold.py`.
- **B) PnP world-fix cov inflation** — `87725cb` *"Vision: inflate analytic PnP world-fix covariance 2x (PNP_FIX_COV_INFLATION)"* (`src/racer/localization.py` + `scripts/characterize_perception.py --cov-inflation`).
- (`2e7df71` is a memory-bank commit; not edited here.)

Pre-checks: fast-forward pull, no conflict with the in-tree `race_outcome.py` work. Unit tests for the
changed modules **green**: `tests/test_finish_hold.py tests/test_localization.py tests/test_race_outcome.py` → **33 passed**.

---

## TL;DR
- **B (PnP cov inflation): MEASURED — the change UNDER-DELIVERS on the course; do NOT ship K=2.0 as-is.**
  K=1.0 reproduces the perception-char baseline bit-for-bit (N=312 solved, 143 catastrophic, 5 leaked).
  Inflating to the shipped **K=2.0 only moves good-fix rejection 21%→17% (<3 m) / 13%→11% (<1 m)** — the
  predicted ~2.6% never materialises — **and pushes catastrophic leak 1.6%→2.2% (of all) past the ≤~2%
  ceiling**. Good-fix rejection *plateaus* (~17%/~11%) for all K≥2; the over-rejected good fixes are
  **attitude-lever-dominated, not PnP-dominated**, so PnP inflation is the wrong lever. Fallback per the
  laptop HANDOFF — bump `NavigatorConfig.attitude_noise_std` (1.0°→~1.5°) — is the correct next move.
- **A (finish_hold): code-verified + unit-green, LIVE verification BLOCKED by a sim-side flight regression.**
  The module + wiring are correct and unit-tested, but the drone **cannot complete the course it flew 6/6
  on 2026-06-07** — deterministically flies **~1.5 m under gate 1** at 50 Hz (pilot GUI: *"flying UNDER the
  gate, not even trying to go through it"*) and **backflips at launch** at 100 Hz. finish_hold is post-FINISH
  code → never triggers, so no 6/6 self-certification was producible. **Exhaustively NOT our side:** flight
  code, gains, frames/attitude decode, start pad pose, ACRO mode, and numpy are all byte-identical to the
  6/6 (table in §A). Identical inputs → different motion ⇒ **sim-side** (build/timing/dynamics); not either
  offline change (vision OFF → cov path never exercised; finish_hold post-FINISH). See §A for the full
  elimination + the retracted attitude-convention false lead + the reverted `--rate` experiment.

---

## B — PnP world-fix covariance inflation  ✅ measured (authoritative before/after)

### Commands (per-gate course bundles from `data/runs/20260607_194615_course_60s`)
```
# baseline + shipped default, then the rest of the sweep, per gate g0..g5:
for K in 1.0 2.0 1.5 2.5 3.0:
  .venv\Scripts\python.exe scripts\characterize_perception.py ^
    --bundle handoff\perception-char-2026-06-08\pg\course_g<G> ^
    --weights models\gate_yolo11s_curriculum_v2.pt ^
    --map handoff\shadowpc-firstcontact-2026-06-02\track_map.json ^
    --cov-inflation <K> --json cov_sweep\char_g<G>_k<K>.json
# course-level aggregation (sum GATE TRADE-OFF counts across the 6 gate bundles):
.venv\Scripts\python.exe handoff\shadowpc-cov-finishhold-verify-2026-06-09\cov_sweep\aggregate_tradeoff.py
```
Scripts kept in `cov_sweep/` (`run_sweep.sh`, `run_sweep_extra.sh`, `aggregate_tradeoff.py`); raw per-gate
`GATE TRADE-OFF` blocks in `cov_sweep/sweep_k*_stdout.txt`; per-frame dumps `cov_sweep/char_g*_k*.json`.

### Course-level GATE TRADE-OFF — before/after (summed over all 6 per-gate bundles)
N = 312 solved/associated fixes; thresholds match the new block (GOOD `|fix|<1 m`, CATASTROPHIC `|fix|≥3 m`,
gate at χ²₀.₉₉₉ = 16.27). good<3 m column added to reconcile against perception-char's "20%".

| K (cov_inflation) | good-fix rej `<1 m` | good-fix rej `<3 m` | catastrophic leak (of cat) | (of all solved) |
|---|---|---|---|---|
| **1.0** (pre-change baseline) | 13/103 = **13%** | 35/169 = **21%** | 5/143 = **3.5%** | **1.6%** |
| 1.5 | 12/103 = 12% | 31/169 = 18% | 7/143 = 4.9% | 2.2% |
| **2.0** (shipped default) | 11/103 = **11%** | 29/169 = **17%** | 7/143 = **4.9%** | **2.2%** |
| 2.5 | 11/103 = 11% | 29/169 = 17% | 9/143 = 6.3% | 2.9% |
| 3.0 | 11/103 = 11% | 29/169 = 17% | 10/143 = 7.0% | 3.2% |

**Baseline reproduces perception-char exactly:** N=312, 143 catastrophic (46%), 5 leaked, 169 good `<3 m`
with 21% (~20%) rejected — all match `handoff/perception-char-2026-06-08/README.md`. The earlier "1.6%"
and "3.5%" leak figures are the *same 5 fixes*, just different denominators (of-all-solved vs of-catastrophic).

### Why it under-delivers (diagnosed)
- **Plumbing is correct** — `--cov-inflation` reaches the gate; per-gate median `maha` responds to K
  (ratio @2.0/@1.0 = 0.58–0.98 across gates).
- **But the over-rejected good fixes barely respond.** Of the 13 good fixes rejected at K=1.0, only **2**
  recover at K=2.0 (the marginal ones, maha just over 16.27); the deeply-rejected ones stay rejected
  (e.g. g0 31.7→30.6, g4 370→191, g5 151→89). Their error lives in the axis where the **attitude
  lever-arm term dominates the covariance — and that term is (correctly) NOT inflated** by this change.
- Net: good-fix rejection plateaus at ~17% (`<3 m`)/~11% (`<1 m`) for all K≥2, while **catastrophic leak
  rises monotonically** (1.6%→3.2% of all). There is **no knee** on the PnP-inflation axis that hits the
  good-rej `<5%` goal while holding leak `≤~2%`. The laptop sizing assumed PnP-dominance (f≈3.5); the
  course data refutes that for the residual good-fix population.

### Recommendation
1. **Do not adopt `PNP_FIX_COV_INFLATION = 2.0` on this evidence.** K=1.0 has the lowest leak (1.6%) and
   essentially the same good-fix yield. If anything ≤1.5 is the only relaxation that keeps leak ≤~2%, and
   it barely helps good-rej.
2. The good-fix over-rejection is **attitude-lever-dominated** → use the laptop HANDOFF's fallback lever:
   bump `NavigatorConfig.attitude_noise_std` 1.0°→~1.5°. NOTE: `characterize_perception.py` does **not**
   expose `attitude_noise_std` (only `--cov-inflation`), so confirming that lever needs a small script
   change (thread `attitude_noise_std` through the per-frame `gate_pose_to_world_position` call) before a
   re-measure — out of scope for this verify pass.
3. The PnP-inflation knob still has a *real but small* effect and is harmless ≤1.5 for VQ2 vision-only; the
   call is to pick the lever that matches where the good-fix error actually lives (attitude), not to push K.

---

## A — finish_hold (autonomous post-finish drain)  ⚠️ code OK, live verification BLOCKED by a sim flight regression

### Code review — correct + unit-tested
- `src/racer/finish_hold.py`: `sim_finish_confirmed` (finished flag / valid `race_finish_time_ns` / active
  gate ran off the end), `finish_drain_done` (cap vs post-confirm flush), `drain_until_finish` (pump+hold
  loop behind injected I/O seams). `tests/test_finish_hold.py` green.
- `scripts/fly_vq1.py`: after `Mission.run` returns FINISHED (or `sim_finish_confirmed`), and only when
  `not dry_run and --finish-hold-s > 0`, it pumps `mission.step(navigator())` (records inbound RACE_STATUS,
  holds position) until the terminal status is captured, then disarms. Prints the target live line
  `terminal RACE_STATUS captured at +X.XXs (finished=..., time=...s)`. Default `--finish-hold-s 1.0`,
  early-exits on capture (bigger cap is free). Hold sends the FINISHED hold setpoint; since the sim pauses
  physics at finish, it should not nudge the drone into the final gate — **but this needs a real finish to
  prove, which we could not reach (see below).**

### Live runs — BLOCKED: the `--faithful` course flight regressed (NOT finish_hold, NOT any offline change)
Six launches on the local sim (`udp:127.0.0.1:14550`); none reached a finish, so finish_hold's post-FINISH
code never executed. The drone CANNOT fly the course it flew 6/6 on 2026-06-07.

**Observed failure modes (deterministic):**
- **50 Hz** (committed default; runs `041625`, `124631`, `161… fly6`): stable takeoff, clears gate 0
  cleanly, then flies **~1.5 m UNDER gate 1** (drone D≈+6.6 m vs gate-1 D=+5.07 m) and limit-cycles
  N≈−42…−57, never threading g1; race ends ~40 s in. `final state RUN`, `gate_index 1/6`, `col=0`.
  Pilot GUI: *"much more stable… flying UNDER the gate, not even trying to go through it."*
- **100 Hz** (run `161601`): **side-backflip at launch** (rolls to 124° while still near origin), clips a
  gate, recovers, then diverges → HARD COLLISION, abort at gate 0. Pilot GUI confirmed the backflip.
- Gate centres (N,E,D): g0 `(−23.3,−0.4,−0.03)`, g1 `(−46.9,−2.5,+5.07)` … g5 `(−159.2,−4.4,+26.0)` —
  a steadily **descending** course; the symptom is a **vertical overshoot of the descent**.

**Root-cause elimination — everything on OUR side is byte-identical to the canonical 6/6:**
| suspect | verdict | evidence |
|---|---|---|
| flight-control code | identical | `git diff 8cadb42→HEAD` over src/racer = only finish_hold/localization/race_outcome/rl_plant; controller/mission/navigator/frames/state_estimator untouched |
| controller gains | identical | `controller_config` byte-equal (canonical meta vs failed meta) |
| attitude decode (frames.py, mavlink_client.py) | identical | last changed 2026-06-04, before the 6/6 |
| **start pad attitude** | **identical** | both `q=[0.001,−0.155,0,−0.988]` → euler `(−0,−17.8,−179.9)` at origin |
| flight mode | identical | tlog HEARTBEAT `custom_mode=0` both; pilot confirms **ACRO** (no self-leveling) |
| numpy / scipy | identical | 2.4.6 / 1.17.1 dated 2026-06-01; the 2026-06-08 torch+ultralytics install did NOT touch numpy |

→ **Identical inputs (start pose, mode, our code+gains+env) produce different motion ⇒ the sim's
input→output response differs vs when the 6/6 was flown.** The alt loop is the project's known
delay-sensitive element ("rung-1 alt limit cycle was delay-driven"), consistent with a sim-side
**timing/dynamics** difference manifesting as vertical overshoot (50 Hz) / instability (100 Hz).
The ONE thing not verifiable: the **sim build at the 6/6** (never recorded); current build = **1.0.3364**
(user reports no manual update, same course, no settings).

**CORRECTION (recorded honestly):** an interim claim that the *attitude convention* flipped 180° was a
mistake — it compared the canonical's pre-race RESIDUE pose (logged ~326 s before GO, at pos≈52 m) against
today's pad pose. Properly aligned at the pad, the start attitudes are identical (above). No frame
transform is warranted.

**Speculative `--rate`→100 Hz edit: tried then REVERTED.** The canonical meta showed `rate_hz=100` vs our
50 Hz default, so faithful→100 Hz was hypothesised and patched into `fly_vq1.py`; the re-fly **backflipped
and collided at gate 0** (100 Hz is *worse*, not better), so the edit was reverted. Working tree is clean
(only the pre-existing race_outcome trio). Rate is not the fix.

**Disposition:** finish_hold is **code-verified + unit-green** but **cannot be live-verified** until the
course flight is restored. The regression is conclusively NOT in either offline change and NOT in our
flight stack — it is sim-side (build/timing/dynamics). Re-flying or re-tuning the controller to the
current sim is a separate effort beyond this verify pass. When a clean 6/6 is achievable again, re-fly
`fly_vq1.py --faithful --label course_finishhold` (default `--finish-hold-s 1.0`; bump to 2.0 only if the
live line prints `!! no terminal RACE_STATUS within 1.0s`) and grade with `race_outcome.py` →
EXPECT finished=True, CLEAN FINISH=True, gates 6 (clean 6 / contact 0), ~35.3 s, final gate NO contact.

---

## Files (in this dir)
- `cov_sweep/char_g{0..5}_k{1.0,1.5,2.0,2.5,3.0}.json` — per-gate per-frame characterize dumps (maha reflects K).
- `cov_sweep/sweep_k*_stdout.txt` — raw per-gate `GATE TRADE-OFF` blocks per K.
- `cov_sweep/aggregate_tradeoff.py` — course-level summation + dual-threshold reconciliation (the table above).
- `cov_sweep/run_sweep.sh`, `run_sweep_extra.sh` — the sweep drivers.
- `finishhold_flies/fly2,fly3_stalled_gate1.log` — 50 Hz: clear g0, fly under g1 (reproduced).
- `finishhold_flies/fly5_100hz_backflip_collide_g0.log` — 100 Hz: launch backflip → collide g0.
- `finishhold_flies/fly6_50hz_under_gate1.log` — 50 Hz committed baseline, ACRO-mode-check run (= fly2/3).

## Open items for the commander
1. **B:** don't take K=2.0 on the course evidence; pursue the `attitude_noise_std` lever (needs the small
   `characterize_perception` thread-through first). K=1.0 stays the lowest-leak point.
2. **A (finish_hold):** code-verified + unit-green; **live-unproven** because the course flight regressed.
   Re-fly + grade the instant a clean 6/6 is achievable again — one good run self-certifies the 6/6, the
   finish time, and the no-nudge check.
3. **Flight regression (separate from both offline changes):** same `--faithful` config that flew 6/6 on
   2026-06-07 now (a) flies ~1.5 m **under gate 1** at 50 Hz and (b) **backflips at launch** at 100 Hz.
   Exhaustively eliminated on our side (code, gains, frames decode, start pad pose, ACRO mode, numpy all
   byte-identical to the 6/6). Identical inputs → different motion ⇒ **sim-side** (build/timing/dynamics);
   build now `1.0.3364`, 6/6-era build not recorded. Symptom localises to the **delay-sensitive alt loop**
   (vertical overshoot of the descending course). Next-step options if pursuing: (i) confirm/compare the
   sim build vs the 6/6 era; (ii) decisive plant test — same actuator outputs vs different attitude-rate
   in the first 0.5 s of takeoff ⇒ proves sim-plant change; (iii) if treating current sim as ground truth,
   re-tune the alt loop (alt_offset / vz source / kd_alt) — but that is re-tuning to a changed sim, not a
   bug fix in our pilot.
