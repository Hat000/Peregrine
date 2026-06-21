> 🚩 **SUPERSEDED 2026-06-19 (511e85c) re: the σ_p0 bar.** Any "σ_p0 ≲ 0.08 / NO-GO vs 0.08 / near-field-gate-estimator pivot" conclusion in this report is OVERTURNED — the 0.08 bar was a `margin_envelope.py` double-count of the drone (real gate clearance 0.75−r ≈ 0.37–0.47 m). The REAL bar is **σ_p0_lat ≲ 0.15 (p99 ≲ 0.45)**; measured 0.15–0.20 = MARGINAL-PASSING, NOT NO-GO. Gate-4 is a REACH/PASS-RATE problem; RL stays the tool (the "pivot off RL" is RETRACTED). The engineering + measurements below STAND; only the bar and its GO/NO-GO verdict are corrected. → MEMORY.md:8 + memory/project_rl_increment_history.md §inc8-2026-06-19.
# inc8 RECENTER RUN — REPORT (2026-06-18)

**Session:** inc8 RECENTER LAUNCH — sync fix to Adroit, submit, monitor `success_rate`.
**Job:** `3276449` (`peregrine_inc8_recenter`, RUNTAG=rc1), node adroit-h11g3, partition gpu.
**Fix under test:** gated `through_centering_reward` (R1-to-centre cross-track lateral component restored;
RW_TC=10.0, look-at on, rw_centering OFF) — built+merged on `main` @ 9cecf64 / ebdbf42.

---

## VERDICT (headline)

🟢 **FLIGHT RECOVERED — `success_rate` OFF ZERO. This is the first flying inc8.** Seed 0 (4000 updates):
**`FLIGHTCHECK_VERDICT=FLEW`** → `success_rate_max=0.538`, `n_passed_gates_max=3.03`. The auto early-stop did
NOT fire; the gate PASSED and the job proceeded to seeds 1 & 2.

- The inc8 lineage was `success_rate≡0` / `l_episode~1.5` (died at gate-0) across EVERY prior run incl. the
  S2-seed2 parent + warm-start. The ONLY change here is `+env.rw_through_centering=10.0` (the restored
  R1-to-centre cross-track lateral pull). Flight recovered → the dropped through-approach centering WAS the
  policy-gap root cause. Confirmed by construction (single-variable) + the FLEW verdict.
- σ_p0 (terminal centring miss) is now MEANINGFUL for the first time (laps complete) — measure offline via
  contact_true_eval on the seed checkpoints.

**ALL 3 SEEDS FLEW (RC=0 each, `INC8_RECENTER_DONE`) — tightly clustered, lineage-wide gap fixed:**

| seed | success_rate_max | n_passed_gates_max | verdict |
|------|------------------|--------------------|---------|
| 0 | 0.538 | 3.03 | FLEW |
| 1 | 0.513 | 2.98 | FLEW (RC=0) |
| 2 | 0.504 | 2.96 | FLEW (RC=0) |

3/3 fly, success_rate ~0.50–0.54, n_passed ~2.96–3.03 — the recovery is reproducible across seeds, not a
seed-0 fluke. The lineage-wide policy-gap (success_rate≡0 across ALL prior runs incl. S2-seed2 parent +
warm-start) is RESOLVED by the through-centering restore.

**GPU util:** mean ~24–28% (sampler CSV parse slightly garbled but clearly low) — the 2048-env PPO did not
saturate the GPU (rollout/CPU-bound), but ran fine at ~41 min/seed, ~2 h total. Not zero → AUP-safe.

---

## 1. SYNC (DONE — md5-verified)

Pushed via the adroit-connector serve daemon (tar+gz+base64 chunks, LF-normalized sbatch) to
`/scratch/network/fl3689/peregrine_repo/rl/`. All 4 files md5-matched local↔remote:

| file | md5 |
|------|-----|
| rl/inc8_reward.py | a435ff435f4b7e331ac99ce4fba2df27 |
| rl/peregrine_racing_inc8.py | b5dc24c8b73f76efe6186f4f7d3a46b7 |
| rl/inc8_tb_trace.py | a86c8736ff69900de4151d028940c4e5 |
| rl/peregrine_inc8_recenter.sbatch | 2ab53a958d91381246d22a75feb38ed7 |

**Daemon recovery (#76):** on entry the serve daemon was wedged — port 8765 held by a stale PID whose
in-memory token ≠ `.daemon.json` (token-mismatch → `[daemon] unauthorized`, adroit.py:461). 4 overlapping
serve processes. Fix per #76: force-killed all stale serve PIDs (51996/75736 already exited; killed
34248/47792), removed stale `.daemon.json`, launched one fresh serve → Duo approved by Fengyou →
authenticated (fresh token, port 8765 open). Exec verified (adroit5 login node).

## 2. SUBMIT (DONE)

`sbatch --export=ALL,RUNTAG=rc1 …/peregrine_inc8_recenter.sbatch` → **Submitted batch job 3276449**.
PRECHECK_RC=0 (3-update wiring/NaN/dim check passed; obs_dim 20, through-centering ON). Seed 0 training
started: 4000 updates, 2048 envs, rw_through_centering=10.0, ~1.6 it/s (~44 min/seed).

## 3. MONITOR (trajectory)

**Seed 0 live tqdm (success_rate climbs monotonically off zero, holds):**

| update | success_rate | l_episode | note |
|--------|--------------|-----------|------|
| 8      | 0.00 | 1.7 | warmup |
| 117    | 0.13 | 1.9 | OFF ZERO |
| 486    | 0.29 | 3.4 | climbing |
| 857    | 0.36 | 2.9 | |
| 1182   | 0.39 | 2.9 | |
| 1560   | 0.45 | 2.9 | |
| 2345   | 0.46 | 2.8 | plateau forming |
| 3133   | 0.41 | 2.8 | |
| 3532   | 0.48 | 2.9 | |
| 3934   | 0.52 | 2.9 | end |

`l_episode` climbs 1.7→~2.9 s and HOLDS (vs lineage death at ~1.5 s = gate-0). `survive_rate=0.00`
throughout = metric-definition artifact (completed laps count as success, not survive; not a concern).

## 4. READS (seed-0 TB trace)

**FLIGHTCHECK:** `success_rate_max=0.53809  n_passed_gates_max=3.02832` → course-completion PRIMARY GO met.
End-of-run (step 3990): success_rate 0.521, n_passed_gates 2.91.

- **success_rate trajectory:** 0 → 0.13 (u100) → 0.32 (u400) → 0.45 (u1500) → plateau ~0.45–0.54. Flies but
  not every lap (n_passed ~3.0 of the full course). Genuine, stable flight — no collapse.
- **n_passed_gates:** 0 → ~3.0. The drone clears the early/mid course (no longer dies at gate-0).
- **Pointing RETAINED (criterion #3 met, rebalanced):** terminal_pointing 0→~0.057, lockband_pointing
  ~0.017, pointing_rate ~0.03, fix_rate ~0.006 — all NON-ZERO, so the look-at primitive still fires. BUT
  much lower than the pointing-optimized non-flying lineage (which hit pointing_rate 0.46 / fix 0.17–0.23).
  This is the intended rebalance: the policy now spends its budget FLYING THE COURSE instead of pointing a
  resetting drone. band_az ~37–58°, band_el ~66–79° (loose lock) — consistent with the lower pointing.
- **estim_err_ip_m ~0.17** = the estimator in-plane RMS ceiling (NOT policy-controlled; σ_p0 needs offline
  contact_true_eval, not this TB scalar).
- **through_centering term:** NOT surfaced as its own column in inc8_tb_trace.py (the traced `centering` tag
  = `inc8_centering` = the near-gate rw_centering term, which is OFF → 0.000 throughout, as designed). The
  through_centering restore is confirmed ACTIVE indirectly: precheck (3 updates, through-centering ON)
  passed clean (no NaN/dim), and it is the sole single-variable change that turned ≡0 into FLEW. A direct
  scalar trace would need a tb_trace tag add (follow-up nicety, not blocking).
- **RW_TC=10.0 read:** recovered flight cleanly but success_rate plateaus ~0.5 (not ~1.0) and n_passed ~3.0
  of the full course → there is headroom. Cannot tell over- vs under-centering from the trace (no
  through_centering column), but the partial completion suggests a **RW_TC sweep is the natural follow-up**
  (the sbatch supports `RW_TC=<w>` via --export; e.g. try 6.0 and 14.0 to bracket). Not a blocker — flight
  is recovered, which was the GO.

## 5. MEMORY-DELTA (text only — do NOT commit memory/)

```
inc8 RECENTER RUN (job 3276449, RUNTAG=rc1, adroit-h11g3, 2026-06-18): 🟢 FIRST FLYING inc8 — the
lineage-wide policy-gap (success_rate≡0 across ALL prior runs incl. S2-seed2 parent + warm-start) is FIXED.
- ALL 3 FRESH seeds FLEW (RC=0): success_rate_max 0.538/0.513/0.504, n_passed_gates_max 3.03/2.98/2.96.
  Tight cluster = reproducible, not a fluke. FLIGHTCHECK auto-gate PASSED on seed 0; ran all 3.
- The ONLY change vs the non-flying S0 baseline = +env.rw_through_centering=10.0 (restored R1-to-centre
  cross-track lateral pull, ORTHOGONAL to arc-Γ). Single-variable → dropped through-approach centering WAS
  the root cause. rw_centering stayed OFF (S3-cliff driver; layer later). look-at on, warmup=200, signs
  ++yaw=-3.0 ++pitch=3.0.
- Pointing RETAINED but rebalanced DOWN (terminal_pointing~0.057, lockband~0.017, pointing_rate~0.03,
  fix~0.006 — all >0 but << the old pointing-optimized non-flying lineage's 0.46/0.17-0.23). Intended: budget
  now goes to flying the course, not pointing a resetting drone. band_az~37-58 band_el~66-79 (loose lock).
- σ_p0 NOW MEASURABLE for the first time (laps complete) → next = offline contact_true_eval on the seed
  checkpoints for the physical terminal-centring miss (TB estim_err_ip~0.17 = estimator RMS floor, NOT σ_p0).
- success_rate plateaus ~0.5 / n_passed ~3.0 of full course = HEADROOM → RW_TC sweep is the follow-up
  (sbatch supports RW_TC=<w> via --export; bracket 6.0/14.0). Not a blocker — flight = the GO, achieved.
- FOOTGUN: inc8_tb_trace.py does NOT surface a through_centering column (traced 'centering'=inc8_centering=
  the OFF near-gate rw_centering term=0.000); restore confirmed indirectly (clean precheck + single-var FLEW).
  Add a tb tag for a direct trace (nicety). GPU util low ~24-28% (rollout-bound), ~41min/seed.
- OPS: 4 fix files md5-verified to /scratch peregrine_repo/rl via serve daemon. #76 recurred — wedged
  port-8765 listener (stale token → 'unauthorized'); fix = kill ALL stale serve PIDs + rm .daemon.json +
  fresh serve (one Duo). Checkpoints in diffaero/outputs/train/inc8_recenter_seed{0,1,2}_rc1.
```

## 6. NEXT (suggested)

1. **σ_p0 eval** — offline `contact_true_eval.py` on `inc8_recenter_seed{0,1,2}_rc1` checkpoints (the physical
   terminal-centring miss; now meaningful since laps complete). Gate-4 closure needs σ_p0_lat ≲ 0.08 m.
2. **RW_TC sweep** — `--export=ALL,RW_TC=6.0,RUNTAG=rc_w6` and `RW_TC=14.0` to push success_rate past ~0.5
   and bracket over/under-centering (the partial completion suggests headroom).
3. Optional: add an `inc8_through_centering` tag to `inc8_tb_trace.py` for a direct restore trace.
4. Checkpoints staged on Adroit `/scratch/.../outputs/train/inc8_recenter_seed*_rc1/` — pull a flying actor
   for the POC / vertical-slice if promoting.
