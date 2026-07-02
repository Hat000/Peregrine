# COMMANDER HANDOFF — Gen 8 boot briefing (2026-07-01) · LAPTOP→SHADOWPC MIGRATION

**Canary:** you are the OVERALL COMMANDER (Gen 8). Address **Fengyou** by name in every message (missing name = context degradation → rotate). Read `COMMANDER.md` (craft) + `MEMORY.md` (state) FIRST. This doc is self-contained: it embeds the current state + session deltas that are NOT yet formally banked into the memory files, plus the ShadowPC boot mechanics.

---
## 0. MIGRATION (why this doc exists)
As of 2026-07-01 all work moves **laptop → ShadowPC** — ShadowPC is now the SINGLE work machine (it already runs the official VQ2 eval sim; now it runs all dev + automation too). **The copy-paste relay is RETIRED** — no more bouncing between machines. **All committed work is pushed to origin** (verified: `git log --branches --not --remotes` is empty). Nothing is stranded on the laptop except one untracked duplicate of `tests/test_ff_owns_vertical.py` in the `naughty-lamport` worktree (already on origin via `42b9952` — ignore it).

**Fable directive (Fengyou, 2026-07-01):** Fable 5 is REINSTATED and expires **2026-07-07**. Use it heavily — for judgment-dense work AND to land fixes / refly faster. Route mechanical work (banking, git, flight-ops) to Sonnet; reserve Fable for the hard analyses + verification. (Commander's standing read: spend the premium window on the hardest problems, not on a consumption counter.)

---
## 1. THE THREAD — VQ2 slow-lap bring-up
Goal (Fengyou): **land ONE successful slow flight, then as many moonshots as we can muster.** Slow-is-smooth first; speed ramp after.

**A19 (2026-07-01) = the drain-to-empty receiver fix WORKED.** The detector is fed and the seeker flies itself for the first time (A19c `cmds=64 pursuit=22`, vs A18 `0/0`). First fully-observed active-pursuit flight. Frame supply is no longer the blocker.

**Master lever discovered — GPU CONTENTION on `detect`:** cost swings **32 ms (un-contended) ↔ 234 ms (contended, co-located sim)**, and that single variable drives loop rate + freeze + recorder cutout together. The operator's short-pre-GO protocol mitigates it (starves the GPU less). detect=cuda:0 confirmed (not CPU-fallback); 234 ms is contention, not model cost.

**Flight-ender diagnosed + FIXED — the vertical bang-bang:** at gate acquisition the alt-hold thrust slammed bang-bang between its clip rails (0.05↔0.60) every 2–4 ticks → net climb OVER the gate → gate lost → yaw-search. Root cause: the alt-hold damps thrust with `kd_alt=3.0` against `vel[2]` = the DEAD-RECKONED vertical velocity (no baro, ODOMETRY blocked, only sparse floor_height pins z). Differentiating noisy z → vz amplifies noise ~14×/tick. **This is the vertical twin of `ff_owns_horizontal`** (which we fixed for the horizontal axes and explicitly left live for vertical). Key lesson: **detector-fed ≠ estimator-accurate ≠ controller-fed** — the controller eats estimator output, not frames.

**337c554 yaw-sign test still PENDING** — the drone lost the gate to the climb before a clean off-axis-gate-2 yaw test. Reachable once the vertical fix keeps it on the gate.

---
## 2. BRANCHES ON ORIGIN (exact SHAs — all pushed)
| Branch | SHA | What |
|---|---|---|
| `claude/loving-galileo-92f020` | tip `c325585` (flight code = `db4e016`) | **Flight stack**: drain-to-empty UDP receiver + load-cut (detect_cached/decimation/device) + pre-warm + vision-timing/video-thread instrumentation. Latest MEMORY on this branch. |
| `vq2-ff-owns-vertical-2026-07-01` | `42b9952` | **VERTICAL FIX (the must-have)** — DONE, green_gate GREEN-except-known, offline-validated (rail-slam flips 19→3 on real A19c arc, 17→0 synthetic, thrust median 0.267≈hover). 6 files, default-off. NOT flown. |
| `vq2-detect-cut-a20-2026-07-01` | `fda3a29` | A20 detect knobs (imgsz/half, default byte-id). Based on `a013705` (older vertical intermediate). **Accuracy verdict PENDING** (see §4). |
| `claude/optimistic-banzai-208d3d` | `de334e7` | **Fixes the ONE known green_gate RED** (`test_diagnose_session` fixture guard) — MERGE to retire the standing RED. |
| `claude/condescending-hofstadter-0ea844` | `dbda6a9` | Flaky `fit_vp` perf-guard fix (the task chip). |
| `claude/cool-heyrovsky-e08624` | `9fe1aa7` | A MEMORY.md trim-under-budget (reconcile into the memory reconciliation). |

**ff_owns_vertical fix form:** default-off `Controller.ff_owns_vertical` (+ `ff_vertical_kd_alt=0.5` live-retune knob). When ON + `sp.accel_ned` present: position loop on floor-corrected z; vertical-align `vz_t` → a ramping `z_target`; damping against a LOW-PASSED diff-of-z, NOT `vel[2]`. Enabled ONLY in `deploy_profile.vq2_case_c` controller_overrides. OFF path byte-identical.

---
## 3. NEXT ACTIONS (priority order)
1. **FLY A20 = the vertical fix ALONE — branch `vq2-ff-owns-vertical-2026-07-01 @ 42b9952`** (it IS flight-stack `db4e016` + the ACTIVE vertical fix; no reconciliation needed). NO launch procedure in the box (the operator flies better than we script; just point at the branch + config):
   `python rl/fly_rl.py --gate-seeker --deploy-profile vq2_case_c --seeker-detector yolo --seeker-weights models/gate_clean_ens_course_L110.pt`
   READ (footage-grounded — the estimate is fiction): thrust arc should sit near hover **0.2656 with NO 0.05/0.60 rail-slam** (fix working); does it **hold altitude on the gate + thread it**; does it reach off-axis gate 2 for the **337c554 yaw test**. `ff_vertical_kd_alt` (default 0.5) is a safe live-retune knob if the alt channel rings.
2. **Detect-cut is a DEFERRED opt-in lever — do NOT bundle it now.** a20detect verdict: `imgsz=416` = −51% detect time but **loses long-range recall (100% ≤24 m → 69% beyond)**; `imgsz=320` worse (degrades from 16 m); `half=True` unmeasured (no CUDA on the laptop). Numbers are CPU + a SUBSTITUTE weight (`curriculum_v3`; the flight weight `L110.pt` is a gitignored artifact, not local) — relative trends transfer, absolutes don't. Deploy later ONLY if the loop needs it AND after CUDA + flight-weight validation on ShadowPC. ⚠️ The detect branch `fda3a29` carries `a013705` (an OLDER vertical fix) but NOT the vq2_case_c ENABLE (only in `42b9952`) — so DON'T fly `fda3a29` (vertical fix would be present-but-disabled). To deploy the cut: take the detect-knob delta from `fda3a29` (detector.py/navigator.py/fly_rl.py imgsz plumbing), apply on `42b9952`, flip the `vq2_case_c` seam to `imgsz=416`. Its accuracy numbers live ONLY in the transcript + here (the harness blocked the subagent from writing its FINDINGS file).
3. **Retire the known green_gate RED**: merge `claude/optimistic-banzai-208d3d @ de334e7` (the diagnose-test fixture guard) → green_gate goes fully GREEN.
4. **Reconcile memory properly** (see §7) once the A20 fly is read.

---
## 4. BACKGROUND AGENTS (the migration "don't forget")
- `vfix` (Opus) — DONE. Vertical fix on `42b9952`. Self-recovered a worktree collision.
- `a20detect` (Sonnet) — DONE. `fda3a29` detect knobs (default byte-id), pushed. Accuracy verdict + the "deferred lever" decision are in §3.2. green_gate GREEN-except-known. (Harness blocked it from writing its FINDINGS file — its verdict lives in §3.2 + the transcript only.)

---
## 5. BAND-AID REGISTER (technical debt — pay down AFTER the slow lap)
The band-aids almost all pay for TWO root debts: weak denied-state estimation, and a VQ1-heritage controller not built for denied state. **A better vision system (incoming — see §6) + a denied-state estimator/controller retires most in one pass.** Do NOT pay them down mid-bring-up.

TRUE fixes: drain-to-empty receiver · detector pre-warm · `detect_cached` dedup · `ff_owns_horizontal` · `ff_owns_vertical` · gyro/odo sign corrections · `cmd_rate_scale=0.4` (calibration for the sim's 2.5× rate gain).
BAND-AIDS: `kp_att` 10→4 softening · `body_rate_slew` limit · `hold_last_demand` bridge (masks ~75% pose-drop — better vision retires it) · `egress_thrust_floor`/`egress_freeze_attitude` guards · `min_trust_elevation` (borderline) · A20 detect-cut imgsz/half (only if accuracy holds, else band-aid).

---
## 6. OPEN STRATEGIC QUESTIONS (Fengyou, 2026-07-01)
- **A significantly better vision system is INCOMING.** ASK Fengyou: what is it + when? It attacks the ROOT (state quality) → better detections → better estimator → retires several band-aids + may moot the detect-cut. Sequence the merge order around it.
- **Retraining / twin refit:** the current twin is NOT accurate for VQ2 RL — training gives privileged state; VQ2 denies it; add the 2.5× rate gain + the ~14 Hz choked loop + vision cadence/noise. **Refit the twin to the VQ2 logs BEFORE any RL.** We now have A19c logs to refit against.
- **Never trained a policy on vision+IMU only** (the denied observation). Pure vision→control RL scored **~0%** in our pre-flight research → the path is **bootstrap** (classical/privileged first, distill toward vision), not end-to-end. This is the moonshot retraining target.
- **Controller architecture:** patched VQ1-heritage controller. For SPEED, evaluate (a) image-space visual servo vs (b) RL-on-vision+IMU. NOT needed for the slow lap.

---
## 7. MEMORY RECONCILIATION (do early on ShadowPC)
The `~/.claude/projects/.../memory/` mirror is machine-local and will NOT transfer — the ShadowPC commander must rebuild it. The most-recent committed memory is on `claude/loving-galileo-92f020` (main's memory is stale at 2026-06-27). Steps:
1. Copy repo `memory/` (from `claude/loving-galileo-92f020`) into `~/.claude/projects/C--Users-...-Anduril/memory/`.
2. Reconcile the `cool-heyrovsky` MEMORY.md trim (`9fe1aa7`) — MEMORY.md was over its 24.4 KB budget; route bloated/superseded detail DOWN to sub-indices (do NOT use /consolidate-memory).
3. Bank THIS session's deltas (all embedded above): A19 drain-fix-worked + GPU-contention master lever · vertical bang-bang diagnosis + `ff_owns_vertical` fix · band-aid register · Fable reinstatement · migration-to-ShadowPC · drop-launch-procedure-in-fly-boxes · worktree-isolation-for-code-delegations · detector-fed≠estimator-accurate≠controller-fed.

---
## 8. LESSONS THIS GEN (fold into COMMANDER.md)
- **Spawn load-bearing code delegations in their OWN worktree** (`isolation:'worktree'`). Two concurrent background agents collided in the session's primary worktree (`naughty-lamport`) — recovered with zero loss (one linear chain), but it cost untangling. Isolate by default for code-mutating agents.
- **detector-fed ≠ estimator-accurate ≠ controller-fed** — trace the full `frames→detector→estimator→controller` chain before concluding a subsystem is "fed."
- **Empirical over analytical** — read the footage + the thrust arc, never the dead-reckoned estimate (fiction on VQ2).
- **Don't rely on mid-task SendMessage to a background agent for load-bearing directives** — `a20detect` correctly treated the commander's mid-task "push your branch" message as a possible prompt-injection and refused it (good security instinct). Put everything in the INITIAL prompt; push an agent's branch yourself rather than asking it mid-run.

---
## 9. SHADOWPC BOOT MECHANICS
1. `git clone` / `git pull` `github.com/Hat000/Peregrine`.
2. Create the venv + `pip install` (the repo's requirements). venv note: on the laptop it lived at `<repo>/.venv` (main checkout, NOT worktrees).
3. Rebuild the `~/.claude` memory mirror per §7.
4. `git checkout main` → read this handoff → follow §3.
5. Directives still active: no launch procedure in fly boxes · sim = no physical risk (commander authorizes measurement flights directly) · slow-is-smooth · minimize band-aids · GPU-contention is the detect master lever.
