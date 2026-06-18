---
name: project-fullstack-burn
description: The 2-day opus-ultracode burn plan (~2026-06-18+) — full-stack integration center of gravity, the 3 VERIFIED wiring blockers, the data-staging critical-path-zero, the prove-then-saturate PIVOT, ranked lanes + cut list. Source = survey workflow wjsevwxfx + adversarial critic.
metadata:
  type: project
---

# Full-stack integration burn plan (2-day opus-ultracode burn, ~2026-06-18+)

**Source:** survey workflow `wjsevwxfx` (8 agents, 45 candidates, 6 dims) + adversarial critic, 2026-06-17. **Commander-adopted WITH the critic's PIVOT.** Trigger: weekly credit reset (~15h out from 2026-06-17), then a 2-day full-burn window.

## STATUS (pre-reset, 2026-06-17)
- **Pre-reset substrate = DONE** (the cheap, no-data half of the pivot): `scripts/diagnose_session.py` — the auto-judge; R_y(π) mirror canary now an always-on ATTENTION gate — MERGED `c07e886`. `scripts/green_gate.py` — merge-safety gate: stack-guard (torch must import or RED, else the +L test importorskips away = deceptive green) + 3 load-bearing invariants (OFF==inc7 AST, +L sign asserted-RAN, VQ1 import guard) + test-count sentinel **baseline 933** + diff-scoped pytest via merge-base — MERGED. The L-effort replay rig + TB adjudicator stay DEFERRED until the spike proves the loop (per the pivot).
- **RL lane:** warm-start running on Adroit (job 3276071, fine-tuning the S2-seed2 ckpt); result reads against the VQ2 wire.
- **Data staging — DONE via the artifact pipe (2026-06-17):** inc8-best + inc7 checkpoints LOCAL; detector `best.pt` (vq2_pose_8kp_r1clean, mAP50 0.991) + 5 curated flight bundles published to GitHub release **`burn-artifacts-2026-06-17`** (171 MB, 6 assets). Spike bundle = `flight_bundle_*_atspd_s2_s30_p3` (FINISHED, 6 gates, 0 collisions, 30 m/s). ⚠️ **verify the chosen bundle is FRAMES-bearing before the spike** — the report only confirmed 436 frames for the multigate bundle (`simopsA_c02_f1`); the in-repo `handoff/shadowpc-postfix-dataset-*` bundles were OBS-ONLY (`debug_obs.jsonl`+`meta.json`, no frames), so do NOT assume.
- 🚩 **ARTIFACT PIPE (live, canonical — replaces the cross-machine scp/Duo dance):** checkpoints / weights / curated recordings ship as GitHub *release assets* under dated `burn-artifacts-YYYY-MM-DD` tags (OUTSIDE .git → no bloat; 2 GB/file cap). Pull anywhere: `gh release download <tag> --repo Hat000/Peregrine --dir _staged`. Re-publish a file with `--clobber`. NEVER git-add binaries (`*.pt`/`*.pth`/`data/runs/`/`runs/` gitignored). Adroit's login node has internet → its checkpoints can join the same pipe. **gh recipe / gotchas:** NOT preinstalled (laptop + ShadowPC both lacked it → `winget install --id GitHub.cli`, lands at `C:\Program Files\GitHub CLI\gh.exe`, NOT on PATH → prepend per shell); no `gh login` needed — feed `GH_TOKEN` from `git credential fill host=github.com` (Hat000 PAT, repo scope, in Windows Credential Manager), but `GH_TOKEN` does NOT persist across PowerShell tool calls → re-set each call; PowerShell mangles `gh --jq '\(...)'` → use `gh ... --json | ConvertFrom-Json`; `MSYS_NO_PATHCONV=1` for Adroit `/scratch` scp (Git Bash rewrites the path).
- 🚩 **BURN OPS (earned this session):** (1) agent worktrees (`isolation:worktree`) come **STALE-BASED** — 3× (warm-start/diagnose/green-gate, all cut from old bases missing recent merges) → ALWAYS `git diff --stat main..<branch>` before merging; selective-checkout ONLY the agent's new/intended files; a naive merge reverts intervening work; prefer ADDITIVE agent tasks. (2) `*.pth` was NOT gitignored (only `*.pt`) → FIXED `3738193` (binaries were one `git add -A` from bloating the 636MB .git). (3) burn box needs `pyserial` or `pymavlink` tlog-grade degrades to the `meta.json` fallback (loses per-gate granularity). (4) `MSYS_NO_PATHCONV=1` for Adroit `/scratch` pulls (Git Bash rewrites the path).

## Center of gravity (the thesis)
The deployed VQ2 racer IS **inc7-policy ⊕ C2 gate-relative estimator ⊕ VQ2 8-kpt detector** stacked — and those three have **NEVER run in one closed loop.** The burn's one job: close + harden that loop offline at compute speed, before the judged wire.

## 🚩 THREE VERIFIED WIRING BLOCKERS (file:line)
1. `fly_rl.py:561` hardwires `detector=None` / `use_vision=False` / `use_given_position=True` → the deploy path never runs vision.
2. `estimator_obs.py:67` explicitly does NOT build `obs[17:20]` → **no inc8 (20-dim) checkpoint can be flown today.** The triple is built ONLY in the no-render emul (`_confidence_triple` @ `estimator_emul.py:307`, SIGMA_REF=0.05 / TAU_STALE=0.10).
3. `detector.py:33` hardcodes `N_CORNERS=4` against the merged 8-kpt VQ2 fork (`blender_gen/contract.py:56`, flip_idx [1,0,3,2,5,4,7,6], inner 0-3 claimed IPPE-identical) → shape mismatch / silent corner mis-order (the corner_to_center 180° bug that has bitten twice — VERIFY the IPPE order, don't assume).

## 🚩 CRITICAL-PATH-ZERO — DATA/CHECKPOINT STAGING (the silent burn-killer)
VERIFIED by the critic: the checkout has **NO inc7 actor + NO inc8 checkpoint** (only detector YOLO `.pt` in models/), and `data/runs/` has only 3 trivial items — NOT the corpus the plan drives on (`*_l3atspeed_*`, `*_atspd_*`, gate-0 head-on N~13k, 240-frame real-multigate). Ranks 1,2,3,5,6,7,12 ALL assume a flyable policy + rich corpus that are NOT present. **Before ANY fan-out, stage:** inc7 actor + inc8-best (S2-seed2) `.pt` + sidecars (Adroit /scratch), VQ2 detector best.pt (gitignored / origin/vq2-data / unmerged 0b99157), the recorded corpus (ShadowPC-local + origin/vq2-data). Needs Fengyou relay at hour 0. **Without it the Day-1 anchors fan out over fixtures that don't exist and burn credits producing nothing.**

## The PIVOT (commander-adopted): prove-then-saturate, NOT build-substrate-first
The survey's instinct (build ~15h of unattended substrate first) **inverts the risk** — building infra to service a loop nobody has proven closes, over fixtures that aren't staged, is a token-sink with zero race-day payoff. Correct order:
0. **Stage data/checkpoints** (critical-path-zero, hour 0, Fengyou).
1. **Day-1 FIRST action = VERTICAL-SLICE SPIKE:** ONE real frame → PnP → gate-relative KF → obs[0:20] → one actor forward pass, on ONE real bundle with the real checkpoint, asserted to a **frozen golden vector**. Proves the loop closes, closes blocker #2 (the 20-dim deploy builder), and surfaces the cold-start-no-position + emul-vs-live-faithfulness unknowns the sweeps silently assume away.
2. **Pre-reset, build ONLY cheap substrate:** `scripts/diagnose_session.py` (compose existing `race_outcome` + `frame_residual_report` R_y(π) canary + `verify_bundle` + debug_obs → one verdict; primitives verified present in scripts/); the diff-scoped green-gate (pytest + byte-identical invariants: OFF==inc7 AST, +L 4.77e-7, the VQ1-const import guard, a test-count sentinel); size/pointer/date sentinels. **DEFER** the L-effort replay rig + the TB GO/NO-GO adjudicator until the spike proves their target loops are real.
3. **THEN saturate** (Day-1→Day-2, onto the PROVEN loop): estimator emul-fidelity audit (does the emul KF the whole S-ladder trusts match the real Navigator→PnP→gate_relative_inplane_fix→RewindKF chain?); full-stack fault-injection (8+ cross-module surfaces); the offline p90/p99-vs-radius selection matrix; the detector INT8/latency leaderboard with CO-MEASURED keypoint accuracy; the ~20-constant C2 recalibration sweep (CV-backed).
4. **Build unattended orchestration onto the proven loop:** Adroit SLURM auto-orchestrator (CONDITIONAL on VQ2 un-gating RL; hard human-confirm before first auto-submit — AUP suspension blast radius); promote the ShadowPC sim-ops runner (partA_cycle.py + recoveries) from handoff/ to scripts/ + harden.
5. **POC** (Day-2, AFTER 1+2+sim-ops): instrumented self-localizing full lap on VQ1-slow; commander authorizes the measurement flights (sim = no physical risk), explicit POC sign-off is Fengyou's gate.

## Hold behind the VQ2 wire (~tomorrow) + non-RL fallback
Build case-C deploy PLUMBING (pays either way) but do NOT pre-commit appo/case-C training — the S3 cliff is case-C-specific + dissolves if VQ2 streams position (matches standing directive). 🚩 **If the wire SLIPS, queue a concrete non-RL fallback so the budget doesn't idle:** the MonoRace (arXiv 2601.15222) synthesis lane + the detector INT8/keypoint-regression leaderboard — both unblocked + token-hungry.

## Critic's MISSING lanes (add these — they were absent from the raw plan)
- **Golden end-to-end numeric regression fixture** — freeze a full detector→PnP→KF→obs→action golden vector (1e-6) the green-gate runs every merge. The machine-check that actually catches silent stack-level corruption (the +L / R_camera_from_body sin-flip / corner_to_center class). The plan builds scaffolding to PRODUCE traces but never FREEZES one as an invariant.
- **MonoRace synthesis** — memory flags it "read FIRST"; appeared nowhere. Extract its mono gate-PnP+IMU arch deltas, the offline IoU extrinsic-calibration recipe (== our ε_vert boresight lever / fix-ACCURACY), the appearance-robust detector path. De-risks Round-2.
- **VQ2-wire ingestion + cold-start / self-loc-init** — the scored wire has NO position → where does the KF get its first absolute anchor? The S-ladder ran on FLAT GT-anchor; deployment cold-start is unaddressed. Author a DORMANT wire-parser lane that fires when the sim drops.
- **Attitude-frame reconciliation vs the NEW VQ2 wire** — attitude+rates are the ONLY trusted case-C inputs; a single attitude-frame error silently rotates every gate-relative fix (the R_y(π) conjugation footgun lives here).
- **Kill-switch / safe-abort runtime supervisor** (#50/#63) — detect estimator divergence (covariance blowup / innovation monitor) → safe-hover/clean-abort on the JUDGED path. Race-day value > several offline sweeps.

## Cut list (do NOT build this window)
- Dual near/far vision + runtime selector (over-engineering; build the SINGLE near-field candidate as a gated arm + MEASURE the 12m floor first — it may just confirm 12m is irreducible, itself valuable).
- x86→Orin parity harness (no hardware until Sept; fold a golden-bundle pin into the INT8 leaderboard).
- Heteroscedastic per-corner σ detector head (gated on the photoreal dataset that doesn't exist yet — Round-2 neutral lesson).
- VQ2 4-gate re-render+retrain (Fengyou's call, gated on official arena ~06-29; #72 HDRI-fallback bug to fix when it runs).
- Cloud GitHub-Actions CI (torch/diffaero too heavy → false confidence; the local diff-scoped gate is the right form).
- appo / case-C A1-A3 verdict committed now (gated on the VQ2 wire — directive).

## Prompts in flight (fit-check)
- **Warm-start build** (BUILT + merged 11cb840; OFF byte-identical, 871 tests; NOT launched) = the RL lane; correctly gated on VQ2 (rank-8 + cut-list confirm: hold appo; warm-start-from-S2seed2 is the cheap RL next-step IF VQ2 un-gates).
- **ShadowPC sim-ID** (sent) = PLANT-layer twin fidelity (diffaero plant vs official sim dynamics). Complementary to the burn's ESTIMATOR-layer twin fidelity (emul KF vs real chain). Both wanted, different layers.

## Memory corrections surfaced by the survey
- Test suite is **~900** (collect-only), not 723 (stale) — sentinel fixed in MEMORY.md.
- Broken pointer fixed: `[[rl-increment-history]]` → `[[project-rl-increment-history]]` (project_phase2_rl_vision_decisions.md).
- MEMORY.md over the 24.4KB limit — the full line-6 (inc8 saga) compression remains pending the VQ2 re-bank.
