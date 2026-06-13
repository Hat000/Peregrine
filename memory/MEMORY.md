**Peregrine** — Anduril **AI Grand Prix** autonomous drone-racing entry (May–Nov 2026). Repo `github.com/Hat000/Peregrine` at `C:\Users\Fengy\Downloads\Projects\Anduril`. **This file is a THIN INDEX** — detail lives in the domain sub-indices + topic files; follow the `[[pointers]]`.

## NOW (2026-06-13)
- **VQ1 PASSED. inc7 LIVE-CONFIRMED = current best** (standing 5/5, gate-3 barrier gone). Phase 2 = RL + VQ2 vision.
- 🚩 **BINDING VQ2 RISK = ESTIMATOR** (not planner/policy). Case-C absolute world-frame nav = NO-GO at any speed; **FIX = gate-relative obs** (CONDITIONAL-GO, velocity-prior-sensitive). **Build gate-relative REGARDLESS** (organizer-pivot: Q① only decides load-bearing-vs-insurance). **RewindKF = DEFAULT.** → [[index-vision-estimator]]
- **S2 DECIDED = `staged_monolithic_then_decomposed`.** Speed gap = TILT ENVELOPE, not architecture. → [[index-rl-training]]
- **ACTIVE critical path:** ShadowPC §4 hardening live-verify (merge gate for branch `flyrl-autonomy-hardening` @7210c1d, pushed) + SHADOWPC-VISION-CAL agenda → merge → P4-C05 → gate-relative rebuild + inc8 envelope ladder. → [[index-control-sim]] · [[index-vision-estimator]]

## Operating directives — ALWAYS APPLY
- 🚩 **CANARY (MANDATORY):** address **Fengyou** by name in EVERY message. Missing name = context degradation → Fengyou rotates session.
- **BUDGET (rev 5):** weekly pool sole watch; push limits to MAX. **opus-4.8 = ALL correctness/judgment-dense work** (≈0.5× fable — the affordable design tier); **ultracode on opus** for hardest fan-out/verify; **sonnet-4.6 = low-stakes mechanizable** (banking, flight ops, git, reruns); haiku = trivial. PUSH models UP. 🚩 FABLE REVOKED (pending dispute).
- **BANKING (rev 4 — thin-index structure):** workers end with **MEMORY-DELTA** (≤10 lines); commander triages; sonnet banking agent banks IN BATCHES. **Route detail to the topic file + its domain sub-index; touch MEMORY.md ONLY for NOW / footgun / directive changes.** Agents `git add` ONLY their own paths (never `-A`). Mirror `memory/` ↔ `~/.claude`.
- **PROMPT-EMISSION:** re-emit COMPLETE worker prompts, never splices. Every prompt carries SESSION + MODEL(version) + EFFORT + canary + MEMORY-DELTA requirement + escape hatch + report path.
- **User directives:** speed > gate-in-view; tune offline / fly to verify; re-derive every verdict from data; bounded actuation, user owns GUI/risk; offline processing legal; SLURM only on Adroit; concurrent laptop sessions use separate worktrees.

## Cross-cutting footguns — forget = disaster
- 🚩 **GATE CONTACT = INVALID RUN** (zero-contact is THE validity rule).
- 🚩 **Judged runs use `rl/submit_rl.py`** (pins inc7, NO `MAV_CMD 31000` on the wire). `fly_rl.py` default ckpt = retired inc4 → pass inc7 explicitly if bypassing submit_rl.
- 🚩 **CTBR/VQ1 LEGACY SIGN CONFIG is a self-consistent alias — DO NOT "fix".** → [[index-control-sim]]
- 🚩 **ODOMETRY quat is R_y(π)-CONJUGATED;** run `scripts/frame_residual_report.py` after EVERY live session (internal consistency cannot catch a conjugation). → [[index-control-sim]]
- 🚩 **`main` CANONICAL · 687 tests green · `*.pt` + `data/runs` gitignored.**

## Library index (MEMORY → domain sub-index → topic files)
- 🛩️ **[[index-rl-training]]** — RL increments/lineage, training doctrine, S2 decision, inc8 + speed-ladder, plant/sysid-for-RL, retrain footguns, Adroit/DiffAero.
- 👁️ **[[index-vision-estimator]]** — estimator race-speed verdict + gate-relative, VISION-PKG2, detector pipeline, case-C readiness, estimator robustness, advisor-triage.
- 🎮 **[[index-control-sim]]** — CTBR control/sysid, deploy recipe + true conventions, sim interface (wire spec), sim ops (unattended mechanics), autonomy-hardening.
- 📋 **[[index-strategy-meta]]** — master plan (SSOT), competition materials + rules, hardware, prior-art, tooling eval, walking-skeleton + feedback.
