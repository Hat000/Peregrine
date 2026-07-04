# PEREGRINE — GEN 9 COMMANDER BOOT BRIEFING
**Handoff from Gen 8 (Opus 4.8) · 2026-07-04 · ShadowPC**

You are booting as the next **OVERALL COMMANDER** of Peregrine. Read this whole document before you touch anything. It is long on purpose — it is the difference between you continuing seamlessly and you re-deriving two days of hard-won ground. Everything below is what Gen 8 wishes it had known on turn one.

---

## 0. THE CANARY — OBEY THIS BEFORE ALL ELSE
**Address Fengyou by name in EVERY message you send him. No exceptions, every single message.**
This is the standing signal that your context is intact. If you ever notice you've dropped it, treat that as a red flag that something in your context is broken and re-ground yourself. It is not optional politeness — it is a liveness check Fengyou relies on.

---

## 1. WHO YOU ARE
- You are the **overall commander** of **Peregrine**, Anduril's **AI Grand Prix** autonomous drone-racing entry. You run the whole effort: diagnosis, design, delegation, and judgment.
- You are an **Opus-class** model (Gen 8 was Opus 4.8). You do the judgment-dense work yourself — diagnosis, control/estimation design, verification, commander decisions — and you **delegate** the mechanical and the parallelizable work to subagents (see §8).
- Your scarcest resource is **your own context window**. Protect it. Push deep dives, code tracing, and flight execution into background subagents; keep the conclusions, not the raw file dumps, in your context.
- You operate as a **single continuous session** with background delegation. Fengyou drives one conversation with you; you fan out to agents. (This is the "COMMANDER MODEL rev 2" — single session + background delegation; copy-paste only for the ShadowPC handoff of results.)

---

## 2. WHO FENGYOU IS (and the single most important working principle)
- **Fengyou** (fl3689@princeton.edu) is the human operator and your commanding partner. He sits at the ShadowPC, launches the sim, and **watches the flights with his own eyes**.
- **HIS EYES ARE GROUND TRUTH. VIDEO TRUMPS TELEMETRY, ALWAYS.** This is not a soft preference — it is the hardest-learned lesson of the whole campaign (see §7.1). When the operator's visual read of a flight conflicts with any metric you computed, **the metric is wrong.** Full stop.
- **You NEVER read video frames yourself.** You read instrumentation (logs, nav_estimate.jsonl, tlogs) only. When a flight needs a visual verdict, you render the onboard video and give Fengyou the **absolute path** and tell him exactly what to look for. He renders the verdict; you never claim to have "seen" the drone's behavior.
- He is technical and precise. He derives physics himself (he correctly derived the yaw–roll–pitch gimbal coupling from scratch). Give him real engineering answers, not hand-waving. When he asks "how do we solve this," he wants your actual solution reasoning, then execution.
- He gives crisp directives and expects them applied durably as standing rules. He escalates when he knows something you don't; you escalate to him when you're genuinely blocked on a decision only he can make (usually a physical/visual fact, or a risk/aggressiveness tradeoff).
- He values momentum. Once he green-lights a direction, drive it autonomously and bring him **results**, not a stream of check-ins. But honor explicit checkpoints you promised him (e.g. "I'll show you the offline proof before we fly").

---

## 3. THE MISSION
**Land ONE successful VQ2 slow lap in the simulator. "Slow is smooth" first — a clean completed lap beats a fast crash.**
Everything is subordinate to that one outcome. We are not optimizing lap time yet; we are trying to *complete a lap at all*. The whole campaign is a sequence of removing the thing that currently kills the run.

---

## 4. WHERE WE ARE RIGHT NOW ⟵ READ THIS FIRST IF NOTHING ELSE

### The state of play (2026-07-04)
We fly VQ2 in the sim. The drone currently: **lifts off cleanly → clears gate 0 → and (as of today) finally yaws the RIGHT way toward gate 1.** The immediately-blocking problem is now the **turn** onto the next gate — specifically making it a clean, coordinated, banked turn instead of an over-yawed skid.

### The single biggest thing that happened today: THE YAW SIGN IS FIXED
For **two-plus days** the drone rotated *opposite* to every yaw command — it could never point at the next gate, so it orbited. We believed this was an inverted actuation sign ("the VQ2 wire inverts yaw"). **That belief was wrong and cost us two days.** The truth, resolved 2026-07-04 by Fengyou's eyes across two clean flights:
- The "wire inverts yaw → `body_rate_sign` yaw = −1" model was built on the **cmd-vs-gyro same-sign metric, which is a FOOTGUN** — it is *invariant to the sign flip* (post-sign command vs the estimator's mirrored gyro always opposes, whatever your sign). Only the ~2.1× *magnitude* from that measurement was real; the *sign* was an artifact.
- The eyes-confirmed answer: **`vq2_case_c` yaw = `body_rate_sign (1,1,1)` (yaw = +1) + `yaw_steer_mode="A"`** (a controller R_cur-yaw recovery). Mode B (`yaw=−1`) yawed the wrong way; **mode A (`yaw=+1`) yaws correctly toward the gate.** `brs_yaw=+1` was arguably right since the A22 change; we chased a measurement ghost.
- **Committed and locked:** `400d212` ("yaw sign EYES-CONFIRMED — mode A"). The 6 yaw-sign tests were honestly reconciled to the non-inverting model with explicit "OPERATOR-EYES-RESOLVED, NOT the metric" provenance notes.

### The immediate pending action
The **coordinated-turn package (commit `4cb3f98`)** is built, offline-proven, verified by Gen 8, and **awaiting Fengyou's go to fly.** As of this handoff Gen 8 has just presented the proof + magnitudes to Fengyou and asked "OK to fly?" **If you are booting mid-stream: the next action is almost certainly to spawn a fresh Sonnet pilot and fly `4cb3f98`, then bring Fengyou the video.** (Flight command + pilot brief template in §10.)

### What the coordinated-turn package (`4cb3f98`) changes (all `vq2_case_c`-scoped, VQ1 byte-identical)
Operator eyes on the mode-A flight (run `20260704_120357`): the turn fired *absurdly early* (halfway from start to gate 1, nearly clipping gate 1), used *far too much yaw and too little roll*, and ended *facing sideways* (lost all gates). Diagnosis found:
1. **The designed coordinated pass-turn never fired at all.** An old A31 **orbit-breaker (a failsafe *brake*)** ran the turn instead — because the pass-turn arms on getting within `pass_arm_range_m=3.0` of the gate, but **tracked range floors at ~4.3m** (vision is lost as the gate fills/exits the FOV on the pass — the gate physically blocks the camera). So the pass never armed and the brake took over.
2. **The roll was starved by "omega-budget starvation" (NOT plain aero coupling).** The orbit-brake *freezes* the yaw setpoint while the drone yaws hard → the attitude-error becomes yaw-dominated → after the body-rate norm-clip (`max_body_rate=4`) the yaw component *eats the budget and starves roll* (roll cmd 1.19 when yaw-error is 0° → 0.59 at 90°, matching the flight's ~0.6). Also `total_accel_cap=2.0` was throttling `image_lat_cap=3.0` so the raised lateral cap was inert.
3. Fengyou's gimbal-coupling insight is real too: pitched nose-down + a flat body-yaw-right induces roll-left and pitch-up (`roll̇ ≈ r·tan θ`). The fix — a *banked* coordinated turn — cancels it.

The three fixes (interdependent; built as **one** package because A needs C to matter):
| Fix | Change | Purpose |
|---|---|---|
| **A** unthrottle roll | `total_accel_cap_mps2` 2.0 → **3.0** (lateral-first; forward stays governed by `forward_accel_mps2=0.65`, verified not to balloon) | lets the full ~17° bank command through, still slow forward |
| **B** cut yaw | `pursuit_yaw_slew_rps` & `visual_yaw_rate_cap_rps` 1.5 → **0.9** | roll-led turn; smaller yaw error → un-starves the roll |
| **C** real turn owns it | `pass_arm_range_m` 3.0 → **4.5** (arms at the ~4.3m vision floor) + `orbit_guard_rad` 1.75 → **3.0** (172°, demotes the orbit-brake to a rare failsafe) | the *coordinated* pass-turn fires at the gate plane; brake stops pre-empting it |

Offline proof (`handoff/a36_coordinated_turn_proof.py`) passes — deliberately **unit-level against the real controller+seeker, not a full kinematic sim** (an earlier synthetic "passed for the wrong reason"). It proves the *mechanisms* (arms at floor, yaw cut, roll survives at small yaw-error, orbit-brake demoted). **The emergent behavior — does it actually bank and keep the gate in frame — is what the confirm-fly + Fengyou's eyes verify.** Suite: 1761 passed, 11 pre-existing baseline failures, 0 new.

### After this turn works, the known remaining blockers
- **Completing the gate-1 turn cleanly** (this fly tests it). Then likely iterate roll-response magnitudes (kp/roll-cap) if the bank is still short.
- **Between-gate estimator drift** — this is the architectural risk (see §5, §6). The controller is map-free and transfers; drift in the self-localized position between gates is the deep problem VIO would eventually address (but VIO is deferred until *after* a lap on the classical servo — see §14).
- **Gate 2 is a HIGH gate — must climb to it.** Later gates are significantly higher. The vertical channel (A28) is solved for gate 0/1; the climb to gate 2 is next.
- Pose latency showed a suspicious **flat ~293ms pin** on the last run (likely a clock/staleness artifact like the old `pose_age=1.0` bug, separate from the real ~66ms vision obs-age). Chase it after the turn settles.

---

## 5. THE VQ2 PROBLEM — TECHNICAL FOUNDATION

**VQ2 is the self-localizing (case-C) race.** The wire is deliberately impoverished:
- **DENIED on the wire:** POSITION, VELOCITY, ATTITUDE, LOCAL_POSITION_NED, ODOMETRY, GATE_INFO/map. No magnetometer, no barometer.
- **AVAILABLE and authoritative:** `HIGHRES_IMU` (accelerometer + gyro only), the **vision gate detector** (our YOLO pose model), and `RACE_STATUS.active_gate_index` (the sim crediting a gate pass — the one piece of authoritative external truth).
- Consequence: the stack must **SELF-LOCALIZE.** Attitude comes from an IMU AHRS; yaw and altitude are pinned by vision; position comes from a gate-relative chain. There is **no given position and no map.**

**Because of this, on the VQ2 wire the estimator's own output is THE DEFENDANT, not evidence.** When you're debugging, never treat the estimator's reported position/attitude/velocity as ground truth to reason from — it is the thing under test. Get ground truth from Fengyou's eyes first, *then* analyze telemetry. (Memory: `estimator-telemetry-is-defendant`, `ask-human-observation-before-data-dive`.)

**Gate geometry priors (load-bearing for the detector/PnP):** gates are **upright** (roll = pitch = 0), a **concentric-square** target — inner square 1.5m, outer 2.72m — with an **8-keypoint** layout. This enables a gravity-constrained upright 4-DOF PnP fit. Gates appear as **glowing RED on a dark warehouse** (not purple/orange — recon-confirmed). Gate spacing is **~23.7–38.5m**; **no next gate is ever more than ~30m away** (this is why the A34 absolute range cap of 35m works — anything the detector "sees" beyond 35m is a garbage mis-depth lock).

**The control handshake (live-confirmed):** `ARM(400)` + `SET_ATTITUDE_TARGET` with the body-rate mask at ~100Hz, `ControlMode.BODY_RATE`. The body-rate uplink pre-scales by **`cmd_rate_scale = 0.4` (= 1/2.5)** to compensate the wire's ~2.5× realization gain. The **gyro is fully sign-negated on all three axes** vs the code's FRD assumption, corrected at the wire by `gyro_sign = (-1,-1,-1)` before the AHRS (magnitude ~2.1× is handled separately by `cmd_rate_scale`; only the sign is corrected here).

---

## 6. THE CONTROL / ESTIMATION STACK (how a flight actually works)

**Architecture philosophy (Fengyou's, load-bearing):**
- **Map-free classical servo FIRST; tightly-coupled gate-landmark VIO AFTER a lap** (not generic SLAM, not speculative). The controller is map-free and *transfers*; the risk is between-gate estimator drift. Land the lap on the classical servo, then add VIO as the between-gate estimator if drift demands it. The vertical filter (A28) is literally the vertical marginal of that future VIO, so the work compounds.
- **Aim for LESS preprogrammed/open-loop behavior, not more.** We already have a spawn-egress maneuver; don't add canned maneuvers. Prefer closed-loop servoing.
- **Don't hand-roll estimation that's long-solved** — use references/prior art (Mahony 2008 AHRS, Solà ESKF, Lee et al. SO(3) geometric control, PX4/ArduPilot bounded-distrust). 
- **Weight measurements by consistency; don't hard-reject.** Hard gates are crude and starve the servo; soft (Cauchy/Huber) weighting is preferred.
- **A robust start (no crash) protects the AHRS** by keeping the drone out of the contact-spike regime that corrupts attitude.

**The vertical channel (SOLVED, A28, commit `00569c2`).** A 2-state complementary filter (z_off, vz_rel; IMU predicts, fresh gate-poses correct via α-β, innovation-gated) + a single-PD vertical law with the old `vz_t` term removed. This killed the ground↔ceiling divergence that was rocketing the drone into the ceiling. Operator: "MUCH calmer vertically, coasted through gate 0." Later hardened by A34 (absolute range cap → clean pose feed) and A35 (a magnitude-gated trust floor so a sustained honest offset can't be soft-weighted into the wrong sign; and a liftoff hover-floor so it doesn't sink on the line).

**The horizontal/pursuit channel.** Image-servo: the seeker rolls toward the gate's apparent azimuth and yaws to point; a pass-turn state machine (refine-to-real-gate + hold-until-pointed) handles the turn onto the next gate. This is the channel currently under active work (the coordinated-turn package).

**The detector (flight default).** `models/vq2_darkred_negv1_2026-07-02_fp16_384x640.engine` — a TensorRT fp16 384×640 engine, parity-exact with the .pt, ~2× faster (82% good-fix vs the old 32% flight-default we flew for too long). **It requires the `.venv` python (tensorrt lives there).** Never fly the raw `.pt` again (offline/parity only). Detector runs in an async daemon worker (`async_detect`) so the ~30Hz control loop never blocks on the ~250ms GPU-contended detect stall.

**Current `vq2_case_c` profile state (the curated ON bundle; all the below are the live values):**
- `gyro_sign (-1,-1,-1)`, `cmd_rate_scale 0.4`, `async_detect True`.
- Yaw (EYES-CONFIRMED): `body_rate_sign (1,1,1)`, `yaw_steer_mode "A"`.
- Turn (coordinated package `4cb3f98`): `total_accel_cap_mps2 3.0`, `image_lat_cap_mps2 3.0`, `pursuit_yaw_slew_rps 0.9`, `visual_yaw_rate_cap_rps 0.9`, `pass_arm_range_m 4.5`, `orbit_guard_rad 3.0`, `forward_accel_mps2 0.65`, lateral-first budget on.
- Range: `track_abs_range_cap_m 35.0` (A34 — hard-discard >35m garbage locks).
- Vertical: `gate_pd_vertical True`, `kp_alt 0.0`, `kp_gate 0.04`, `ff_vertical_kd_alt 0.06`, `use_zoff_big_trust True`, `zoff_reseed_min_w 0.3`, `alt_thrust_lo 0.05`, `hold_thrust_lo_frac 1.00`.
- `hover_thrust 0.2656` (a measured fit, good to ~2.3% of g). **The plant CAN thrust below free-fall** (Fengyou-confirmed net-down authority to arrest overshoot) — but sub-0.15 collective is *unmeasured* (floor-smack risk).

---

## 7. HARD-WON LESSONS & FOOTGUNS (each one cost real time)

**7.1 — Trust the operator's eyes over ANY yaw-sign telemetry metric.** The "cmd-yaw vs raw-gyro same-sign" and "realized-rotation-vs-azimuth toward-gate" metrics BOTH misled us — repeatedly, confidently, and in the same direction. They read "yaw is correct" while the drone was visibly yawing the wrong way. The body-rate is logged *post-sign* and the plant response makes these metrics invariant to the very flip you're testing. **Never adjudicate a yaw sign with telemetry. Fly it, render it, ask Fengyou's eyes.** This single lesson, learned late, is why the yaw sign finally got fixed.

**7.2 — A sign that's burned us both ways: never flip a sign on one run's offline correlation.** The A22 "regression" was declared off a run that *predated its own fix commit and wasn't even flying the sign A22 thought it was.* Gen 8 then flipped the sign twice more chasing telemetry. The discipline that finally worked: make the sign a **flag-gated, fully-reversible knob**, ship two candidate settings behind a selector, and let the operator's eyes pick the winner in ≤2 flights. Don't hard-code a sign you can't verify offline — make it a toggle and let the physical oracle decide.

**7.3 — The estimator is the defendant (see §5).** On a position-denied wire, estimator output is under test, not evidence. Also: a self-localized position readout drifting to *thousands of meters* over a flight is almost certainly **dead-reckoning drift**, not the drone physically there. Trust the onboard video for where it went.

**7.4 — Ask the human's direct observation BEFORE a deep telemetry dive.** Fengyou's sensory read reframes the whole analysis and saves you from chasing artifacts. Get his eyes first.

**7.5 — Metrics/harnesses can "pass for the wrong reason."** Twice, offline checks passed while not actually reproducing the failure. Prefer **unit-level proofs that exercise the real controller/seeker code** over elaborate synthetic sims that can quietly diverge from reality. When an agent says a proof passed, ask *what mechanism* it proved.

**7.6 — Fly the TRT engine with the `.venv` python.** For days we flew the slow raw `.pt` because `python rl/fly_rl.py` resolved to a tensorrt-less env → ~120ms+ pose age. The engine needs `C:/Users/Shadow/Peregrine/.venv/Scripts/python.exe` (tensorrt 10.16.1.11). `vq2yolo-venv` is train/eval only (no tensorrt). Keep `--seeker-weights` explicit (a footgun-guard requires it so a retrain can't silently fly a stale engine).

**7.7 — Reset the sim via RETURN-TO-MENU, not the vq2ctl restart function.** Restart needed a manual override every flight and could leave a degraded/short-flight state. Use `to-menu → waiting → go` (§10) and verify a clean fresh-armed state (probe + screenshot) before every launch.

**7.8 — The stale-sim gotcha.** The probe can report `WAITING` while the window is actually a crashed/paused race. Confirm the visual state with a screenshot; don't trust the probe alone.

**7.9 — `--ignore-collisions` for tuning/confirm flights.** A single glancing threat-2 *environment* collision (id 1002, a light wall/floor graze during an aggressive bank) trips `final_state=CRASH` and truncates otherwise-good runs. `--ignore-collisions` (already built, `rl/fly_rl.py`) logs the clip but keeps flying so you can observe the turn. Normal collision physics still apply; the operator's eyes are the real crash judge.

**7.10 — PowerShell here-string quote mangling.** `@'...'@` with embedded double quotes splits args mid-string. Write payloads to files instead. (You're on Windows; the Bash tool is Git Bash / POSIX sh — each takes its own syntax.)

**7.11 — Pilot subagent transcripts can vanish** (an old pilot could not be resumed). Don't rely on resuming a pilot; be ready to **spawn a fresh Sonnet pilot with the full sim-ops brief** each time (template in §10).

**7.12 — `tests/test_nav_estimate_log.py::EXPECTED_KEYS` is a recurring footgun** — any new nav-log field must be added there or the suite breaks. (Current baseline: 46 keys.)

**7.13 — The gate blocks the camera on the pass.** Vision (and thus tracked range) is lost as you fly through a gate; range floors at ~4m. Don't gate turn logic on getting closer than the vision floor. Use `RACE_STATUS.active_gate_index` (authoritative plane-crossing) or a range threshold above the floor — but mind that the wire pass event can *lead* the physical plane by ~9m.

---

## 8. STANDING DIRECTIVES & PROCESS (how Fengyou wants you to work)

**Delegation model:**
- **Flight EXECUTION → Sonnet subagents.** Reset + attach + GO + render is a mechanical checklist. Reserve Opus/Fable for diagnosis, design, and build.
- **Diagnosis / design / build → Opus or Fable dives** (background subagents). Use them to preserve your own context.
- **The Fable/dive process:** write crystal-clear briefs; **flag your hypotheses AS hypotheses** (don't state guesses as fact); enable a **two-way, ask-don't-assume** channel (instruct every dive to ask you when unsure via SendMessage to `main`; you escalate to Fengyou when *you* don't know); **resume** resumable subagents (`SendMessage` to the agentId) rather than re-spawning and losing their context.
- **When a dive hands back a meta-message uncommitted** (stops before finishing the suite/commit), **you take over the verify + commit** rather than re-sending. Gen 8 did this repeatedly.
- **A dive that catches its own flaw and STOPS before committing is doing the right thing** — reward it, resolve the fork, don't punish the pause.

**Engineering discipline:**
- **Minimize band-aids.** Prefer real fixes and proven methods/prior art over patches. Don't build new thresholds when an existing flag does the job.
- **Verify, don't trust.** Independently confirm an agent's commit before you fly it: check the commit is real and on-branch, confirm the profile values, run the focused tests + the offline proof yourself. Gen 8 caught nothing wrong doing this — but the discipline is why the yaw win is trustworthy.
- **Every VQ2 change is `vq2_case_c`-scoped and VQ1/case-A byte-identical.** VQ1 (`vq1_case_a`) is the legacy proven baseline; it must never change. New flags default OFF (byte-identical when off). Keep byte-identity pins in tests.
- **New control fixes: flag-gated + offline proof + confirm-fly.** The offline proof gates the *mechanism*; the fly + Fengyou's eyes gate the *emergent behavior*.
- **Convert relative dates to absolute** when you record anything durable.

---

## 9. THE ENVIRONMENT (ShadowPC)
- **ShadowPC** is a Shadow **cloud VM**: passthrough **RTX 2000 Ada**, **4 cores**, **no iGPU**. Windows 11. The **sim and the detector SHARE the one GPU** — the sim's render load starves guest CUDA (confirmed not fixable guest-side), which is why detect latency swings 32↔234ms and why the async-detect decouple (loop 3→15.5Hz) is the win. The GPU is not the bottleneck; GPU *arbitration* is.
- **Pin these or things break:** `albumentations==2.0.8` / `albucore==0.0.24`; `torch 2.12.0+cu126`. (A fresh venv was missing albumentations and a CUDA torch broke the flight weight — both fixed, keep pinned.)
- **The flight venv:** `C:/Users/Shadow/Peregrine/.venv/Scripts/python.exe` (has tensorrt — REQUIRED for the engine). **The vision/train venv:** `C:/Users/Shadow/vq2yolo-venv/Scripts/python.exe` (train/eval only, NO tensorrt). Never confuse them.
- **Platform notes for you:** primary shell is **PowerShell** (Windows PowerShell 5.1 — no `&&`/`||` chaining, no ternary; use `;` and `if`). A **Bash tool** (Git Bash / POSIX sh) is also available — use `/dev/null`, forward slashes, heredocs there. Prefer the dedicated file/search tools over shell `grep`/`cat`/`find`.
- **The sim:** `FlightSim.exe → DCGame`. Bring-up keys (verified): `Enter → Enter → Down×2 → Enter×2`. Dual-instance UDP split = "armed-but-deaf" (a real gotcha). Fullscreen auto-minimizes.

---

## 10. TOOLS, FILES, COMMANDS (the operational kit)

**Canonical flight command** (the standing default — the venv is load-bearing):
```
C:/Users/Shadow/Peregrine/.venv/Scripts/python.exe rl/fly_rl.py --gate-seeker --deploy-profile vq2_case_c --seeker-detector yolo --seeker-weights models/vq2_darkred_negv1_2026-07-02_fp16_384x640.engine --ignore-collisions
```
(Drop `--ignore-collisions` only for a scored/final run. Run from the active worktree — see §12.)

**Render onboard video WITH detector boxes** (the deliverable for Fengyou's eyes):
```
C:/Users/Shadow/Peregrine/.venv/Scripts/python.exe handoff/tools/render_cmd_indicator.py <session_dir> <out.mp4> --yolo models/vq2_darkred_negv1_2026-07-02.pt
```
Note: `--yolo` REQUIRES a weights arg (bare `--yolo` errors); pass the `.pt` for the boxes (parity-identical for visualization). `render_cmd_indicator.py` overlays per-frame command indicators + frame-lag banners — use it instead of the stock renderer.

**Sim ops — `scripts/vq2ctl.py`** (the canonical 3379 sim-ops CLI): `fly` = any-state → fresh race; `to-menu` = Esc, Left×2, Down×4, Enter → main menu; `waiting` = Down×4 → waiting room; `go` = start race. Pause-menu pin is LEFT×2. **Use return-to-menu, NOT restart** (§7.7).

**Fresh Sonnet pilot brief template** (spawn one per fly; they're mechanical and don't retain context):
> Fly ONE VQ2 flight, do NOT edit code. Worktree `<path>`, branch `<branch>` (there may be an intentional uncommitted profile change — verify the profile prints the expected values before flying; do NOT git-reset). Reset via return-to-menu (`vq2ctl to-menu → waiting → go`), confirm fresh-armed by probe + screenshot. Fly the canonical command (`.venv` python, TRT engine, `--ignore-collisions`). Render with `--yolo`. Report: absolute mp4 path; passed gate 0?; survival time / final_state / max active_gate_index; median pose latency; any VISIBLE hard crash. Do NOT judge yaw direction — that's the operator's eyes. STOP and report if anything is off; don't improvise.

**Key source files (in the VQ2 worktree):**
- `src/racer/deploy_profile.py` — `vq2_case_c()` is where every VQ2 fix is gated. This is your primary control surface.
- `src/racer/gate_seeker.py` — the seeker: detection, pursuit, pass-turn state machine, orbit-breaker.
- `src/racer/vertical_estimator.py` — the A28 complementary filter + A35 trust floor.
- `src/racer/controller.py` — attitude/body-rate controller; `omega *= body_rate_sign` (logged post-sign); `_decoupled_body_rate` builds R_cur from `odo_att_sign`.
- `rl/fly_rl.py` — the flight entry (`--gate-seeker`, `--deploy-profile`, `--seeker-detector`, `--seeker-weights`, `--ignore-collisions`).
- `handoff/a36_coordinated_turn_proof.py`, `handoff/a36_offline_turn_proof.py` — the turn offline proofs.
- `tests/test_vq2_*.py` — focused per-fix tests + byte-identity pins.

---

## 11. THE MEMORY SYSTEM
You have a persistent file-based memory at `C:\Users\Shadow\.claude\projects\C--Users-Shadow-Peregrine\memory\`. `MEMORY.md` is the index (one line per memory) loaded each session; each fact is its own file with frontmatter (`type: user|feedback|project|reference`). **Write durable, non-obvious facts there; update the index line.** Before saving, check for an existing file to update rather than duplicate; delete memories that turn out wrong. The yaw saga, the plant/thrust facts, the process directives, the campaign state — all live there. **When a recalled memory names a file/flag/value, verify it still exists before acting** (memories reflect what was true when written). The single most important memory to read first: `vq2-yaw-sign-inverted-a22-regression.md` (now the RESOLVED yaw story + the metric-footgun lesson).

---

## 12. GIT / WORKTREE STATE
- The VQ2 work lives in worktree **`C:/Users/Shadow/Peregrine/.claude/worktrees/agent-a08b24f288de69e40`**, branch **`vq2-gate2-turn-dive`**. (Gen 8's own commander session ran from a *different* worktree, `ecstatic-kepler-30e9c7` — don't confuse them; the flyable code is in the `agent-a08b24f288de69e40` tree.)
- **HEAD (2026-07-04): `4cb3f98`** feat(vq2): A36 coordinated turn (A+B+C) — awaiting fly.
- Recent lineage: `4cb3f98` (coordinated turn) → `400d212` (yaw sign eyes-confirmed, mode A) → `fe49580` (yaw selector off/A/B) → `8884ead` (A36 turn package) → `458223d` (A35 vertical trust floor + liftoff) → `80902d4` (A34 abs range cap) → `00569c2` (A28 vertical solved).
- **Commit only when Fengyou asks or when locking a verified win.** End commit messages with the `Co-Authored-By: Claude` trailer. Keep VQ1 byte-identical in every commit.

---

## 13. IMMEDIATE NEXT ACTIONS (in order)
1. **Fly the coordinated-turn package `4cb3f98`** (pending Fengyou's go — Gen 8 just asked "OK to fly?"). Spawn a fresh Sonnet pilot (§10), return-to-menu reset, TRT engine, `--ignore-collisions`, render with `--yolo`.
2. **Bring Fengyou the video** with the specific question: *right after gate 0, does it now **bank** (roll-led) into the turn, keep the gate in frame, and not over-yaw/face-sideways?* His eyes are the verdict.
3. **If the bank is good** → lock it (already committed) and move to *completing* gate 1 → then the climb to the HIGH gate 2. **If the bank is still short** → the next lever is the roll-response side (kp / roll-cap); re-engage the dive. **If it over-yaws or faces sideways still** → re-diagnose the pass-turn geometry (is it firing? is `hold-until-pointed` engaging?).
4. **Chase the ~293ms flat pose-latency pin** once the turn settles (likely a clock/staleness artifact).

---

## 14. THE ROADMAP AFTER THE TURN
- **Complete gate 1, then gate 2 (a HIGH gate — climb).** Later gates are higher still.
- **Between-gate estimator drift** is the deep risk. When the classical servo can't hold the lap because of drift, that is the trigger to build the **tightly-coupled gate-landmark VIO** — *after* the lap, not before, and not generic SLAM. The A28 vertical filter is its vertical marginal; the work compounds.
- **Don't add open-loop maneuvers.** Solve with closed-loop servoing and robust estimation (proven references).
- The **YOLO detector** can later be improved (n/m sizes + ensemble, fold in real frames, re-test blooms on real red-bloom frames) — but parked until there's real data; don't start speculatively. Never overwrite the champion weights.

---

## FINAL WORD
Two-plus days of this campaign were spent chasing a yaw-sign ghost that telemetry kept "confirming." The lesson that broke it: **build the reversible experiment, then let Fengyou's eyes decide.** When you're stuck on something physical, stop computing metrics and get the operator's visual ground truth. Keep the CANARY. Keep VQ1 byte-identical. Fly slow and smooth. Land the lap.

*— Gen 8 (Opus 4.8), 2026-07-04*
