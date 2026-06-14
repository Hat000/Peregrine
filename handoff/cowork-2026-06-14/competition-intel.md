# PEREGRINE COWORK — FINDINGS MEMO
## Anduril "AI Grand Prix" Autonomous Drone-Racing Competition Intelligence

**Prepared for:** Fengyou (to relay to commander)
**Date:** 2026-06-14
**Effort:** MAX · opus-4.8 · deep-research methodology (direct action, no sub-agents)
**Source posture:** Public material is *abundant*, not scarce. The escape hatch is NOT needed. Critically, the document we held as an "internal spec" (`VADR-TS-002`, `260508_Technical_Spec_0002.pdf`) is in fact the **official, publicly-hosted Technical Specification**, linked directly from the competition's Official Rules page and from Elodin's write-up. Every competitor has it. That raises confidence on Q1–Q3 substantially.

---

## BLUF (Bottom Line Up Front)

The "AI Grand Prix" is real, live, and well-documented (Anduril + Drone Champions League + Neros + JobsOhio; $500K + an Anduril job; 1,000+ teams). It is a **three-phase** competition: a **virtual qualifier in a DCL-built simulator** (what we submit to now), a **physical qualifier** (Sept 2026, SoCal), and a **live final** (Nov 2026, Columbus OH). Phases 2–3 fly **identical Neros drones**.

**The single most important correction to our internal framing:** our "~100 TOPS onboard compute" figure is real and official — **but it describes the *physical* Neros drone's onboard AI module, not the virtual-qualifier evaluation box.** The virtual qualifier (the thing we are coding against today) runs in a DCL simulator that the spec says executes on **Windows 11 + a standard PC + a discrete desktop GPU with 8 GB VRAM** — i.e., **CUDA-class desktop hardware, not an embedded Jetson and not CPU-only.**

**Consequence for our latency axis:** It closes — comfortably — in *both* contexts. On the virtual-qualifier desktop GPU our ~15–25 ms vision path is trivially under the ≤50 ms budget (the ~115 ms CPU-only path is never forced). The "~100 TOPS → Jetson Orin" mapping is *correct as a device-class inference for the physical drone* (Orin NX ≈ 100 TOPS) and remains the right design target for sim-to-real portability — but it should not be mistaken for the qualifier's eval hardware, which is more capable. **Net: stop treating latency as a live risk for the qualifier; treat "≤50 ms at INT8 on an Orin-class module" as the portability constraint for the physical rounds.**

---

## Q1 — (LOAD-BEARING) Evaluation hardware: GPU/CUDA-class or CPU-only? Does "~100 TOPS" → Jetson Orin?

**Answer:** There are **two distinct compute contexts**, and conflating them is the trap:

1. **Virtual Qualifier eval (now):** GPU/CUDA-class **desktop**, not embedded, not CPU-only. The official Technical Spec (VADR-TS-002 §5.1) states verbatim: *"The DCL Simulator software runs on Windows 11 with a standard PC and a decent GPU with 8 GB VRAM. Currently we do not support Linux OS,"* and *"Participants may assume a Python-based runtime environment. Python 3.14.2 is known to operate correctly."* So the evaluation/runtime machine is a **Windows 11 desktop with a discrete ~8 GB-VRAM GPU** (RTX-3060/4060-class envelope). This is **more** compute than an Orin, not less. CUDA/TensorRT/onnxruntime-gpu are the natural inference paths.

2. **Physical drone (Sept/Nov rounds):** This is where "~100 TOPS" actually lives. The official FAQ says verbatim: *"the drone will use an AI-focused compute module (more capable than a Raspberry Pi), with a rough indication around **~100 TOPS** depending on configuration and power limits,"* and that the module includes *"onboard AI compute with RAM/SSD, an FPV camera feed, IMU sensors … and connectivity modules such as Wi-Fi/Bluetooth."* That profile (≈100 TOPS, CUDA-class, RAM/SSD, camera/IMU, Linux edge module) maps cleanly to an **NVIDIA Jetson Orin NX** (the Orin NX 16 GB is the canonical 100-TOPS edge module; AGX Orin = up to 275 TOPS; Orin Nano = 40–67 TOPS). The Orin NX is also the industry-standard ~100-TOPS drone mission computer (e.g., Neousys FLYC-300).

**So: CONFIRM the "~100 TOPS → Jetson Orin" mapping — but scope it to the physical drone.** Most likely device: **Jetson Orin NX (16 GB)-class.** Neros has not publicly named the exact module (they describe a "custom integrated flight computer"), so the specific part is an *inference*, not a sourced fact. Caveat for sizing: "100 TOPS" is the **sparse INT8** marketing figure; **dense INT8 ≈ ~50 TOPS**, and FP16 lower — budget the real model against ~50 TOPS dense, not 100.

**Latency verdict (the payoff):** The latency axis **closes**.
- Virtual qualifier: desktop GPU → 15–25 ms is far under ≤50 ms. No contest. The CPU-only ~115 ms scenario is **not** the operative constraint and should be retired as a qualifier risk.
- Physical drone: ≤50 ms at 640×360 is achievable on Orin NX **if engineered for it** (TensorRT INT8, small detector/segmentation backbone). Existence proof: the A2RL-winning MonoRace stack ran gate segmentation onboard and its control net at 500 Hz on the flight controller (see Q4).
- **Binding constraint going forward:** design the perception model to hit ≤50 ms at INT8 on an Orin-class module so the *same Python stack* ports from the generous desktop sim to the constrained drone (the FAQ confirms the Python approach is meant to transfer to the physical drone).

**Residual unknown (worth one verification when the sim package drops):** whether the qualifier's GPU is fully available to our inference or shared with the simulator's renderer (the 8 GB VRAM is one pool). Plan for **~3–5 GB VRAM headroom** for our model alongside the renderer, and confirm CUDA visibility on first run.

**Evidence/Sources:** VADR-TS-002 §5.1 (official spec); theaigrandprix.com FAQ ("onboard compute … ~100 TOPS"; "AI compute module"); Notebookcheck/NVIDIA (Orin NX 16 GB = 100 TOPS); Hackster/Waveshare (AGX Orin = 275 TOPS); Neousys FLYC-300 (Orin NX drone mission computer); Elodin harness (desktop GPU-rendered sim).

**Confidence:**
- Eval is GPU/CUDA-class desktop (Windows 11, 8 GB-VRAM GPU), not CPU-only, not Jetson: **HIGH** (official spec, verbatim).
- "~100 TOPS" describes the physical drone, and the latency axis closes: **HIGH.**
- Specific device = Jetson Orin NX-class: **MEDIUM** (well-grounded inference from the official ~100-TOPS figure + industry norm; not a named source).

---

## Q2 — Does the timed phase stream ground-truth pose, or must we self-localize?

**Answer: No ground-truth pose is streamed during the timed run. The onboard estimator is LOAD-BEARING, not insurance.** You must self-localize from vision + IMU.

- Official FAQ, verbatim: *"Will there be access to coordinates or absolute positions? Teams may receive **limited coordinate information for the starting position** in the virtual qualifier. Beyond that, you should expect to fly **without coordinate/position data**."* And: *"The drone will not 'know' the track. Teams must detect gates and navigate using onboard sensing (primarily vision). Gate position details may be provided only at a rough level."*
- VADR-TS-002 corroborates from the protocol side: §3.3 *"No geographic coordinates… GPS simulation is not available and absolute global position is not exposed."* The supported telemetry messages (§4.3) are `HEARTBEAT`, `ATTITUDE`, `HIGHRES_IMU`, `TIMESYNC` (sim→client) — **there is no `LOCAL_POSITION_NED`, `GLOBAL_POSITION_INT`, or `ODOMETRY` message.** Position is simply not on the wire.

**Important nuance — what you DO get vs. what you must estimate:**
- **Available:** attitude/orientation + angular rates (`ATTITUDE`), full IMU (`HIGHRES_IMU`: accel, gyro, mag, baro), and the spec's §4.5 telemetry list explicitly includes **"linear velocities"** and "system status flags." So you likely get **body/NED velocity and attitude for free** — but **not position**.
- **Must estimate:** absolute/gate-relative **position**. With a known start (origin = arming point) plus provided velocity, you can dead-reckon position by integrating velocity, but drift forces **vision-based gate detection + gate-relative pose** as the real localization anchor. The known gate geometry (1.5 m square opening) is your scale/PnP reference.
- **Control vs. feedback asymmetry:** you *can command* `SET_POSITION_TARGET_LOCAL_NED` (a position setpoint in the armed-origin frame), but you are *not given your position estimate back*. So position **control** is offered while position **feedback** is withheld — which is exactly what makes the estimator load-bearing.

**Cross-source caveat:** the third-party Elodin practice rig exposes a convenience `world pose` to its solver, but its author explicitly flags this as a deviation, noting the official contract is *"No GPS, no depth, no motor RPM."* Also, the FAQ ("accelerometer, gyroscope, likely motor RPM") and the spec ("linear velocities") and Elodin ("no motor RPM") disagree at the margin on exactly which derived quantities are streamed. **Treat IMU + attitude as guaranteed; treat NED linear velocity as probably-available-but-verify against the real sim package; treat position as always self-estimated.**

**Evidence/Sources:** theaigrandprix.com FAQ (coordinates/SLAM questions); VADR-TS-002 §3.3, §4.3, §4.5; Elodin harness (telemetry contract caveats).

**Confidence: HIGH** that no position ground-truth is streamed and self-localization is required. **MEDIUM** on the exact velocity/RPM fields available (verify on first sim run).

---

## Q3 — Race-start vs. arming; telemetry reliability/rates; submission entrypoint; the autonomy rule

**(a) Start vs. arming ordering.** Arming sets the coordinate origin: VADR-TS-002 §3.8 — `MAV_FRAME_LOCAL_NED` origin (0,0,0) is *"a fixed physical point on the ground (usually where the drone armed)."* The example session (§6) is: init MAVSDK → connect to sim endpoint → sim emits `HEARTBEAT` → client streams control commands → telemetry/vision returned. **Well-grounded inference:** the *timed lap is gate-triggered, not arming-triggered.* Scoring is gate-based ("runs must successfully pass gates to count"; "once all gates are passed, the run time is locked") within an 8-minute cap (§8.3), and the course has an explicit **start gate**. So you can arm, take off, stabilize your estimator, then commit through the start gate — pre-start positioning appears "free" within the 8-minute window. Exact arming handshake is not pinned in the public spec (no `COMMAND_LONG`/arm message is listed; the Elodin rig auto-arms and takes off), so arming may be abstracted by the harness. *(Verify when the sim drops.)*

**(b) Telemetry reliability & rates.**
- **Transport is UDP** (§4.2, §4.6) → **best-effort, no delivery or ordering guarantee.** Build for dropped/reordered packets on *every* stream.
- **Video is chunked JPEG over UDP:5600** with a 24-byte little-endian header (`frame_id` u32, `chunk_id` u16, `total_chunks` u16, `jpeg_size` u32, `payload_size` u32, `sim_time_ns` u64). **You must reassemble frames and tolerate missing chunks** (drop incomplete frames rather than block). 30 Hz, 640×360.
- **Rates:** physics 120 Hz; **command rate < 100 Hz**; **heartbeat ≥ 2 Hz** (you must maintain it — a client responsibility); vision 30 Hz. **Per-message `ATTITUDE`/`HIGHRES_IMU` rates are NOT individually specified** — plan for "high-rate, ≤120 Hz, exact TBD," subscribe and measure on first run.
- **Time sync:** use `TIMESYNC` + the per-frame `sim_time_ns` to align vision↔IMU on **sim time, not wall-clock** — essential for any VIO/velocity integration.

**(c) Submission entrypoint contract.** The public spec does **not** define a formal submission entrypoint signature (no required `main()`/file layout). It specifies: runtime (Python 3.14.2 known-good, Windows 11, other environments allowed, C/Cython extensions expected to be permitted), the **interface** (MAVLink 2 via `c_library_v2` / MAVSDK-compatible over UDP; video on UDP:5600), and client responsibilities (establish MAVLink, maintain heartbeat, send control, process telemetry + vision). The control surface is **`SET_POSITION_TARGET_LOCAL_NED`** and/or **`SET_ATTITUDE_TARGET`** at < 100 Hz, riding on top of a **stabilized inner loop** (spec §5.3 "Pilot Commands → Stabilized Controller"; Elodin runs a real **Betaflight SITL** inner loop — strong signal the official sim does likewise). **The exact "upload your code" packaging/entrypoint is not yet public** — it arrives via the teams portal (`teams.theaigrandprix.com`) and email per the FAQ, and the official VQ1 simulator package was still pending as of the latest public writeups. *(This is the main open mechanics question; flag for the commander.)*

**(d) Prohibited human/sim interaction (the autonomy rule).** Unambiguous and strict:
- VADR-TS-002 §7 (Compliance): *"human interaction during the flight which the participants submit as a timed run is grounds for immediate disqualification."*
- FAQ: *"Using connectivity to pilot the drone would result in disqualification (the flight must be autonomous)."* (i.e., no teleoperation over the drone's Wi-Fi/Bluetooth.)
- Official Rules: no hardware modifications; tampering with / circumventing / disrupting the competition = DQ (Race Organizer's sole discretion); one team per person; **Russian citizens barred**; employees of Anduril/DCL/Neros ineligible. **FLOSS and generative-AI tools are allowed but must be disclosed in writing**; entry must be original and contain no malicious code (you retain ownership; partners get a judging-period license).
- **What's allowed:** full **offline** iteration between runs — export flight logs, retune, resubmit. Only *live, in-the-loop human input during a timed run* is prohibited. (This is exactly the MonoRace "offline optimization from onboard logs" workflow — see Q4.)

**Evidence/Sources:** VADR-TS-002 §3.8, §4.2–4.6, §5.1–5.3, §6, §7, §8.3; theaigrandprix.com FAQ; theaigrandprix.com Official Rules; Elodin harness/README (Betaflight inner loop; per-tick callback model).

**Confidence:** Autonomy rule & DQ conditions: **HIGH.** UDP/rates/video framing: **HIGH** (from spec). Gate-triggered timing inference: **MEDIUM-HIGH.** Submission-packaging entrypoint: **LOW** (not yet public).

---

## Q4 — Course/gate specs, scoring, prior editions, deadlines, known entrants/approaches

**Course & gate geometry (VADR-TS-002 §3.6–3.8):**
- **Drone chassis:** 280 × 280 × 160 mm.
- **Gate:** outer frame 2700 × 2700 × 260 mm; **inner opening 1500 × 1500 mm** (depth 260 mm). Generous clearance — a 0.28 m drone through a 1.5 m square. Round 1 gates are visually highlighted/desaturated background.
- **Camera:** monocular forward FPV, 640 × 360, pinhole no distortion, intrinsics `fx=fy=320`, `cx=320`, `cy=180`, **pitched +20° up**, 30 Hz. **Spec FoV bug to internalize:** it states "VFoV = 90°," but the intrinsics imply **HFoV = 90°, VFoV ≈ 58.7°** — independently confirmed by Elodin, who follow the intrinsics. The +20° up-tilt biases near-approach gates toward the lower image; tune detection/cropping for it.
- **Frames:** NED throughout; body X-fwd/Y-right/Z-down; body↔IMU = identity; camera shares the body origin. (Rotate into your CV library's convention.)
- **Physics:** rigid-body, thrust/drag/gravity/collision, 120 Hz; deterministic; **no wind** in the virtual qualifiers.

**Scoring (FAQ + spec §8):** Pure **time-trial**, fastest **valid** time advances; **all gates must be passed** for a run to count; collisions (collision physics) end the run; balance speed vs. missed-gate risk. Max run duration **8 minutes**; run time locks once all gates are cleared. Matches our "fastest valid lap, zero gate contact" framing.

**Two virtual rounds (FAQ):** Round 1 = intentionally simple, few gates, desaturated env, highlighted gates (Elodin models a 3-gate course). Round 2 = significantly harder, **real 3D-scanned environment**, more visually complex. Same drone/interface across both; complexity rises via realism. Round 2 cutoff ~end of July (approx).

**Timeline / deadlines:**
- Interface specs published 2nd half of March; **Technical Spec VADR-TS-002 dated 2026-05-08**; official **VQ1 simulator package targeted "May"** but was **running late** as of the latest public writeups (Elodin published their rig precisely because the official drop slipped). As of today (2026-06-14) we are **mid-virtual-window; the official sim is out or imminent — confirm via the portal/newsletter.**
- Virtual qualifier window ≈ **May–July 2026**; **Physical qualifier Sept 2026, Southern California** (two weeks, **indoor**, consistent lighting, obstacles/visual distractions); **Final: Nov 2026, Columbus OH** (indoor, possible spectator camera-flash disturbances).

**Prize/structure:** $500K pool split among top finishers at the Ohio final; **top scorer eligible for an Anduril job** (must be 18+, may require US security clearance/relocation; ineligible/declining members get **+$10K each**, up to 8); **top-10 at Ohio guaranteed ≥ $5K**. Teams of up to 8; no entry fee; **1,000+ teams within 24 h of launch**. University performers reaching the physical qualifier get in-person Anduril screening.

**Partners / why this exists:** Anduril (host; recruiting funnel tied to its Arsenal-1 Ohio facility), **DCL** (operator + simulator builder; its "AI vector module" is integrated into the platform), **Neros** (identical drone hardware — Archer-class US FPV), **JobsOhio** (economic-development backer).

**Prior editions & directly-applicable prior art:**
- **AlphaPilot / AIRR (Lockheed Martin + Drone Racing League, 2019)** — the canonical predecessor: fully autonomous, **no GPS/no data-relay/no human**, **NVIDIA Jetson AGX Xavier** onboard with stereo cameras; **TU Delft MAVLab won**. Establishes the onboard-Jetson lineage our Q1 inference rests on.
- **UZH "Swift" (Kaufmann/Scaramuzza et al., *Nature*, Aug 2023)** — deep-RL, **onboard vision + IMU**, first to beat human champions; **sim-trained then transferred** — methodologically the template for this competition.
- **A2RL — Abu Dhabi Autonomous Drone Racing (2025/2026)** — concurrent autonomous championship; AI reached/then beat top human pilots.
- **★ MonoRace (arXiv 2601.15222, de Croon / De Wagter / Ferede / Blaha et al., TU Delft MAVLab, 21 Jan 2026)** — **the most directly applicable reference for our exact setup.** A **monocular-camera + IMU** stack (no external motion capture) that **won A2RL 2025**, beating all AI teams and three human world champions at up to 100 km/h. Its recipe maps almost 1:1 onto AI-GP constraints: (1) **NN gate segmentation fused with a drone dynamics model** for state estimation (no GPS, no stereo); (2) **offline optimization that exploits the *known* gate geometry** to refine state-estimation and **camera-calibration** parameters **purely from onboard flight logs** — which AI-GP explicitly enables (rough gate positions given; logs exportable; offline iteration allowed); (3) a small guidance/control NN. Read this first.

**Known entrants/approaches:** No competitor stacks are publicly named (1,000+ teams; expect the usual academic powerhouses — TU Delft MAVLab, UZH RPG, ETH — to be in the mix, unconfirmed). The one concrete public technical artifact is **Elodin's open-source practice rig** (`github.com/elodin-sys/ai-grand-prix`; Elodin is a YC-backed aerospace-sim startup — a *tooling vendor*, not a competitor) — Elodin physics + **Betaflight SITL** + a spec-matched FPV camera, with a single `autopilot(update) -> RCCommand` solver you edit. Useful to start iterating before the official sim, but note its caveats: it speaks Betaflight UDP not MAVLink (needs a shim), exposes ENU not NED, and exposes a convenience world-pose the official sim won't. **Community/forum footprint is otherwise thin** — the Hacker News thread is dead (2 points, no comments); no substantive Reddit/Discord competitor discussion surfaced.

**Confidence:** Geometry/scoring/rounds/timeline/partners/prize: **HIGH** (official spec + FAQ + rules). Prior-art relevance: **HIGH.** Specific named entrants: **LOW** (not public).

---

## Sources

**Official / primary**
- AI Grand Prix Technical Specification, VADR-TS-002 Issue 00.02, 2026-05-08 (the uploaded PDF; publicly hosted): https://www.theaigrandprix.com/wp-content/uploads/2026/05/260508_Technical_Spec_0002.pdf
- AI Grand Prix — official site & FAQ: https://www.theaigrandprix.com/
- AI Grand Prix — Official Rules: https://www.theaigrandprix.com/official-rules
- Anduril announcement: https://www.anduril.com/news/anduril-launches-the-ai-grand-prix-a-global-autonomous-drone-race
- Drone Champions League announcement: https://dronechampionsleague.com/theaigrandprix2026-announcement/

**Third-party technical**
- Elodin — open-source AI Grand Prix practice rig (write-up): https://www.elodin.systems/post/elodin-ai-grand-prix-race-sim-harness
- Elodin — repo: https://github.com/elodin-sys/ai-grand-prix

**Compute / hardware**
- NVIDIA Jetson Orin NX 16 GB = 100 TOPS (Notebookcheck): https://www.notebookcheck.net/NVIDIA-Jetson-Orin-NX-16-GB-Module-launches-worldwide-with-100-TOPS-AI-performance.687003.0.html
- NVIDIA Jetson AGX Orin = 275 TOPS (Hackster): https://www.hackster.io/news/nvidia-launches-275-tops-jetson-agx-orin-developer-s-kit-at-1-999-bbb5ff80e050
- Neousys FLYC-300 (Orin NX drone mission computer, industry reference): https://www.neousys-tech.com/en/product/product-lines/intelligent-video-analytics/flyc-300
- Neros Technologies (drone hardware): https://www.neros.tech/

**Prior editions / prior art**
- AlphaPilot: Autonomous Drone Racing (RSS paper): https://www.roboticsproceedings.org/rss16/p081.pdf
- DRL RacerAI / AIRR (Jetson AGX Xavier onboard): https://www.prnewswire.com/news-releases/drone-racing-league-launches-drl-racerai-the-first-ever-autonomous-racing-drone-300933437.html
- UZH Swift — Champion-level drone racing using deep RL (*Nature*, 2023): https://www.nature.com/articles/s41586-023-06419-4
- MonoRace — Robust Monocular AI, won A2RL 2025 (arXiv, Jan 2026): https://arxiv.org/abs/2601.15222
- A2RL human-vs-drone championship (DroneDJ): https://dronedj.com/2026/01/27/a2rl-human-vs-drone-championship/

**Press / context**
- DRONELIFE: https://dronelife.com/2026/01/28/anduril-launches-500k-drone-racing-ai-grand-prix/
- TechCrunch: https://techcrunch.com/2026/01/27/anduril-has-invented-a-wild-new-drone-flying-contest-where-jobs-are-the-prize/
- Built In: https://builtin.com/articles/anduril-ai-grand-prix-2026
- Ohio Tech News: https://www.ohiotechnews.com/ai-grand-prix-anduril-ohio-fall-2026/

---

## MEMORY-DELTA (bankable, ≤10 lines)

1. AI Grand Prix is real & public: Anduril + DCL + Neros + JobsOhio; $500K + Anduril job; 1,000+ teams; phases = virtual qualifier → physical qualifier (Sept 2026, SoCal, indoor) → final (Nov 2026, Columbus OH).
2. Our "internal spec" = the OFFICIAL public Technical Spec **VADR-TS-002** (hosted on theaigrandprix.com); all competitors have it.
3. **Q1 RESOLVED — latency axis CLOSES.** Virtual-qualifier eval = Windows 11 desktop + discrete ~8 GB-VRAM GPU (CUDA-class), NOT CPU-only and NOT a Jetson. Our ~15–25 ms GPU path ≪ 50 ms; ~115 ms CPU path is never forced.
4. "~100 TOPS" is OFFICIAL but describes the **physical Neros drone's** onboard AI module (FAQ), most likely **Jetson Orin NX-class (~100 TOPS sparse INT8 ≈ ~50 dense)** — design target for sim-to-real, not the qualifier eval box.
5. **Q2 RESOLVED — no pose ground-truth streamed** (only maybe a start coordinate). Get attitude + IMU (+ likely NED velocity); position must be self-estimated from vision (gate PnP via known 1.5 m gate) → **onboard estimator is load-bearing.**
6. Interface = MAVLink2/MAVSDK over **UDP (best-effort)** + chunked-JPEG video on UDP:5600 (reassemble, tolerate drops); rates: physics 120 Hz, commands <100 Hz, heartbeat ≥2 Hz, vision 30 Hz; align on `sim_time_ns`/TIMESYNC; control via SET_POSITION_TARGET_LOCAL_NED / SET_ATTITUDE_TARGET onto a Betaflight-style inner loop.
7. **Autonomy rule (strict):** any human interaction during a timed run = immediate DQ; no teleop over drone connectivity; no hardware mods. Offline log-based retuning between runs IS allowed. FLOSS + genAI allowed but must be disclosed in writing.
8. Scoring = time-trial, fastest VALID time, ALL gates required, collisions end run, 8-min cap; Round 1 simple/highlighted gates, Round 2 3D-scanned/harder; no wind in sim.
9. **Spec camera-FoV bug:** stated "VFoV=90°" is wrong; intrinsics give **HFoV=90°, VFoV≈58.7°**; camera pitched **+20° up** (Elodin-confirmed). Use intrinsics.
10. **Read MonoRace (arXiv 2601.15222, TU Delft MAVLab, won A2RL 2025):** monocular+IMU, NN gate-seg + drone-model state estimation, offline calibration from onboard logs using known gate geometry — near-1:1 to our constraints. Open gaps to verify when official sim drops: exact submission/entrypoint packaging; GPU/VRAM share vs. renderer; precise per-message IMU/velocity rates & arming handshake.
