---
name: project-ai-grand-prix
description: "User is competing in Anduril's AI Grand Prix autonomous drone racing competition; long-running engagement spanning May–November 2026."
metadata: 
  node_type: memory
  type: project
  originSessionId: 381822b6-b523-4990-a2cf-3739f8593be8
---

User is a competitor in the AI Grand Prix, an autonomous drone racing competition hosted by Anduril Industries.

> **⚠️ STALE STRATEGY SECTIONS (flagged 2026-05-28):** [[project-master-plan]] is the single source of truth and SUPERSEDES the "Strategic angle" and parts of "Technical envelope" below. Specifically: velocity is NOT confirmed in telemetry (derive it); **VFoV=90° is mislabeled — it's HFoV; VFoV≈58.7°**; the old "hybrid CasADi-prior + RL-refines" strategy is replaced by Track-A model-based (RL parked, but RL-on-surrogate is now the recommended speed-UPGRADE given the user's strength — see master-plan Review-validation #12); "VQ2 needs VIO" → a lightweight linear pos/vel KF suffices (attitude given). The competition-fact lines below remain valid.

**Why:** User explicitly asked me to be present "for the entirety of the competition." Stakes are explicitly low — "fun project to see how far I can go" — but they want to try their best. Not aiming for a top finish; perfection is not the goal.

**How to apply:** Treat this as a months-long collaboration. Default to pragmatic, ship-something-that-works choices over theoretically optimal ones. Be a thinking partner — bring "many bright ideas" but don't over-engineer. Always keep the relative cost/benefit of VQ1 (completion-only) vs VQ2 (timed) in mind when proposing work.

**Timeline (absolute dates):**
- Virtual Qualifier 1 (VQ1) — opens May 2026, currently OPEN as of 2026-05-27
- Virtual Qualifier 2 (VQ2) — opens June 2026, closes mid-to-late July 2026
- Physical Qualifier — September 2026 (California, real drones)
- Grand Prix Final — November 2026 (Ohio, real drones, audience)

**Technical envelope (from spec VADR-TS-002, issue 00.02, 2026-05-08):**
- Race interface: MAVLink2 over UDP, MAVSDK-compatible
- Inputs: 640×360 FPV at 30 Hz (JPEG over UDP:5600, chunked), ATTITUDE + HIGHRES_IMU + velocity, HEARTBEAT, TIMESYNC. NED frame.
- Outputs: SET_POSITION_TARGET_LOCAL_NED or SET_ATTITUDE_TARGET (no direct rotor commands)
- Physics 120 Hz, command rate <100 Hz, 8-min max run
- Camera intrinsics: pinhole, 640×360, cx=320, cy=180, fx=fy=320, VFoV=90°, **tilted 20° upward** (recent change in 00.02)
- Gate inner square: 1500 mm × 1500 mm (PnP target); outer 2700 mm; depth 260 mm
- Drone chassis: 280×280×160 mm
- No GPS, no depth, no battery state, no RPM
- Python 3.14.2+, Windows only, RTX 2060 Super / 16 GB RAM / 8 GB VRAM minimum

**Stage difficulty gap:**
- VQ1: <10 gates, "highlighted," high SNR, guidance aids ON, judged on completion alone. Crude stack passes.
- VQ2: 10–20 gates, complex lighting, guidance aids OFF, elevation changes, judged on fastest time.

See [[reference-competition-materials]] for source documents.

**User skill profile (corrected 2026-05-28):** user is **RL-strong**, less comfortable with classical control. This reweights recommendations toward learning-based approaches; don't assume classical-control fluency. (Earlier they self-described "robotics/control"; the later, more specific signal is RL > classical.)

**Strategic angle (evolved through research phase):**
- Perception: monocular PnP using known gate geometry (1500 mm inner square) + given intrinsics — one detected gate yields full 6-DOF pose. VQ1 = classical OpenCV color/contour detector. VQ2 = **YOLOv8/11-pose** trained to output the 4 gate corners as keypoints → solvePnP (YOLO's AGPL is NOT a practical blocker for a private non-distributed competition entry; YOLO-pose is light/fast/well-tooled and accepted).
- Course is deterministic but NO map is given → EXPLORE mode builds an ordered gate map; data association by position+sequence (not appearance, since gates look identical). VQ2 needs lightweight VIO (IMU integration + gate-pose correction) for "which gate is next."
- Control/planning: **hybrid** — CasADi time-optimal trajectory as the prior (optimal for the *modeled* dynamics), then **RL refines it** (residual/warm-started) to capture *true* dynamics. Don't throw away the spline; warm-start RL with it.
- RL training on Adroit: build a surrogate sim via **system identification** (chirp/PRBS/doublet excitation; note we identify the sim+its onboard controller since we command setpoints) + **domain randomization** to close the transfer gap. RL infra: lean JAX route (MuJoCo MJX / Brax) since our surrogate dynamics are simple & self-identified; Isaac Lab as batteries-included fallback. Both run well on Adroit (NVIDIA/Linux).
- VQ1 floor needs almost none of this: reactive fly-at-most-centered-gate + position-target control passes completion-only judging.

See [[reference-prior-art]] for the surveyed projects/libraries and license verdicts.
