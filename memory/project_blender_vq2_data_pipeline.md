---
name: project-blender-vq2-data-pipeline
description: Build a Blender-based vision training-data pipeline matched to the known VQ2 appearance (glowing-red gates, dark warehouse), replacing broad domain randomization with high-fidelity appearance matching.
metadata:
  type: project
---

Fengyou 2026-06-29 directive: build a vision training-data pipeline in BLENDER matched to the now-known VQ2 appearance.

**KEY PRINCIPLE:** the sim is DETERMINISTIC and the look is FIXED — do NOT domain-randomize broadly; MATCH ONE appearance with HIGH FIDELITY.

**Ground-truth reference:** recon frames at `handoff/vq2-recon-2026-06-29` (origin branch `vq2-recon-2026-06-29`). Appearance profile:
- Glowing-red emissive gates
- Scene mean gray ~36/255 (very dark warehouse)
- Blue floor lane-lines
- Regular floor grid + ceiling truss
- Bloom profile: saturated core R≥250, ~3px crisp inner / ~12px outer halo

**Pipeline:**
1. Parametric Blender replica of the warehouse + gate geometry
2. Auto-labeled render harness: sample camera pose → render → project known 1.5 m gate geometry to pixel-accurate corner/keypoint labels
3. Intrinsics: fx=fy=320, cx=320, cy=180, 640×360 resolution, mount +20°
4. Output: labeled dataset for learned detector training

**PURPOSE:** train a LEARNED detector to beat the classical red-glow detector's hard limits (far-gate >~15 m detection floor, oblique angles, motion blur). The classical detector (`src/racer/vision/red_glow_detector.py`) stays the reliable baseline; learned extends the envelope.

**Tooling:** Blender driven via the Blender MCP (needs Blender running + addon server at localhost:9876). Use `mcp__Blender__execute_blender_code` for scene construction and rendering.

Links [[index-vision-estimator]].
