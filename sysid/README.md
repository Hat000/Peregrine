# Plant sysID handoff (RL-commander → ShadowPC), 2026-07-15

Purpose: validate/tune the DiffAero training plant against the real VQ2 sim by replaying **one identical open-loop command program** through both and overlaying the drone's response. The RL-commander already captured the DiffAero side; this branch delivers the program so ShadowPC can capture the VQ2 side.

Files:
- `sysid_program.csv` — **the exact program to replay** (1092 rows @ 40 Hz, ~27.3 s). Columns `t,a_thrust,a_roll,a_pitch,a_yaw` = the RAW policy action vector fed each tick (NOT wire units). `a_thrust∈[-1,1]` (hover trim ≈ −0.6), `a_roll/pitch/yaw∈[-1,1]`. Replay **byte-identical** — do not regenerate or resample.
- `sysid_program_segments.json` — the excitation map (per-axis step / doublet / chirp on roll→pitch→yaw, then thrust steps, with hover-trim rests between). Read it to know which rows are which maneuver and to place safety margin.
- `sysid_program.py` — the generator, for provenance only (regenerate ONLY from this if you must; must be byte-identical to the CSV above).

The capture spec (what to build/log/output, footguns, deliverable) is in the RL-commander's handoff prompt delivered via Fengyou. Deliverable back: `sysid_vq2_log.csv` (+ raw 117 Hz IMU dump), committed here or relayed.
