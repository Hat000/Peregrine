# AI Grand Prix — autonomy stack

Anduril AI Grand Prix entry. Autonomous navigation of a simulated drone
through a sequence of gates over a MAVLink + UDP-JPEG interface.

## Spec
`260508_Technical_Spec_0002.pdf` — VADR-TS-002 issue 00.02, 2026-05-08.

## Layout
- `src/racer/` — autonomy stack (perception, control, frames, MAVLink client)
- `scripts/` — smoke tests and utilities
- `tests/` — unit tests
- `data/` — recorded runs (gitignored)

## Setup (Python 3.13)

The autonomy/ML stack runs on **Python 3.13**, not the spec's 3.14: stable
PyTorch ships CUDA wheels for 3.13 (`cp313`), while 3.14 is nightly-only until
PyTorch 2.10. The spec's "3.14.2" applies to the comms layer only and explicitly
allows other environments. Use the same 3.13 on the Azure VM and Adroit.

```powershell
# Install Python 3.13 if needed:  winget install Python.Python.3.13
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest   # expect 5 passed (tests/test_frames.py)
```

## Smoke tests
With the DCL simulator running:

```powershell
python scripts\smoke_mavlink.py udp:127.0.0.1:14550
python scripts\smoke_video.py 30
```

## Pipeline (intended)

```
JPEG-UDP rx ─► gate detector ─► solvePnP ─► gate pose ─┐
                                                       ▼
MAVLink in ──► drone state ─────────► state machine ──► controller ─► MAVLink out
```

VQ1 uses a color/contour gate detector + position-target controller.
VQ2 swaps in a learned detector and (optionally) attitude-target
control without changing the rest of the stack.

## Open questions to confirm on first connection
- Spec lists "linear velocities" in telemetry (4.5) but no velocity-bearing
  message in the supported table (4.3). Which message actually carries it?
- Exact MAVLink host:port — not in spec, presumably documented in DCL portal.
- Whether `LOCAL_POSITION_NED` is emitted even though it isn't listed.

## Notes
- Camera is tilted **+20° upward** about body Y (spec sec 3.8, added in
  issue 00.02 on 2026-05-08). All transforms live in `racer.frames`.
- Coordinate convention is NED (Z-down); intrinsic 3-2-1 (yaw, pitch, roll)
  Euler for body↔world.
