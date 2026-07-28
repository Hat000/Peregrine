# AI Grand Prix — autonomy stack

> ## 🏁 This project is CLOSED (2026-07-28)
>
> The competition ended without reaching the 20-gate goal. The best flight of 699 recorded
> reached **gate 9**, and there was a hard wall at gate 6.
>
> **→ Start with [`POSTMORTEM.md`](POSTMORTEM.md).** It carries the honest result, everything
> that was established, everything that was refuted (with the numbers), and — most usefully —
> the thirteen ways the instruments produced confident, reproducible, wrong answers.
>
> Every number in it recomputes from this repo: `python3 scripts/adjudicate/corpus_summary.py`
>
> All historical branches are preserved as `archive/*` tags rather than branches
> (`git tag -l 'archive/*'`, then `git switch -c <name> archive/<name>` to reopen one).
> Nothing was deleted.

Anduril AI Grand Prix entry. Autonomous navigation of a simulated drone
through a sequence of gates over a MAVLink + UDP-JPEG interface.

## Spec
`260508_Technical_Spec_0002.pdf` — VADR-TS-002 issue 00.02, 2026-05-08.

## Layout
- `src/racer/` — autonomy stack (perception, control, frames, MAVLink client)
- `scripts/` — smoke tests, utilities, and the cohort-adjudication tooling
  (`adjudicate/`, `vision_horizon/`, `postimpact/` — all accept `PEREGRINE_RUNS=<path>`)
- `tests/` — unit tests (2077 passing)
- `data/runs/` — **the flight corpus: 729 run directories, 699 recorded flights, committed.**
  Every claim in `POSTMORTEM.md` is recomputable from it.
- `handoff/` — the three final investigation reports (geometric detectability, the two
  obstacles, and the speed refutation)

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
pytest   # expect 125 passed
```

## Smoke tests
With the DCL simulator running:

```powershell
python scripts\smoke_mavlink.py udp:127.0.0.1:14550
python scripts\smoke_video.py 30
```

## Recording (run for every sim contact)
The course is deterministic, so one good recording replays into many offline iterations
(system-ID, mapping, racing-line fitting, detector auto-labels). Start the sim first, then:

```powershell
python scripts\record_session.py --label first_contact      # Ctrl-C to stop
python scripts\record_session.py --label probe --seconds 60 # fixed duration
```

Writes `data/runs/<stamp>_<label>/`: `mavlink.tlog` (standard pymavlink tlog — readable by
MAVExplorer/QGC; the sim clock rides inside the frames), `video.bin` + `video_index.jsonl`
(bit-exact JPEGs, seekable), and `meta.json` (clock bridge + stats). On exit it prints the
MAVLink message-type histogram and which hedge fields (position/velocity/mag/baro) appeared
— a built-in first-contact `msg_audit` that resolves risk R1. Replay in code via
`racer.recording.RecordingReader` (`.iter_mavlink()`, `.frames()`, `.iter_jpeg()`).

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
