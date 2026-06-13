"""rl/submit_rl.py — the COMMITTED, submission-safe entrypoint for the RL race entry.

This is the launch path to point the organizer's harness at (or to invoke verbatim on the
judged host). It PINS the autonomy-hardened flags so a run can NEVER:
  * emit a sim-control command (MAV_CMD 31000) on the judged wire   -> §7 DQ      [R1]   (--no-auto-reset)
  * load the gitignored bridge map absent from a clean checkout     -> crash-armed [R2]   (--no-bridge)
  * ship the retired aero-blind inc4 actor                          -> slow/invalid       (--checkpoint inc7)
  * block the control loop on per-tick debug file I/O                                      (--no-debug-obs)
and it fixes --flights 1 (the submission shape).

fly_rl.py's DEFAULTS are now these same safe values, so a bare `python rl/fly_rl.py` is also
safe — but this wrapper pins them EXPLICITLY so a future defaults drift, or a stray forwarded
flag, cannot silently regress the submission. The safety flags are appended LAST, so they win
over anything the caller forwards (argparse: last occurrence wins; --no-auto-reset is also a
hard override of --dev-auto-reset). Non-safety flags (--endpoint, --label, --max-seconds, ...)
are forwarded verbatim.

    python rl/submit_rl.py                 # judged run: standing start, passive wait for GO
    python rl/submit_rl.py --label myrun   # forward extra (non-safety) flags

For a DEV batch that needs the bridge / auto-reset / multi-flight / forensics, call
``rl/fly_rl.py`` directly with the dev flags — NOT this wrapper.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "src"))
sys.path.insert(0, str(_HERE.parent / "scripts"))

import fly_rl  # noqa: E402

# Absolute path so the checkpoint resolves regardless of the caller's CWD.
_CKPT = _HERE / "checkpoints" / "stage1_inc7_actor.pth"

# Authoritative safety pins (appended LAST -> win over any forwarded flag).
SAFE_PINS = [
    "--checkpoint", str(_CKPT),
    "--no-bridge",        # standing start; map-free (R2)
    "--no-auto-reset",    # never emit a sim-control command on the judged wire (R1 DQ)
    "--no-debug-obs",     # no per-tick blocking file I/O
    "--flights", "1",     # the submission shape
]


def build_argv(extra: list[str]) -> list[str]:
    """Forwarded flags first, safety pins last (so the pins are authoritative)."""
    return [*extra, *SAFE_PINS]


def main() -> int:
    if not _CKPT.exists():
        print(f"FATAL: submission checkpoint missing: {_CKPT}", file=sys.stderr)
        return 2
    sys.argv = ["fly_rl.py", *build_argv(sys.argv[1:])]
    print(f"[submit_rl] launching submission-safe entry: {' '.join(sys.argv[1:])}")
    return fly_rl.main()


if __name__ == "__main__":
    raise SystemExit(main())
