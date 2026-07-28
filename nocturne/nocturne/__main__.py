"""``python -m nocturne`` — start the screenless shell."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from .config import Config
from .tts import available_backends


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="nocturne",
        description="A screenless, listen-and-type shell for Claude Code.",
    )
    p.add_argument("--config", help="path to nocturne.toml")
    p.add_argument("--voice", help="TTS backend: sapi | piper | elevenlabs | claude | auto")
    p.add_argument("--mode", choices=["bedtime", "focused"], help="starting mode")
    p.add_argument("--cwd", help="working directory for the Claude Code session")
    p.add_argument("--resume", metavar="SESSION_ID", help="resume a session by id")
    p.add_argument("--continue", dest="cont", action="store_true",
                   help="continue the most recent session in the cwd")
    p.add_argument("--list-voices", action="store_true",
                   help="print available TTS backends and exit")
    p.add_argument("--no-spotify", action="store_true", help="disable Spotify")
    p.add_argument("--driver", choices=["cli", "sdk"],
                   help="cli = your Claude subscription (default); sdk = API key/credits")
    p.add_argument("--permission-mode", dest="permission_mode",
                   choices=["default", "acceptEdits", "plan", "bypassPermissions"],
                   help="how tool permissions are handled (headless)")
    p.add_argument("--debug", action="store_true",
                   help="echo the raw CLI stream, stderr, and events to the terminal")
    return p.parse_args(argv)


def _apply_overrides(cfg: Config, args: argparse.Namespace) -> None:
    if args.voice:
        cfg.tts.backend = args.voice
    if args.mode:
        cfg.bedtime.mode = args.mode
    if args.cwd:
        cfg.agent.cwd = args.cwd
    if args.resume:
        cfg.session.session_id = args.resume
    if args.cont:
        cfg.session.resume_last = True
    if args.no_spotify:
        cfg.spotify.enabled = False
    if args.driver:
        cfg.agent.driver = args.driver
    if args.permission_mode:
        cfg.agent.permission_mode = args.permission_mode


def _preflight(cfg: Config) -> None:
    if cfg.agent.driver == "cli":
        from .claude_cli import locate_cli
        path = locate_cli(cfg.agent.claude_cli_path)
        if not path:
            print("[nocturne] Claude Code CLI not found. Install it or set "
                  "agent.claude_cli_path in nocturne.toml.", file=sys.stderr)
        else:
            print(f"[nocturne] driving Claude Code on your subscription: {path}")
            print("[nocturne] (if it says 'not logged in', run `claude` once and "
                  "sign in — no API key needed.)")
    else:  # sdk driver = API credits
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("[nocturne] warning: sdk driver needs ANTHROPIC_API_KEY "
                  "(billed as API credits).", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if args.debug:
        os.environ["NOCTURNE_DEBUG"] = "1"
    cfg = Config.load(args.config)
    _apply_overrides(cfg, args)

    if args.list_voices:
        avail = available_backends(cfg.tts)
        print("Available TTS backends:", ", ".join(avail) or "(none — no-op fallback only)")
        return 0

    _preflight(cfg)

    # Import late so --list-voices works without the SDK / audio deps.
    from .app import NocturneApp

    app = NocturneApp(cfg)
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        print("\n[nocturne] interrupted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
