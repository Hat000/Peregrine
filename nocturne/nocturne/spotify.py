"""Spotify as the thinking indicator.

WORKING resumes playback (fade in); SPEAKING pauses or ducks it (fade out).
Because music plays for exactly as long as real work takes, it is a genuine
"still thinking" signal during long agentic runs.

Playback control needs Spotify Premium and an active device. We handle 403
PREMIUM_REQUIRED and "no active device" gracefully: on any failure the
controller degrades to the local ambient pad (via ``AudioEngine``) so the app
still has a thinking indicator without Spotify. Current playback is snapshotted
on start and restored on exit so we don't leave a random playlist running.

The Web API has no fade primitive, so fades are emulated by ramping the device
volume over a few steps.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

from .config import SpotifyConfig

_SCOPE = "user-modify-playback-state user-read-playback-state"


@dataclass
class _Snapshot:
    was_playing: bool = False
    context_uri: str | None = None
    track_uri: str | None = None
    progress_ms: int | None = None
    device_id: str | None = None
    volume: int | None = None


class SpotifyController:
    def __init__(self, cfg: SpotifyConfig, ambient=None) -> None:
        self.cfg = cfg
        self._ambient = ambient
        self._sp = None
        self._ok = False           # Spotify usable for playback control
        self._degraded = False     # fell back to ambient
        self._snap = _Snapshot()
        self._device_id = cfg.device_id or None
        self._base_volume = 100

    # ------------------------------------------------------------------ #
    async def connect(self) -> bool:
        if not self.cfg.enabled:
            self._use_ambient()
            return False
        try:
            self._sp = await asyncio.to_thread(self._build_client)
            await asyncio.to_thread(self._snapshot)
            self._ok = True
            return True
        except Exception as e:  # missing creds, no network, auth declined, ...
            print(f"[nocturne] Spotify unavailable ({_short(e)}); using local ambient.")
            self._use_ambient()
            return False

    def _build_client(self):
        import spotipy
        from spotipy.oauth2 import SpotifyOAuth

        cid = os.environ.get(self.cfg.client_id_env)
        secret = os.environ.get(self.cfg.client_secret_env)
        if not cid or not secret:
            raise RuntimeError("missing SPOTIFY client id/secret env vars")
        auth = SpotifyOAuth(
            client_id=cid, client_secret=secret,
            redirect_uri=self.cfg.redirect_uri, scope=_SCOPE,
            open_browser=False,
            cache_path=os.path.expanduser("~/.config/nocturne/.spotify-cache"),
        )
        return spotipy.Spotify(auth_manager=auth, requests_timeout=10)

    def _snapshot(self) -> None:
        pb = self._sp.current_playback()
        if not pb:
            return
        dev = pb.get("device") or {}
        self._snap = _Snapshot(
            was_playing=bool(pb.get("is_playing")),
            context_uri=(pb.get("context") or {}).get("uri"),
            track_uri=(pb.get("item") or {}).get("uri"),
            progress_ms=pb.get("progress_ms"),
            device_id=dev.get("id"),
            volume=dev.get("volume_percent"),
        )
        if not self._device_id:
            self._device_id = dev.get("id")
        if self._snap.volume is not None:
            self._base_volume = self._snap.volume

    # ------------------------------------------------------------------ #
    # State transitions
    # ------------------------------------------------------------------ #
    async def on_working(self) -> None:
        """Music on."""
        if self._degraded or not self._ok:
            if self._ambient:
                self._ambient.ambient_start()
            return
        try:
            await asyncio.to_thread(self._start_playback)
        except Exception as e:
            self._downgrade(e)

    async def on_speaking(self) -> None:
        """Music out of the way of the voice."""
        if self._degraded or not self._ok:
            if self._ambient:
                (self._ambient.ambient_duck(self.cfg.duck_volume / 100)
                 if self.cfg.duck_instead_of_pause else self._ambient.ambient_stop())
            return
        try:
            if self.cfg.duck_instead_of_pause:
                await asyncio.to_thread(self._set_volume, self.cfg.duck_volume)
            else:
                await asyncio.to_thread(self._fade_and_pause)
        except Exception as e:
            self._downgrade(e)

    async def on_idle(self) -> None:
        """Silence while waiting on the user."""
        await self.on_speaking()

    # ------------------------------------------------------------------ #
    def _start_playback(self) -> None:
        dev = self._device_id
        if self.cfg.duck_instead_of_pause:
            self._sp.volume(self._base_volume, device_id=dev)
            self._sp.start_playback(device_id=dev)
            return
        # start (optionally a specific thinking playlist), then fade volume up
        self._set_volume(max(1, self.cfg.duck_volume))
        if self.cfg.thinking_context_uri:
            self._sp.start_playback(device_id=dev, context_uri=self.cfg.thinking_context_uri)
        else:
            self._sp.start_playback(device_id=dev)
        self._fade_volume(self._base_volume)

    def _fade_and_pause(self) -> None:
        self._fade_volume(max(0, self.cfg.duck_volume))
        self._sp.pause_playback(device_id=self._device_id)
        # restore the volume level so a later resume isn't silent
        self._set_volume(self._base_volume)

    def _fade_volume(self, target: int, steps: int = 5) -> None:
        try:
            cur = self._sp.current_playback()
            start = ((cur or {}).get("device") or {}).get("volume_percent", self._base_volume)
        except Exception:
            start = self._base_volume
        for i in range(1, steps + 1):
            v = int(start + (target - start) * i / steps)
            self._set_volume(v)

    def _set_volume(self, v: int) -> None:
        self._sp.volume(max(0, min(100, int(v))), device_id=self._device_id)

    # ------------------------------------------------------------------ #
    def _downgrade(self, err: Exception) -> None:
        if self._degraded:
            return
        self._degraded = True
        reason = _classify(err)
        print(f"[nocturne] Spotify control failed ({reason}); switching to local ambient.")
        self._use_ambient()
        if self._ambient:
            self._ambient.ambient_start()

    def _use_ambient(self) -> None:
        self._degraded = True

    async def restore(self) -> None:
        if self._ambient:
            self._ambient.ambient_stop()
        if not self._ok or not self.cfg.restore_on_exit:
            return
        try:
            await asyncio.to_thread(self._restore_blocking)
        except Exception:
            pass

    def _restore_blocking(self) -> None:
        snap = self._snap
        if snap.volume is not None:
            try:
                self._set_volume(snap.volume)
            except Exception:
                pass
        try:
            if snap.was_playing and snap.context_uri:
                self._sp.start_playback(
                    device_id=snap.device_id or self._device_id,
                    context_uri=snap.context_uri,
                    offset={"uri": snap.track_uri} if snap.track_uri else None,
                    position_ms=snap.progress_ms or 0,
                )
            elif not snap.was_playing:
                self._sp.pause_playback(device_id=snap.device_id or self._device_id)
        except Exception:
            pass


def _classify(err: Exception) -> str:
    s = str(err).lower()
    if "premium" in s or "403" in s:
        return "Premium required"
    if "no active device" in s or "404" in s or "device" in s:
        return "no active device"
    return _short(err)


def _short(err: Exception) -> str:
    s = str(err)
    return (s[:80] + "…") if len(s) > 80 else s or type(err).__name__
