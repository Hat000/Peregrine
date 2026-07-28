"""Away notifications.

If Nocturne needs your input (a confirmation gate) and you've drifted off, it
can send a phone push instead of talking into an empty room. Supports ntfy
(just a topic) and Pushover (user + token via env). All best-effort and
non-blocking; failures are swallowed.
"""

from __future__ import annotations

import asyncio
import os
import urllib.parse
import urllib.request

from .config import SessionConfig


class AwayNotifier:
    def __init__(self, cfg: SessionConfig) -> None:
        self.cfg = cfg
        self.kind = (cfg.away_notifier or "none").lower()

    @property
    def enabled(self) -> bool:
        return self.kind in ("ntfy", "pushover")

    async def ping(self, message: str, title: str = "Nocturne needs you") -> None:
        if not self.enabled:
            return
        try:
            await asyncio.to_thread(self._send, message, title)
        except Exception:
            pass

    def _send(self, message: str, title: str) -> None:
        if self.kind == "ntfy":
            if not self.cfg.ntfy_topic:
                return
            url = f"{self.cfg.ntfy_server.rstrip('/')}/{self.cfg.ntfy_topic}"
            req = urllib.request.Request(
                url, data=message.encode("utf-8"), method="POST",
                headers={"Title": title, "Priority": "high"},
            )
            urllib.request.urlopen(req, timeout=10).close()
        elif self.kind == "pushover":
            token = os.environ.get("PUSHOVER_TOKEN")
            user = os.environ.get("PUSHOVER_USER")
            if not (token and user):
                return
            data = urllib.parse.urlencode({
                "token": token, "user": user, "message": message,
                "title": title, "priority": 1,
            }).encode("utf-8")
            req = urllib.request.Request(
                "https://api.pushover.net/1/messages.json", data=data, method="POST")
            urllib.request.urlopen(req, timeout=10).close()
