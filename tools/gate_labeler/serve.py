"""VQ2 gate hand-labeling server -- STDLIB ONLY (http.server + json), no pip installs.

Run:
    <venv>\\Scripts\\python.exe tools/gate_labeler/serve.py --frames <img_dir> --labels <out_dir> [--port 8000]

Serves the single-page UI (index.html, same directory) at http://localhost:<port>, lists the
.png/.jpg frames in --frames, and on save writes a YOLO-pose .txt (same basename) into
--labels via labelio.encode_label (the pure, unit-testable encoder). Existing labels are
loaded back onto the canvas for resume-editing.

API:
    GET  /                    -> index.html
    GET  /api/frames          -> {"frames":[{"name","labeled"}], "labels_dir"}
    GET  /api/label?name=F    -> {"exists": bool, "gates": [...]}          (decoded to pixels)
    GET  /frames/F            -> image bytes
    POST /api/save            -> body {"name", "img_w", "img_h", "gates":[{inner,outer,occluded}]}
                                 gates=[] writes an EMPTY file (intentional negative frame).
"""
from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

TOOL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOL_DIR))
import labelio  # noqa: E402  (pure encoder; same directory)

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}

FRAMES_DIR: Path = None  # set in main()
LABELS_DIR: Path = None
SEEDS_DIR: Path = None   # optional detector pre-labels (scripts/seed_labels_with_m.py)


def list_frames():
    return sorted(p.name for p in FRAMES_DIR.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def safe_frame(name: str) -> Path:
    """Reject path tricks: the name must be a plain basename of an existing frame."""
    if not name or Path(name).name != name or name.startswith("."):
        raise ValueError(f"bad frame name: {name!r}")
    p = FRAMES_DIR / name
    if not (p.is_file() and p.suffix.lower() in IMAGE_EXTS):
        raise ValueError(f"unknown frame: {name!r}")
    return p


def label_path(name: str) -> Path:
    return LABELS_DIR / (Path(name).stem + ".txt")


def seed_path(name: str) -> Path | None:
    """Detector pre-label for this frame, if seeding is enabled and one exists.

    SAFETY: a seed is NEVER a label. It is served only when no human label exists, is flagged
    ``seed: true``, and leaves the frame counted as UNLABELED -- so unverified machine output
    cannot masquerade as a hand label or silently enter the training set.
    """
    if SEEDS_DIR is None:
        return None
    p = SEEDS_DIR / (Path(name).stem + ".txt")
    return p if p.is_file() else None


class Handler(BaseHTTPRequestHandler):
    # -- plumbing ------------------------------------------------------------------
    def _send(self, code, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode("utf-8"))

    def _err(self, msg, code=400):
        self._json({"error": str(msg)}, code)

    def log_message(self, fmt, *args):  # quieter: only non-GET or errors
        if not (args and str(args[1]).startswith("2") and self.command == "GET"):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # -- routes --------------------------------------------------------------------
    def do_GET(self):
        url = urlparse(self.path)
        try:
            if url.path in ("/", "/index.html"):
                self._send(200, (TOOL_DIR / "index.html").read_bytes(),
                           "text/html; charset=utf-8")
            elif url.path == "/api/frames":
                labeled = {p.stem for p in LABELS_DIR.glob("*.txt")}
                seeded = ({p.stem for p in SEEDS_DIR.glob("*.txt")}
                          if SEEDS_DIR is not None else set())
                self._json({
                    "frames": [{"name": n, "labeled": Path(n).stem in labeled,
                                # seeded is reported ONLY while unlabeled: once a human saves,
                                # the label wins and the seed is irrelevant
                                "seeded": (Path(n).stem in seeded
                                           and Path(n).stem not in labeled)}
                               for n in list_frames()],
                    "labels_dir": str(LABELS_DIR),
                    "seeds_dir": (str(SEEDS_DIR) if SEEDS_DIR is not None else None),
                })
            elif url.path == "/api/label":
                name = parse_qs(url.query).get("name", [""])[0]
                safe_frame(name)
                lp = label_path(name)
                if lp.exists():
                    self._json({"exists": True, "seed": False,
                                "gates": labelio.decode_label(lp.read_text())})
                else:
                    sp = seed_path(name)
                    if sp is None:
                        self._json({"exists": False, "seed": False, "gates": []})
                    else:
                        # exists:false keeps the frame UNLABELED; seed:true tells the UI these
                        # points are unverified detector output to be corrected, not accepted.
                        self._json({"exists": False, "seed": True,
                                    "gates": labelio.decode_label(sp.read_text())})
            elif url.path.startswith("/frames/"):
                p = safe_frame(url.path[len("/frames/"):])
                self._send(200, p.read_bytes(), MIME[p.suffix.lower()])
            else:
                self._err("not found", 404)
        except ValueError as e:
            self._err(e, 404)
        except Exception as e:  # keep the labeler alive on any handler bug
            self._err(e, 500)

    def do_POST(self):
        if urlparse(self.path).path != "/api/save":
            return self._err("not found", 404)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            name = body["name"]
            safe_frame(name)
            img_w = int(body.get("img_w", labelio.IMAGE_WIDTH))
            img_h = int(body.get("img_h", labelio.IMAGE_HEIGHT))
            warning = None
            if (img_w, img_h) != (labelio.IMAGE_WIDTH, labelio.IMAGE_HEIGHT):
                # Normalization stays correct (divide by ACTUAL size), but the VQ2 contract
                # is 640x360 -- flag loudly so a wrong-source frame dir gets noticed.
                warning = (f"frame is {img_w}x{img_h}, contract is "
                           f"{labelio.IMAGE_WIDTH}x{labelio.IMAGE_HEIGHT}")
                sys.stderr.write(f"WARNING [{name}]: {warning}\n")
            text = labelio.encode_label(body.get("gates", []), img_w, img_h)
            lp = label_path(name)
            lp.write_text(text)  # "" for negatives: empty file marks intentional background
            self._json({"ok": True, "path": str(lp),
                        "rows": len(text.splitlines()), "warning": warning})
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            self._err(e, 400)
        except Exception as e:
            self._err(e, 500)


def main():
    global FRAMES_DIR, LABELS_DIR, SEEDS_DIR
    ap = argparse.ArgumentParser(description="VQ2 gate hand-labeling tool (stdlib only)")
    ap.add_argument("--frames", required=True, help="directory of .png/.jpg frames to label")
    ap.add_argument("--labels", required=True, help="output directory for YOLO-pose .txt labels")
    ap.add_argument("--seeds", default=None,
                    help="OPTIONAL dir of detector pre-labels (scripts/seed_labels_with_m.py). "
                         "Served only when no human label exists, flagged as unverified, and the "
                         "frame stays UNLABELED until you save -- seeds never become labels on "
                         "their own. Keep this dir SEPARATE from --labels.")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    FRAMES_DIR = Path(args.frames).resolve()
    LABELS_DIR = Path(args.labels).resolve()
    if not FRAMES_DIR.is_dir():
        sys.exit(f"--frames dir not found: {FRAMES_DIR}")
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    if args.seeds:
        SEEDS_DIR = Path(args.seeds).resolve()
        if SEEDS_DIR == LABELS_DIR:
            sys.exit("--seeds must NOT be the same dir as --labels (seeds are unverified)")
        if not SEEDS_DIR.is_dir():
            sys.exit(f"--seeds dir not found: {SEEDS_DIR}")

    n = len(list_frames())
    print(f"gate_labeler: {n} frames in {FRAMES_DIR}")
    print(f"gate_labeler: labels -> {LABELS_DIR}")
    if SEEDS_DIR is not None:
        n_seed = len(list(SEEDS_DIR.glob("*.txt")))
        print(f"gate_labeler: seeds  <- {SEEDS_DIR}  ({n_seed} unverified pre-labels)")
    print(f"gate_labeler: open http://localhost:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
