"""Local CC0 asset registry + Polyhaven fetcher for the photoreal VQ2 backend (pure-Python).

The Blender backend dresses each frame with REAL assets -- HDRI environment maps (image-based
lighting + photographic backgrounds), PBR floor textures, and photoscanned props (boxes, chairs,
barrels, crates, jerrycans, plants, ...) -- all CC0 from polyhaven.com. This module is the bridge:

  * SETUP (run once, needs network): ``fetch_all(root)`` downloads a curated, reproducible asset set
    into ``root`` (default ``<repo>/assets_vq2``). Idempotent -- existing files are skipped. Uses only
    the stdlib (urllib), so it runs in Blender's bundled Python too. See RUN_GUIDE.md.
  * RENDER (no network): ``AssetLibrary(root)`` enumerates what is on disk -- ``hdris()`` /
    ``floor_texsets()`` / ``prop_gltfs()`` -- which the backend samples per frame. People are
    procedural (no external asset), built in ``bpy_photoreal``.

Nothing here imports bpy, so it stays importable + unit-testable on the laptop.
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

_API = "https://api.polyhaven.com"
_HDRI_EXTS = (".hdr", ".exr")

# Curated, reproducible asset manifest (all CC0 on polyhaven.com).
N_INDOOR_HDRIS = 26                      # spread across the 'indoor' category for broad domain coverage
FLOOR_TEXTURES = ("concrete_floor_02", "asphalt_pit_lane", "brushed_concrete")
PROP_MODELS = (
    "cardboard_box_01", "Barrel_01", "Barrel_02", "WoodenChair_01", "GreenChair_01", "SchoolChair_01",
    "WoodenTable_01", "CoffeeTable_01", "cement_bag", "plastic_crate_01", "plastic_crate_02",
    "wooden_crate_01", "old_military_crate", "metal_jerrycan_green", "metal_toolbox", "metal_trash_can",
    "potted_plant_01", "potted_plant_02", "ladder_sectioned_01", "old_tyre", "metal_stool_01",
)


def default_root() -> Path:
    """``<repo>/assets_vq2`` (repo root = four parents up from this file: .../src/racer/vision/blender_gen)."""
    return Path(__file__).resolve().parents[4] / "assets_vq2"


# --------------------------------------------------------------------------- fetch (setup, network)
def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    with urllib.request.urlopen(url, timeout=180) as r, open(dest, "wb") as f:
        f.write(r.read())


def fetch_hdris(root: Path, n: int = N_INDOOR_HDRIS, res: str = "2k") -> int:
    """Download ``n`` indoor HDRIs (spread across the category for variety) into ``root/hdris``."""
    out = root / "hdris"
    names = list(_get_json(f"{_API}/assets?type=hdris&categories=indoor").keys())
    step = max(1, len(names) // max(1, n))
    picks = names[::step][:n]
    got = 0
    for hid in picks:
        try:
            files = _get_json(f"{_API}/files/{hid}")
            url = files["hdri"][res]["hdr"]["url"]
            _download(url, out / f"{hid}_{res}.hdr")
            got += 1
        except Exception:
            continue
    return got


def fetch_texture(root: Path, tid: str, res: str = "2k") -> bool:
    """Download a PBR floor texture set (diffuse / normal-gl / arm) into ``root/textures/<tid>``."""
    out = root / "textures" / tid
    try:
        files = _get_json(f"{_API}/files/{tid}")
    except Exception:
        return False
    ok = False
    for api_key, local in (("Diffuse", "diff.jpg"), ("nor_gl", "nor.jpg"), ("arm", "arm.jpg")):
        try:
            _download(files[api_key][res]["jpg"]["url"], out / local)
            ok = True
        except Exception:
            continue
    return ok


def fetch_model(root: Path, mid: str, res: str = "1k") -> bool:
    """Download a glTF model (+ its .bin and textures, preserving relative paths) into ``root/models/<mid>``."""
    out = root / "models" / mid
    try:
        files = _get_json(f"{_API}/files/{mid}")
        g = files["gltf"][res]["gltf"]
    except Exception:
        return False
    try:
        _download(g["url"], out / os.path.basename(g["url"]))
        for rel, info in g.get("include", {}).items():
            _download(info["url"], out / rel.replace("/", os.sep))
        return True
    except Exception:
        return False


def fetch_all(root: Path | str | None = None) -> dict:
    """Download the full curated asset set. Returns counts. Idempotent (skips existing files)."""
    root = Path(root) if root else default_root()
    n_h = fetch_hdris(root)
    n_t = sum(fetch_texture(root, t) for t in FLOOR_TEXTURES)
    n_m = sum(fetch_model(root, m) for m in PROP_MODELS)
    return {"hdris": n_h, "textures": n_t, "models": n_m, "root": str(root)}


# --------------------------------------------------------------------------- registry (render, local)
class AssetLibrary:
    """Enumerate the on-disk CC0 assets the backend samples per frame (no network)."""

    def __init__(self, root: Path | str | None = None):
        self.root = Path(root) if root else default_root()

    def hdris(self) -> list[str]:
        d = self.root / "hdris"
        return [str(p) for p in sorted(d.glob("*")) if p.suffix.lower() in _HDRI_EXTS] if d.is_dir() else []

    def floor_texsets(self) -> list[dict]:
        """Each floor texture set as {'diff','nor','arm'} absolute paths (only those present)."""
        base = self.root / "textures"
        sets = []
        if base.is_dir():
            for d in sorted(p for p in base.iterdir() if p.is_dir()):
                s = {k: str(d / f"{k if k != 'diff' else 'diff'}.jpg") for k in ("diff", "nor", "arm")
                     if (d / f"{k}.jpg").exists()}
                if "diff" in s:
                    sets.append(s)
        return sets

    def prop_gltfs(self) -> list[str]:
        d = self.root / "models"
        return [str(p) for p in sorted(d.glob("*/*.gltf"))] if d.is_dir() else []

    def available(self) -> bool:
        return bool(self.hdris())


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Fetch CC0 assets for the VQ2 photoreal backend")
    ap.add_argument("--root", default=None, help="asset root (default <repo>/assets_vq2)")
    a = ap.parse_args(argv)
    print(fetch_all(a.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
