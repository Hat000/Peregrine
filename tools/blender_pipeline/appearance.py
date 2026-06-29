"""Load the VQ2 appearance params (the FIXED look the Blender scene + mock renderer match).

Thin typed wrapper over ``appearance_params.json`` so both the mock renderer and the future
Blender scene-builder read ONE source of truth. The JSON is the canonical machine-readable
form; this module just parses + validates it and exposes the few fields the renderer needs.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

PARAMS_PATH = os.path.join(os.path.dirname(__file__), "appearance_params.json")


@dataclass(frozen=True)
class AppearanceParams:
    """The subset of appearance numbers the renderer (mock or Blender) consumes."""

    # gate glow
    gate_emission_rgb: tuple[int, int, int] = (255, 56, 16)
    gate_emission_strength: float = 8.0
    gate_bar_thickness_frac: float = 0.175
    # bloom
    bloom_crisp_edge_px: int = 3
    bloom_soft_edge_px: int = 12
    core_threshold: int = 250
    ambient_red_wash_rgb: tuple[int, int, int] = (46, 40, 32)
    # scene
    background_rgb: tuple[int, int, int] = (10, 10, 10)
    dark_structure_rgb: tuple[int, int, int] = (24, 26, 26)
    scene_mean_gray: float = 36.0
    # other elements
    lane_line_rgb: tuple[int, int, int] = (222, 198, 80)
    floor_grid_rgb: tuple[int, int, int] = (45, 45, 48)
    floor_base_rgb: tuple[int, int, int] = (8, 8, 8)
    floor_grid_spacing_m: float = 1.0
    ceiling_light_rgb: tuple[int, int, int] = (237, 238, 238)
    green_marker_rgb: tuple[int, int, int] = (80, 252, 80)
    raw: dict = field(default_factory=dict, repr=False)

    @staticmethod
    def _rgb(v) -> tuple[int, int, int]:
        t = tuple(int(c) for c in v)
        assert len(t) == 3, f"expected RGB triple, got {v}"
        return t  # type: ignore[return-value]


def load_appearance(path: str | None = None) -> AppearanceParams:
    """Parse ``appearance_params.json`` -> :class:`AppearanceParams`. Missing keys fall back to
    the dataclass defaults (which mirror the recon numbers), so an older/partial JSON still loads."""
    p = path or PARAMS_PATH
    with open(p, "r") as f:
        raw = json.load(f)
    g = raw.get("gate", {})
    bl = raw.get("bloom", {})
    sc = raw.get("scene", {})
    ll = raw.get("lane_lines", {})
    fl = raw.get("floor", {})
    st = raw.get("structure", {})
    R = AppearanceParams._rgb
    d = AppearanceParams.__dataclass_fields__  # for defaults
    return AppearanceParams(
        gate_emission_rgb=R(g.get("emission_rgb", d["gate_emission_rgb"].default)),
        gate_emission_strength=float(g.get("emission_strength", d["gate_emission_strength"].default)),
        gate_bar_thickness_frac=float(g.get("bar_thickness_frac", d["gate_bar_thickness_frac"].default)
                                      if "bar_thickness_frac" in g else d["gate_bar_thickness_frac"].default),
        bloom_crisp_edge_px=int(bl.get("crisp_edge_px", d["bloom_crisp_edge_px"].default)),
        bloom_soft_edge_px=int(bl.get("soft_edge_px", d["bloom_soft_edge_px"].default)),
        core_threshold=int(bl.get("core_threshold_for_labels", d["core_threshold"].default)),
        ambient_red_wash_rgb=R(bl.get("ambient_red_wash_rgb", d["ambient_red_wash_rgb"].default)),
        background_rgb=R(sc.get("background_rgb", d["background_rgb"].default)),
        dark_structure_rgb=R(sc.get("dark_structure_rgb", d["dark_structure_rgb"].default)),
        scene_mean_gray=float(sc.get("mean_gray", d["scene_mean_gray"].default)),
        lane_line_rgb=R(ll.get("color_rgb", d["lane_line_rgb"].default)),
        floor_grid_rgb=R(fl.get("grid_line_rgb", d["floor_grid_rgb"].default)),
        floor_base_rgb=R(fl.get("floor_base_rgb", d["floor_base_rgb"].default)),
        floor_grid_spacing_m=float(fl.get("grid_spacing_m", d["floor_grid_spacing_m"].default)),
        ceiling_light_rgb=R(st.get("ceiling_light_rgb", d["ceiling_light_rgb"].default)),
        green_marker_rgb=R(st.get("green_marker_rgb", d["green_marker_rgb"].default)),
        raw=raw,
    )
