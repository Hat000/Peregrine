"""Scenario-preset schema + loader (pure-Python, validated).

A preset is a JSON file with four sections: ``viewpoint`` (camera envelope -> geometry.py),
``augment`` (post-render aug -> augment.py), ``appearance`` (gate colour / lighting / background
domain-randomization -> the bpy backend), and ``render`` (Cycles/Eevee + motion blur + glare ->
the bpy backend). The pure-Python core consumes viewpoint + augment; the appearance + render
sections are read by the Blender backend. Validation is strict (unknown keys + out-of-range
values raise) so a typo in a preset fails loudly instead of silently doing nothing on ShadowPC.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from .augment import AugmentConfig
from .geometry import ViewpointConfig

PRESETS_DIR = Path(__file__).resolve().parent / "presets"
_ENGINES = {"CYCLES", "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"}
_BG_MODES = {"arena", "indoor", "outdoor", "mixed"}
_LIGHTING_MODES = {"daylight", "dark_arena", "auto"}


@dataclass(frozen=True)
class AppearanceConfig:
    """Gate-material + lighting + background domain-randomization (Blender backend)."""

    use_vq1_red: bool = True               # anchor gate colour at the extracted VQ1 orange-red
    gate_hue_jitter: float = 0.0           # fraction of the hue wheel sampled around the base (0..0.5)
    gate_sat_range: tuple = (0.85, 1.0)    # HSV saturation range
    gate_val_range: tuple = (0.75, 1.0)    # HSV value range
    gate_metallic_range: tuple = (0.0, 0.3)
    gate_roughness_range: tuple = (0.2, 0.7)
    gate_emission_prob: float = 0.0        # chance the gate is slightly emissive (self-lit look)

    sun_elevation_range_deg: tuple = (15.0, 80.0)
    sun_intensity_range: tuple = (1.0, 6.0)
    color_temp_range_k: tuple = (4500.0, 7500.0)
    use_hdri: bool = False
    hdri_dir: str | None = None            # folder of .hdr/.exr env maps (ShadowPC-local); None -> sky

    # --- photoreal real-asset dressing (Blender backend; see bpy_photoreal + assets) -------------
    photoreal: bool = False                # True -> HDRI world + PBR floor + real props/people (the
    #                                        validated 2026-06-15 look); False -> legacy procedural sky/walls
    assets_dir: str | None = None          # root of the CC0 asset set (default: <repo>/assets_vq2)
    gate_emission_range: tuple = (0.25, 0.5)   # small always-on gate glow (vivid, not washed out)
    prop_count_range: tuple = (8, 14)      # real photoscanned props scattered OFF the gate corridor
    people_count_range: tuple = (2, 5)     # procedural clothing-tinted mannequin people

    # Lighting MODE (the dominant realism lever; see the VQ2 plan, axis B). The real A2RL x DCL
    # venue is a DARK indoor arena with bright, often coloured spotlights + high dynamic range --
    # NOT uniform daylight. 'dark_arena' = low ambient + N hard spotlights; 'daylight' = the sun
    # path; 'auto' = pick per frame (bias indoor). Consumed by bpy_materials.add_lighting/setup_world.
    lighting_mode: str = "daylight"        # daylight | dark_arena | auto
    ambient_strength_range: tuple = (0.2, 2.0)   # world background strength (low => dark arena)
    n_spotlights_range: tuple = (1, 3)     # hard spots in dark_arena mode (ints)
    spotlight_energy_range: tuple = (200.0, 3000.0)   # W per spot
    colored_light_prob: float = 0.4        # chance a spot is coloured (DCL stage lighting)

    background_mode: str = "arena"         # arena|indoor|outdoor|mixed
    clutter_max: int = 6                   # random background props
    floor_wall_texture_prob: float = 0.5

    exposure_range: tuple = (-0.5, 0.5)    # film exposure EV
    gamma_range: tuple = (0.9, 1.1)
    sensor_noise_prob: float = 0.3


@dataclass(frozen=True)
class RenderConfig:
    """Cycles/Eevee render settings (Blender backend)."""

    engine: str = "CYCLES"
    samples: int = 64                      # Cycles path-trace samples (preview low, final high)
    use_gpu: bool = True
    use_denoise: bool = True
    motion_blur: bool = True
    motion_blur_shutter: float = 0.5       # frames; longer -> more streak
    use_glare: bool = True                 # compositor glare/bloom for bright lights
    film_transparent: bool = False


@dataclass(frozen=True)
class ScenarioPreset:
    name: str
    description: str = ""
    # Fraction of frames rendered as HARD NEGATIVES (background/confusers, NO gate, empty label).
    # The sim-to-real evidence (MonoRace + the YOLOv11 DR study, ~35% negatives) makes this a
    # first-class lever, not an afterthought. 0 => all positives (legacy). The `negatives` preset
    # sets 1.0; the recommended overall mix targets ~25-30% negatives across arms.
    negative_fraction: float = 0.0
    viewpoint: ViewpointConfig = field(default_factory=ViewpointConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    appearance: AppearanceConfig = field(default_factory=AppearanceConfig)
    render: RenderConfig = field(default_factory=RenderConfig)

    def to_dict(self) -> dict:
        return {
            "name": self.name, "description": self.description,
            "negative_fraction": self.negative_fraction,
            "viewpoint": asdict(self.viewpoint), "augment": asdict(self.augment),
            "appearance": asdict(self.appearance), "render": asdict(self.render),
        }


def _section(cls, d: Any, where: str):
    """Build a dataclass from a dict, rejecting unknown keys (strict schema)."""
    if d is None:
        return cls()
    if not isinstance(d, dict):
        raise ValueError(f"preset section '{where}' must be an object, got {type(d).__name__}")
    known = {f.name for f in fields(cls)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"preset section '{where}' has unknown keys: {sorted(unknown)}")
    # normalise lists -> tuples for the tuple-typed range fields
    kw = {}
    for f in fields(cls):
        if f.name in d:
            v = d[f.name]
            kw[f.name] = tuple(v) if isinstance(v, list) else v
    return cls(**kw)


def _validate(p: ScenarioPreset) -> ScenarioPreset:
    if not p.name or not isinstance(p.name, str):
        raise ValueError("preset 'name' must be a non-empty string")
    if not (0.0 <= p.negative_fraction <= 1.0):
        raise ValueError("negative_fraction must be in [0, 1]")
    vp = p.viewpoint
    if not (0.0 < vp.range_min_m < vp.range_max_m):
        raise ValueError(f"viewpoint range invalid: [{vp.range_min_m}, {vp.range_max_m}]")
    if vp.range_skew <= 0:
        raise ValueError("viewpoint.range_skew must be > 0")
    ap = p.appearance
    if not (0.0 <= ap.gate_hue_jitter <= 0.5):
        raise ValueError("appearance.gate_hue_jitter must be in [0, 0.5]")
    if ap.background_mode not in _BG_MODES:
        raise ValueError(f"appearance.background_mode must be one of {sorted(_BG_MODES)}")
    if ap.lighting_mode not in _LIGHTING_MODES:
        raise ValueError(f"appearance.lighting_mode must be one of {sorted(_LIGHTING_MODES)}")
    if not (0.0 <= ap.colored_light_prob <= 1.0):
        raise ValueError("appearance.colored_light_prob must be in [0, 1]")
    for lo_hi in (ap.gate_sat_range, ap.gate_val_range, ap.sun_intensity_range,
                  ap.color_temp_range_k, ap.exposure_range, ap.gamma_range):
        if not (len(lo_hi) == 2 and lo_hi[0] <= lo_hi[1]):
            raise ValueError(f"appearance range must be (lo<=hi), got {lo_hi}")
    rc = p.render
    if rc.engine not in _ENGINES:
        raise ValueError(f"render.engine must be one of {sorted(_ENGINES)}")
    if rc.samples < 1:
        raise ValueError("render.samples must be >= 1")
    if not (0.0 < rc.motion_blur_shutter <= 2.0):
        raise ValueError("render.motion_blur_shutter must be in (0, 2]")
    return p


def preset_from_dict(d: dict) -> ScenarioPreset:
    known = {"name", "description", "negative_fraction",
             "viewpoint", "augment", "appearance", "render"}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"preset has unknown top-level keys: {sorted(unknown)}")
    p = ScenarioPreset(
        name=d.get("name", ""),
        description=d.get("description", ""),
        negative_fraction=float(d.get("negative_fraction", 0.0)),
        viewpoint=_section(ViewpointConfig, d.get("viewpoint"), "viewpoint"),
        augment=_section(AugmentConfig, d.get("augment"), "augment"),
        appearance=_section(AppearanceConfig, d.get("appearance"), "appearance"),
        render=_section(RenderConfig, d.get("render"), "render"),
    )
    return _validate(p)


def load_preset(name_or_path: str) -> ScenarioPreset:
    """Load a preset by bare name (``presets/<name>.json``) or by explicit path."""
    p = Path(name_or_path)
    if not p.exists():
        p = PRESETS_DIR / f"{name_or_path}.json"
    if not p.exists():
        raise FileNotFoundError(
            f"preset not found: {name_or_path} (looked in {PRESETS_DIR}); "
            f"available: {sorted(x.stem for x in PRESETS_DIR.glob('*.json'))}"
        )
    return preset_from_dict(json.loads(p.read_text()))


def available_presets() -> list[str]:
    return sorted(x.stem for x in PRESETS_DIR.glob("*.json")) if PRESETS_DIR.exists() else []
