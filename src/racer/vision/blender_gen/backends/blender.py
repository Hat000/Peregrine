"""bpy/Cycles photoreal render backend (ShadowPC). Drop-in for ProceduralBackend.

Composes the four bpy leaf modules (``bpy_camera`` / ``bpy_scene`` / ``bpy_materials`` /
``bpy_render``) behind the ``dataset.RenderBackend`` protocol. The geometry/labels are owned by
``geometry.py`` (pure Python) and NEVER read back from Blender -- this backend only turns each
FrameSpec's optical gate poses into photoreal pixels, so a Cycles dataset and a procedural dataset
share byte-identical labels.

Everything bpy is imported lazily inside ``__init__`` / ``render`` so the module imports fine on a
Blender-less laptop (the laptop uses ProceduralBackend). Run it via ``render_entry.py`` under
``blender --background --python``.

Per-frame protocol: keep the camera + gate-mesh template across frames (cheap), rebuild the
randomized material / lighting / world / background / gate instances each frame, render to a temp
PNG, read it back as BGR, and delete the per-frame objects so memory does not grow over a long run.

SILHOUETTE EXCEPTION to "labels are never read back from Blender". Corner POSITIONS still are not --
geometry.py stays the single source of those, and the keypoint values are untouched. But two things
genuinely cannot be computed in closed form and are therefore MEASURED with a throwaway Workbench
object-id render (:mod:`racer.vision.blender_gen.bpy_idmask`) taken right after the beauty render:

  1. the gate's true projected SILHOUETTE -- the gate is a 0.26 m deep prism, so close and off-axis
     the camera sees its inner side walls and no flat quad describes it (only with ``--seg``/
     ``--masks``, which adds the second opening pass);
  2. WHICH GATES ARE VISIBLE AT ALL. Occlusion coverage is an area question; the renderer's z-buffer
     already answers it exactly, with props, people, the floor, nearer gates and the frame edge all
     accounted for. This runs on EVERY frame -- pose labels must not depend on whether ``--seg`` was
     passed -- and replaces the last surviving corner-count drop (see the method docstring).
"""
from __future__ import annotations

import numpy as np

from racer.frames import R_camera_from_body, R_world_from_body

from ..config import ScenarioPreset
from ..geometry import FrameSpec


def _optical_velocity(frame: FrameSpec) -> np.ndarray:
    """Body NED velocity expressed in the camera optical frame (for motion-blur streaking)."""
    R_wb = R_world_from_body(frame.roll, frame.pitch, frame.yaw)
    return R_camera_from_body() @ R_wb.T @ np.asarray(frame.body_vel_ned, dtype=np.float64)


class BlenderBackend:
    """Stateful Cycles/Eevee backend. Construct ONCE (sets up camera + gate template), then call
    ``render`` per frame. Must be constructed inside a running Blender (``import bpy`` succeeds)."""

    def __init__(self, preset: ScenarioPreset):
        import bpy  # noqa: F401  (only available inside Blender)

        from .. import bpy_camera, bpy_materials, bpy_render, bpy_scene

        self._bpy = bpy
        self._cam_mod = bpy_camera
        self._scene_mod = bpy_scene
        self._mat_mod = bpy_materials
        self._render_mod = bpy_render
        self.preset = preset

        bpy_scene.clear_scene()
        self.scene = bpy.context.scene
        self.cam = bpy_camera.setup_camera(self.scene)
        self.template = bpy_scene.build_gate_template()
        # hide the bare template from renders (we render duplicated instances of its mesh)
        try:
            self.template.hide_render = True
        except Exception:
            pass
        # Label-only proxy for the see-through hole; always built (it costs one 4-vert mesh) so the
        # id pass never has to mutate the scene graph mid-render.
        self.opening_template = bpy_scene.build_opening_template()
        self._frame_objects: list = []

        # Set by dataset.generate_dataset when seg/mask output is requested: it adds the OPENING
        # pass (class 1) on top of the structure pass, which now runs on every frame regardless.
        self.emit_silhouettes = False
        self._silhouettes: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        # Occlusion bookkeeping, printed once at the end of a run by render_entry. A run that
        # suddenly reports a large 'hidden' count is dropping gates again -- the fingerprint of the
        # bug this backend has now had four times. 'unmeasured' means the id pass failed and the
        # occlusion check did NOT run for those gates, which is silent data damage if ignored.
        self._census: dict[str, int] = {}
        # (gate_id, range_m, unoccluded_px) for the frame just rendered -- diagnostics only, so a
        # threshold study can read the raw measurement instead of guessing at it.
        self.last_gate_areas: list[tuple[int, float, float]] = []

        # Photoreal path: HDRI + PBR floor + real props/people (the validated look). Active only when
        # the preset asks for it AND the CC0 asset library is on disk; otherwise the legacy procedural
        # sky/walls path runs. Never affects gate poses / labels -- only pixels + raycast visibility.
        self._pr_mod = None
        self._lib = None
        self._prop_cache = None
        if getattr(preset.appearance, "photoreal", False):
            from .. import assets, bpy_photoreal
            self._pr_mod = bpy_photoreal
            self._lib = assets.AssetLibrary(preset.appearance.assets_dir)
            if not self._lib.available():
                print("[vq2] WARNING: photoreal=True but no HDRIs found under "
                      f"{self._lib.root} -- falling back to the procedural look. Run assets.fetch_all.")
                self._pr_mod = None
            else:
                # import every prop glTF ONCE; per-frame spawns duplicate (shared mesh) -- cheap
                self._prop_cache = bpy_photoreal.PropCache(self.scene, self._lib.prop_gltfs())

    # -- intrinsics self-check (the ShadowPC <=1 px gate) --------------------------------
    def intrinsics_error_px(self) -> float:
        return float(self._cam_mod.projection_max_error_px(self.scene, self.cam))

    # -- per-frame cleanup ---------------------------------------------------------------
    def _purge_frame_objects(self) -> None:
        bpy = self._bpy
        for obj in self._frame_objects:
            try:
                bpy.data.objects.remove(obj, do_unlink=True)
            except Exception:
                pass
        self._frame_objects = []
        # drop now-orphan mesh/light/material/image data so memory does not grow over a long run
        try:
            bpy.data.orphans_purge(do_recursive=True)
        except Exception:
            pass

    # -- the RenderBackend protocol ------------------------------------------------------
    def render(self, frame: FrameSpec, preset: ScenarioPreset, rng: np.random.Generator) -> np.ndarray:
        self._silhouettes = {}
        if self._pr_mod is not None:
            return self._render_photoreal(frame, preset, rng)
        return self._render_legacy(frame, preset, rng)

    # -- true-silhouette segmentation targets (optional protocol extension) --------------
    def gate_silhouettes(self) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """``{gate_id: (ring_mask, opening_mask)}`` measured on the frame just rendered.

        Keyed by ``gate_id``, NOT by list position: ``augment_frame`` rebuilds and REORDERS the
        GateRender list (labelled first, then the rest), so any index-based pairing would silently
        hand gate A's mask to gate B under augmentation. gate_ids are unique within a frame (each
        row of the course appears at most once), and ``augment._recompute`` preserves them.
        """
        return self._silhouettes

    def occlusion_census(self) -> dict[str, int]:
        """``{labelled, hidden, unmeasured}`` accumulated over the whole run (both splits).

        Printed at the end of a render. Read it: a jump in ``hidden`` is the fingerprint of gates
        being dropped again, and ANY ``unmeasured`` means some frames were written without the
        occlusion check running at all.
        """
        return dict(self._census)

    def _resolve_occlusion_and_silhouettes(self, frame: FrameSpec, gate_objs: list) -> None:
        """Measure each candidate gate's TRUE unoccluded silhouette, drop the ones that are hidden,
        and (when ``emit_silhouettes``) keep the masks as the segmentation target.

        WHY THE VISIBILITY DECISION LIVES HERE. "Is enough of this gate visible to label it?" is an
        AREA question, and once something can stand in front of the gate the only honest area is the
        one the renderer's z-buffer produces -- props, people, the floor, nearer gates and the frame
        edge all take their bite before we count a single pixel. Every closed-form substitute this
        module has tried was a corner count in disguise and every one of them poisoned the dataset by
        handing a rendered gate to training as background (see bpy_photoreal.occlude_blocked_keypoints
        and geometry._has_labellable_area). Pass A is the measurement; we were already paying for it
        in seg mode, so it now runs unconditionally.

        UNCONDITIONALLY is load-bearing: gate this on ``emit_silhouettes`` and the POSE LABELS would
        depend on whether ``--seg`` was passed. Two runs of the same preset and seed would disagree
        on which gates are labelled, and nothing on disk would say why. Measured price for buying
        that away on a pose-only run: 33.78 s -> 36.35 s per 10 vq1_partial frames, i.e. +0.26 s/frame
        (+7.6%) on a 128-sample Cycles frame -- and pose labels then diff byte-identical with and
        without --seg, which was verified on disk rather than assumed.

        Never raises: a failed id pass must degrade to "keep every gate the extent rule accepted,
        loudly", not kill a multi-hour render and not silently start dropping gates.
        """
        from .. import bpy_idmask, bpy_scene
        from ..geometry import apply_measured_occlusion, unoccluded_area_px

        # Candidates = whatever the (occlusion-blind) on-screen-extent rule already accepted. The
        # rest still RENDER, and still occlude -- painted black -- in the id pass.
        wanted = [(i, gr) for i, gr in enumerate(frame.gates) if gr.visible]
        self.last_gate_areas = []
        if not wanted:
            return
        ids = [int(gr.gate_id) for _, gr in wanted]
        if len(set(ids)) != len(ids):
            # Duplicate gate_ids would make the silhouette dict lossy and mis-pair masks after
            # augment (which reorders the gate list). Bail loudly rather than write a poisoned label.
            print(f"[vq2] WARNING: duplicate gate_ids {ids} in one frame -- skipping the id pass.")
            self._census["unmeasured"] = self._census.get("unmeasured", 0) + len(wanted)
            return

        try:
            # Pass A: the gate STRUCTURE, as the camera sees it. No opening proxies exist yet, and
            # they are hide_render by construction anyway -- a solid proxy sitting in a near gate's
            # opening would black out a far gate seen through it.
            rings = bpy_idmask.render_id_masks(self.scene, [gate_objs[i] for i, _ in wanted])
        except Exception as exc:                      # pragma: no cover - defensive on ShadowPC
            print(f"[vq2] WARNING: gate id pass A failed ({type(exc).__name__}: {exc}); this frame "
                  f"keeps every on-screen gate UNCHECKED for occlusion and gets no seg masks.")
            self._census["unmeasured"] = self._census.get("unmeasured", 0) + len(wanted)
            return

        # THE DROP. A gate whose measured silhouette is essentially gone is behind something solid;
        # labelling it would teach the detector to see gates through walls. The decision itself is
        # pure Python in geometry.apply_measured_occlusion so the laptop suite can test it -- the
        # last round of this bug hid in a module the tests could not import.
        for k, (_, gr) in enumerate(wanted):
            self.last_gate_areas.append((int(gr.gate_id), float(gr.range_m),
                                         unoccluded_area_px(rings[k])))
        for key, n in apply_measured_occlusion([gr for _, gr in wanted], rings).items():
            self._census[key] = self._census.get(key, 0) + n
        kept = [(k, gr, rings[k]) for k, (_, gr) in enumerate(wanted) if gr.visible]

        if not self.emit_silhouettes or not kept:
            return
        try:
            # Pass B: the SEE-THROUGH HOLE, for the SURVIVORS only -- one render per group of
            # openings that cannot overlap on screen (see bpy_idmask.disjoint_batches). Gates stay
            # in the scene, painted black, so the hole is correctly eaten by the near-side inner
            # wall on an oblique gate. Proxies are registered for purge as they are CREATED, not
            # after the loop -- a throw halfway through would leak objects into every later frame.
            openings = []
            for _, gr, _ in kept:
                obj = bpy_scene.instance_opening(self.opening_template, gr.R_cam_gate, gr.t_cam_gate)
                self._frame_objects.append(obj)
                openings.append(obj)
            open_masks: list[np.ndarray | None] = [None] * len(kept)
            quads = [np.asarray(gr.keypoints_px, dtype=float) for _, gr, _ in kept]
            for batch in bpy_idmask.disjoint_batches(quads):
                targets = [openings[j] for j in batch]
                hidden = [o for j, o in enumerate(openings) if j not in batch]
                for j, m in zip(batch, bpy_idmask.render_id_masks(self.scene, targets, hidden=hidden)):
                    open_masks[j] = m
            for j, (_, gr, ring) in enumerate(kept):
                self._silhouettes[int(gr.gate_id)] = (ring, open_masks[j])
        except Exception as exc:                      # pragma: no cover - defensive on ShadowPC
            print(f"[vq2] WARNING: gate id pass B failed ({type(exc).__name__}: {exc}); "
                  f"no seg masks for this frame (pose labels are unaffected).")
            self._silhouettes = {}

    # -- photoreal path: HDRI world + PBR floor + real props/people (validated 2026-06-15) ---------
    def _render_photoreal(self, frame: FrameSpec, preset: ScenarioPreset,
                          rng: np.random.Generator) -> np.ndarray:
        ap, rc = preset.appearance, preset.render
        PR = self._pr_mod

        # 1. HDRI environment (image-based lighting + photographic background, optical-frame oriented)
        hdris = self._lib.hdris()
        PR.setup_hdri_world(self.scene, rng, hdris[int(rng.integers(len(hdris)))],
                            strength=float(rng.uniform(0.7, 1.3)))

        # 2. PBR floor a good way BELOW the gates (so gates float in the air, ground visible below)
        floor_y = PR.floor_below_gates(frame, rng)
        texsets = self._lib.floor_texsets()
        texset = texsets[int(rng.integers(len(texsets)))] if texsets else {}
        self._frame_objects.append(PR.add_floor(self.scene, floor_y, texset, rng))

        # 3. solid vivid gate(s) at their exact optical poses (labels unchanged). Keep the object
        #    list POSITIONALLY aligned with frame.gates -- the id pass indexes into it.
        gate_mat = PR.solid_gate_material(rng, ap)
        gate_objs = [self._scene_mod.instance_gate(self.template, gr.R_cam_gate, gr.t_cam_gate, gate_mat)
                     for gr in frame.gates]
        self._frame_objects += gate_objs

        # 4. real props + mannequin people, scattered OFF the gate corridor (sides/background).
        #    Props are spawned as duplicates of the once-imported cache (shared mesh data -> fast).
        n_prop = int(rng.integers(int(ap.prop_count_range[0]), int(ap.prop_count_range[1]) + 1)) \
            if self._prop_cache.available() else 0
        for _ in range(n_prop):
            x, z = PR._side_xz(rng)
            self._frame_objects += self._prop_cache.spawn(x, z, floor_y, rng)
        n_ppl = int(rng.integers(int(ap.people_count_range[0]), int(ap.people_count_range[1]) + 1))
        for _ in range(n_ppl):
            x, z = PR._side_xz(rng)
            self._frame_objects += PR.add_person(self.scene, x, z, floor_y, rng)

        # 5. honest labels: downgrade any gate corner a prop/person actually blocks (raycast)
        PR.occlude_blocked_keypoints(self.scene, frame)

        # 6. clean render (NO in-render motion blur / glare; AgX + exposure variety)
        PR.configure_clean_render(self.scene, rng, rc, ap)
        image = self._render_mod.render_to_bgr(self.scene)
        # 7. id pass BEFORE the purge -- it needs the very objects we are about to delete. This is
        #    also where a gate that is actually HIDDEN loses its label (see the method docstring);
        #    step 5's raycast only marks corners.
        self._resolve_occlusion_and_silhouettes(frame, gate_objs)
        self._purge_frame_objects()
        return np.ascontiguousarray(image)

    # -- legacy procedural path (no assets) --------------------------------------------------------
    def _render_legacy(self, frame: FrameSpec, preset: ScenarioPreset,
                       rng: np.random.Generator) -> np.ndarray:
        ap, rc = preset.appearance, preset.render

        # 1. world + lighting + background (fresh randomization per frame)
        self._mat_mod.setup_world(rng, ap)
        self._frame_objects += list(self._mat_mod.add_lighting(rng, ap))
        bg_objs = list(self._scene_mod.build_background(rng, ap))
        if bg_objs:
            bg_mat = self._mat_mod.background_material(rng, ap)
            for o in bg_objs:
                self._mat_mod.apply_material(o, bg_mat)
        self._frame_objects += bg_objs

        # 2. one shared gate material per frame (real gates on a course look identical)
        gate_mat = self._mat_mod.gate_material(rng, ap)
        gate_objs = []
        for gr in frame.gates:                                   # place every in-band gate (occlusion renders)
            obj = self._scene_mod.instance_gate(self.template, gr.R_cam_gate, gr.t_cam_gate, gate_mat)
            gate_objs.append(obj)
        self._frame_objects += gate_objs

        # 3. render config (+ camera-motion blur from the optical-frame velocity)
        self._render_mod.configure_render(self.scene, rc, ap)
        if rc.motion_blur:
            self._render_mod.set_motion_blur(
                self.scene, gate_objs + bg_objs, self.cam, _optical_velocity(frame), rc.motion_blur_shutter
            )

        # 4. render -> BGR, then tear down this frame's objects
        image = self._render_mod.render_to_bgr(self.scene)
        # Same occlusion resolution as the photoreal path, and for the same reason: this path has no
        # props, but a far gate can still sit entirely behind a NEARER gate's frame, and it would
        # otherwise be labelled through it. Runs before the purge; see the photoreal path.
        self._resolve_occlusion_and_silhouettes(frame, gate_objs)
        self._purge_frame_objects()
        return np.ascontiguousarray(image)
