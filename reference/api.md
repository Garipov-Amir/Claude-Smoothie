# `bpy_stylized_kit` API reference

`scripts/bpy_stylized_kit.py`. Import with:

```python
import sys; sys.path.insert(0, "<repo>/blender-stylized-3d/scripts")
import bpy_stylized_kit as k
```

## Scene setup

- `new_scene(clear=True)` — wipe the default cube/camera/light and orphaned data-blocks.
- `set_stylized_color_management()` — force "Standard" view transform (called automatically by `render_still`).
- `set_world_background(color=(0.04,0.045,0.05), strength=1.0)` — flat dark studio backdrop.

## Primitives & mesh

- `add_primitive(kind, location=(0,0,0), rotation=(0,0,0), scale=(1,1,1), name=None, **extra)` — `kind` in `cube|sphere|ico_sphere|cylinder|cone|torus|plane`. `**extra` forwards to the underlying `bpy.ops.mesh.primitive_*_add` (e.g. `subdivisions=3` for `ico_sphere`, `vertices=8` for `cylinder`/`cone`).
- `shade_smooth(obj, auto_smooth_angle=None)` / `shade_flat(obj)`
- `add_subsurf(obj, levels=2, render_levels=2)`
- `add_bevel(obj, width=0.02, segments=2, limit_method="ANGLE")`
- `add_decimate(obj, ratio=0.5)` — push toward low-poly.
- `add_solidify_outline(obj, thickness=0.02, color=(0.02,0.02,0.02))` — inverted-hull ink outline (toon look).
- `join_objects(objects, name="Model")` — merge parts into one object; do this last, after per-part materials are assigned.
- `mirror_x(obj)` — add a Mirror modifier across local X (model one half of a symmetric creature/prop, mirror at the end).

## Polygon / loft-based mesh building (preferred for organic characters)

See `polygon_modeling.md` for the full technique writeup — summary here:

- `loft_profile(rings, segments=16, cap_start=True, cap_end=True, name="Loft")` — low-level: bridges a list of `{"co": (x,y,z), "radius": r_or_(rx,ry)}` rings into one continuous mesh.
- `build_profile_body(spine, segments=16, name="Body")` — friendlier wrapper: `spine` is `[(position, radius), ...]` (or `(position, radius_x, radius_y)`). Use for a torso+neck+head as one continuous part instead of stacked spheres.
- `rounded_cap_points(base_co, base_radius, tip_co, n=2)` — `n` intermediate `(position, radius)` points with a quarter-circle falloff, for inserting between a spine's last full-radius point and its tapering tip. Without this, a straight linear taper to a near-zero tip reads as a sharp CONE (found and fixed in the loop-iteration testing — e.g. a humanoid's head); with it, the same taper reads as a rounded dome. Usage: `spine = [..., (head_co, head_r)] + k.rounded_cap_points(head_co, head_r, top_co) + [(top_co, 0.02)]`.
- `add_tapered_limb(start, end, start_radius, end_radius=None, segments=10, name="Limb", bend=None, bend_radius=None)` — a straight or single-bend tapered capsule-like limb (arm/leg/tail/quill), capped both ends.
- `boolean_op(objects, operation="UNION", name="Merged")` — Boolean modifier (exact solver) with `operation` in `UNION`/`INTERSECT`/`DIFFERENCE`; preserves each source's per-face material assignment (verified for `UNION` in `_smoke_test_loft.py`).
- `boolean_union(objects, name="Merged")` — thin wrapper: `boolean_op(objects, "UNION", name)`.
- `assign_material_by_region(obj, mat, region_fn)` — paint multiple materials onto one continuous mesh by world-space face position; `region_fn(world_space_face_center) -> bool`.
- `world_bounds(objects)` — true world-space bounding box `(center, size)` from actual geometry (not object origins); used internally by `render_turntable`/`frame_camera_on_objects` and useful directly when framing a custom camera shot.

## Materials

- `build_hand_painted_material(name, base_color, variation=0.15, saturation_boost=1.15, cavity_strength=0.45, roughness=0.55, metallic=0.0, rim_color=(0.55,0.70,0.95), rim_strength=0.12)` — see `hand_painted_shader.md` for what each knob does. `base_color` is `(r,g,b)` in 0..1 — use `hex_to_rgb("#RRGGBB")`.
- `apply_material(obj, mat, slot_index=None)` — with `slot_index=None` (the common one-material-per-part call), appends `mat` as a new slot AND repoints every face at it. It used to only append, which silently rendered the object flat white/gray whenever the mesh already carried an empty slot 0 with faces still pointing at it — found on a `carve_from_silhouettes` INTERSECT result, where Blender's Boolean modifier (EXACT solver) left a phantom empty slot 0 on a mesh that had never had a material, even though the source volumes had none either (a Blender quirk, not something the toolkit's boolean code does explicitly). Pass an explicit `slot_index` to append without touching face indices (e.g. building up regions ahead of `assign_material_by_region`).
- `hex_to_rgb(hex_color)` → `(r,g,b)` in 0..1

## Image-driven geometry (no ML reconstruction)

- `add_displace_from_heightmap(obj, image_path, strength=0.3, midlevel=0.5)` — needs the object to have UVs (smart-project or unwrap first if it's not already a plane from `extrude_silhouette`).
- `extrude_silhouette(mask_image_path, name="Cutout", depth=0.3, resolution=128)` — the silhouette extruded to a flat slab (`extrude_silhouette_volume(axis="top")` under the hood — see its entry and `polygon_modeling.md` for why this replaced an older Displace-modifier version that produced a jagged fringe no resolution/bevel could fix). Result is centered on the origin along Z (spans `-depth/2..+depth/2`), not resting on the ground. Shallow relief, not a full volume — use `carve_from_silhouettes` below for genuine 3D from multiple photos.
- `extrude_silhouette_volume(mask_image_path, axis="front", size=2.0, depth=2.5, resolution=48, name="Volume")` — a solid prism through `axis` ("front"/"side"/"top") shaped by the mask's silhouette (not a relief bump) — the "generalized cone" that `carve_from_silhouettes` intersects. Reads the mask directly via `bpy.data.images` (no PIL needed inside Blender).
- `carve_from_silhouettes(views, size=2.0, depth_margin=1.3, resolution=48, name="Carved")` — classical shape-from-silhouette / visual-hull reconstruction: intersects `extrude_silhouette_volume` prisms from ≥2 views (`views` = `[{"mask": path, "axis": "front"|"side"|"top"}, ...]`). See `polygon_modeling.md`'s "Multi-view silhouette carving" section for the framing/alignment requirement and what it fundamentally can't recover (concavities invisible from every view).

## Lighting, camera, render, export

- `add_three_point_lighting(target=(0,0,0), radius=6.0, key_energy=120, fill_energy=40, rim_energy=70, bounce_energy=18, key_color=..., fill_color=..., rim_color=..., bounce_color=...)` — tuned low by default to avoid Cycles blowout on flat stylized colors. Includes a dim underside "Bounce" light (`bounce_energy=0` to disable) — without it every object's bottom renders near-black regardless of actual geometry there, since Key/Fill/Rim all sit above the target (found in loop-iteration testing: a confirmed-symmetric carved shape's bottom looked flat/cut purely from lighting).
- `setup_camera(location=(5,-5,3.5), look_at=(0,0,0), lens=50)`
- `frame_camera_on_objects(cam_obj, objects, margin=1.3)` — rough auto-frame heuristic; check the render and adjust manually if parts are cropped.
- `render_still(filepath, resolution=(1024,1024), samples=64, engine="CYCLES", transparent=False)`
- `render_turntable(filepath_prefix, objects, frames=8, radius=None, height=None, resolution=(768,768), samples=48)` — writes `<prefix>_00.png ... _NN.png`, one per angle; sets up its own camera/lighting/background, call after building the model. `height` is an offset above the objects' true geometric center (via `world_bounds`), not world Z=0. Leave `radius`/`height` as `None` (default) to auto-fit from the objects' actual size (`radius = size*1.7`, `height = size*0.15`) — pass explicit values only for deliberate manual framing control. (A fixed 6.0/3.0 default used to be hardcoded here; it silently clipped elongated props like a sword's blade tip out of frame — found in loop-iteration testing.)
- `export_glb(filepath, selected_only=False)`
- `save_blend(filepath)`

## Conventions worth keeping

- Assign materials to individual parts *before* `join_objects` — material slots merge correctly that way.
- One `build_hand_painted_material` call per distinct color region; reuse the returned material object across parts that share a color rather than rebuilding the node graph each time.
- Always call `set_world_background()` before rendering (or let `render_turntable` do it) — otherwise the world defaults to plain gray and stylized colors look flatter than intended.
