---
name: blender-stylized-3d
description: Generate 3D models in Blender from a text description or a reference photo — stylized hand-painted-PBR props/creatures (Workflows A/B), or a full game-ready realistic character (Workflow C: SDF sculpt on an anatomical reference body, wrapped base-mesh retopology, UVs, high→low normal/AO/curvature bakes, PBR texture set, UE-compatible skeleton + skinning + walk cycle, LOD0–LOD4, FBX/GLB export with a validation report). Use when the user asks to create, model, texture, rig or export a 3D asset/character, "make a 3D model of X", "turn this photo into a 3D model", or wants a .blend/.fbx/.glb.
---

# Blender stylized 3D generation

Builds stylized 3D assets by writing and running real Blender Python (`bpy`)
scripts against a headless Blender — no external AI 3D-generation service.
Two honest input paths:

- **Text description → 3D**: you write a one-off `bpy` script that assembles
  primitives/modifiers into the described shape, matching the toolkit's style.
- **Photo → 3D**: there is no neural reconstruction here. `scripts/image_prep.py`
  extracts a color palette, a grayscale heightmap, and a foreground silhouette
  from the photo using classic (non-ML) image processing, and the Blender side
  turns those into a **bas-relief** or **cutout/extrusion** stylized model —
  think "carved medallion" or "shadow-box cutout" of the photo, not a full
  turnaround-accurate 3D reconstruction. Say this plainly to the user before
  starting so expectations are set correctly.

Default style is **hand-painted PBR**: `bpy_stylized_kit.build_hand_painted_material`
fakes painted-texture look via baked-in AO cavity shading, warm/cool fresnel rim
tint, and noise-driven color/roughness variation — no image textures required.

**Honest fidelity ceiling**: Workflows A/B produce stylized blockout-tier
assets — correct silhouette/proportions, clean continuous topology,
procedural materials. For a *game-ready character* use **Workflow C**
below: it runs the full production chain (sculpt → retopo → UV → bake →
texture → rig → LOD → export) and passes engine-readiness checks. Anatomy
comes from a CC0 reference body and the topology from a wrapped base mesh,
so proportions and edge flow are right; likeness/personality, hair cards
and facial rigging are still an artist's job. Say so; don't oversell.

## Prerequisites

Blender must be installed (`brew install --cask blender` on macOS). Check with:

```bash
scripts/run_blender.sh --help >/dev/null 2>&1 && echo ok
```

`scripts/image_prep.py` needs Pillow in system Python3 (`pip3 install pillow` if missing).

## Workflow A — from a text description (quick/blocky primitive assembly)

Use for simple props, hard-surface objects, or fast rough iteration. For
anything organic or where the model needs to actually read as the described
character/creature, use **Workflow A2** below instead — plain primitive
assembly reads as a "bead necklace" of separate blobby shapes.

1. Read `reference/api.md` for the full `bpy_stylized_kit` function list.
2. Write a new script under a scratch/output dir (not inside `scripts/` —
   that directory is the reusable library, keep generated one-offs separate),
   importing the kit:
   ```python
   import sys, os
   sys.path.insert(0, "<repo>/blender-stylized-3d/scripts")
   import bpy_stylized_kit as k

   k.new_scene()
   body = k.add_primitive("ico_sphere", location=(0,0,1), name="Body")
   ...
   mat = k.build_hand_painted_material("Skin", k.hex_to_rgb("#E07A3E"))
   k.apply_material(body, mat)
   model = k.join_objects([body, ...], name="Creature")
   k.set_world_background()
   k.render_turntable("<out>/preview", [model], frames=4)
   k.save_blend("<out>/model.blend")
   k.export_glb("<out>/model.glb")
   ```
3. Decompose the description into primitives + modifiers (subsurf for
   organic/rounded forms, bevel for hard-surface edges, mirror for symmetric
   creatures/props). Assign one `build_hand_painted_material` call per
   distinct color region — reuse materials across parts that share a color.
4. Run it: `scripts/run_blender.sh <your_script.py>`
5. **Always render a preview and look at it** (`Read` the PNG) before calling
   the model done — bpy code that runs without error can still produce wrong
   geometry (wrong scale, inside-out normals, parts floating apart, etc.).
   Iterate: adjust the script, rerun, re-view.

## Workflow A2 — continuous polygon modeling (preferred for organic/recognizable characters)

Builds each major body part as ONE continuous lofted surface instead of a
stack of primitives — see `reference/polygon_modeling.md` for the full
technique. Preferred whenever the model needs to actually be recognizable
(a specific character/creature), not just a rough blob.

```python
import sys, os
sys.path.insert(0, "<repo>/blender-stylized-3d/scripts")
import bpy_stylized_kit as k

k.new_scene()

# torso + neck + head as ONE continuous piece — not a sphere-for-body +
# sphere-for-head. Each entry is (position, radius); add more points for a
# smoother silhouette, keep radius changes gradual between adjacent points.
body = k.build_profile_body([
    ((0, 0, 0.0), 0.05),   # taper to a point at the base
    ((0, 0, 0.3), 0.45),   # hips
    ((0, 0, 0.8), 0.50),   # chest
    ((0, 0.05, 1.3), 0.35),  # neck
    ((0, 0.1, 1.65), 0.42),  # head
    ((0, 0.15, 1.95), 0.05), # taper to a point at the crown
], segments=16, name="Body")

mat_main = k.build_hand_painted_material("Main", k.hex_to_rgb("#1B63D8"))
k.apply_material(body, mat_main)

# a second color region on the SAME continuous mesh (no separate object):
mat_belly = k.build_hand_painted_material("Belly", k.hex_to_rgb("#FBD7A8"))
k.assign_material_by_region(body, mat_belly, lambda p: p.y > 0.05 and p.z < 1.5)

# a continuous tapered limb, optionally bent at an elbow/knee
arm = k.add_tapered_limb((0.5, 0.05, 1.1), (0.75, 0.15, 0.55), 0.12, 0.09,
                          bend=(0.68, 0.1, 0.85), name="Arm")
k.apply_material(arm, mat_main)

# fuse into one fully watertight mesh (materials survive the union)
model = k.boolean_union([body, arm], name="Creature")
k.shade_smooth(model, auto_smooth_angle=40)

k.set_world_background()
k.render_turntable("<out>/preview", [model], frames=6)
k.save_blend("<out>/model.blend")
k.export_glb("<out>/model.glb")
```

Build each limb/tail/prop the same way (`add_tapered_limb`, or
`build_profile_body` for anything not a simple 2-3-point taper), assign
materials per-part before unioning (or `assign_material_by_region` for
color variation within one part), then either `boolean_union` everything
for a fully seamless mesh or leave parts as slightly-overlapping separate
objects if a union looks wrong on complex geometry — check `polygon_modeling.md`'s
"Joining parts" section for the tradeoff. Render and view before calling it
done, same as Workflow A.

## Workflow B — from a photo

1. Preprocess the image (system python, not Blender):
   ```bash
   python3 scripts/image_prep.py <photo.png> --outdir <out>/prep --colors 6
   ```
   Output: `palette.json` (hex colors), `heightmap.png` (grayscale), `silhouette.png`
   (foreground mask — heuristic background-difference threshold; if the photo
   has a busy/non-flat background, tell the user results will be rougher and
   suggest a pre-cutout PNG with real alpha for a clean mask).
2. Pick a technique based on what the user wants:
   - **Cutout / medallion relief** (recognizable silhouette, works for
     portraits, logos, characters, animals): `k.extrude_silhouette(mask_path,
     depth=..., resolution=...)`. Raise `resolution` (default 128, the smoke
     tests used 48) for smoother edges on faces/curved silhouettes.
   - **Bas-relief from the whole image** (landscapes, textured scenes): build
     a subdivided plane and `k.add_displace_from_heightmap(plane, heightmap_path,
     strength=...)`.
3. Apply `build_hand_painted_material` using colors from `palette.json`
   (`k.hex_to_rgb(...)`) instead of guessing colors.
4. Render a turntable preview and view it before finishing. If the silhouette
   mask looks wrong (check `silhouette.png` visually first — it's cheap), fix
   the mask (ask the user for a cleaner photo, or pass an explicit `threshold`
   to `image_prep.py`) rather than pushing bad geometry downstream.

## Workflow B2 — from multiple photos (real 3D, not a flat relief)

If the user has ≥2 consistently-framed photos of the same subject (e.g.
front + side, ideally + top), `carve_from_silhouettes` builds a genuine
volumetric shape via classical shape-from-silhouette (visual-hull)
reconstruction — see `reference/polygon_modeling.md` for the technique and
its real limits (concavities invisible from every view can't be recovered;
this is a property of the technique itself, not a shortcut). Still no
neural reconstruction, consistent with Workflow B.

1. `python3 scripts/image_prep.py --outdir <out>/prep --view front=<front.jpg> --view side=<side.jpg>` (add `--view top=<top.jpg>` for a 3rd view — meaningfully tightens the result vs. 2 views).
   **The photos must be consistently framed** — same subject scale/centering
   across all of them — or the carve comes out wrong (often just empty).
2. `carved = k.carve_from_silhouettes([{"mask": "<out>/prep/silhouette_front.png", "axis": "front"}, {"mask": "<out>/prep/silhouette_side.png", "axis": "side"}], size=2.0)`
3. Same as Workflow B from here: material from `palette.json`, render and view before finishing. Output is blocky at the silhouette edges by default — raise `resolution` or follow with `add_bevel`/`add_subsurf` for a softer look.

## Workflow C — game-ready realistic character (full production pipeline)

Use when the ask is a character that has to *work in a game*: clean
deforming topology, UVs, baked maps, PBR textures, skeleton + weights,
LODs, FBX/GLB. Full technique, numbers and pitfalls:
`reference/game_ready_character.md` — read it before changing anything.

```bash
pip install bpy scikit-image scipy pillow        # if no Blender install; or use Blender's python
python scripts/game_character/build_character.py <out> --res 2048   # ~5 min on 4 cores
python scripts/game_character/verify_export.py <out>/export/SK_Character.glb
cat <out>/export/report.json                                          # validation numbers
```

Stages (each re-runnable with `--from <stage>`): `highpoly` (SDF sculpt
on an anatomical CC0 reference body, clothing/hair/gear on top, 4.3 M tris)
→ `lowpoly` (a clean base topology *wrapped* onto the sculpt — booted
variant when the costume has boots — plus gear pieces, UVs, LOD2/LOD4)
→ `bake` (per piece: normal/AO/curvature/position, merged in one atlas)
→ `textures` (BaseColor, Normal GL/DX, ORM, Height) → `lookdev`
(engine-style material, eyes) → `rig` (63-bone UE-named skeleton, heat
skinning, rigid gear, walk cycle) → `export` (LOD0–4, FBX per engine, GLB,
`report.json` incl. watertightness and self-intersections per piece).

To make a *different* character, edit the data, not the pipeline: body
shape via the reference's morph targets (`reference_body.build`: muscle,
stature) or the procedural body (`humanoid.build(body="procedural")`),
outfit in `costume.py` (garment masks + `region_id` + `dress`), materials in
`texture.py` (recipes per region). Everything positional keys off the
landmarks (`landmarks.py`), not coordinates. Always look at: the clay sheet
of the high-poly, the true-edge wireframe close-ups (head, hands, feet,
crotch, shoulder) **with self-intersecting faces highlighted**, the textured
turnaround, the walk frames + stress poses, and the report
(`watertight: true`, `self_intersecting_face_pairs: 0` on every LOD).

## Style toolkit

- `build_hand_painted_material` — default look, see `reference/hand_painted_shader.md`
  for the technique and every tunable knob (`variation`, `cavity_strength`,
  `rim_strength`, `saturation_boost`).
- Low-poly: `add_decimate(obj, ratio=0.3-0.6)` + `shade_flat(obj)` instead of
  smooth/subsurf.
- Toon/outline: `add_solidify_outline(obj, thickness=0.02)` for an inverted-hull
  ink line on top of any material.
- `set_stylized_color_management()` is called automatically by `render_still` —
  it forces Blender's "Standard" view transform instead of the default AgX,
  because AgX desaturates flat stylized colors toward gray/white. Don't
  override `scene.view_settings.view_transform` back to Filmic/AgX for this
  style of asset.
- Lighting: `add_three_point_lighting` defaults are tuned low (key=120W) to
  avoid blowing out flat colors under Cycles — if a render looks washed out,
  lower energies further before suspecting the material. It also adds a dim
  underside bounce light by default — without it, a render's bottom-facing
  surfaces look flat/cut/near-black regardless of the actual geometry there
  (found via a false-alarm "asymmetric geometry" bug that turned out to be
  pure lighting — verify with vertex coordinates before trusting a render's
  read of symmetry, the same way `world_bounds`/framing bugs got caught).

## Verification checklist (do this every time)

1. Script ran with exit code 0 and no Python traceback in the Blender log.
2. Rendered preview PNG viewed with `Read` — geometry matches the ask, no
   inverted normals (black patches), no exploded/disconnected parts, colors
   aren't blown out to white or muddy-gray.
3. For photo-driven models: silhouette/heightmap visually checked against the
   source photo before spending render time on the 3D result.
4. `.glb` exported if the user wants to use the asset elsewhere (game engine,
   web viewer, AR) — `.blend` alone doesn't travel outside Blender.

## Live-viewing in the Blender GUI (optional)

`mcp_server/` wraps a running (non-headless) Blender session as an MCP server
so you can drive it live and the user can watch. See `mcp_server/README.md`
for setup — it's a separate, opt-in path; the headless workflow above is the
primary, always-available one and doesn't need it.

## Troubleshooting

- **`enum "..." not found` errors from `obj.modifiers.new(...)`**: Blender's
  modifier/enum names changed between versions (this happened with auto-smooth
  between 4.0 and 4.1+). Check `bpy.context.scene` in the error's enum list
  and update `bpy_stylized_kit.py`, don't work around it per-script.
- **Render looks washed-out / pale**: almost always light energy too high
  under Cycles (linear radiance clips to white past 1.0 regardless of view
  transform) — lower `key_energy`/`fill_energy`/`rim_energy`, not the material.
- **Silhouette mask is empty or inverted**: the background-difference heuristic
  in `image_prep.py` failed to estimate the background from the image corners
  (e.g. the subject touches a corner). Pass `--colors` aside, you can also
  call `make_silhouette_mask(img, path, threshold=<value>)` directly with an
  explicit threshold, or ask the user for a photo with a clear background.
