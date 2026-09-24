# Continuous polygon modeling (loft/boolean technique)

`build_profile_body`, `add_tapered_limb`, `boolean_union`, and
`assign_material_by_region` in `scripts/bpy_stylized_kit.py`. This is the
preferred technique for organic/recognizable characters — see
`SKILL.md`'s "Workflow A2" for when to reach for it over primitive assembly.

## Honest scope

This gets you from "stack of overlapping spheres and cones" to a clean,
continuous-topology **stylized blockout**: correct silhouette and
proportions, smooth single-surface bodies and limbs, no primitive seams.
It does not get you to hand-sculpted/hand-painted production-character
quality — that requires digital sculpting, manual retopology, hand UV work,
and hand-painted texture maps, a different (human-artist) pipeline
entirely. Don't oversell output built this way as more than a stylized
blockout/game-prop tier asset.

## How the loft works

`loft_profile(rings, segments, cap_start, cap_end, name)` is the low-level
builder everything else wraps:

1. Each ring is a dict `{"co": (x,y,z), "radius": r_or_(rx,ry)}`.
2. For each ring, a **tangent** is computed from its neighbors (average
   direction for interior rings, single-neighbor direction at the ends) —
   this is what lets a loft follow a curved spine, not just a straight one.
3. Two vectors perpendicular to the tangent (`basis1`, `basis2`) are built
   via cross products against a reference "up" vector (world Z, falling
   back to world X when the tangent is nearly vertical, so the basis never
   degenerates) — the same general idea as the `to_track_quat` calls
   already used for cameras/lights elsewhere in the kit, just computed
   manually here since raw vertex positions are needed, not an object
   transform.
4. `segments` vertices are placed around each ring using those two basis
   vectors, consecutive rings are bridged into quad faces, and the first/
   last ring gets an n-gon cap (skipped automatically when that ring's
   radius is ~0, since a zero-radius "ring" is already a point).
5. `bmesh.ops.remove_doubles` welds coincident vertices (this is what makes
   a zero-radius end ring resolve into a clean point/taper instead of a
   cluster of unwelded overlapping verts), then
   `bmesh.ops.recalc_face_normals` fixes winding/normal direction — don't
   try to hand-reason about face winding when adding new callers of
   `loft_profile`, just trust recalc.

`build_profile_body(spine, segments, name)` converts a friendlier
`[(position, radius), ...]` (or `(position, radius_x, radius_y)`) spine
into `rings` and calls `loft_profile`. Use this for anything that reads as
one continuous body part — a torso+neck+head is one spine, not a sphere
for the body and a separate sphere for the head.

`add_tapered_limb(start, end, start_radius, end_radius, segments, bend,
bend_radius)` is `build_profile_body` specialized for a 2- or 3-point
limb (straight, or bent once at an elbow/knee), capped at both ends so it
stands alone as a watertight capsule-like shape.

## Picking spine control points

- More points = smoother silhouette but more geometry; 4-6 points is
  usually enough for a torso+head, 2-3 for a limb.
- Radius changes should be gradual between adjacent points — a big radius
  jump over a short spine distance produces a sharp cone-like taper rather
  than a smooth bulge; add an intermediate point to soften it.
- **A straight linear taper to a near-zero tip reads as a sharp CONE, not a
  rounded cap** — found this rendering a humanoid's head during loop-iteration
  testing (a chest→neck→head spine tapering directly to a point produced a
  pointed dunce-cap head, not a dome). Use `rounded_cap_points(base_co,
  base_radius, tip_co, n=2)` between the last full-radius point and the tip
  to get a proper rounded cap instead: `spine = [..., (head_co, head_r)] +
  k.rounded_cap_points(head_co, head_r, top_co) + [(top_co, 0.02)]`.
- A point's radius can be `(rx, ry)` for a non-circular cross-section
  (e.g. a flatter front-to-back torso) — `rx` follows `basis1`, `ry`
  follows `basis2`; for a roughly-vertical spine, `basis1`/`basis2` land
  close to world X/Y, so `rx` reads as "left-right" and `ry` as
  "front-back" in that common case (verify with a render, not by
  assumption, if the spine bends far from vertical).

## Joining parts: overlap, or `boolean_union`

Two options for how separately-lofted parts (body, arm, leg, tail) meet:

- **Let them overlap slightly** (like the old primitive-assembly approach)
  — simplest, no risk of a boolean failure, but leaves a visible (if much
  softer than primitive-soup) seam where surfaces intersect.
- **`boolean_union(objects, name)`** — fuses them into one fully watertight
  mesh via the Boolean modifier (`UNION`, exact solver). Verified (see
  `scripts/_smoke_test_loft.py`) that Blender's boolean union **preserves
  each source object's per-face material assignment**, remapped into the
  merged object's combined material list — so build each part with its own
  `build_hand_painted_material` first, then union; you don't need
  `assign_material_by_region` just to keep per-part colors through a union.
  Boolean ops can still fail or produce artifacts on complex
  self-intersecting or non-manifold geometry — if a union produces a
  broken-looking result, check the two source meshes render cleanly on
  their own first before assuming the union call is at fault.

`assign_material_by_region(obj, mat, region_fn)` is for the different case
of wanting multiple colors on ONE already-continuous mesh without a second
object at all (e.g. a lofted body that's blue on top and peach on the
belly) — `region_fn` receives each face's world-space center and returns
whether it belongs to that material.

## Why extrude_silhouette no longer uses a Displace modifier

`extrude_silhouette` originally built a subdivided plane and pushed it with
a Displace modifier reading the mask as a height texture (`strength=depth,
midlevel=0.0`), then Solidify for thickness. It reliably produced a jagged
"fringe" of partial-height spikes all around the silhouette boundary —
confirmed by directly inspecting the evaluated mesh's vertex Z coordinates,
not just eyeballing the render (worth remembering as a technique in
general: a render that looks asymmetric or noisy is not proof the geometry
is — check the actual vertex/mesh data before trusting what the shading
shows you, the same lesson that caught the lighting-only issue documented
in `SKILL.md`'s style-toolkit notes). Raising `resolution` made the fringe
*worse* (finer but denser), and even a large, non-angle-limited `add_bevel`
(width 0.08, 6 segments, `limit_method="NONE"`) barely changed it — ruling
out both "just sample finer" and "smooth it after the fact" as fixes.

Root cause: Blender's Displace modifier bilinearly interpolates the image
texture at each vertex by default (`Texture.use_interpolation`). A raster
silhouette's boundary is essentially never aligned with the mesh's
rectangular vertex grid, so a large fraction of boundary vertices land
partway between "inside" and "fully outside" the mask and get pushed a
partial, inconsistent amount — a property of *interpolated sampling on a
grid that doesn't conform to the shape*, not of resolution or edge
sharpness, so no amount of subdivision or beveling fixes it.

Fix: `extrude_silhouette` now delegates to `extrude_silhouette_volume`
(`axis="top"`), which samples the mask as a hard binary value per grid
cell via `_sample_mask_foreground` in `scripts/bpy_stylized_kit.py` — no
texture interpolation anywhere, so boundary cells snap cleanly to full
height or none. The remaining edge
irregularity is honest, expected pixel-grid quantization (a blocky/faceted
boundary, worse at low `resolution`, better at high) — a completely
different and much smaller artifact than the old interpolation fringe.

## Live-bridge gotcha: Blender's Python process persists across calls

Iterating on `bpy_stylized_kit.py` through `mcp_server/send_to_blender.py`
against a live GUI session (see `mcp_server/README.md`) is fast, but the
Blender process — and its Python module cache — stays alive across every
call, unlike `scripts/run_blender.sh` which is a fresh process each time.
A plain `import bpy_stylized_kit` after the first call returns the already-
cached module, not a re-read of the file: edit the toolkit, re-run the
same test, and it silently keeps using the OLD code. This looks exactly
like a fix not working and cost real time to notice while diagnosing the
issue above (the first "verification" render after the fix still showed
the fringe — because it wasn't actually running the fixed code). Always
force a reload when testing a toolkit change against a live bridge:
`send_to_blender.py --with-toolkit` now does this automatically; doing it
by hand looks like `import bpy_stylized_kit as k; import importlib; importlib.reload(k)`.

## Multi-view silhouette carving

`carve_from_silhouettes(views, size, depth_margin, resolution, name)` in
`scripts/bpy_stylized_kit.py`. Real volumetric reconstruction from multiple
photos of the same subject, instead of `extrude_silhouette`'s flat relief
from one photo.

**Where this comes from**: researched how commercial neural 3D-generation
tools (Tripo AI's TripoSG, Microsoft's TRELLIS) actually work before adding
this — both are billion-parameter generative models trained on 500K+ 3D
assets; there's no algorithm to "reverse engineer" out of a trained deep
model, and matching their quality isn't achievable through procedural
scripting. But TRELLIS's core idea — fuse information from multiple views
into one 3D structure — has a genuine **classical, non-ML** counterpart:
**shape-from-silhouette / visual-hull reconstruction**, standard computer
vision since the 1990s. Each view's silhouette back-projects into a 3D
"generalized cone"; the intersection of those cones across views is the
object's visual hull. That's exactly what this implements — no neural net,
consistent with the rest of this skill.

**How it's built**:

1. `extrude_silhouette_volume(mask, axis, size, depth, resolution, name)`
   reads the mask via `bpy.data.images` + `image.pixels` (not PIL — PIL
   isn't available inside Blender's bundled Python, which is why
   `image_prep.py` runs under system `python3` instead). It samples a
   `resolution`×`resolution` grid of foreground/background, builds a 2D
   quad mesh of only the foreground cells in the plane perpendicular to
   `axis`, then `bmesh.ops.extrude_face_region` + `translate` sweeps that
   whole flat region `depth` units through `axis` — a proper single-pass
   extrusion of a (possibly multi-island, possibly holed) 2D region, so
   the only new geometry created is the true outer wall of the resulting
   prism (no leftover internal faces the way naively unioning many small
   per-pixel boxes would leave). This is a straight-prism **orthographic
   simplification** of the true perspective visual cone — appropriate for
   roughly-orthographic reference photos (zoomed-in, centered subject),
   not for wide-angle/close-up perspective shots.
2. `carve_from_silhouettes` builds one volume per view with the same
   `size`/`depth`, then `boolean_op(volumes, "INTERSECT", name)`.

**Hard requirement**: all views must be **consistently framed** — same
subject scale and centering across photos (e.g. all cropped so the subject
fills the same fraction of frame, roughly centered). `size` is shared
across every view's volume specifically so they overlap correctly at the
origin; mismatched framing carves a wrong shape (often empty, if the
volumes don't overlap at all) rather than failing loudly, so a first carve
attempt is worth a quick render before trusting it.

**Gotcha found in loop-iteration testing**: a first carve of a non-trivial
shape (a synthetic toy-car silhouette, front+side) rendered completely
flat white despite `apply_material` being called with a real red material —
`boolean_op`'s `INTERSECT` (via Blender's EXACT boolean solver) left the
result mesh with a phantom empty material slot 0, and `apply_material`
used to only *append* a new material slot without repointing any faces at
it, so the red material sat unused in slot 1 while every face still
pointed at the empty slot 0. Confirmed by inspecting `obj.material_slots`
directly (`[None, 'CarPaint']`), not by guessing from the render. Fixed:
`apply_material` now repoints every face to the slot it just appended —
see its docstring in `scripts/bpy_stylized_kit.py`. This could have
affected any `boolean_op` result (union or intersect), not just carving.

**Fundamental limits of the technique itself** (not this implementation):

- Needs ≥2 views; 2 (front+side) leaves real ambiguity along whichever axis
  wasn't captured (a limb positioned diagonally between the two silhouette
  planes can vanish or balloon) — 3 views (front+side+top) meaningfully
  tightens the result.
- **Concavities invisible from every view can't be recovered** — a visual
  hull is always convex-ish along each silhouette direction; a dimple or
  hollow that doesn't show up in any single silhouette's outline is gone.
  This is the textbook limitation of shape-from-silhouette, true of every
  implementation of the technique, not a shortcut taken here.
- Output is blocky/voxel-stepped at the silhouette edges (visible in
  `_smoke_test_carve.py`'s renders) — raise `resolution` for smoother
  edges at the cost of more geometry, or follow with `add_bevel`/`add_subsurf`
  for a softer stylized read.

## Framing renders correctly

`render_turntable`/`frame_camera_on_objects` use `world_bounds(objects)` —
the true world-space bounding box computed from each object's actual mesh
geometry — as the camera's look-at center, not the object's origin. This
matters specifically for lofted meshes: `loft_profile` places vertices at
absolute world coordinates and leaves the object's own origin at world
`(0,0,0)`, so anything that used `obj.matrix_world.translation` as a stand-
in for "where the geometry is" would aim the camera at the wrong point
(this was a real bug here — a first version of `render_turntable` did
exactly that and cropped renders to the bottom third of the model until
`world_bounds` replaced the origin-averaging).

That fixed *where* the camera looks, but not *how far back* it needs to be.
`render_turntable`'s `radius`/`height` also default to `None` and auto-fit
from `world_bounds`' `size` (the largest bounding-box dimension) rather
than a fixed number — a hardcoded radius/height tuned for one kind of
subject (character-sized, roughly as tall as it is wide) silently clips a
very differently-shaped or -sized object out of frame. Found this exact
failure in loop-iteration testing: a stylized sword prop (tall and thin,
not character-proportioned) rendered with its blade tip cropped off at a
fixed radius/height that had worked fine for every character test so far.
Leave `radius`/`height` as `None` unless deliberately framing a specific shot.
