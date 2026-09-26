# Game-ready character pipeline (Workflow C)

`scripts/game_character/` — a complete, code-only production pipeline for
realistic humanoid game characters: sculpt → retopology → UVs → bake →
PBR textures → skeleton + skin + animation → LOD chain → FBX/GLB export →
validation report. Every stage is a plain Python module driving Blender
(`bpy`), so it runs headless, on CI, or in a cloud container.

**The character is data.** A JSON *spec* (`spec.py`, presets in
`presets/*.json`) sets the body (MakeHuman sex/age/muscle/weight/height,
ethnicity, any of ~250 named morph modifiers), skin/eye/hair colors, hair and
beard style, an outfit of garment layers from a library (shirt, trousers,
vest, belt, boots/shoes, gloves, bracers — each with shape options and a
material), and separate pieces (ponytail, bun, long beard, horns, tusks,
pauldrons, pouch). Every stage reads the same spec, so one code path builds a
ranger, a scout, a dwarf, an orc or a demon. The agent-facing workflow
(request → spec → preview → build → review) is the
[`humanoid-character` skill](../skills/humanoid-character/SKILL.md).

```bash
# one command, all stages (≈15 min at 2K textures on 4 CPU cores, 7 of them the sculpt)
python scripts/game_character/build_character.py <out_dir> --spec orc_warrior --res 2048
# 1–2 min clay preview of a spec before committing to a build
python scripts/game_character/preview_spec.py my_character.json preview.png
# re-run from any stage after an edit (the driver refuses a shape change after highpoly)
python scripts/game_character/build_character.py <out_dir> --from textures --res 4096
# check the shipped files stand on their own
python scripts/game_character/verify_export.py <out_dir>/export/SK_Character.glb
```

Runs under Blender's bundled Python *or* the `bpy` pip wheel (`pip install bpy
scikit-image scipy pillow`) — the pip route is what works on machines with no
Blender install. EEVEE/Workbench need an EGL/OpenGL context that headless
containers usually lack; everything here renders and bakes with **Cycles on
the CPU**.

## Stage map

| Stage | Module(s) | Output | Senior checklist it satisfies |
|---|---|---|---|
| spec | `spec.py`, `presets/*.json`, `preview_spec.py`, `list_modifiers.py` | `spec.json` | the brief as data: validated, stored with the build |
| highpoly | `reference_body.py`, `landmarks.py`, `sdf.py`, `humanoid.py`, `costume.py`, `pieces.py`, `build_highpoly.py` | `highpoly.npz` (~4 M tris) + `highpoly_<piece>.npz` | forms → planes → details; cloth with thickness and hems; ~1 mm detail |
| lowpoly | `wrap.py` (reference body) or `retopo.py` (procedural), `piece_lowpoly.py`, `uvs.py`, `build_lowpoly.py` | `lowpoly.blend` (LOD0/LOD2/LOD4 + pieces, UVs) | quads with animation loops, closed, no self-intersections, seams hidden, texel density |
| bake | `bake.py`, `build_bake.py` | `bakes_<res>.npz` | synced tangents, cage/ray distance, dilation, bake cleanup |
| textures | `texture.py`, `build_textures.py` | `textures/*.png` | PBR ranges, ID-driven smart materials, detail normals, ORM packing |
| lookdev | `lookdev.py`, `eyes.py`, `build_lookdev.py` | `lookdev.blend` | engine-style material (textures only), separate eyes |
| rig | `rig.py`, `build_rig.py` | `rigged.blend` | UE-compatible skeleton, twist bones, ≤4 influences, test walk |
| export | `lods.py`, `build_export.py`, `verify_export.py` | `export/*.fbx`, `*.glb`, `report.json` | LOD0–LOD4, per-engine files, automated validation |

---

## 1. Sculpt — SDF "digital clay" (`sdf.py`, `humanoid.py`, `costume.py`)

A code-only stand-in for ZBrush. The model is an ordered list of operations
(smooth union / smooth subtraction / intersection / arbitrary field edits)
that can be evaluated on any grid.

- **Anatomical base (default)**: the body is the limit surface of the
  MakeHuman base mesh (CC0) turned into a signed distance field
  (`reference_body.MeshSDF`: dense subdivision samples + normals,
  point-to-plane near the surface, a coarse inside/outside grid filled by
  connectivity for far points). Its shape is blended **exactly like
  MakeHuman's macro sliders** (`reference_body.macro_targets`): the
  `universal-<sex>-<age>-<muscle>-<weight>` targets with bilinear weights
  over sex × age (baby 1 / child 11 / young 25 / old 90 years) × muscle ×
  weight, `<ethnicity>-<sex>-<age>` targets for the face/body blend, the
  `proportions` targets, then any named **modifier** (`ear-shape-pointed`,
  `upperlegs-height-decr`, … one side-less stem drives both sides), then a
  uniform scale to `height_m`. Targets are fetched on demand from
  MakeHuman's repository and cached (a new body needs ~20–60 small files).
  Anatomy is correct by construction (skull, face, hands, feet, muscle
  masses, age proportions), which primitive sculpting never reached. The
  skeleton and finger chains come from the reference's joint helpers;
  `landmarks.py` measures eyes, nose tip, lips, chin, ears, neck, crotch and
  **limb radii** (upper arm, forearm, thigh, calf, a radius-vs-height leg
  profile) on the surface, and every later stage (garment edges and fold
  sizes, hairline, piece placement, texture color zones, eye bones, LOD
  protection) keys off those landmarks instead of hard-coded coordinates —
  that is what lets one costume library fit a 1.10 m goblin and a 1.96 m orc.
- **Procedural body** (`body="procedural"`) is the primitive sculpt below,
  tuned to 33 ANSUR-II anthropometric targets (`anthropometry.py`).
- **Primitives**: oriented ellipsoids (muscle masses, fat pads), round cones
  (limbs, fingers, bones), round boxes (buckle, pouch, soles), tori (boot
  cuff), `Custom` (any distance function, e.g. eyelids, sole outline, the
  head loft).
- **Smooth booleans** (polynomial smin, blend radius `k`) make masses flow into
  each other like clay. Rule of thumb: `k` ≈ 30–60 % of the smaller form's
  radius for muscles, 2–5 mm for crisp anatomy (lips, lids, ear folds).
- **Anatomy order**: skeleton landmarks → big masses (ribcage, pelvis,
  cranium) → muscles on top (pecs, lats, deltoids, biceps/triceps, quads,
  hamstrings, calves) → carves (eye sockets, spine groove, nostrils).
- **Head = planes first**: a loft of horizontal superellipse sections driven
  by profile curves (front-view width, side-view front and back silhouettes,
  "boxiness" per section — `HEAD_PROFILE`). Stacking ellipsoids for a head
  does not converge; silhouettes-first does. Features (brow, nose, lips,
  eyelids, ears, jaw corner, cheek pads) go on top.
- **Eyelids** are a sphere slightly larger than the eyeball with an
  almond-shaped opening cut through it along the view axis — gives the lid
  margin real thickness, which is what catches light.
- **Cloth = offset shells**: inside a garment's region mask the field is
  lowered by thickness + ease, so the surface grows outward; the mask's
  ~1–2 mm falloff at hems/cuffs/neckline creates the ledge a bake turns into a
  crisp layered edge. Garments stack (shirt → trousers → jerkin → belt).
- **Folds** are displacement in garment-local coordinates (distance along the
  limb × angle around it, `tube_coords`), with **ridged noise stretched around
  the limb** and amplitude concentrated where cloth compresses: inner elbow,
  back of the knee, stacking above cuffs/boot tops. Pure sine folds read as
  corrugated cardboard — don't.
- **Hair** is a sculpted "hair cap": offset above a hairline curve (by azimuth
  around the head), feathered edge, strand clumps as anisotropic noise along
  the combing direction.
- **Polygonization**: `sparse_polygonize` — coarse pass finds surface bricks,
  only those are evaluated at 1–1.2 mm (in parallel), marching cubes per
  brick, welded. A whole clothed character at 1.2 mm ≈ 4.3 M triangles in
  ~2 min / < 2 GB instead of a 20+ GB dense grid.

### 1b. The costume library (`costume.py`) and pieces (`pieces.py`)

- **A garment is a mask + an edit + a material.** Each garment type is a
  function of the body's landmarks and the spec options: the shirt mask is a
  torso region from the neck base down to below the belt line (below the
  hip joints for an untucked shirt), sleeve ends at a fraction of the arm
  length (`arm_param`: long = at the wrist, short = 30 % down the upper arm,
  none = at the shoulder), the neckline crew/V/scoop; trousers end at the ankle, below the knee
  or mid thigh; boots at ankle/calf/knee height from the leg profile. Sizes
  (ease, fold amplitude and wavelength, cuff and sole thickness) scale with
  the limb radii and stature, so the same code fits any body.
- **Layering is fixed**, like dressing a real person: shirt → trousers →
  vest → bracers → gloves → belt → footwear, an untucked shirt moved above the
  trousers. Each layer offsets the model built so far, so a vest over a
  shirt stands off the shirt, not the skin.
- **A region registry** (`register(name, kind, material, mask)`) records every
  layer in build order; `region_id(P)` evaluates them on any point set (the
  texture stage calls it per texel), last layer wins. Kinds (`cloth`, `hair`,
  `footwear`, `sole`, `metal`, `piece`) drive the wrap allowances and the
  booted template; materials (`linen`, `cloth`, `wool`, `leather`, `metal`,
  `fur` + color) pick the texture recipe.
- **Hair styles** are thickness profiles over the hairline curve (buzz,
  short, mohawk = a midline strip; receding/straight/rounded hairlines);
  bald skips the layer. A short beard is a sculpted shell on the jaw with a
  feathered top edge; stubble is texture only.
- **Pieces are separate SDF models** (`pieces.py`), not edits of the body:
  ponytail and bun (tubes seated on the back of the head with a clearance
  from the neck), long beard (tapered tube from the chin, held clear of the
  chest), horns (swept tubes: curved, straight, ram spiral), tusks, pauldrons
  (a spherical cap + rolled rim + rivets, axis perpendicular to the upper arm
  and tilted up), pouch. Each piece records its bounds, bind target
  (a bone, or averaged skin weights under it), material and how its game mesh
  is made (see 2c); each is polygonized into its own `highpoly_<name>.npz`.

## 2. Retopology

Two routes, picked by the body the sculpt was built on.

### 2a. Reference body: wrap a base topology (`wrap.py`) — the default

Studios don't retopologize every realistic character from scratch: they keep
one animation-tested **base mesh** and wrap it onto each new sculpt or scan
(R3DS Wrap, shrink-wrap + relax). Edge loops stay where the face and joints
need them, UV seams stay hidden in the same places, and every character built
this way shares one topology (weights, blend shapes, UV masks carry over).

The template is the MakeHuman base mesh (CC0): 13 378 quads, closed (eye
sockets and a mouth bag included), no self-intersections, UV islands for
torso+legs, head, arms, feet, mouth and sockets. The sculpt sits on its limit
surface, so every template vertex has an exact anchor:

1. **Anchor**: Catmull-Clark limit position + normal of each control vertex
   (one subsurf level with *limit surface* on; Blender keeps control vertices
   first in the result).
2. **Offset to the outside of the sculpt**: ray from 4 mm inside along the
   limit normal; the first exit through the high-poly is the outer surface of
   whatever was sculpted on top. A hit counts only if it faces the same way
   and lies within the **material's allowance**, read from the same region
   registry that sculpted the costume: garments/footwear/pieces' contact
   28 mm, hair 45 mm × head scale, and **bare skin casts no ray at all** —
   the sculpt there *is* the template's own limit surface, so its offset is
   exactly 0. (With a 4 mm skin allowance, rays from a closed lip line went
   through the other lip; on a heavy-jawed orc that crossed 180 face pairs.)
3. **Interior islands** (eye sockets, mouth bag — the small UV islands inside
   the head) and a 2-ring band around them (lids, lips) stay on the body,
   behind eyeballs and lips.
4. **Covered detail is re-cast from a smoothed base**: where a garment stands
   off the body (trousers bridging the gluteal cleft) rays from the detail fan
   out and land out of order; there the template is Taubin-smoothed first and
   rays are cast from that base.
5. **Untrusted offsets** are filled harmonically from their neighbors,
   **mirror-averaged** (the game mesh stays symmetric, which also lets the LOD
   decimator work symmetrically), and **no vertex may cross the mirror plane**:
   with the left half at x ≥ 0.6 mm and the right at x ≤ −0.6 mm the halves
   cannot pass through each other where the sculpt bridges the midline.

**Booted variant.** Toes cannot be laid onto a boot without folding: fixed
topology, five separate digits, one toe box. Every smoothing/inflation scheme
tried left hundreds to thousands of crossing faces (Taubin, uniform and
cotangent implicit fairing + normal offsets, balloon inflation). Studios keep
a "shoe" variant of the base mesh for this, and so does `boot_feet`: each
foot is cut at the closed metatarsal-head edge loop (32 edges, the most
distal ≥ 24-edge loop behind the ball joint) and capped with a domed quad
grid built as a Coons patch of that ring (10 × 6, mirror-exact on both feet).
The wrap then stretches that toe box over the sculpted boot. The variant is
picked by the spec: any footwear layer → booted, barefoot → toes kept.

Result: LOD0 ≈ 11.9 k verts / 23.8 k body tris, closed, 0 non-manifold, 0
degenerate, **0 self-intersecting face pairs** on every preset (see the
gallery in `examples/game_character/`); the stage takes ~20–60 s including
the pieces.

### 2c. Pieces: three low-poly recipes (`piece_lowpoly.py`)

Separate pieces have no template, so each declares how its game mesh is made:

- **`box`** (pouch): an 8-vertex cage from the piece's own frame,
  projected onto the high-poly; LOD4/2/0 = the cage and 1/2 Catmull-Clark
  subdivisions, each re-projected.
- **`dome`** (pauldrons — thin shells): a two-layer spherical-cap cage
  (8 around × 2 rings, outer and inner layer + rim) generated from the
  piece's analytic frame (center, axis, radius, thickness, cap angle).
  LOD4 is the cage, LOD2/LOD0 are 1/2 *simple* subdivisions followed by
  exact placement: every vertex carries a `shell`
  attribute (1 outer, 0 inner, interpolated by the subdivision) and is put
  back at radius `r + t·shell` and, on the rim, at the cap angle. Projecting
  a 5 mm shell onto its high-poly folded it (both layers snapped to the
  nearest side), and classifying layers by normal failed at the rim.
- **`remesh`** (organic: hair, beard, horns, tusks): **QuadriFlow** per
  connected part, at unit scale, with a face budget by area share; the result
  is shrink-wrapped to the high-poly, checked for crossing faces, and falls
  back to a self-checking collapse decimation of the high-poly if it folds.
  LOD2/LOD4 are decimations of LOD0 to 25 % / 8 % (never below 48 / 24
  triangles).

All pieces are UV-unwrapped into the body's atlas and carried through the
bake, texture, rig and LOD stages with the body.

### 2b. Procedural body: a designed cage (`retopo.py`)

The primitive-sculpted body (`humanoid.build(body="procedural")`) has no
template, so its topology is laid out like a manual retopo:

- **Ring counts** (cage level; LOD0 doubles them): trunk + head 16, arms and
  legs 10, fingers/thumb 4, palm 10; head azimuth slots packed toward the
  front (`HEAD_AZ`).
- **Joint loops** bracket every joint — ≥ 3 loops across each bend after one
  subdivision.
- **Junctions** (all quads): crotch split 16 → 2 × 10, the arm plugged into a
  2 × 3-face hole in the torso side, hand rings rotated to give 5 dorsal + 5
  palmar verts, four finger tubes sharing their webbing edges, quad-grid caps.
- **Face loops**: two insets around each eye continuing into a closed socket
  "bag" behind the lids (so the mesh is watertight), one around the lips.
- **Placement**: ray-cast cage, Catmull-Clark subdivision, re-projection +
  tangential relax; radial from the bone for limbs/trunk, nearest-point for
  palm/fingers/crotch; spike and collapse guards (including quad diagonals).

This cage is tied to the procedural body's proportions and pose; on the
reference body (elbows bent 46°, different head) it folded badly, which is
why the reference route wraps a template instead.

## 3. UVs (`uvs.py`)

- Seams where a character artist hides them: back of the head to the crown
  center, neck base, torso sides (front/back split below the armpit),
  shoulder rings, arm undersides, wrists, palmar side of hands and every
  finger, crotch rings, inner legs, boot tops.
- Wrapped template: seams are the template's UV discontinuities (hidden
  where MakeHuman's layout hides them); the mesh is re-unwrapped with ABF
  along them and packed by this pipeline. Designed cage: seams are authored on
  the cage and every subdivision LOD inherits the UVs. Either way decimated
  LODs carry the UVs along, so **one texture set serves every LOD**.
- **Gear pieces share the atlas**: the pouch is smart-projected and packed
  together with the body (multi-object edit), one material, one draw call.
- Texel density equalized (`average_islands_scale`), then the head islands
  get **1.6×** (the face is what players look at), then packed (margin 0.35 %
  of the atlas ≈ 7 px at 2K / 14 px at 4K).
- Targets: coverage 60–75 % (this layout: 63 %), zero overlap (mirrored UVs
  would stop asymmetric wear/decals), checker test uniform on body and denser
  on the head.

## 4. Baking (`bake.py`)

- **Triangulate the low-poly before baking** and ship that triangulation:
  the engine rebuilds the same MikkTSpace tangents the map was baked with.
- Selected-to-active, cage extrusion ~1 cm, max ray ~2.5 cm: long enough for
  the buckle/pouch, short enough not to catch the wrong surface in armpits,
  crotch and between fingers.
- Maps: tangent normal (OpenGL +Y), object-space normal, AO (20 cm), curvature
  (Cycles *pointiness* of the high-poly via an emission bake), object-space
  **position** (float) — the key to texturing, see below.
- **Bake cleanup** (`texture.clean_normal_bake`): texels whose tangent normal
  points sideways/backwards (z < 0.15: a ray hit the wrong surface — nostrils,
  under a flap) are refilled from good neighbors by normalized convolution.
  ~0.5 % of texels on this character.
- **Bake per piece, then merge** ("match by mesh name" in Substance/Marmoset
  terms): the body low-poly bakes only from the body high-poly, the pouch
  only from the pouch high-poly, so a body texel next to the pouch never
  catches the pouch's surface. All high-polys stay render-visible, so AO
  still gets the contact shadow of the pouch on the hip. The per-piece maps
  are merged in the shared atlas by nearest UV island (each piece keeps its
  own dilated margin), and the merged `part` map tells the texture stage
  which texels are pouch — the leather recipe no longer leaks onto the
  trousers around it.
- DirectX-convention engines (Unreal): use `*_Normal_DirectX.png` (green
  flipped). Unity/glTF/Blender: OpenGL.

## 5. PBR textures in numpy (`texture.py`)

Every texel knows its 3D position (position bake), facing (object normal),
occlusion (AO) and convexity (curvature). Smart materials are then just
functions:

- **Material ID** = `costume.region_id(P)`: the *same masks that sculpted the
  garments*, evaluated per texel — lines up with the hems in the bake exactly,
  like an ID map baked from high-poly vertex colors; pieces come from the
  bake's `part` map instead (exact, no mask). Soft-blended at borders
  (0.6 px) to avoid aliasing.
- **A material library, dispatched per region** by the region's kind and the
  spec's `material` + `color` (albedo / roughness / metal / height):
  - *skin* (spec: color, lips, freckles, stubble, roughness): base tone, the
    three facial color zones derived from it (yellowish forehead, red middle
    third: nose/cheeks/ears, blue-grey beard shadow scaled by `stubble`),
    lips, painted eyebrows, freckles, pores (cellular noise) in roughness +
    height, oily T-zone;
  - *hair* (caps, beard, ponytail, bun): anisotropic strand noise combed
    front-to-back on the crown and downward on the sides, beard and pieces;
    a feathered hairline where darkened scalp shows between strands;
  - *linen* / *cloth*: plain weave in UV space at physical scale (texel size
    from the position map), slubs, sweat/dirt at pits/collar/cuffs,
    AO/cavity grime (cloth: finer, cleaner);
  - *leather*: grain, wrinkles, darker cavities, lighter + smoother worn
    edges from curvature, scratches, **stitch rows at a fixed distance from
    the garment border measured in 3D** (KD-tree), mud rising from the
    ground; grain size, wear, roughness, mud and stitch distance per garment
    (`LEATHER_USE`: a belt is finer and more worn than a vest);
  - *wool* twill, *fur* (strands and tufts, matte), *metal* (measured
    reflectance for brass/iron/steel/silver/gold/bronze, polished edges,
    tarnish in cavities, grime), *horn* (growth rings banded by distance from
    the head, streaks, polished edges; horns and tusks), rubber *sole*.
- **Detail normals**: the height channel → tangent-space normal (gradients in
  UV space divided by meters-per-texel) → combined with the baked normal by
  **Reoriented Normal Mapping**.
- **Outputs**: `BaseColor` (sRGB), `Normal_OpenGL` / `Normal_DirectX`,
  `ORM` (R = AO, G = roughness, B = metallic — glTF/UE packing), `Height`,
  `SkinMask` (subsurface weight for the preview/engine), `T_Eye_BaseColor`.
- PBR sanity: albedo never below ~0.02 or above ~0.95 sRGB, dark leather
  0.15–0.30, skin 0.45–0.80, metals use measured reflectance with metallic 1.

## 6. Skeleton, skinning, animation (`rig.py`)

- **Skeleton**: UE mannequin naming (`root`, `pelvis`, `spine_01..03`,
  `neck_01`, `head`, `clavicle/upperarm/lowerarm/hand_{l,r}`, 3 phalanges ×
  5 fingers, `thigh/calf/foot/ball`, `upperarm/lowerarm/thigh/calf_twist_01`,
  `eye_{l,r}`) — 63 bones, single root at the origin. Unreal's IK Retargeter and
  Unity's Humanoid avatar map these names automatically.
- **Roll convention**: every hinge flexes about its **local X** (Z points
  back/dorsal). `+X` on thigh = leg back, `+X` on calf = knee bend, `−X` on
  lowerarm/fingers = flex, `+Z` on the left upperarm lowers it from the A-pose.
- **Skinning** = bone heat (Baran & Popović 2007) implemented directly:
  `(L + M·H) w = M·H·p` with a clamped cotangent Laplacian, vertex areas M,
  heat `H = c/d²` from the nearest *candidate* bone. Candidates come from
  per-vertex part labels (a thigh vertex can't be pulled by the other thigh,
  the torso side can't be claimed by the hanging arm) — the job Pinocchio's
  visibility test does. On the wrapped template the labels come from the
  nearest bone segment (arm and leg roots slightly penalized, never the other
  side's limb) cleaned by majority vote over the mesh; on the designed cage
  from the cage's own labels. One sparse LU, one solve per bone: ~2 s.
- **Pieces are rigid.** Hair, beard, horns and tusks are bound 100 % to the
  `head` bone. Hard gear resting on the skin (pouch, pauldrons) gets one
  weight set for the whole piece — the average of the skin weights under it
  (a pouch uses only its top 20 %, the belt loop; a pauldron all of it) — so
  it rides the body rigidly instead of bending with every skin vertex it
  touches. The pelvis bone's heat segment spans both hip joints
  (`EXTRA_SEGMENTS`), otherwise the thigh claims the side of the hip and a
  belt pouch swings with the leg (71 % thigh before, 65 % pelvis after).
  Every piece then joins the body mesh of each LOD (one skinned mesh and one
  draw call per LOD).
- **Twist bones** take a linear share (up to 60 %) of their parent along the
  segment — the candy-wrapper fix for forearm/upper-arm/thigh/calf twist.
- **Engine cleanup**: ≤ 4 influences per vertex, weights < 0.01 pruned,
  normalized to 1. LOD1–4 get weights by Data Transfer from LOD0 + the same
  cleanup. Eyes are rigid-skinned to the eye bones (exported as skinned meshes
  so eye tracking works).
- **Test animation**: an in-place 32-frame walk (legs/arms in counter phase,
  swing-phase knee flexion, pelvis bob + yaw, spine counter-rotation) exported
  with the files; stress poses (arms raised, deep squat, fists) are the
  deformation check.

## 7. LODs (`lods.py`)

| LOD | How (wrapped template) | Tris (Ranger, incl. pouch) | Typical screen size |
|---|---|---|---|
| LOD0 | the wrapped base mesh, triangulated | 24.5 k | close-up / hero |
| LOD1 | LOD0 collapse-decimated 50 % | 12.3 k | 0.5 |
| LOD2 | LOD0 decimated to 25 % | 6.1 k | 0.25 |
| LOD3 | LOD2 decimated 50 % | 3.1 k | 0.12 |
| LOD4 | LOD0 decimated to 6.25 % | 1.5 k | 0.06 / crowds |

(The designed-cage route uses cage × 2 / cage × 1 / cage subdivisions for
LOD0/2/4 instead.) Decimation is symmetric, slowed down on the face and
hands with a protection group whose weight is capped below 1, and
**self-checking**: a collapse can fold a narrow crease (gluteal cleft,
crotch) into itself, so each result is tested for crossing faces and the
source vertices around any crossing are frozen and the decimation redone
(one retry with a few dozen frozen vertices is typical). Pieces bring their
own LOD0/LOD2/LOD4 (section 2c) and are joined into each body LOD; LOD1 and
LOD3 are made from the joined mesh.
Eyes: 720-tri spheres on LOD0–2, 168-tri on LOD3–4. All LODs share the
material and UV layout.

## 8. Export (`build_export.py`, `verify_export.py`)

- `SK_Character.fbx`: skeleton + LOD0–LOD4 + eyes named `*_LOD#` (Unity builds
  the LODGroup automatically) + the walk take. Settings: `FBX_SCALE_ALL`,
  −Z forward / Y up, tangent space on, no leaf bones, Y/X bone axes, no mesh
  modifiers baked (the armature binding travels as a skin).
- `SK_Character_LOD#.fbx`: one per LOD for Unreal's LOD import (LOD0 carries
  the animation).
- `SK_Character.glb`: LOD0 + eyes + skeleton + walk + textures (2K JPEG
  color/ORM, PNG normals) for web/engine preview.
- `verify_export.py` re-imports each file into an empty scene and checks
  skeleton, skin, animation and embedded images survive.

## 9. Validation report (`export/report.json`)

Per LOD: tris/verts, quads/n-gons, **watertight** (no boundary and no
non-manifold edges), **self-intersecting face pairs within a mesh piece**
(BVH overlap of faces that share no vertex) reported separately from
**contacts between pieces** (gear resting on / sunk into the body is how game
gear is built), degenerate faces, loose verts, UVs inside 0–1, UV coverage and
overlap, max influences, unweighted verts, normalized weights, applied
transforms. Source topology: quad ratio, poles by valence, quad-angle
deviation. Skeleton: bone count, single root at origin, required humanoid
bones present. Textures: sizes and power-of-two.

Current build, all five LODs: watertight, **0 self-intersecting face pairs**,
0 non-manifold, 0 boundary, 0 degenerate, 0 n-gons, UVs in 0–1, ≤ 4
influences, 0 unweighted, weights normalized; UV overlap 0 on LOD0/1/3 and
< 0.2 % on the decimated LOD2/LOD4 (collapses across UV seams). LOD0 source:
100 % quads, 136 × valence-3 and 120 × valence-5 poles, median quad-corner
deviation 8°. `verify_export.py` re-imports GLB and FBX: 63 bones, skinned,
walk take, textures embedded in the GLB.

## Budgets worth knowing

| Target | LOD0 tris | Textures |
|---|---|---|
| Mobile hero | 8–15 k | 1–2 × 1K–2K |
| Current-gen NPC | 20–40 k | 2K–4K set |
| Current-gen hero | 40–100 k (+ hair cards) | 2–4 × 4K sets (head separate) |
| Cinematic | 100 k+ | UDIMs 4K–8K |

## Honest limits of this pipeline

- **Body space** is MakeHuman's: humans of any sex, age, build and ethnicity
  and near-human races via morph modifiers. Extreme cartoon proportions,
  tails, wings, extra limbs and non-bipeds are out of reach of one wrapped
  template.
- **Likeness** of a specific person is not a goal; faces are MakeHuman
  faces steered by modifiers.
- **Hair** is a sculpted cap (buzz/short/mohawk) plus rigid pieces
  (ponytail, bun, long beard) with painted strands — not hair cards or
  strands, no long flowing hair or braids.
- **Garments are skin-tight shells** (a few mm) from a fixed library: fine
  for fitted clothing, but anything that bridges the legs or hangs free
  (skirts, robes, capes, long coats) needs its own mesh pieces and cloth
  sim or bones. No helmets or masks yet.
- **Face**: the template has a mouth bag but there are no teeth/tongue
  meshes, no facial rig or blend shapes (a jaw/brow/lip rig or ARKit-style
  shape keys would be next — the fixed template topology makes them
  reusable across characters). Eyes are simple spheres.
- **Rest pose** is the reference's: arms 41° down, elbows bent 46° with the
  forearms forward. Engines retarget from it fine, but a stricter A-pose
  (elbows ~15°) would need the reference reposed before sculpting.
- **Materials** are procedural; realistic, but no scanned skin/fabric data.
  **Subsurface scattering** is provided as a mask; the look depends on the
  engine's skin shader.
- Belt buckles are part of the body sculpt; everything else that is hard
  (pouch, pauldrons, horns, tusks) is a separate piece.

## Lessons learned building it (bugs worth not repeating)

- **Offset surfaces clip at primitive blocks.** Cloth/hair offsets move the
  surface up to a few cm from where a primitive put it; primitives must be
  evaluated with a margin ≥ the edit band or the offset surface is sliced flat
  at the block boundary (showed up as a flat "plate" on the head).
- **A loft is not a distance field.** Superellipse sections have the right
  zero set but a collapsing metric near the end caps; anything built on top
  (hair offset) balloons there. Fix: true distance to dense surface samples
  (KD-tree), sign from the implicit function, gradient-normalized value near
  the surface.
- **skimage marching-cubes winding is already outward** for a negative-inside
  SDF. Flipping it inverted every high-poly normal — the tangent map came out
  olive instead of lavender. Check the signed volume.
- **Garment masks must exclude everything else explicitly** — the shirt mask
  reached the underside of the chin and grew a plate there.
- **Thighs fuse once trousers add ease**, which moves the crotch down to the
  knees for any ray-based retopo. Carve a thigh gap in the sculpt.
- **Adding a bmesh custom-data layer invalidates existing BMVert references.**
  Create layers before creating geometry.
- **Subsurf interpolates integer attributes** (part labels) — re-derive labels
  from the nearest cage vertex afterwards.
- **`delete(context="FACES_ONLY")` leaves wire edges**; after subdivision they
  become loose verts and Blender's bone-heat solver fails for the whole mesh.
- **The Cycles Wireframe node shows render triangles, not quads.** Review
  topology with a Wireframe *modifier* on a copy.
- **Stitches/edge effects must use 3D distance**: in UV space every island
  seam looks like a garment border.
- **A toe cannot become a toe box.** Folding the template's five toes onto
  one boot surface failed with every smoothing/inflation scheme; swap to a
  "shoe" variant of the base mesh (cut at a closed ring, quad-cap it).
- **A vertex with inverted decimation weight 0 is never collapsed** by
  Blender's collapse decimator (the group's factor does not matter). Fully
  "protecting" the face and hands froze them, and a dense base mesh could
  not get below 75 %. Cap protection weights below 1.
- **Relaxing a fold in place oscillates** instead of resolving it; fix the
  cause (where rays start, what may cross the midline) instead.
- **Seed flood fills from the right place**: the "most forward face" on the
  character's left was a fingertip (hands hang in front), not the toe, and
  the toe cut tried to delete the whole body. Restrict seeds by region, and
  sanity-check the size of what a fill selects.
- **Unbounded KD queries on a 10 M-cell grid** took 170 s per stage; bounded
  queries near the surface + a connectivity fill for the rest give identical
  signs in 3.7 s.
- **A ray from bare skin is a wrong hit**: limit offsets per material, and
  where the sculpt is the template's own surface (bare skin), cast none.
- **Search for a landmark inside a bounded window.** The chin was found as
  "the lowest point of the face profile"; on a stockier body the search
  wandered down the torso and landed at the groin, and the shirt mask (which
  never touches the chin) vanished. Bound the window relative to other
  landmarks and assert the result's distance.
- **QuadriFlow is scale-sensitive and fails silently** (`CANCELLED`) on
  small parts: it runs a manifold check at its own tolerance. Remesh each
  connected part separately at unit scale, clean the input (merge doubles,
  dissolve degenerates, fill holes), give it a budget by area share, verify
  the result has no crossing faces, and keep a decimation fallback.
- **Thin shells can't be projected**: a 5 mm pauldron shell snapped both
  layers to the same side. Build it from its analytic frame and carry the
  layer (outer/inner) as a vertex attribute through subdivision.
- **An orientation built from the wrong reference is wrong everywhere**: the
  pauldron's axis was taken from the torso's outward normal; it must be
  perpendicular to the upper arm (plus a lift), or it floats off the
  shoulder on every body.
- **Texture regions follow the geometry masks, not the style name**: the
  mohawk's shaved sides were textured as hair because the style reused the
  short-hair cap mask; region masks must be exactly the sculpted strip.

