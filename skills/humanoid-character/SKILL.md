---
name: humanoid-character
description: Build a game-ready 3D humanoid character in Blender — any human (man, woman, child, elder, any build or ethnicity) or humanoid fantasy race (elf, dwarf, orc, goblin, demon…) — from a text description or reference, as data: an anatomical body from MakeHuman sliders and morphs, sculpted clothing, footwear, hair and beard, separate pieces (horns, tusks, ponytail, long beard, pauldrons, pouch), wrapped watertight quad topology, UVs, high→low bakes, PBR textures, UE-named skeleton with skinning and a walk cycle, LOD0–LOD4, FBX/GLB export and a validation report. Use when asked to create, model or make a character, NPC, hero, avatar or humanoid creature for a game or real-time use, or to change one (body, outfit, colors), or to turn a character description into .fbx/.glb.
---

# Game-ready humanoid characters

A character is a **spec** (JSON): body sliders + outfit + pieces + colors.
One command turns it into a production asset:

```
<out>/export/SK_Character.fbx          all LODs + skeleton + walk (Unity LODGroup by _LOD# suffix)
<out>/export/SK_Character_LOD0..4.fbx  one file per LOD (Unreal LOD import)
<out>/export/SK_Character.glb          LOD0 + eyes + skeleton + walk + textures
<out>/export/report.json               validation: watertight, self-intersections, UVs, weights, skeleton
<out>/textures/T_Character_*.png       BaseColor, Normal_OpenGL/_DirectX, ORM, Height, SkinMask; T_Eye_BaseColor
```

The code lives in this repository: `REPO/scripts/game_character/`, where
`REPO` is the repository root (this skill's real folder is
`REPO/skills/humanoid-character/`; if it was reached through a symlink such
as `.claude/skills/humanoid-character`, resolve it with `realpath`).
Technique details: `REPO/reference/game_ready_character.md`.
Spec fields: [`reference/spec.md`](reference/spec.md).

## Setup (once per machine)

```bash
pip install bpy scikit-image scipy pillow      # or use Blender's bundled python
export GC_REF_CACHE=~/.cache/game_character_ref  # optional; MakeHuman CC0 data is downloaded here on first use
```

Runs headless on the CPU (Cycles). The first build of a new body shape
downloads the MakeHuman targets it needs (a few MB from
raw.githubusercontent.com) — the network must allow that host.

## Workflow

### 1. Turn the request into a spec

Write `<out>/spec.json` (or start from a preset in
`REPO/scripts/game_character/presets/`: `ranger`, `scout`, `dwarf_smith`,
`orc_warrior`, `horned_demon`). Map the description, don't ask for every
detail — choose sensible values and say what you chose:

- **Who**: `body.sex`, `body.age`, `body.height_m` (realistic for the age:
  a 10-year-old ≈ 1.38 m, or `null` for MakeHuman's own), `muscle`, `weight`.
- **Race / silhouette**: `body.modifiers` (short legs, pointed ears, heavy
  jaw, big hands…) — the archetype table in `reference/spec.md` has
  starting values for elf, dwarf, orc, goblin, demon, child, elder, brute.
- **Face**: nose/mouth/chin/ear/eye modifiers, `ethnicity` blend.
- **Colors**: skin, lips, iris, hair, every garment — sRGB hex.
- **Outfit**: garment layers (shirt, trousers, vest, belt, boots/shoes,
  gloves, bracers) with sleeves/neck/length options and materials.
- **Pieces**: ponytail/bun, long beard, horns, tusks, pauldrons, pouch.
  Omitted fields come from the Ranger preset — lists too: write
  `"pieces": []` when there are none, or the character gets the Ranger's pouch.
- **Things outside the library** (see Limits): pick the closest supported
  option, and tell the user what was substituted.

**From a reference image**: Read it and write down what it shows before
mapping — apparent sex, age, build (thin/average/heavy, muscular), height
cues, face features (nose, jaw, ears), hair style and color, each visible
garment (sleeve length, neckline, trouser length, footwear height), gear.
Sample colors instead of guessing names:

```bash
python -c "from PIL import Image; im=Image.open('ref.png').convert('RGB'); print('#%02x%02x%02x' % im.getpixel((X, Y)))"
```

(pick pixels in lit, mid-tone areas — not highlights or shadows). Then
preview and compare side by side with the image.

Validate early — errors name the field and the allowed values:

```bash
cd REPO/scripts/game_character
python -c "import spec; spec.resolve('<out>/spec.json')"
python list_modifiers.py ear chin        # find modifier names
```

### 2. Preview the silhouette (1–2 min)

```bash
python preview_spec.py <out>/spec.json <out>/preview.png
```

Clay render at a 3 mm voxel: front, 3/4, side, back, two face close-ups.
**Look at it** (Read the PNG). Check proportions and silhouette against the
request, that garments start and end where they should, pieces sit where
they should (no horn inside the skull, ponytail clear of the back). Iterate
on the spec until it reads right — this loop is cheap; the full build is not.

### 3. Build (≈ 12–15 min at 2K on 4 cores)

```bash
python build_character.py <out> --spec <out>/spec.json --res 2048
python render_showcase.py <out> <out>/showcase     # review sheets
python verify_export.py <out>/export/SK_Character.glb
```

Stages: `highpoly` (sculpt, ~7 min) → `lowpoly` (wrap the base topology,
pieces, UVs, LODs) → `bake` → `textures` → `lookdev` → `rig` → `export`.
Restart from a stage with `--from <stage>`:

| change | restart from |
|---|---|
| any color, material, skin freckles/stubble/roughness, iris | `--from textures` |
| body, garments (type/length/sleeves…), hair or beard style, pieces | `--from highpoly` (the driver refuses a later start) |
| rig/animation code | `--from rig` |
| export settings | `--from export` |

### 4. Review before calling it done

Read every sheet in `<out>/showcase/`: `highpoly` (sculpt), `topology`
(real edges: face loops, hands), `textures` (BaseColor, normal, ORM, UV
layout), `beauty` (engine-style material), `head`, `lods`, `walk`, `poses`
(arms up, crouch, fists). Then the validation report:

```bash
python check_report.py <out>          # PASS/FAIL per check, exit 1 on failure
```

It checks, per LOD:

- `watertight: true`, `self_intersecting_face_pairs: 0`, `non_manifold_edges: 0`,
  `degenerate_faces: 0`, `ngons: 0`
- `uv_inside_0_1: true`, `uv_overlap_fraction` ≈ 0 (LOD0 exactly 0)
- `max_influences ≤ 4`, `unweighted_verts: 0`, `weights_normalized: true`
- `piece_contact_face_pairs` > 0 is fine (a pouch resting on a hip)
- skeleton: 63 bones, `single_root`, no `missing_humanoid_bones`;
  textures power of two; all FBX/GLB files written

Fix what fails (below), rebuild, re-check. Report numbers to the user, not
adjectives.

### 5. Deliver

Give the file paths, the LOD triangle counts, texture set, bone count and
the report summary; mention engine notes: Unreal uses `*_Normal_DirectX.png`
and imports `SK_Character_LOD#.fbx` as LODs; Unity/glTF use the OpenGL
normal map and `SK_Character.fbx` (LODGroup from the suffixes).

## When something looks wrong

| symptom | likely cause → fix |
|---|---|
| body folds or spikes in preview | too many modifiers at ±1 → reduce to ≤ 0.7 |
| garment edge in the wrong place | check `height_m`/age (garments follow landmarks); shorten/lengthen via `length`, `sleeves`, `tuck` |
| a piece intersects the body in preview | size too big for this head/body (`size`, `length`) |
| `self_intersecting_face_pairs` > 0 on LOD0 | the wrap log (`lowpoly` stage) names the region; usually an extreme modifier or a garment bridging two limbs — tone it down |
| bake errors (black/blotchy spots in the normal map) | a garment offset > 2.5 cm or a piece too close to the body; thinner `thickness` |
| weird skin colors on cloth | the region masks: a garment's material comes from its own layer; check `material`/`color` of that layer |
| walk looks broken | check `poses`/`walk` sheets; skinning follows part labels, extreme proportions can mislabel — reduce leg/arm modifiers |

## Limits (say so, and substitute)

- **Long flowing hair** beyond ponytail/bun, **braids**, **skirts, robes,
  dresses, capes, long coats** (cloth that bridges the legs or hangs free
  needs separate cloth meshes and simulation), **helmets/masks**, **tails,
  wings, extra limbs, non-bipeds** — not in the library. Nearest
  substitutes: ponytail/bun, trousers + vest, pauldrons.
- Likeness of a **specific real person** is not a goal of this pipeline.
- Body shapes are MakeHuman's space: human and near-human; extreme
  cartoon proportions will fold.
- No facial rig/blend shapes, no teeth/tongue, eyes are simple spheres.
- Materials are procedural (no scanned textures); SSS is a mask.
- Rest pose is MakeHuman's (arms 41° down, elbows bent ~46°).

## How it works (one paragraph)

The body is MakeHuman's CC0 base mesh with its macro targets blended exactly
like MakeHuman (sex × age × muscle × weight, ethnicity, proportions) plus
named modifiers, turned into a signed distance field; garments, footwear,
hair and beard are sculpted onto it as offset shells whose heights and
widths follow the body's own landmarks and limb radii; pieces are separate
SDF models seated on the surface. The game mesh is the same base topology
**wrapped** onto the finished sculpt (booted variant under footwear),
validated watertight and self-intersection-free; pieces are quad-remeshed
(QuadriFlow). Maps are baked per piece into one atlas, textures authored
per region from the spec's materials, the skeleton comes from the body's
joint helpers, weights from bone-heat diffusion, LODs by self-checking
decimation. Details and lessons learned: `REPO/reference/game_ready_character.md`.
