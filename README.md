# blender-stylized-3d

A Claude Code skill + optional MCP connector for generating 3D models in
Blender from a text description or a reference photo — stylized props and
creatures, and (Workflow C) a complete **game-ready realistic character**:
sculpt on an anatomical reference body → wrapped quad base topology (watertight,
no self-intersections) → UVs → baked normal/AO/curvature → PBR texture set →
UE-compatible skeleton, skinning and a walk cycle → LOD0–LOD4 → FBX/GLB with a
validation report. See [`reference/game_ready_character.md`](reference/game_ready_character.md)
and [`examples/game_character/`](examples/game_character/). Default
look is hand-painted PBR (baked cavity shading, fresnel rim tint, painterly
noise variation) — no image textures required, no external AI 3D-generation
API, no neural image-to-3D reconstruction.

- **[`SKILL.md`](SKILL.md)** — the skill itself; auto-loaded by Claude Code
  when it's placed under a project's `.claude/skills/`, or invoke directly.
- **`scripts/bpy_stylized_kit.py`** — the reusable `bpy` helper library
  (primitives, modifiers, hand-painted material builder, lighting/camera/
  render/export). See [`reference/api.md`](reference/api.md) for the full list.
- **`scripts/image_prep.py`** — turns a photo into a palette + heightmap +
  silhouette mask using classic image processing (system Python, Pillow).
- **`scripts/run_blender.sh`** — runs a generation script headless (Blender binary, or the `bpy` pip module).
- **`scripts/game_character/`** — the game-ready character pipeline (Workflow C); entry point `build_character.py --spec <spec|preset>`.
- **[`skills/humanoid-character/`](skills/humanoid-character/SKILL.md)** — a second skill: any humanoid character (human or fantasy race) from a JSON spec — body sliders, outfit, pieces, colors — to FBX/GLB. Link it into `.claude/skills/humanoid-character` the same way as this folder.
- **`mcp_server/`** — optional connector to drive a *live*, visible Blender
  session instead of headless batch runs. See `mcp_server/README.md`.

## Quick start

```bash
# one-time
brew install --cask blender          # if not already installed
pip3 install -r requirements.txt     # Pillow (+ mcp, only if using the live connector)

# text -> 3D: write a generation script using bpy_stylized_kit (see SKILL.md), then
scripts/run_blender.sh my_script.py

# photo -> 3D
python3 scripts/image_prep.py photo.png --outdir /tmp/prep
# then a generation script that calls k.extrude_silhouette(...) or
# k.add_displace_from_heightmap(...) using /tmp/prep's outputs (see SKILL.md)
```

## Using this as an installed Claude Code skill

Symlink (or copy) this folder into any project's skill directory:

```bash
mkdir -p /path/to/project/.claude/skills
ln -s /Users/amirgaripov/dev/3d-skills/blender-stylized-3d /path/to/project/.claude/skills/blender-stylized-3d
```

It's already linked into this repo's own `.claude/skills/` so it's available
in this session too.

## What "stylized 3D from a photo" actually means here

There is no neural 3D reconstruction in this toolkit (by design — see the
setup conversation that scoped this out in favor of a fully local, free,
Blender-only pipeline). Photo-driven generation uses two classic non-ML
techniques instead:

- **Cutout/medallion extrusion** — a foreground silhouette, extruded and
  displaced into a raised relief. Good for recognizable character/portrait/
  logo shapes.
- **Bas-relief displacement** — a whole image's luminance pushed into a
  plane's surface. Good for scenes/landscapes/textures.

Both are genuinely 3D and genuinely derived from the photo (palette, shape,
and relief all come from the source image), but they are not a full
turnaround-accurate reconstruction of the photographed subject — set that
expectation with whoever's asking before generating.
