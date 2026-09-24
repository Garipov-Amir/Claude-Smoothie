# The hand-painted PBR shader technique

`build_hand_painted_material` in `scripts/bpy_stylized_kit.py` fakes the
classic stylized-game-art "hand painted" look (think WoW/League-of-Legends
era texture painting) procedurally, with no image textures to paint. Node
graph, in order:

1. **Object-space Noise Texture → Color Ramp** produces a soft mottled
   pattern mapped between a darkened and lightened/saturated version of the
   base color. This is what reads as "brush strokes" instead of a flat CG
   fill. Controlled by `variation` (how far the dark/light ends move from the
   base color) and `saturation_boost` (extra saturation pushed into the light
   end — real hand-painted textures push saturation in lit areas rather than
   just going lighter).

2. **Ambient Occlusion node → Color Ramp → Multiply** bakes soft occlusion
   directly into the albedo (`cavity_strength` controls how dark the occluded
   end gets). This is the single biggest contributor to the "painted" read —
   real hand-painted textures have shading baked into the diffuse map instead
   of relying purely on realtime lighting, and this reproduces that even
   though the material is otherwise using dynamic lighting.

3. **Fresnel → Multiply(rim_strength) → Add** tints grazing angles with a
   cool rim color (`rim_color`, default a soft blue). Cheap warm/cool
   separation is a classic stylized-lighting trick — keep `rim_strength` low
   (0.08–0.2) or it reads as a sci-fi rim light instead of a subtle tint.

4. The same noise drives a second Color Ramp feeding **Roughness**, so
   specular response varies slightly across the surface instead of being
   perfectly uniform (another flat-CG tell).

5. `Metallic` is a flat input (default 0) — hand-painted stylized materials
   are almost always non-metal; for metal props, pass `metallic=1.0` and
   expect the noise/AO trick to read more like brushed/worn metal than paint.

## Tuning by material type

- **Skin/organic**: defaults are tuned for this. `cavity_strength` 0.4–0.5, `rim_strength` 0.1–0.15.
- **Cloth/fabric**: raise `variation` to 0.2–0.3 for more visible weave-like mottling, `roughness` 0.6–0.8.
- **Stone/rock**: raise `cavity_strength` to 0.6+ for deep crevice shadows, lower `saturation_boost` toward 1.0.
- **Polished wood/lacquer**: lower `roughness` to 0.25–0.35, keep `variation` low (0.08–0.12) for a cleaner painted-gloss look.
- **Metal**: `metallic=1.0`, `roughness` 0.3–0.5, keep `rim_strength` up around 0.15–0.2 for a stylized specular highlight.

## Why not just use image textures?

Nothing stops you from painting/generating a real texture and plugging it
into `Base Color` instead of the procedural ramp — swap the noise+ramp chain
for an Image Texture node if the user supplies or wants a specific painted
texture. The procedural version exists so a model can look intentionally
painted with zero texture-authoring step, which is the common case when
generating from a text description.

## Why "Standard" view transform, not AgX

Blender's default AgX view transform is a filmic tone-map built for
photoreal work — it deliberately rolls off saturation and contrast in
highlights. Flat, saturated stylized colors run straight through that
roll-off and come out desaturated/pale even at correct exposure. `render_still`
forces `Standard` via `set_stylized_color_management()` so the material's
actual RGB values read through close to as-authored. If a user specifically
wants a filmic/photoreal grade over a stylized model, override this back to
`Filmic` or `AgX` deliberately — don't do it by default for this skill.
