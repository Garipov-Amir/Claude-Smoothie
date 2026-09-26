# Example: game-ready character ("Ranger")

Built entirely by `scripts/game_character/build_character.py` — no manual
steps, no external assets. Realistic-proportioned adult male in game A-pose:
linen shirt, leather jerkin, wool trousers, belt with brass buckle and pouch,
leather boots, short sculpted hair.

```bash
python scripts/game_character/build_character.py /tmp/ranger --res 2048
python scripts/game_character/render_showcase.py /tmp/ranger examples/game_character/out
```

| Sheet | Shows |
|---|---|
| `out/highpoly.jpg` | SDF sculpt, clay, ~4.3 M tris @ 1.2 mm |
| `out/topology.jpg` | LOD0 quads (real edges): body, face loops, hand |
| `out/textures.jpg` | BaseColor · Normal (OpenGL) · ORM · UV layout |
| `out/beauty.jpg` | engine-style material (textures only), Cycles |
| `out/head.jpg` | face close-ups |
| `out/lods.jpg` | LOD0–LOD4 with triangle counts |
| `out/walk.jpg` | exported walk cycle |
| `out/poses.jpg` | deformation stress poses |

`out/SK_Character.glb` is the shipped LOD0 + skeleton + walk + 2K textures (open it in any glTF viewer); `out/report.json` is this build's validation report.

| | LOD0 | LOD1 | LOD2 | LOD3 | LOD4 |
|---|---|---|---|---|---|
| body tris | 43 648 | 21 824 | 10 912 | 5 456 | 2 728 |
| + eyes | 1 440 | 1 440 | 1 440 | 336 | 336 |

All LODs: 0 n-gons, 0 non-manifold edges, 0 degenerate faces, UVs in 0–1
with 0 % overlap, ≤ 4 bone influences, 0 unweighted vertices, weights
normalized. Source topology (LOD0 before triangulation): 100 % quads, median
quad-corner deviation ≈ 2.1°, UV coverage 63 %. Skeleton: 63 bones, UE-mannequin naming, single
root at the origin.

What's shipped (`<out>/export/`): `SK_Character.fbx` (all LODs + skeleton +
walk; Unity auto-builds the LODGroup from the `_LOD#` suffixes),
`SK_Character_LOD#.fbx` (per-LOD for Unreal), `SK_Character.glb` (LOD0 +
skeleton + walk + textures), and `<out>/textures/` (`T_Character_BaseColor`,
`_Normal_OpenGL`, `_Normal_DirectX`, `_ORM`, `_Height`, `_SkinMask`,
`T_Eye_BaseColor`).

Known limits of this example are listed in
`reference/game_ready_character.md` ("Honest limits").
