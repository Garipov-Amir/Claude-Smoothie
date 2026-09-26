# Example: game-ready character ("Ranger")

Built entirely by `scripts/game_character/build_character.py` — no manual
steps. Adult male, 1.80 m, in the reference A-pose: linen shirt, leather
jerkin, wool trousers, belt with brass buckle and a belt pouch, leather
boots, short sculpted hair. The body is the CC0 MakeHuman reference (young
male, a bit above average muscle) sculpted on as an SDF; the game mesh is
its base topology wrapped onto the finished sculpt (booted variant).

```bash
python scripts/game_character/build_character.py /tmp/ranger --res 2048
python scripts/game_character/render_showcase.py /tmp/ranger examples/game_character/out
```

| Sheet | Shows |
|---|---|
| `out/highpoly.jpg` | SDF sculpt, clay, ~4.4 M tris @ 1.2 mm (+ 0.1 M pouch) |
| `out/topology.jpg` | LOD0 quads (real edges): body, face loops, hand |
| `out/textures.jpg` | BaseColor · Normal (OpenGL) · ORM · UV layout |
| `out/beauty.jpg` | engine-style material (textures only), Cycles |
| `out/head.jpg` | face close-ups |
| `out/lods.jpg` | LOD0–LOD4 with triangle counts |
| `out/walk.jpg` | exported walk cycle |
| `out/poses.jpg` | deformation stress poses |

`out/SK_Character.glb` is the shipped LOD0 + skeleton + walk + 2K textures
(open it in any glTF viewer); `out/report.json` is this build's validation
report.

| | LOD0 | LOD1 | LOD2 | LOD3 | LOD4 |
|---|---|---|---|---|---|
| body + pouch tris | 24 540 | 12 268 | 6 132 | 3 066 | 1 530 |
| + eyes | 1 440 | 1 440 | 1 440 | 336 | 336 |

All LODs: **watertight** (0 boundary, 0 non-manifold edges), **0
self-intersecting face pairs** within a mesh piece (the pouch touching the
hip is reported separately as contact), 0 n-gons, 0 degenerate faces, UVs
in 0–1, ≤ 4 bone influences, 0 unweighted vertices, weights normalized.
Source topology (LOD0 before triangulation): 100 % quads, 11 888 vertices.
Skeleton: 63 bones, UE-mannequin naming, single root at the origin. The
pouch hangs from the belt: one rigid weight set, 65 % pelvis.

What's shipped (`<out>/export/`): `SK_Character.fbx` (all LODs + skeleton +
walk; Unity auto-builds the LODGroup from the `_LOD#` suffixes),
`SK_Character_LOD#.fbx` (per-LOD for Unreal), `SK_Character.glb` (LOD0 +
skeleton + walk + textures), and `<out>/textures/` (`T_Character_BaseColor`,
`_Normal_OpenGL`, `_Normal_DirectX`, `_ORM`, `_Height`, `_SkinMask`,
`T_Eye_BaseColor`).

Known limits of this example are listed in
`reference/game_ready_character.md` ("Honest limits").
