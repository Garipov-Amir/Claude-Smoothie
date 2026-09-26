# Character spec reference

A character is one JSON object. Everything is optional: missing fields come
from the default ("Ranger", `scripts/game_character/presets/ranger.json`).
That includes the lists: a spec without `"outfit"` wears the Ranger's
clothes and one without `"pieces"` gets the Ranger's belt pouch — write
`"pieces": []` (or `"outfit": []` for a nude body) when the character has
none.
`spec.resolve()` validates it and names the field and the allowed values on
any mistake. Colors are sRGB `"#rrggbb"` (or `[r, g, b]` in 0..1). Lengths
are meters.

```json
{
  "name": "Scout",
  "body":   { ... },
  "skin":   { ... },
  "eyes":   { ... },
  "hair":   { ... },
  "beard":  { ... },
  "outfit": [ {"type": ..., ...}, ... ],
  "pieces": [ {"type": ..., ...}, ... ]
}
```

## body — the anatomy (MakeHuman blend)

| field | range | default | meaning |
|---|---|---|---|
| `sex` | 0..1 | 1.0 | 0 female, 1 male, 0.5 androgynous |
| `age` | 1..90 | 25 | years; child (≤ 12) and old (≥ 60) proportions come from MakeHuman's age targets |
| `muscle` | 0..1 | 0.725 | 0.5 average; 1 bodybuilder |
| `weight` | 0..1 | 0.5 | 0.5 average; 0 very thin, 1 obese |
| `proportions` | 0..1 | 0.5 | 0 uncommon, 0.5 regular, 1 idealized (heroic) |
| `height_m` | 0.5..2.6 or `null` | 1.80 | final stature without hair; `null` keeps MakeHuman's stature for that age/sex |
| `ethnicity` | weights | caucasian 1 | `{"african": a, "asian": b, "caucasian": c}` — face and body shape blend (skin color is separate) |
| `modifiers` | name → -1..1 | `{}` | any MakeHuman modifier, see below |

**Modifiers** are MakeHuman targets by their side-less stem; one entry
applies to both sides (`ear-shape-pointed` shapes both ears). Use the
positive weight of the stem you want (`nose-scale-horiz-incr: 0.5`, not
`nose-scale-horiz-decr: -0.5`). List them:

```bash
python scripts/game_character/list_modifiers.py            # all ~250
python scripts/game_character/list_modifiers.py ear nose   # filtered
```

Groups: head, forehead, eyebrows, neck, eyes, nose, mouth, ears, chin,
cheek, torso, hip, stomach, buttocks, pelvis, armslegs (hand/foot scale,
limb volumes, `upperlegs-height-*`, `lowerlegs-height-*`).

Useful ones:

| effect | modifiers |
|---|---|
| short legs (dwarf) | `upperlegs-height-decr: 1`, `lowerlegs-height-decr: 0.8` |
| long legs (elf) | `upperlegs-height-incr: 0.5`, `lowerlegs-height-incr: 0.5` |
| big head (goblin, child-like) | `head-scale-vert-incr`, `head-scale-horiz-incr`, `head-scale-depth-incr` |
| pointed / big ears | `ear-shape-pointed: 1`, `ear-scale-incr`, `ear-scale-vert-incr` |
| heavy brow, jaw (orc, brute) | `forehead-nubian-incr`, `chin-prominent-incr`, `chin-width-incr`, `chin-jaw-drop-incr` |
| broad nose | `nose-scale-horiz-incr`, `nose-point-width-incr`, `nose-volume-incr` |
| V-shaped torso | `torso-vshape-incr`, `torso-muscle-pectoral-incr`, `torso-muscle-dorsi-incr` |
| big hands / feet | `hand-scale-incr`, `foot-scale-incr` |
| belly | `stomach-pregnant-incr` (0.2–0.5 for a paunch) |
| thick neck | `neck-scale-horiz-incr`, `neck-scale-depth-incr` |

Keep modifiers within ±1; stacking many at 1.0 can fold the base mesh.

## skin, eyes

| field | default | notes |
|---|---|---|
| `skin.color` | `#b3876d` | base tone; the face color zones (forehead, cheeks/nose, beard shadow) are derived from it |
| `skin.lips` | `#9e5c57` | |
| `skin.freckles` | 0.25 | 0..1 |
| `skin.stubble` | 0.45 | 0..1 beard shadow on the lower face (0 for women, children, beardless races) |
| `skin.roughness` | 0.5 | 0.35 oily/wet … 0.65 dry/matte |
| `eyes.iris` | `#5b6b33` | |

Skin tones: pale `#e9c7b2`, fair `#d9a98c`, tan `#b3876d`, olive `#a07a5c`,
brown `#7d5539`, dark `#4f3324`. Fantasy: orc green `#6f8a4e`, demon red
`#8e3b36`, frost blue-grey `#7d8ea3`, ashen `#8a8580`.

## hair, beard

| field | values | default |
|---|---|---|
| `hair.style` | `bald`, `buzz`, `short`, `mohawk` | `short` |
| `hair.hairline` | `receding` (masculine), `straight`, `rounded` (feminine/young) | `receding` |
| `hair.color` | color | `#2b1d14` |
| `beard.style` | `none`, `stubble` (texture only), `short` (sculpted on the jaw), `long` (adds a `beard_long` piece) | `none` |
| `beard.color` | color; defaults to the hair color | |

Long hair is a piece (`ponytail`, `bun`) on top of a short or buzz cut.
Hair colors: black `#141110`, dark brown `#2b1d14`, brown `#5a3521`, auburn
`#6e3218`, ginger `#8a4a22`, blond `#a8834e`, platinum `#cfc3a6`, grey
`#8d8a86`, white `#d8d6d0`.

## outfit — garment layers (at most one of each type)

Order does not matter; layering is fixed (shirt → trousers → vest →
bracers → gloves → belt → footwear; an untucked shirt goes over the
trousers). `material` ∈ `linen`, `cloth` (fine fabric), `wool`, `leather`,
`metal`, `fur` — it picks the texture recipe, not the shape.

| type | fields (defaults) |
|---|---|
| `shirt` | `sleeves`: long/short/none; `neck`: crew/v/scoop; `tuck`: in/out; `material` linen; `color`; `thickness` 0.0048 |
| `trousers` | `length`: full/knee/shorts; `material` wool; `color`; `thickness` 0.0055 |
| `vest` | `neck`: v/crew; `length`: waist/hip; `material` leather; `color`; `thickness` 0.006 |
| `belt` | `buckle`: brass/iron/steel/silver/gold/bronze/none; `width` 0.04; `material` leather; `color` |
| `boots` | `height`: ankle/calf/knee; `material` leather; `color`; `sole` color |
| `shoes` | `material` leather; `color`; `sole` color |
| `gloves` | `fingers`: full/none (fingerless); `material` leather; `color`; `thickness` 0.002 |
| `bracers` | `material` leather (or metal); `color`; `thickness` 0.0045 |

No footwear = barefoot (the game mesh keeps its toes). Boots or shoes swap
the base mesh to its "shoe" variant (toes replaced by a toe box).

## pieces — separate meshes (own high-poly, low-poly, LODs, rigid skin)

| type | fields (defaults) | bound to |
|---|---|---|
| `pouch` | `side`: left/right; `material` leather; `color` | skin under its belt loop |
| `ponytail` | `length` 0.26 (m, head-scaled); `color` = hair | head |
| `bun` | `size` 1.0; `color` = hair | head |
| `beard_long` | `length` 0.12; `color` = beard | head |
| `horns` | `shape`: curved/straight/ram; `size` 1.0; `color` `#d8ccb0` | head |
| `tusks` | `size` 1.0; `color` `#e6dcc3` | head |
| `pauldrons` | `side`: both/left/right; `material` metal/leather; `metal` iron; `color` | skin of the shoulder |

## Archetype starting points

Tweak from these; presets exist for the first five.

| archetype | body | look |
|---|---|---|
| **ranger** (preset) | male 25, muscle 0.725, 1.80 m | linen shirt, leather vest, wool trousers, calf boots, pouch |
| **scout** (preset) | female 24, muscle 0.6, weight 0.42, 1.68 m | short sleeves, bracers, fingerless gloves, ankle boots, ponytail |
| **dwarf** (`dwarf_smith`) | male 58, muscle 0.85, weight 0.78, 1.38 m, short legs, big hands | bald, long beard, sleeveless shirt, heavy belt |
| **orc** (`orc_warrior`) | male 30, muscle 1.0, 1.96 m, pointed ears, heavy brow/jaw, green skin | mohawk, tusks, bare torso, pauldron, knee boots |
| **demon** (`horned_demon`) | 0.85 male, 1.88 m, red skin, pointed ears | ram horns, short beard, scoop-neck shirt untucked |
| elf | 0.3–0.7 sex, weight 0.3, muscle 0.45, 1.88 m, `ear-shape-pointed: 1`, `ear-scale-vert-incr: 0.6`, long legs | ponytail, fine cloth |
| goblin | male 30, 1.10 m, weight 0.35, head-scale-*-incr 0.6, `ear-scale-incr: 1`, `ear-shape-pointed: 1`, `nose-scale-vert-incr: 0.8` | green-grey skin, bald, shorts, barefoot |
| child | age 8–12, `height_m: null`, muscle 0.5 | shirt short sleeves, shorts, shoes; stubble 0 |
| elder | age 70–85, muscle 0.3, weight 0.55 | grey hair `receding`, long beard, vest |
| heavy brute | male 40, muscle 0.9, weight 0.9, 1.90 m, `stomach-pregnant-incr: 0.3` | buzz cut, stubble 0.8 |
