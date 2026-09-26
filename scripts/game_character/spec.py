"""
spec — the character specification every stage builds from.

A character is data, not code: body shape (MakeHuman macro sliders +
any MakeHuman modifier by name), skin/eyes/hair, an outfit made of garment
layers from the library in `costume.py`, and separate pieces (long hair,
beards, horns, tusks, pauldrons, pouches) from `pieces.py`. A spec may be
partial: everything missing comes from DEFAULT (the "Ranger" example).

    import spec
    S = spec.resolve({"name": "Scout", "body": {"sex": 0.0, "height_m": 1.68}})
    spec.use(S)            # modules read spec.SPEC at call time
    spec.save(S, out_dir)  # stages after the first read out_dir/spec.json

Colors are sRGB, as "#rrggbb" or [r, g, b] in 0..1.
Lengths are meters; body-relative placements are fractions of landmarks.
"""

import copy
import json
import os

import numpy as np

SPEC = {}

MATERIALS = ("linen", "wool", "leather", "metal", "cloth", "fur")
METALS = ("brass", "iron", "steel", "silver", "gold", "bronze")

# garment layers: type -> (defaults, allowed values per field)
GARMENTS = {
    "shirt": ({"sleeves": "long", "neck": "crew", "tuck": "in", "material": "linen", "color": "#b3a387",
               "thickness": 0.0048},
              {"sleeves": ("long", "short", "none"), "neck": ("crew", "v", "scoop"), "tuck": ("in", "out")}),
    "vest": ({"neck": "v", "length": "hip", "material": "leather", "color": "#452a18", "thickness": 0.0060},
             {"neck": ("v", "crew"), "length": ("waist", "hip")}),
    "trousers": ({"length": "full", "material": "wool", "color": "#40402f", "thickness": 0.0055},
                 {"length": ("full", "knee", "shorts")}),
    "belt": ({"material": "leather", "color": "#361f12", "buckle": "brass", "width": 0.040, "thickness": 0.0055},
             {"buckle": METALS + ("none",)}),
    "boots": ({"height": "calf", "material": "leather", "color": "#301f13", "sole": "#2a2622"},
              {"height": ("ankle", "calf", "knee")}),
    "shoes": ({"material": "leather", "color": "#3a2616", "sole": "#2a2622"}, {}),
    "gloves": ({"fingers": "full", "material": "leather", "color": "#3b2618", "thickness": 0.0020},
               {"fingers": ("full", "none")}),
    "bracers": ({"material": "leather", "color": "#4a2e1a", "thickness": 0.0045}, {}),
}

# separate mesh pieces (own high-poly, own low-poly, rigid skin): see pieces.py
PIECES = {
    "pouch": ({"side": "right", "material": "leather", "color": "#6b4729"}, {"side": ("left", "right")}),
    "ponytail": ({"length": 0.26, "color": None}, {}),
    "bun": ({"size": 1.0, "color": None}, {}),
    "beard_long": ({"length": 0.12, "color": None}, {}),
    "horns": ({"shape": "curved", "size": 1.0, "color": "#d8ccb0"}, {"shape": ("curved", "straight", "ram")}),
    "tusks": ({"size": 1.0, "color": "#e6dcc3"}, {}),
    "pauldrons": ({"side": "both", "material": "metal", "metal": "iron", "color": "#6d6e70"},
                  {"side": ("both", "left", "right")}),
}

HAIR_STYLES = ("bald", "buzz", "short", "mohawk")
HAIRLINES = ("receding", "straight", "rounded")
BEARD_STYLES = ("none", "stubble", "short", "long")

DEFAULT = {
    "name": "Ranger",
    "body": {
        "sex": 1.0,           # 0 female .. 1 male (MakeHuman gender blend; 0.5 = androgynous)
        "age": 25.0,          # years, 1..90 (child/old proportions come from MakeHuman's age targets)
        "muscle": 0.725,      # 0..1, 0.5 average
        "weight": 0.5,        # 0..1, 0.5 average (fat and mass)
        "proportions": 0.5,   # 0 uncommon .. 0.5 regular .. 1 idealized (MakeHuman BodyProportions)
        "height_m": 1.80,     # final stature, feet on the ground (without hair)
        "ethnicity": {"african": 0.0, "asian": 0.0, "caucasian": 1.0},
        "modifiers": {},      # any MakeHuman modifier: {"ear-shape": -1.0, "upperlegs-height": -0.6, ...}
    },
    "skin": {"color": "#b3876d", "lips": "#9e5c57", "freckles": 0.25, "stubble": 0.45, "roughness": 0.50},
    "eyes": {"iris": "#5b6b33"},
    "hair": {"style": "short", "hairline": "receding", "color": "#2b1d14"},
    "beard": {"style": "none", "color": None},
    "outfit": [
        {"type": "shirt", "sleeves": "long", "neck": "crew", "material": "linen", "color": "#b3a387"},
        {"type": "trousers", "length": "full", "material": "wool", "color": "#40402f"},
        {"type": "vest", "neck": "v", "length": "hip", "material": "leather", "color": "#452a18"},
        {"type": "belt", "material": "leather", "color": "#361f12", "buckle": "brass"},
        {"type": "boots", "height": "calf", "material": "leather", "color": "#301f13"},
    ],
    "pieces": [
        {"type": "pouch", "side": "right", "material": "leather", "color": "#6b4729"},
    ],
}


class SpecError(ValueError):
    pass


def color(c):
    """sRGB float triple from '#rrggbb' or [r, g, b]."""
    if isinstance(c, str):
        h = c.lstrip("#")
        if len(h) != 6:
            raise SpecError(f"color {c!r}: expected #rrggbb")
        return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)
    a = np.asarray(c, np.float32)
    if a.shape != (3,):
        raise SpecError(f"color {c!r}: expected 3 components")
    return np.clip(a if a.max() <= 1.0 else a / 255.0, 0, 1)


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k not in ("modifiers", "ethnicity"):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _items(entries, table, what):
    res = []
    for i, e in enumerate(entries or []):
        if isinstance(e, str):
            e = {"type": e}
        t = e.get("type")
        if t not in table:
            raise SpecError(f"{what}[{i}]: unknown type {t!r} (known: {', '.join(table)})")
        defaults, allowed = table[t]
        unknown = set(e) - set(defaults) - {"type"}
        if unknown:
            raise SpecError(f"{what}[{i}] ({t}): unknown fields {sorted(unknown)} (known: {sorted(defaults)})")
        g = _merge(defaults, e)
        g["type"] = t
        for k, vals in allowed.items():
            if g[k] not in vals:
                raise SpecError(f"{what}[{i}] ({t}).{k} = {g[k]!r}: expected one of {vals}")
        if "material" in g and g["material"] not in MATERIALS:
            raise SpecError(f"{what}[{i}] ({t}).material = {g['material']!r}: expected one of {MATERIALS}")
        res.append(g)
    return res


def resolve(user=None):
    """Full, validated spec from a partial one (dict, JSON path or preset name)."""
    if isinstance(user, str):
        user = load(user)
    user = copy.deepcopy(user or {})
    S = _merge({k: v for k, v in DEFAULT.items() if k not in ("outfit", "pieces")},
               {k: v for k, v in user.items() if k not in ("outfit", "pieces")})
    unknown = set(user) - set(DEFAULT)
    if unknown:
        raise SpecError(f"unknown top-level fields {sorted(unknown)} (known: {sorted(DEFAULT)})")
    S["outfit"] = _items(user.get("outfit", DEFAULT["outfit"]), GARMENTS, "outfit")
    S["pieces"] = _items(user.get("pieces", DEFAULT["pieces"]), PIECES, "pieces")
    b = S["body"]
    for k, lo, hi in (("sex", 0, 1), ("muscle", 0, 1), ("weight", 0, 1), ("proportions", 0, 1),
                      ("age", 1, 90), ("height_m", 0.5, 2.6)):
        if k == "height_m" and b[k] is None:      # MakeHuman's own stature for this age/sex
            continue
        if not (lo <= float(b[k]) <= hi):
            raise SpecError(f"body.{k} = {b[k]} outside {lo}..{hi}")
    eth = {k: float(b["ethnicity"].get(k, 0.0)) for k in ("african", "asian", "caucasian")}
    if sum(eth.values()) <= 0:
        raise SpecError("body.ethnicity: weights must not all be 0")
    b["ethnicity"] = eth
    for k, v in b["modifiers"].items():
        if not (-1.0 <= float(v) <= 1.0):
            raise SpecError(f"body.modifiers[{k!r}] = {v}: expected -1..1")
    if S["hair"]["style"] not in HAIR_STYLES:
        raise SpecError(f"hair.style {S['hair']['style']!r}: expected one of {HAIR_STYLES}")
    if S["hair"]["hairline"] not in HAIRLINES:
        raise SpecError(f"hair.hairline {S['hair']['hairline']!r}: expected one of {HAIRLINES}")
    if S["beard"]["style"] not in BEARD_STYLES:
        raise SpecError(f"beard.style {S['beard']['style']!r}: expected one of {BEARD_STYLES}")
    types = [g["type"] for g in S["outfit"]]
    for t in set(types):
        if types.count(t) > 1:
            raise SpecError(f"outfit: {t!r} listed twice")
    if "boots" in types and "shoes" in types:
        raise SpecError("outfit: boots and shoes both listed")
    # garments are fitted shells over the skin: under a top, flatten the
    # nipples (as game characters are built) unless the spec sets it itself
    if {"shirt", "vest"} & set(types):
        b["modifiers"] = dict(b["modifiers"])
        b["modifiers"].setdefault("nipple-flatten", 1.0)
    if not 0.0 <= float(b["modifiers"].get("nipple-flatten", 0.0)) <= 1.0:
        raise SpecError("body.modifiers['nipple-flatten']: expected 0..1")
    # colors that default to another one
    if S["beard"]["color"] is None:
        S["beard"]["color"] = S["hair"]["color"]
    if S["beard"]["style"] == "long" and not any(p["type"] == "beard_long" for p in S["pieces"]):
        S["pieces"].append(_items([{"type": "beard_long"}], PIECES, "pieces")[0])
    for p in S["pieces"]:
        if "color" in p and p["color"] is None:
            p["color"] = S["beard"]["color"] if p["type"] == "beard_long" else S["hair"]["color"]
    # every color must parse
    for where, c in _colors(S):
        try:
            color(c)
        except SpecError as e:
            raise SpecError(f"{where}: {e}") from None
    return S


def _colors(S):
    for k in ("color", "lips"):
        yield f"skin.{k}", S["skin"][k]
    yield "eyes.iris", S["eyes"]["iris"]
    yield "hair.color", S["hair"]["color"]
    yield "beard.color", S["beard"]["color"]
    for i, g in enumerate(S["outfit"]):
        for k in ("color", "sole"):
            if k in g:
                yield f"outfit[{i}].{k}", g[k]
    for i, p in enumerate(S["pieces"]):
        if "color" in p:
            yield f"pieces[{i}].color", p["color"]


def garment(S, t):
    """The outfit layer of type t, or None."""
    return next((g for g in S["outfit"] if g["type"] == t), None)


def pieces(S, t=None):
    return [p for p in S["pieces"] if t is None or p["type"] == t]


def footwear(S):
    return garment(S, "boots") or garment(S, "shoes")


def use(S):
    SPEC.clear()
    SPEC.update(S)
    return SPEC


def load(path):
    """A spec from a JSON file, or a preset name (presets/<name>.json)."""
    if not os.path.exists(path):
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets", f"{path}.json")
        if not os.path.exists(p):
            raise SpecError(f"no spec file or preset named {path!r}")
        path = p
    with open(path) as fh:
        return json.load(fh)


def save(S, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "spec.json"), "w") as fh:
        json.dump(S, fh, indent=2)


def from_out(out_dir):
    """The resolved spec a build directory was made with (DEFAULT if none)."""
    p = os.path.join(out_dir, "spec.json")
    return resolve(load(p) if os.path.exists(p) else {})


def body_key(S):
    """Stable hash of everything that shapes the body mesh (cache key)."""
    import hashlib
    return hashlib.sha1(json.dumps(S["body"], sort_keys=True).encode()).hexdigest()[:12]


def shape_key(S):
    """Everything that changes geometry (colors and materials stripped): two
    specs with the same key share sculpt, topology, UVs and bakes."""
    import hashlib
    drop = ("color", "sole", "material", "metal", "lips", "iris", "freckles", "stubble", "roughness")
    strip = lambda d: {k: v for k, v in d.items() if k not in drop}
    key = {"body": S["body"], "hair": strip(S["hair"]), "beard": strip(S["beard"]),
           "outfit": [strip(g) for g in S["outfit"]], "pieces": [strip(p) for p in S["pieces"]]}
    return hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
