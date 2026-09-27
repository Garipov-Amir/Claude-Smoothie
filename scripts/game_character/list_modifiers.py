"""
List the MakeHuman modifiers usable in a spec's body.modifiers (side-less
target stems, applied to both sides), optionally filtered.

    python list_modifiers.py [substring ...]
    python list_modifiers.py ear nose     # every ear/nose modifier
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    import humanoid
    import reference_body as RB
    idx = RB.modifier_index(humanoid.REF_CACHE)
    words = [a for a in sys.argv[1:] if a != "--"]
    keys = sorted(k for k in idx if not words or any(w in k for w in words))
    groups = {}
    for k in keys:
        groups.setdefault(idx[k][0].split("/")[1], []).append(k)
    for g, ks in sorted(groups.items()):
        print(f"[{g}]")
        for k in ks:
            print("   ", k)
    print(f"{len(keys)} modifiers")
