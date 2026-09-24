#!/usr/bin/env python3
"""
Send Python code to a running Stylized3D MCP Bridge (blender_addon.py) and
print the JSON response. For interactively driving a live Blender GUI
session from the command line without hand-writing socket code each time.

Usage:
    python3 send_to_blender.py "result = 1 + 1"
    python3 send_to_blender.py --file my_script.py
    python3 send_to_blender.py --port 9877 "result = bpy.context.scene.name"

Each call is a fresh exec() namespace (only `bpy` and `result` are
pre-defined) — re-import bpy_stylized_kit in every call that needs it:
    import sys; sys.path.insert(0, "<repo>/blender-stylized-3d/scripts"); import bpy_stylized_kit as k

GOTCHA — Blender's Python process stays alive across every call (unlike
headless scripts/run_blender.sh, which is a fresh process each time), so
plain `import bpy_stylized_kit` after the first call returns the module
Python already cached in sys.modules, NOT a re-read of the file — editing
bpy_stylized_kit.py on disk and re-running silently keeps using the OLD
code. This looks exactly like "my fix didn't work" and cost real time
figuring out in loop-iteration testing before the cause was clear. Use
--with-toolkit (which now always force-reloads) or add
`import importlib; importlib.reload(k)` yourself after importing whenever
you've edited the toolkit since the bridge was started.
"""
import argparse
import json
import socket
import sys

TOOLKIT_DIR = "/Users/amirgaripov/dev/3d-skills/blender-stylized-3d/scripts"


def send(code, host="127.0.0.1", port=9876, timeout=60):
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
    except (ConnectionRefusedError, OSError) as e:
        return {"ok": False, "error": f"could not reach bridge at {host}:{port} ({e}). "
                                       f"Is Blender open with the Stylized3D bridge started?"}
    f = sock.makefile("rwb")
    f.write((json.dumps({"code": code}) + "\n").encode("utf-8"))
    f.flush()
    line = f.readline()
    sock.close()
    if not line:
        return {"ok": False, "error": "empty response from bridge"}
    return json.loads(line.decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("code", nargs="?", help="Python code to execute")
    ap.add_argument("--file", help="read code from this file instead of the code argument")
    ap.add_argument("--port", type=int, default=9876)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--with-toolkit", action="store_true",
                     help="prepend sys.path.insert + `import bpy_stylized_kit as k`")
    args = ap.parse_args()

    if args.file:
        with open(args.file) as fh:
            code = fh.read()
    elif args.code:
        code = args.code
    else:
        ap.error("pass code as an argument or use --file")

    if args.with_toolkit:
        code = (
            f"import sys; sys.path.insert(0, {TOOLKIT_DIR!r})\n"
            "import bpy_stylized_kit as k\n"
            "import importlib; importlib.reload(k)  # bridge's Python persists across calls — always reload\n"
        ) + code

    resp = send(code, host=args.host, port=args.port)
    print(json.dumps(resp, indent=2))
    sys.exit(0 if resp.get("ok") else 1)


if __name__ == "__main__":
    main()
