#!/usr/bin/env python3
"""
MCP server that bridges to a LIVE Blender session running blender_addon.py.

Setup:
  1. pip3 install mcp
  2. In Blender: install & enable blender_addon.py, then in the 3D viewport
     sidebar (N) > "Stylized3D" tab, click Start.
  3. Register this server, e.g.:
       claude mcp add blender-stylized-3d -- python3 <repo>/blender-stylized-3d/mcp_server/server.py

This is the opt-in live-viewing path. The headless scripts/run_blender.sh
workflow described in ../SKILL.md is the primary path and does not need this.
"""
import json
import os
import socket

from mcp.server.fastmcp import FastMCP

HOST = os.environ.get("STYLIZED3D_HOST", "127.0.0.1")
PORT = int(os.environ.get("STYLIZED3D_PORT", "9876"))
TOOLKIT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

mcp = FastMCP("blender-stylized-3d")


def _send(code: str, timeout: float = 35.0) -> dict:
    try:
        with socket.create_connection((HOST, PORT), timeout=5) as sock:
            sock.settimeout(timeout)
            sock.sendall((json.dumps({"code": code}) + "\n").encode("utf-8"))
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = sock.recv(65536)
                if not chunk:
                    break
                buf += chunk
    except (ConnectionRefusedError, socket.timeout, OSError) as e:
        return {"ok": False, "error": f"could not reach Blender bridge at {HOST}:{PORT} ({e}). "
                                       f"Is Blender open with the Stylized3D bridge started?"}
    if not buf:
        return {"ok": False, "error": "empty response from Blender bridge"}
    return json.loads(buf.decode("utf-8"))


@mcp.tool()
def execute_code(code: str) -> str:
    """Execute arbitrary Python in the live Blender session (bpy is pre-imported).
    Assign to a variable named `result` to get a value back in the response.
    To use the stylized-model helper library, first run:
        import sys; sys.path.insert(0, <path from get_toolkit_path>); import bpy_stylized_kit as k
    """
    return json.dumps(_send(code))


@mcp.tool()
def get_toolkit_path() -> str:
    """Absolute path to the directory containing bpy_stylized_kit.py, for sys.path.insert in execute_code."""
    return TOOLKIT_DIR


@mcp.tool()
def get_scene_info() -> str:
    """List objects currently in the live Blender scene (name, type, location)."""
    code = (
        "result = [{'name': o.name, 'type': o.type, 'location': list(o.location)} "
        "for o in bpy.data.objects]\n"
    )
    return json.dumps(_send(code))


@mcp.tool()
def render_preview(filepath: str) -> str:
    """Render the live scene's active camera view to a PNG at filepath, using the
    stylized color management (Standard view transform) from bpy_stylized_kit."""
    code = (
        "import sys\n"
        f"sys.path.insert(0, {TOOLKIT_DIR!r})\n"
        "import bpy_stylized_kit as k\n"
        f"k.render_still({filepath!r})\n"
        f"result = {filepath!r}\n"
    )
    return json.dumps(_send(code))


if __name__ == "__main__":
    mcp.run()
