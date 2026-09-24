# Live-viewing connector (optional)

This wraps a **running, GUI** Blender session as an MCP server so Claude can
drive it live and you can watch the model get built in the viewport. It's
opt-in — the headless workflow in [`../SKILL.md`](../SKILL.md) is the primary
path and works without any of this.

## How it works

- `blender_addon.py` is a Blender addon that opens a localhost socket and
  executes incoming Python on Blender's main thread (the only thread that can
  safely call `bpy`).
- `server.py` is an MCP stdio server that forwards tool calls to that socket.
  Each call opens a short-lived connection, sends `{"code": "..."}`, and gets
  back `{"ok": true/false, "result"/"error": ...}`.

## Setup

1. **Install the MCP SDK** for the server process:
   ```bash
   pip3 install mcp
   ```
2. **Install the addon in Blender**: Edit > Preferences > Add-ons > Install...,
   choose `blender_addon.py`, enable "Stylized3D MCP Bridge".
3. **Start the bridge**: in the 3D viewport, press `N` for the sidebar, open
   the "Stylized3D" tab, click **Start**. Default port is 9876 (change it in
   the panel if that's taken; then set `STYLIZED3D_PORT` to match when you
   register the server below).
4. **Register the server** with Claude Code:
   ```bash
   claude mcp add blender-stylized-3d -- python3 /Users/amirgaripov/dev/3d-skills/blender-stylized-3d/mcp_server/server.py
   ```
   (run from an interactive session — this session can't run `claude mcp add` itself).

   Or skip the `mcp` SDK entirely and talk to the bridge directly —
   `mcp_server/send_to_blender.py` is a small standalone CLI client (no
   dependencies beyond the standard library):
   ```bash
   python3 mcp_server/send_to_blender.py --with-toolkit "result = bpy.context.scene.name"
   python3 mcp_server/send_to_blender.py --file my_script.py --with-toolkit
   ```
   This is how the live-GUI verification below was actually done.

## Tools exposed

- `execute_code(code)` — run arbitrary Python in the live session. Use this
  for everything; it's the general-purpose escape hatch.
- `get_toolkit_path()` — path to `sys.path.insert` so `execute_code` calls can
  `import bpy_stylized_kit as k`.
- `get_scene_info()` — list current objects.
- `render_preview(filepath)` — render the active camera to a PNG.

## What's actually been verified

**Confirmed working against a real, live Blender GUI session** (not just
background-mode simulation): installed `blender_addon.py` via Preferences >
Add-ons > Install from Disk, started the bridge from the N-panel's
"Stylized3D" tab (`Status: running`, `Stylized3D bridge listening on
127.0.0.1:9876`), then drove it from a plain Python TCP client sending the
exact `{"code": ...}` protocol `server.py`'s tools use:
- `bpy` access against the real open scene (`{'scene': 'Scene', 'objects':
  ['Camera', 'Cube', 'Light']}` — matched what was on screen).
- Built a `bpy_stylized_kit.build_profile_body` loft + hand-painted material
  live — appeared in the viewport immediately (confirmed via a screenshot:
  the new object and its lights showed up in the outliner and 3D view).
- `render_still` through the same connection produced a correctly shaded
  PNG.
- Confirms `bpy.app.timers` really does get ticked by Blender's GUI event
  loop on schedule (the one thing background-mode testing structurally
  could not verify — see below for how that was tested instead).

Before this, the socket/protocol logic (accept loop, JSON framing, main-
thread code execution, error handling, connection reuse) was already
tested end-to-end against the real production code in `--background` mode
via `mcp_server/_test_bridge_server.py` (manually pumping `_drain_queue()`
since background mode has no window event loop to tick `bpy.app.timers`)
and `mcp_server/_test_client.py`. Between the two, both halves — the wire
protocol and the live GUI timer integration — are now independently
confirmed.

**Still not installed/tested**: `server.py`'s own `mcp` SDK wrapper —
installing the `mcp` package needs network access this sandboxed
environment doesn't have for PyPI. It's a thin, well-established pattern
(`@mcp.tool()` decorators forwarding to `_send()`, which is exactly what
`_test_client.py` and the live test above already exercise directly) — low
risk, but flagged rather than silently assumed correct.

**A separate, incompatible Blender MCP tool may also appear in some
environments** (tool names like `mcp__Blender__execute_blender_code`,
bundled Python API docs, screenshot/summary tools) — it happened to default
to the same port 9876 in one environment this was built in, which caused a
confusing timeout when tried against this addon. It is a different,
unrelated project with its own wire protocol; this addon does not
implement it and the two are not interchangeable. Use the plain TCP
`{"code": ...}` protocol (via `server.py`, or directly as shown above) to
talk to `blender_addon.py` specifically.

## Caveats

- One Blender instance, one bridge, one port — this isn't designed for
  concurrent sessions.
- The addon must stay running (Blender open, bridge started) for the whole
  time the server is in use; if Blender closes or the bridge is stopped,
  calls fail with a clear "could not reach Blender bridge" error rather than
  hanging.
- `execute_code` runs with full `bpy` access — same trust level as any other
  script you'd run in Blender. Don't point this at a Blender session with
  unsaved work you can't afford to lose without saving first.
