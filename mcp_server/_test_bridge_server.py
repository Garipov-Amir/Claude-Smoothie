"""
Runs the REAL blender_addon.py production code (register, start the bridge)
inside Blender background mode, then manually pumps _drain_queue() from this
script's own main thread for a fixed window instead of relying on
bpy.app.timers (which needs Blender's GUI event loop ticking — it does not
tick in --background mode, so this substitutes a manual loop calling the
exact same function bpy.app.timers would have called, on the same thread
bpy.app.timers would have called it from). This validates the socket accept
loop, JSON framing, main-thread code execution, and error handling — the
actual bug-prone parts — without needing a GUI window this session has no
way to open or click into.

The one thing this does NOT verify is that Blender's GUI event loop
actually calls a registered bpy.app.timers callback on schedule — that's
standard, widely-documented Blender API behavior, not bridge-specific logic.

Usage: scripts/run_blender.sh mcp_server/_test_bridge_server.py <port> <duration_seconds>
"""
import sys
import os
import time

MCP_DIR = os.path.dirname(os.path.abspath(__file__))
if MCP_DIR not in sys.path:
    sys.path.insert(0, MCP_DIR)
import blender_addon as addon  # noqa: E402
import bpy  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:]
port = int(argv[0])
duration = float(argv[1])

addon.register()
bpy.context.scene.stylized3d_port = port
result = bpy.ops.stylized3d.start_server()
print("START_SERVER_RESULT", result)

start = time.time()
ticks = 0
while time.time() - start < duration:
    addon._drain_queue()
    ticks += 1
    time.sleep(0.05)

print("BRIDGE_TEST_DONE ticks=", ticks)
bpy.ops.stylized3d.stop_server()
