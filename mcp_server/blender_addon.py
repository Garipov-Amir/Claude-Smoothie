"""
Stylized3D MCP Bridge — Blender addon.

Runs a small localhost socket server inside a live (GUI) Blender session so an
external MCP server (mcp_server/server.py) can send it Python to execute,
letting Claude drive the session live while the user watches. This is the
opt-in "live view" path — the headless scripts/run_blender.sh workflow is the
primary, always-available one and doesn't need this addon.

Install: Blender > Edit > Preferences > Add-ons > Install..., pick this file,
enable "Stylized3D MCP Bridge". Then in the 3D viewport, open the sidebar
(press N) > "Stylized3D" tab > Start.

Protocol: newline-delimited JSON over TCP. Request: {"code": "<python>"}.
Response: {"ok": true, "result": "<repr(result) if you set a `result` var>"}
       or {"ok": false, "error": "<traceback>"}.
"""

bl_info = {
    "name": "Stylized3D MCP Bridge",
    "author": "3d-skills",
    "version": (1, 0, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Stylized3D",
    "description": "Socket bridge so an external MCP server (Claude) can drive this Blender session live",
    "category": "Development",
}

import bpy
import json
import os
import queue
import socket
import sys
import threading
import traceback

TOOLKIT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
if TOOLKIT_DIR not in sys.path:
    sys.path.insert(0, TOOLKIT_DIR)

_server_socket = None
_server_thread = None
_stop_event = threading.Event()
_request_queue = queue.Queue()  # (code, result_holder_dict, threading.Event)
_running = False


def _send(conn_file, payload):
    conn_file.write((json.dumps(payload) + "\n").encode("utf-8"))
    conn_file.flush()


def _handle_client(conn, addr):
    conn_file = conn.makefile("rwb")
    try:
        while not _stop_event.is_set():
            line = conn_file.readline()
            if not line:
                break
            try:
                req = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError as e:
                _send(conn_file, {"ok": False, "error": f"bad json: {e}"})
                continue
            code = req.get("code", "")
            result_holder = {}
            done = threading.Event()
            _request_queue.put((code, result_holder, done))
            done.wait(timeout=30)
            if not done.is_set():
                _send(conn_file, {"ok": False, "error": "execution timed out (30s)"})
            else:
                _send(conn_file, result_holder)
    except (ConnectionResetError, OSError):
        pass
    finally:
        conn.close()


def _accept_loop(sock):
    sock.settimeout(1.0)
    while not _stop_event.is_set():
        try:
            conn, addr = sock.accept()
        except socket.timeout:
            continue
        except OSError:
            break
        threading.Thread(target=_handle_client, args=(conn, addr), daemon=True).start()


def _drain_queue():
    """Runs on Blender's main thread via bpy.app.timers — the only safe place for bpy calls."""
    try:
        while True:
            code, result_holder, done = _request_queue.get_nowait()
            local_ns = {"bpy": bpy, "result": None}
            try:
                exec(code, local_ns)
                result_holder["ok"] = True
                result_holder["result"] = repr(local_ns.get("result"))
            except Exception:
                result_holder["ok"] = False
                result_holder["error"] = traceback.format_exc()
            done.set()
    except queue.Empty:
        pass
    return 0.05 if _running else None


class STYLIZED3D_OT_start_server(bpy.types.Operator):
    bl_idname = "stylized3d.start_server"
    bl_label = "Start Stylized3D Bridge"

    def execute(self, context):
        global _server_socket, _server_thread, _running
        if _running:
            self.report({"INFO"}, "Bridge already running")
            return {"CANCELLED"}
        _stop_event.clear()
        port = context.scene.stylized3d_port
        try:
            _server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            _server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            _server_socket.bind(("127.0.0.1", port))
            _server_socket.listen(5)
        except OSError as e:
            self.report({"ERROR"}, f"Could not bind port {port}: {e}")
            return {"CANCELLED"}
        _server_thread = threading.Thread(target=_accept_loop, args=(_server_socket,), daemon=True)
        _server_thread.start()
        _running = True
        bpy.app.timers.register(_drain_queue)
        self.report({"INFO"}, f"Stylized3D bridge listening on 127.0.0.1:{port}")
        return {"FINISHED"}


class STYLIZED3D_OT_stop_server(bpy.types.Operator):
    bl_idname = "stylized3d.stop_server"
    bl_label = "Stop Stylized3D Bridge"

    def execute(self, context):
        global _server_socket, _running
        _running = False
        _stop_event.set()
        if _server_socket:
            try:
                _server_socket.close()
            except OSError:
                pass
            _server_socket = None
        self.report({"INFO"}, "Stylized3D bridge stopped")
        return {"FINISHED"}


class STYLIZED3D_PT_panel(bpy.types.Panel):
    bl_label = "Stylized3D Bridge"
    bl_idname = "STYLIZED3D_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Stylized3D"

    def draw(self, context):
        layout = self.layout
        layout.prop(context.scene, "stylized3d_port")
        layout.label(text=f"Status: {'running' if _running else 'stopped'}")
        layout.operator("stylized3d.start_server", icon="PLAY")
        layout.operator("stylized3d.stop_server", icon="PAUSE")


_classes = (STYLIZED3D_OT_start_server, STYLIZED3D_OT_stop_server, STYLIZED3D_PT_panel)


def register():
    bpy.types.Scene.stylized3d_port = bpy.props.IntProperty(name="Port", default=9876, min=1024, max=65535)
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    global _running
    _running = False
    _stop_event.set()
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.stylized3d_port


if __name__ == "__main__":
    register()
