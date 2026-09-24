#!/usr/bin/env python3
"""
Standalone test client for the Stylized3D MCP Bridge protocol — sends the
exact same newline-delimited-JSON requests server.py's tools send, against
a real running bridge. Retries connecting for a while since the Blender
side needs a moment to start. Writes results to <outdir>/client_result.json.

Usage: python3 _test_client.py <port> <outdir>
"""
import json
import os
import socket
import sys
import time

port = int(sys.argv[1])
outdir = sys.argv[2]
os.makedirs(outdir, exist_ok=True)


def send(sock_file_pair, code, timeout=10):
    sock, f = sock_file_pair
    f.write((json.dumps({"code": code}) + "\n").encode("utf-8"))
    f.flush()
    sock.settimeout(timeout)
    line = f.readline()
    return json.loads(line.decode("utf-8"))


def connect(port, retries=60, delay=0.5):
    last_err = None
    for _ in range(retries):
        try:
            sock = socket.create_connection(("127.0.0.1", port), timeout=2)
            return sock, sock.makefile("rwb")
        except (ConnectionRefusedError, OSError) as e:
            last_err = e
            time.sleep(delay)
    raise RuntimeError(f"could not connect to bridge on port {port} after {retries * delay}s: {last_err}")


results = {}
try:
    pair = connect(port)
    results["connected"] = True

    # 1. basic exec + result round trip
    results["basic_math"] = send(pair, "result = 21 * 2")

    # 2. bpy access actually works (real Blender scene, not a mock)
    results["bpy_access"] = send(pair, "result = bpy.context.scene.name")

    # 3. the toolkit is importable and usable through the bridge, same as a
    #    real modeling session would do — matches how server.py's tools embed
    #    TOOLKIT_DIR as a literal path, not a bare variable
    toolkit_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
    results["toolkit_use"] = send(pair, (
        f"import sys; sys.path.insert(0, {toolkit_dir!r})\n"
        "import bpy_stylized_kit as k\n"
        "k.new_scene()\n"
        "obj = k.add_primitive('cube', name='BridgeTestCube')\n"
        "result = (obj.name, len(bpy.data.objects))\n"
    ))

    # 4. error handling: exceptions in the executed code come back as ok:false
    #    with a traceback, not a crash/hang
    results["error_handling"] = send(pair, "1 / 0")

    # 5. a second request on the SAME connection works (protocol loops, not one-shot)
    results["second_request"] = send(pair, "result = 'still alive'")

except Exception as e:
    results["client_error"] = repr(e)

with open(os.path.join(outdir, "client_result.json"), "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
