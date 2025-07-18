#!/usr/bin/env python3
"""
Simple Master Coordinator for The Real Doomscroll
By Obvious Research

Now always serving an aiohttp endpoint at /next-video
which returns JSON metadata + folder name, and a separate
/videos/{folder_name} endpoint for streaming the video.mp4.
"""

import os
import json
from multiprocessing import Process

from aiohttp import web

# ───────────────────────────────────────────────────────────────
# CONFIGURATION: point this at your root directory of video subfolders
# ─────────────────────────────────────────────────────────────────
VIDEO_ROOT = "/workspace/the-real-doomscroll/back/LTX-Video/generated_content"
CURRENT_VIDEO = 0  # index into sorted subdirs


async def handle_next_video(request):
    """
    aiohttp handler for GET /next-video.
    Returns JSON containing the folder name + metadata.json contents.
    Increments CURRENT_VIDEO each request (wraps around).
    """
    global CURRENT_VIDEO

    # list and sort subdirs (e.g., video_001, video_002, ...)
    try:
        subdirs = sorted(
            d for d in os.listdir(VIDEO_ROOT)
            if os.path.isdir(os.path.join(VIDEO_ROOT, d))
        )
    except FileNotFoundError:
        return web.json_response({"success": False, "message": "VIDEO_ROOT not found"}, status=500)

    if not subdirs:
        return web.json_response({"success": False, "message": "No video subfolders"}, status=404)

    # choose current index, wrap around
    idx = CURRENT_VIDEO % len(subdirs)
    chosen = subdirs[idx]
    folder = os.path.join(VIDEO_ROOT, chosen)

    # locate metadata.json
    meta_path = os.path.join(folder, "metadata.json")
    if not os.path.isfile(meta_path):
        return web.json_response(
            {"success": False, "message": "metadata.json not found in folder"},
            status=500
        )

    # load metadata
    try:
        with open(meta_path, 'r', encoding='utf-8') as mf:
            metadata = json.load(mf)
    except Exception as e:
        return web.json_response(
            {"success": False, "message": f"Error reading metadata: {e}"},
            status=500
        )

    # prepare response JSON
    response = {
        "success": True,
        "folder_name": chosen,
        "metadata": metadata
    }

    # increment for next request
    CURRENT_VIDEO += 1

    return web.json_response(response)


async def handle_video_stream(request):
    """
    aiohttp handler for GET /videos/{folder_name}.
    Streams the video.mp4 file from the given subfolder.
    """
    folder_name = request.match_info.get('folder_name')
    folder = os.path.join(VIDEO_ROOT, folder_name)
    video_path = os.path.join(folder, 'video.mp4')

    if not os.path.isdir(folder):
        return web.Response(status=404, text="Folder not found")
    if not os.path.isfile(video_path):
        return web.Response(status=404, text="video.mp4 not found in folder")

    return web.FileResponse(path=video_path)


def start_server():
    app = web.Application()
    app.router.add_get("/next-video", handle_next_video)
    app.router.add_get("/videos/{folder_name}", handle_video_stream)
    web.run_app(app, port=8080, host="0.0.0.0")


if __name__ == "__main__":
    server_proc = Process(target=start_server, daemon=True)
    server_proc.start()
    print("🟢 aiohttp server listening on port 8080")
    try:
        server_proc.join()
    except KeyboardInterrupt:
        print("\n🔴 Shutting down server...")
        server_proc.terminate()
        server_proc.join()