#!/usr/bin/env python3
"""
Simple Master Coordinator for The Real Doomscroll
By Obvious Research

Now always serving an aiohttp endpoint at /next-video
which returns JSON metadata + video name, and a separate
/videos/{video_name} endpoint for streaming the video.
"""

import os
import json
from multiprocessing import Process

from aiohttp import web

# ───────────────────────────────────────────────────────────────
# CONFIGURATION: point this at your root directory of video subfolders
# ─────────────────────────────────────────────────────────────────
VIDEO_ROOT = "/workspace/videos"  # ← adjust this!
CURRENT_VIDEO = 0  # index into sorted subdirs


async def handle_next_video(request):
    """
    aiohttp handler for GET /next-video.
    Returns JSON containing the metadata + video filename.
    Increments CURRENT_VIDEO each request (wraps around).
    """
    global CURRENT_VIDEO

    # list and sort subdirs
    try:
        subdirs = sorted(
            d for d in os.listdir(VIDEO_ROOT)
            if os.path.isdir(os.path.join(VIDEO_ROOT, d))
        )
    except FileNotFoundError:
        return web.json_response({"success": False, "message": "VIDEO_ROOT not found"}, status=500)

    if not subdirs:
        return web.json_response({"success": False, "message": "No video subfolders"}, status=404)

    # wrap CURRENT_VIDEO in range
    idx = CURRENT_VIDEO % len(subdirs)
    chosen = subdirs[idx]
    folder = os.path.join(VIDEO_ROOT, chosen)

    # locate metadata.json and video file
    meta_path = next(
        (os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".json")),
        None
    )
    vid_filename = next(
        (f for f in os.listdir(folder)
         if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))),
        None
    )

    if not meta_path or not vid_filename:
        return web.json_response(
            {"success": False, "message": "Missing metadata or video in folder"},
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
        "video_name": vid_filename,
        "metadata": metadata
    }

    # increment for next request
    CURRENT_VIDEO += 1

    return web.json_response(response)


async def handle_video_stream(request):
    """
    aiohttp handler for GET /videos/{video_name}.
    Streams the raw video file from the appropriate subfolder.
    """
    video_name = request.match_info.get('video_name')

    # list and sort subdirs
    try:
        subdirs = sorted(
            d for d in os.listdir(VIDEO_ROOT)
            if os.path.isdir(os.path.join(VIDEO_ROOT, d))
        )
    except FileNotFoundError:
        return web.Response(status=500, text="VIDEO_ROOT not found")

    # search all subfolders for matching video
    for sub in subdirs:
        candidate = os.path.join(VIDEO_ROOT, sub, video_name)
        if os.path.isfile(candidate):
            # found, stream it
            return web.FileResponse(path=candidate)

    # not found
    return web.Response(status=404, text="Video not found")


def start_server():
    app = web.Application()
    app.router.add_get("/next-video", handle_next_video)
    app.router.add_get("/videos/{video_name}", handle_video_stream)
    web.run_app(app, port=8080, host="0.0.0.0")


if __name__ == "__main__":
    server_proc = Process(target=start_server, daemon=True)
    server_proc.start()
    print("🟢 aiohttp server listening on port 8080 globally")
    try:
        server_proc.join()
    except KeyboardInterrupt:
        print("\n🔴 Shutting down server...")
        server_proc.terminate()
        server_proc.join()