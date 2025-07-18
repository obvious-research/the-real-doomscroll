#!/usr/bin/env python3
"""
Simple Master Coordinator for The Real Doomscroll
By Obvious Research

Now always serving an aiohttp endpoint at /next-video
which returns JSON metadata + folder name, and separate
endpoints for streaming video, audio, and subtitles.
"""

import os
import json
import time
from multiprocessing import Process

from aiohttp import web
import aiohttp_cors

# ───────────────────────────────────────────────────────────────
# CONFIGURATION: point this at your root directory of video subfolders
# ─────────────────────────────────────────────────────────────────
VIDEO_ROOT = "/workspace/the-real-doomscroll/back/LTX-Video/generated_content"
CURRENT_VIDEO = 0  # index into sorted subdirs
LAST_REQUEST_TIME = None  # Global variable to store the timestamp of the last request


async def handle_next_video(request):
    """
    aiohttp handler for GET /next-video.
    Returns the raw contents of metadata.json for the next video.
    Increments CURRENT_VIDEO each request (wraps around).
    """
    global CURRENT_VIDEO, LAST_REQUEST_TIME
    current_time = time.time()

    # list and sort subdirs (e.g., video_001, video_002, ...)
    try:
        subdirs = sorted(
            d for d in os.listdir(VIDEO_ROOT)
            if os.path.isdir(os.path.join(VIDEO_ROOT, d))
        )
    except FileNotFoundError:
        return web.json_response({"error": "VIDEO_ROOT not found"}, status=500)

    if not subdirs:
        return web.json_response({"error": "No video subfolders found"}, status=404)

    if LAST_REQUEST_TIME is not None and CURRENT_VIDEO > 1:
        watch_duration = current_time - LAST_REQUEST_TIME

        # The watch duration is for the *previous* video.
        previous_video_idx = (CURRENT_VIDEO - 2)
        previous_folder_name = subdirs[previous_video_idx]
        stats_file_path = os.path.join(VIDEO_ROOT, previous_folder_name, 'watch_stats.txt')

        try:
            # Write the duration to the file, creating/overwriting it.
            with open(stats_file_path, 'w', encoding='utf-8') as f:
                f.write(str(watch_duration/13.3))
        except Exception as e:
            # Log an error if writing fails, but don't stop the request.
            print(f"Error writing watch_stats.txt for {previous_folder_name}: {e}")

    # choose current index, wrap around
    idx = CURRENT_VIDEO % len(subdirs)
    chosen_folder_name = subdirs[idx]
    folder_path = os.path.join(VIDEO_ROOT, chosen_folder_name)

    # locate metadata.json
    meta_path = os.path.join(folder_path, "metadata.json")
    if not os.path.isfile(meta_path):
        return web.json_response(
            {"error": f"metadata.json not found in folder {chosen_folder_name}"},
            status=500
        )

    # load metadata
    try:
        with open(meta_path, 'r', encoding='utf-8') as mf:
            metadata = json.load(mf)
    except Exception as e:
        return web.json_response(
            {"error": f"Error reading or parsing metadata: {e}"},
            status=500
        )

    # Update the last request time for the next cycle
    LAST_REQUEST_TIME = current_time

    # increment for the next request
    CURRENT_VIDEO += 1

    # Return the metadata.json content directly
    return web.json_response(metadata)


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


async def handle_audio_stream(request):
    """
    aiohttp handler for GET /audio/{folder_name}.
    Streams the audio.wav file from the given subfolder.
    """
    folder_name = request.match_info.get('folder_name')
    folder = os.path.join(VIDEO_ROOT, folder_name)
    audio_path = os.path.join(folder, 'final_audio.wav')

    if not os.path.isdir(folder):
        return web.Response(status=404, text="Folder not found")
    if not os.path.isfile(audio_path):
        return web.Response(status=404, text="audio.wav not found in folder")

    return web.FileResponse(path=audio_path)


async def handle_subtitles_stream(request):
    """
    aiohttp handler for GET /subtitles/{folder_name}.
    Streams the subtitles.vtt file from the given subfolder.
    """
    folder_name = request.match_info.get('folder_name')
    folder = os.path.join(VIDEO_ROOT, folder_name)
    subtitles_path = os.path.join(folder, 'subtitles.vtt')

    if not os.path.isdir(folder):
        return web.Response(status=404, text="Folder not found")
    if not os.path.isfile(subtitles_path):
        return web.Response(status=404, text="subtitles.vtt not found in folder")

    return web.FileResponse(path=subtitles_path)


def start_server():
    """Initializes and starts the aiohttp web server."""
    app = web.Application()
    app.router.add_get("/next-video", handle_next_video)
    app.router.add_get("/videos/{folder_name}", handle_video_stream)
    app.router.add_get("/audio/{folder_name}", handle_audio_stream)
    app.router.add_get("/subtitles/{folder_name}", handle_subtitles_stream)

    # then set up CORS
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        )
    })

    # apply CORS to all routes
    for route in list(app.router.routes()):
        cors.add(route)

    web.run_app(app, port=8081, host="0.0.0.0")


if __name__ == "__main__":
    server_proc = Process(target=start_server, daemon=True)
    server_proc.start()
    print("🟢 aiohttp server listening on port 8081")
    try:
        server_proc.join()
    except KeyboardInterrupt:
        print("\n🔴 Shutting down server...")
        server_proc.terminate()
        server_proc.join()