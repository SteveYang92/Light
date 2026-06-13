from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..database import get_run, get_video
from ..services.pipeline import _set_main_loop, cleanup_event_queue, get_event_queue
from ..state import get_config

router = APIRouter(prefix="/api/videos/{video_id}/pipeline", tags=["pipeline"])


@router.get("/events")
async def pipeline_events(video_id: str):
    """SSE endpoint for real-time pipeline progress."""
    cfg = get_config()
    video = get_video(cfg.db_path, video_id)
    if video is None:
        raise HTTPException(404, "Video not found")

    queue = get_event_queue(video_id)
    _set_main_loop(asyncio.get_running_loop())

    async def event_stream():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30)
                except TimeoutError:
                    yield ": keepalive\n\n"
                    continue

                data = f"data: {event['stage']}|{event['progress']}|{event['message']}"
                if event.get("chunk") is not None:
                    data += f"|{event['chunk']}|{event.get('total_chunks', 0)}"
                yield f"{data}\n\n"

                if event["stage"] == "done":
                    break
        finally:
            cleanup_event_queue(video_id)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/run")
async def get_pipeline_run(video_id: str):
    """Get latest pipeline run status."""
    cfg = get_config()
    run = get_run(cfg.db_path, video_id)
    if run is None:
        raise HTTPException(404, "No pipeline run found")
    return dict(run)
