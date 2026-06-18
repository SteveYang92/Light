from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from light_subtitle.config import SubtitleConfig

from ..database import delete_chunks, get_video, list_chunks, list_videos, update_video
from ..models import ImportSubmit, UrlSubmit, VideoOut
from ..services.library import delete_video_and_files, get_video_detail, import_existing_output
from ..services.pipeline import _get_video_title, _set_main_loop, start_pipeline_thread
from ..state import get_config

router = APIRouter(prefix="/api/videos", tags=["videos"])


def _chunk_to_out(c: dict) -> dict:
    """Build a sanitized chunk response — no absolute filesystem paths."""
    raw_subs = c.get("subtitles", {})
    subtitle_keys = list(raw_subs.keys()) if isinstance(raw_subs, dict) else list(raw_subs)
    video_path = c.get("video_path", "")
    ext = Path(video_path).suffix.lstrip(".").lower() if video_path else "mp4"
    return {
        "id": c["id"],
        "chunk_index": c["chunk_index"],
        "duration": c.get("duration"),
        "video_ext": ext or "mp4",
        "subtitles": subtitle_keys,
    }


def _video_to_out(db_path: str, v: dict) -> VideoOut:
    from ..database import get_run

    run = get_run(db_path, v["id"])
    chunks = list_chunks(db_path, v["id"])
    thumb = f"/api/videos/{v['id']}/thumbnail" if v.get("thumbnail") else None
    return VideoOut(
        id=v["id"],
        title=v["title"],
        source=v["source"],
        source_url=v.get("source_url"),
        duration=v.get("duration"),
        status=v["status"],
        thumbnail=thumb,
        chunks=[_chunk_to_out(c) for c in chunks],
        run=dict(run) if run else None,
        created_at=v["created_at"],
        updated_at=v["updated_at"],
    )


@router.post("/url")
async def submit_url(body: UrlSubmit):
    """Submit a video URL for download + subtitle generation."""
    from ..database import insert_video

    cfg = get_config()

    title = _get_video_title(body.url)

    video = insert_video(
        cfg.db_path,
        title=title,
        source="url",
        source_url=body.url,
        status="pending",
        config_json=body.model_dump_json(),
    )
    vid = video["id"]

    sub_config = SubtitleConfig(
        input_path=body.url,
        output_dir="",
        target_lang=body.target_lang or None,
        bilingual=body.bilingual,
        whisper_model=body.whisper_model,
        llm_model=body.llm_model,
        llm_base_url=body.llm_base_url,
        diarize=body.diarize,
        annotate=body.annotate,
    )

    _set_main_loop(asyncio.get_running_loop())
    start_pipeline_thread(
        vid,
        sub_config,
        cfg.db_path,
        cfg.data_dir,
        body.video_format,
        cfg.cookies_from_browser,
        cfg.cookies_file,
    )

    return {"id": vid, "title": title, "status": "pending"}


@router.post("/import")
async def import_video(body: ImportSubmit):
    """Import an existing output directory. Automatically discovers video+subtitle files."""
    cfg = get_config()

    if not Path(body.output_dir).exists():
        raise HTTPException(400, f"目录不存在: {body.output_dir}")
    if body.video_path and not Path(body.video_path).exists():
        raise HTTPException(400, f"视频文件不存在: {body.video_path}")

    try:
        video = import_existing_output(
            cfg.db_path,
            cfg.data_dir,
            body.output_dir,
            video_path=body.video_path,
            title=body.title,
        )
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(400, str(e)) from e

    return {"id": video["id"], "title": video["title"], "chunks": len(video.get("chunks", [])), "status": "done"}


@router.get("")
async def list_all_videos():
    """List all videos."""
    cfg = get_config()
    videos = list_videos(cfg.db_path)
    return {"videos": [_video_to_out(cfg.db_path, v) for v in videos]}


@router.get("/{video_id}")
async def get_video_endpoint(video_id: str):
    """Get video detail with chunks."""
    cfg = get_config()
    video = get_video_detail(cfg.db_path, video_id)
    if video is None:
        raise HTTPException(404, "Video not found")
    return _video_to_out(cfg.db_path, video)


@router.delete("/{video_id}")
async def delete_video_endpoint(video_id: str):
    """Delete video and all associated files."""
    cfg = get_config()
    video = get_video(cfg.db_path, video_id)
    if video is None:
        raise HTTPException(404, "Video not found")
    delete_video_and_files(cfg.db_path, cfg.data_dir, video_id)
    return {"ok": True}


@router.get("/{video_id}/thumbnail")
async def get_video_thumbnail(video_id: str):
    """Get video thumbnail."""
    from fastapi.responses import FileResponse

    from .files import _guess_mime

    cfg = get_config()
    video = get_video(cfg.db_path, video_id)
    if video is None:
        raise HTTPException(404, "Video not found")
    thumb = video.get("thumbnail")
    if thumb and Path(thumb).exists():
        return FileResponse(thumb, media_type=_guess_mime(thumb))
    raise HTTPException(404, "No thumbnail")


@router.post("/{video_id}/retry")
async def retry_pipeline(video_id: str):
    """Retry failed pipeline — resumes from the last completed step.

    Does NOT delete downloaded video or segment files.  The runner will
    automatically:
      - reuse the cached download (find_cached_download)
      - reuse existing segments (find_existing_segments)
      - resume each segment from its pipeline_run.json (clone_for_segment
        sets resume=True when the file exists)
    """
    cfg = get_config()
    video = get_video(cfg.db_path, video_id)
    if video is None:
        raise HTTPException(404, "Video not found")

    update_video(cfg.db_path, video_id, status="pending")
    delete_chunks(cfg.db_path, video_id)

    # Rebuild config from stored parameters, falling back to defaults
    stored = {}
    if video.get("config_json"):
        try:
            stored = json.loads(video["config_json"])
        except (json.JSONDecodeError, TypeError):
            pass

    sub_config = SubtitleConfig(
        input_path=video.get("source_url") or "",
        output_dir="",
        resume=True,  # 断点续跑
        target_lang=stored.get("target_lang") or "zh",
        bilingual=stored.get("bilingual", False),
        whisper_model=stored.get("whisper_model", "ggml-large-v3-turbo.bin"),
        llm_model=stored.get("llm_model", "deepseek-v4-flash"),
        llm_base_url=stored.get("llm_base_url", "https://api.deepseek.com"),
        diarize=stored.get("diarize", False),
        annotate=stored.get("annotate", False),
    )

    start_pipeline_thread(
        video_id,
        sub_config,
        cfg.db_path,
        cfg.data_dir,
        stored.get("video_format", ""),
        cfg.cookies_from_browser,
        cfg.cookies_file,
    )
    return {"id": video_id, "status": "pending"}
