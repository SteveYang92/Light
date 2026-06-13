from __future__ import annotations

from pydantic import BaseModel

# ── Request schemas ─────────────────────────────────────


class UrlSubmit(BaseModel):
    url: str
    target_lang: str = "zh"
    video_format: str = ""
    bilingual: bool = False
    diarize: bool = False
    annotate: bool = False
    whisper_model: str = "ggml-large-v3-turbo.bin"
    llm_model: str = "deepseek-v4-flash"
    llm_base_url: str = "https://api.deepseek.com"


class ImportSubmit(BaseModel):
    output_dir: str
    video_path: str = ""
    title: str = ""


# ── Response schemas ────────────────────────────────────


class ChunkOut(BaseModel):
    id: str
    chunk_index: int
    duration: float | None
    video_ext: str = "mp4"
    subtitles: list[str] = []


class VideoOut(BaseModel):
    id: str
    title: str
    source: str
    source_url: str | None
    duration: float | None
    status: str
    thumbnail: str | None
    chunks: list[ChunkOut] = []
    run: dict | None = None
    created_at: str
    updated_at: str


class PipelineEvent(BaseModel):
    stage: str
    progress: float
    message: str
    chunk: int | None = None
    total_chunks: int | None = None


class VideoListOut(BaseModel):
    videos: list[VideoOut]
