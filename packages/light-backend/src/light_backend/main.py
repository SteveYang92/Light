from __future__ import annotations

import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import BackendConfig
from .state import set_config


@asynccontextmanager
async def lifespan(app: FastAPI):
    from .database import init_db

    cfg = BackendConfig(
        data_dir=os.environ.get("LIGHT_DATA_DIR", "./data"),
        port=int(os.environ.get("LIGHT_PORT", "8787")),
        cookies_from_browser=os.environ.get("LIGHT_COOKIES_BROWSER", ""),
        cookies_file=os.environ.get("LIGHT_COOKIES_FILE", ""),
    )
    os.makedirs(cfg.videos_dir, exist_ok=True)
    init_db(cfg.db_path)
    set_config(cfg)
    yield


app = FastAPI(title="Light Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Import routers (after app so no circular dep) ─────

from .routers import files, pipeline_router, videos  # noqa: E402

app.include_router(videos.router)
app.include_router(pipeline_router.router)
app.include_router(files.router)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


def main():
    cfg = BackendConfig(
        data_dir=os.environ.get("LIGHT_DATA_DIR", "./data"),
        port=int(os.environ.get("LIGHT_PORT", "8787")),
        cookies_from_browser=os.environ.get("LIGHT_COOKIES_BROWSER", ""),
        cookies_file=os.environ.get("LIGHT_COOKIES_FILE", ""),
    )
    os.makedirs(cfg.videos_dir, exist_ok=True)
    from .database import init_db

    init_db(cfg.db_path)
    set_config(cfg)
    uvicorn.run(
        "light_backend.main:app",
        host=cfg.host,
        port=cfg.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
