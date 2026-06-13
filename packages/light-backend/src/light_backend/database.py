from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid

_local = threading.local()


def _get_conn(db_path: str) -> sqlite3.Connection:
    if not hasattr(_local, "conn") or _local.conn is None or getattr(_local, "db_path", None) != db_path:
        _local.conn = sqlite3.connect(db_path)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA foreign_keys=ON")
        _local.db_path = db_path
    return _local.conn


def init_db(db_path: str) -> None:
    conn = _get_conn(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS videos (
            id          TEXT PRIMARY KEY,
            title       TEXT NOT NULL,
            source      TEXT NOT NULL,
            source_url  TEXT,
            duration    REAL,
            status      TEXT NOT NULL DEFAULT 'pending',
            thumbnail   TEXT,
            config_json TEXT,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id          TEXT PRIMARY KEY,
            video_id    TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            video_path  TEXT NOT NULL,
            output_dir  TEXT NOT NULL,
            duration    REAL,
            subtitles   TEXT DEFAULT '{}',
            created_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id             TEXT PRIMARY KEY,
            video_id       TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
            status         TEXT NOT NULL DEFAULT 'queued',
            stage          TEXT,
            progress       REAL DEFAULT 0.0,
            total_chunks   INTEGER DEFAULT 0,
            current_chunk  INTEGER DEFAULT 0,
            error_msg      TEXT,
            started_at     TEXT,
            finished_at    TEXT
        );
    """)
    conn.commit()
    # ── Migration: add config_json column for existing DBs ──
    try:
        conn.execute("ALTER TABLE videos ADD COLUMN config_json TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # column already exists


def generate_id() -> str:
    ts = int(time.time() * 1000)
    return f"{ts:013x}{uuid.uuid4().hex[:13]}"


# ── Video CRUD ──────────────────────────────────────────


def insert_video(db_path: str, **kw) -> dict:
    vid = generate_id()
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    conn = _get_conn(db_path)
    conn.execute(
        "INSERT INTO videos "
        "(id, title, source, source_url, duration, status, thumbnail, config_json, created_at, updated_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            vid,
            kw["title"],
            kw["source"],
            kw.get("source_url"),
            kw.get("duration"),
            kw.get("status", "pending"),
            kw.get("thumbnail"),
            kw.get("config_json"),
            now,
            now,
        ),
    )
    conn.commit()
    return get_video(db_path, vid)


def get_video(db_path: str, vid: str) -> dict | None:
    conn = _get_conn(db_path)
    row = conn.execute("SELECT * FROM videos WHERE id=?", (vid,)).fetchone()
    if row is None:
        return None
    return dict(row)


def list_videos(db_path: str) -> list[dict]:
    conn = _get_conn(db_path)
    rows = conn.execute("SELECT * FROM videos ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


def update_video(db_path: str, vid: str, **kw) -> None:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    kw["updated_at"] = now
    pairs = ", ".join(f"{k}=?" for k in kw)
    vals = list(kw.values()) + [vid]
    conn = _get_conn(db_path)
    conn.execute(f"UPDATE videos SET {pairs} WHERE id=?", vals)
    conn.commit()


def delete_video(db_path: str, vid: str) -> None:
    conn = _get_conn(db_path)
    conn.execute("DELETE FROM videos WHERE id=?", (vid,))
    conn.commit()


# ── Chunk CRUD ──────────────────────────────────────────


def insert_chunk(db_path: str, **kw) -> dict:
    cid = generate_id()
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    conn = _get_conn(db_path)
    conn.execute(
        "INSERT INTO chunks (id, video_id, chunk_index, video_path, output_dir, duration, subtitles, created_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (
            cid,
            kw["video_id"],
            kw["chunk_index"],
            kw["video_path"],
            kw["output_dir"],
            kw.get("duration"),
            json.dumps(kw.get("subtitles", {})),
            now,
        ),
    )
    conn.commit()
    return get_chunk(db_path, cid)


def get_chunk(db_path: str, cid: str) -> dict | None:
    conn = _get_conn(db_path)
    row = conn.execute("SELECT * FROM chunks WHERE id=?", (cid,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["subtitles"] = json.loads(d.get("subtitles", "{}"))
    return d


def list_chunks(db_path: str, video_id: str) -> list[dict]:
    conn = _get_conn(db_path)
    rows = conn.execute("SELECT * FROM chunks WHERE video_id=? ORDER BY chunk_index", (video_id,)).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["subtitles"] = json.loads(d.get("subtitles", "{}"))
        result.append(d)
    return result


def delete_chunks(db_path: str, video_id: str) -> None:
    conn = _get_conn(db_path)
    conn.execute("DELETE FROM chunks WHERE video_id=?", (video_id,))
    conn.commit()


# ── Pipeline Run CRUD ───────────────────────────────────


def insert_run(db_path: str, video_id: str) -> str:
    rid = generate_id()
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    conn = _get_conn(db_path)
    conn.execute(
        "INSERT INTO pipeline_runs (id, video_id, status, started_at) VALUES (?,?,?,?)",
        (rid, video_id, "running", now),
    )
    conn.commit()
    return rid


def update_run(db_path: str, rid: str, **kw) -> None:
    pairs = ", ".join(f"{k}=?" for k in kw)
    vals = list(kw.values()) + [rid]
    conn = _get_conn(db_path)
    conn.execute(f"UPDATE pipeline_runs SET {pairs} WHERE id=?", vals)
    conn.commit()


def get_run(db_path: str, video_id: str) -> dict | None:
    conn = _get_conn(db_path)
    row = conn.execute(
        "SELECT * FROM pipeline_runs WHERE video_id=? ORDER BY started_at DESC LIMIT 1", (video_id,)
    ).fetchone()
    if row is None:
        return None
    return dict(row)
