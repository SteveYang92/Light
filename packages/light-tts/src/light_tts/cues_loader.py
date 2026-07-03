from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Cue:
    cue_id: str
    start: float
    end: float
    text: str
    speaker: str
    lang: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def resolve_cues_path(path: str | Path) -> Path:
    """Resolve *path* to a ``cues.json`` file (accepts file or pipeline output dir)."""
    p = Path(path).expanduser()
    if p.is_file():
        return p.resolve()
    if p.is_dir():
        return find_cues_json(p)
    raise FileNotFoundError(
        f"cues.json not found: {p}\n"
        "  Pass a cues.json file or a pipeline output directory, e.g.\n"
        "  --cues output/William_Bill_Maher\n"
        "  --cues output/William_Bill_Maher/cues.json"
    )


def load_cues(cues_path: str | Path, *, lang: str, max_cues: int | None = None) -> list[Cue]:
    """Load subtitle cues from ``cues.json`` filtered by *lang*."""
    path = resolve_cues_path(cues_path)

    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("cues", []) if isinstance(data, dict) else data
    if not isinstance(raw, list):
        raise ValueError(f"Invalid cues.json format: {path}")

    cues: list[Cue] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        cue_lang = str(item.get("lang", ""))
        if cue_lang != lang:
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        cues.append(
            Cue(
                cue_id=str(item.get("cue_id") or item.get("id") or len(cues)),
                start=float(item.get("start", 0)),
                end=float(item.get("end", 0)),
                text=text,
                speaker=str(item.get("speaker", "")),
                lang=cue_lang,
            )
        )
        if max_cues is not None and len(cues) >= max_cues:
            break

    if not cues:
        raise ValueError(f"No cues with lang={lang!r} in {path}")

    empty_speakers = sum(1 for c in cues if not c.speaker)
    if empty_speakers:
        warnings.warn(
            f"{empty_speakers}/{len(cues)} cues have empty speaker — run with --diarize for multi-voice dubbing.",
            stacklevel=2,
        )
    return cues


def find_cues_json(output_dir: str | Path) -> Path:
    """Locate ``cues.json`` under a pipeline output directory."""
    root = Path(output_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")

    direct = root / "cues.json"
    if direct.is_file():
        return direct

    merged = sorted(root.glob("*.cues.json"))
    if len(merged) == 1:
        return merged[0]
    if len(merged) > 1:
        names = "\n".join(f"  {p.name}" for p in merged)
        raise RuntimeError(f"Multiple *.cues.json under {root}:\n{names}\nPass --cues with the file you want.")

    matches = sorted(root.rglob("cues.json"))
    if len(matches) == 1:
        return matches[0]
    if matches:
        # Common mistake: repo-level output/ holding many video runs.
        shallow = sorted(matches, key=lambda p: len(p.relative_to(root).parts))
        preview = "\n".join(f"  {m.relative_to(root)}" for m in shallow[:8])
        more = f"\n  ... and {len(matches) - 8} more" if len(matches) > 8 else ""
        raise FileNotFoundError(
            f"No cues.json at {root}/ — found {len(matches)} nested files.\n"
            f"Point to one pipeline run directory, for example:\n"
            f"  {shallow[0].parent}\n"
            f"Or pass --cues explicitly. Nested matches include:\n{preview}{more}"
        )

    raise FileNotFoundError(
        f"No cues.json under {root}.\n"
        "Run the subtitle pipeline first, e.g.\n"
        "  uv run light-subtitle -i video.mp4 --diarize --target-lang zh -o output/my_run"
    )


def _raw_json_has_lang(path: Path, lang: str) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("cues", []) if isinstance(data, dict) else data
    if not isinstance(raw, list):
        return False
    return any(isinstance(item, dict) and str(item.get("lang", "")) == lang for item in raw)


def find_raw_json(output_dir: str | Path, *, lang: str) -> Path:
    """Locate ``translations/raw.json`` under a pipeline output directory."""
    root = Path(output_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")

    direct = root / "translations" / "raw.json"
    if direct.is_file():
        if not _raw_json_has_lang(direct, lang):
            raise ValueError(f"No cues with lang={lang!r} in {direct}")
        return direct

    matches = sorted(root.rglob("translations/raw.json"))
    valid = [path for path in matches if _raw_json_has_lang(path, lang)]
    if len(valid) == 1:
        return valid[0]
    if valid:
        preview = "\n".join(f"  {m.relative_to(root)}" for m in valid[:8])
        more = f"\n  ... and {len(valid) - 8} more" if len(valid) > 8 else ""
        raise FileNotFoundError(
            f"Multiple translations/raw.json under {root} contain lang={lang!r}.\n"
            f"Pass --cues explicitly. Matches include:\n{preview}{more}"
        )
    if matches:
        raise FileNotFoundError(
            f"Found translations/raw.json under {root}, but none contain lang={lang!r}.\n"
            "Re-run the subtitle pipeline with --target-lang, e.g.\n"
            f"  uv run light-subtitle -i video.mp4 --target-lang {lang} -o {root.name}"
        )

    raise FileNotFoundError(
        f"No translations/raw.json under {root}.\n"
        "Dubbing requires translated cues with punctuation from the LLM translate step.\n"
        "Run the subtitle pipeline with --target-lang, e.g.\n"
        f"  uv run light-subtitle -i video.mp4 --target-lang {lang} -o output/my_run"
    )


def resolve_dub_cues_path(output_dir: str | Path, *, lang: str, explicit: str | Path | None = None) -> Path:
    """Resolve dubbing text from ``translations/raw.json`` only (not display ``cues.json``)."""
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file():
            if not _raw_json_has_lang(path, lang):
                raise ValueError(f"No cues with lang={lang!r} in {path}")
            return path.resolve()
        return find_raw_json(path, lang=lang)
    return find_raw_json(output_dir, lang=lang)
