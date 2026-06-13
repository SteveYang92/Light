import hashlib
import shutil
from pathlib import Path


class ASRCache:
    def __init__(self, cache_dir: Path = Path(".cache/light-regression/asr")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, audio_path: Path) -> Path | None:
        h = self._hash_file(audio_path)
        cached = self.cache_dir / f"{h}.json"
        return cached if cached.exists() else None

    def save(self, audio_path: Path, transcript_path: Path) -> Path:
        h = self._hash_file(audio_path)
        cached = self.cache_dir / f"{h}.json"
        shutil.copy2(transcript_path, cached)
        return cached

    def _hash_file(self, path: Path) -> str:
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
        return sha.hexdigest()[:16]
