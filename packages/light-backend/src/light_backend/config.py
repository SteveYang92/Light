from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class BackendConfig:
    host: str = "0.0.0.0"
    port: int = 8787
    data_dir: str = "./data"
    cookies_from_browser: str = ""
    cookies_file: str = ""

    @property
    def db_path(self) -> str:
        return os.path.join(self.data_dir, "light.db")

    @property
    def videos_dir(self) -> str:
        return os.path.join(self.data_dir, "videos")
