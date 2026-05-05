from __future__ import annotations

from pathlib import Path

from .models import StorageConfig


class ProjectPaths:
    def __init__(self, root: Path, storage: StorageConfig) -> None:
        self.root = root
        self.base_data_dir = root / storage.base_dir
        self.universe_path = self.base_data_dir / storage.universe_file
        self.daily_dir = self.base_data_dir / storage.daily_dir
        self.signals_dir = self.base_data_dir / storage.signals_dir
        self.features_dir = self.base_data_dir / "features"
        self.ml_dir = self.base_data_dir / "ml"
        self.reports_dir = root / storage.reports_dir

    def ensure(self) -> None:
        self.base_data_dir.mkdir(parents=True, exist_ok=True)
        self.universe_path.parent.mkdir(parents=True, exist_ok=True)
        self.daily_dir.mkdir(parents=True, exist_ok=True)
        self.signals_dir.mkdir(parents=True, exist_ok=True)
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.ml_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
