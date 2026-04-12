from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pandas as pd

from stocks_analyzer.cli import _refresh_or_load_universe
from stocks_analyzer.config import load_config
from stocks_analyzer.paths import ProjectPaths
from stocks_analyzer.storage import Storage


class EmptyProvider:
    def get_instruments(self) -> pd.DataFrame:
        return pd.DataFrame(columns=["symbol", "name", "trade_status"])


def _make_workspace_tmp_dir(name: str) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    path = project_root / ".tmp_tests" / f"{name}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_refresh_or_load_universe_falls_back_to_cached_universe() -> None:
    project_root = Path(__file__).resolve().parents[1]
    root = _make_workspace_tmp_dir("update_fallback")
    config = load_config(project_root / "config" / "default.yaml")
    paths = ProjectPaths(root, config.storage)
    storage = Storage(paths)
    cached = pd.DataFrame(
        [
            {"symbol": "600000", "name": "浦发银行", "exchange": "sh", "board": "main_board", "is_st": False, "is_suspended": False}
        ]
    )
    storage.save_universe(cached)

    result = _refresh_or_load_universe(storage, EmptyProvider(), exclude_st=True)

    assert result["symbol"].tolist() == ["600000"]
