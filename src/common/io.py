"""I/O helpers for CSV/JSON/Parquet and checksums."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path | str, payload: Any) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def read_json(path: Path | str) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_df(df: pd.DataFrame, path: Path | str) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix == ".json":
        df.to_json(path, orient="records", indent=2)
    else:
        df.to_parquet(path.with_suffix(".parquet"), index=False)
        path = path.with_suffix(".parquet")
    return path


def read_df(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported file type: {path}")


def file_checksum(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()