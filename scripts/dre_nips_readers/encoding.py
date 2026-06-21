from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


TEXT_ENCODINGS = ("utf-8-sig", "utf-8", "gbk", "gb18030", "latin1")


def read_text_with_fallback(path: str | Path) -> str:
    raw = Path(path).read_bytes()
    errors: list[str] = []
    for encoding in TEXT_ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise UnicodeDecodeError("unknown", raw, 0, 1, "; ".join(errors))


def read_tsv_with_fallback(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    return read_delimited_with_fallback(path, sep="\t", **kwargs)


def read_delimited_with_fallback(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in TEXT_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    return pd.read_csv(path, **kwargs)


def read_json_with_fallback(path: str | Path) -> dict[str, Any]:
    return json.loads(read_text_with_fallback(path))
