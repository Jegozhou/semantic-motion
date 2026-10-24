from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

_T0 = time.time()
VERBOSE = os.environ.get("SM_VERBOSE", "1") != "0"


def log(*args: Any) -> None:
    if VERBOSE:
        print(f"[sm {time.time() - _T0:7.1f}s]", *args, file=sys.stderr, flush=True)


def warn(*args: Any) -> None:
    print("[sm WARNING]", *args, file=sys.stderr, flush=True)


class SMError(RuntimeError):
    """User-facing error with an actionable message."""


def read_json(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _default(o: Any) -> Any:
    try:
        import numpy as np

        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
    except Exception:  # pragma: no cover
        pass
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def write_json(path: str | os.PathLike, data: Any, indent: int | None = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent, default=_default)
        f.write("\n")
    os.replace(tmp, path)


def sha1_file(path: str | os.PathLike, limit: int | None = None) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        n = 0
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
            n += len(b)
            if limit and n >= limit:
                break
    return h.hexdigest()


def sha1_obj(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=_default).encode()).hexdigest()


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def fmt_time(t: float) -> str:
    if t is None or (isinstance(t, float) and math.isnan(t)):
        return "?"
    m = int(t // 60)
    s = t - 60 * m
    return f"{m}:{s:05.2f}"
