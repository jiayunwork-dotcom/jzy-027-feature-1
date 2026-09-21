"""几何档本地文件读写。

按需求“标定当次完成、不另建流水库”：这里只持久化跨距几何
（档距、高差、w、档名），每档一个 JSON 文件，原子替换写入。
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from . import validation
from .catenary import SpanGeometry
from .errors import SpanConflict, SpanNotFound

_DEFAULT_DATA_DIR = os.environ.get("CATENARY_DATA_DIR", "/data")


class SpanStore:
    def __init__(self, data_dir: str | os.PathLike = _DEFAULT_DATA_DIR) -> None:
        # 惰性建目录：导入应用时不触碰文件系统，仅在首次写入时创建。
        self.dir = Path(data_dir)

    def _ensure_dir(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        name = validation.require_span_name(name)
        return self.dir / f"{name}.json"

    def save(self, name: str, geometry: SpanGeometry, *, overwrite: bool = False) -> None:
        path = self._path(name)
        if path.exists() and not overwrite:
            raise SpanConflict(f"档 {name!r} 已登记；如要更新几何请用覆盖方式")
        record = {"name": name, **geometry.to_dict()}
        # 原子写：同目录临时文件写完再 rename。
        self._ensure_dir()
        fd, tmp = tempfile.mkstemp(prefix=f".{name}.", suffix=".tmp", dir=self.dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(record, fh, ensure_ascii=False, indent=2)
                fh.write("\n")
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def get(self, name: str) -> tuple[str, SpanGeometry]:
        path = self._path(name)
        if not path.exists():
            raise SpanNotFound(f"找不到名为 {name!r} 的几何档")
        with path.open(encoding="utf-8") as fh:
            record = json.load(fh)
        return record["name"], SpanGeometry.create(
            record["span"], record["height_difference"], record["weight_per_length"]
        )

    def delete(self, name: str) -> None:
        path = self._path(name)
        if not path.exists():
            raise SpanNotFound(f"找不到名为 {name!r} 的几何档")
        path.unlink()

    def list(self) -> list[dict]:
        if not self.dir.exists():
            return []
        out: list[dict] = []
        for path in sorted(self.dir.glob("*.json")):
            with path.open(encoding="utf-8") as fh:
                record = json.load(fh)
            out.append(record)
        return out
