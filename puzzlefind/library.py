"""多照片索引库。每张照片一个 JSON 文件，落在 config.INDEX_DIR。

查询跨所有照片进行——1000 块碎片分散在十几张照片里，用户不该需要
记得某个编号在哪张图上。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import config, vocabulary
from .models import Piece, PhotoIndex


@dataclass
class QueryResult:
    """一次查询的结果。

    未命中时 unrecognized 会被填上——这是本工具的关键设计：
    「没找到」这句话毫无信息量，但「没找到，而这 5 块是未识别的」
    能把搜索范围从几百块塌缩到个位数。
    """

    found: bool
    code: str
    photo_id: str | None = None
    piece: Piece | None = None
    unrecognized: dict[str, list[Piece]] = field(default_factory=dict)


class Library:
    def __init__(self, index_dir: Path | None = None) -> None:
        self.index_dir = Path(index_dir) if index_dir else config.INDEX_DIR
        self._photos: dict[str, PhotoIndex] = {}

    @property
    def photos(self) -> list[PhotoIndex]:
        return list(self._photos.values())

    @classmethod
    def load(cls, index_dir: Path | None = None) -> Library:
        library = cls(index_dir)
        if not library.index_dir.exists():
            return library
        for path in sorted(library.index_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                index = PhotoIndex.from_dict(data)
            except (json.JSONDecodeError, KeyError, ValueError):
                # 单个损坏的索引文件不该让整个库加载失败
                continue
            library._photos[index.photo_id] = index
        return library

    def save_photo(self, index: PhotoIndex) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        path = self.index_dir / f"{index.photo_id}.json"
        path.write_text(
            json.dumps(index.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._photos[index.photo_id] = index

    def delete_photo(self, photo_id: str) -> bool:
        if photo_id not in self._photos:
            return False
        del self._photos[photo_id]
        path = self.index_dir / f"{photo_id}.json"
        path.unlink(missing_ok=True)
        return True

    def query(self, code: str) -> QueryResult:
        # 走和 OCR 读数同一套归一化，否则人手输入的 A-001 / b403 找不到
        # 索引里的 A-1 / B-403——碎片上印的是不补零的形式。
        target = vocabulary.normalize_ocr_text(code)
        for photo in self._photos.values():
            piece = photo.find(target)
            if piece is not None:
                return QueryResult(
                    found=True, code=target, photo_id=photo.photo_id, piece=piece
                )

        unrecognized = {
            photo.photo_id: photo.unrecognized
            for photo in self._photos.values()
            if photo.unrecognized
        }
        return QueryResult(found=False, code=target, unrecognized=unrecognized)
