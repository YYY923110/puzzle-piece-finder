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

# photo_id 直接当文件名用：data/index/{id}.json、data/photos/{id}.jpg。
MAX_PHOTO_ID_LENGTH = 40
_ILLEGAL_ID_CHARS = frozenset('/\\:*?"<>|')
_RESERVED_ID_NAMES = frozenset(
    ["CON", "PRN", "AUX", "NUL"]
    + [f"COM{i}" for i in range(1, 10)]
    + [f"LPT{i}" for i in range(1, 10)]
)


class InvalidPhotoId(ValueError):
    """photo_id 不能安全地当文件名用。消息直接透给用户看。"""


def sanitize_photo_id(raw: str) -> str:
    """校验用户给的 photo_id，通过则返回去掉首尾空白的结果。

    这层校验以前是白捡的：photo_id 来自 `Path(filename).stem`，而 Path
    顺手剥掉了目录分隔符（`Path("../../x.jpg").stem == "x"`）。改成读一个
    用户自由输入的字段之后，那层意外的保护就没了，得自己做。

    **违反规则一律抛错，绝不静默改写。** 把 `2/3` 悄悄存成 `2_3` 会让用户
    以为存成了自己输入的名字——而名字正是这个功能的全部意义，改写它等于
    把功能悄悄做坏。
    """
    name = raw.strip()
    if not name:
        raise InvalidPhotoId("名字不能为空")
    if len(name) > MAX_PHOTO_ID_LENGTH:
        raise InvalidPhotoId(f"名字不能超过 {MAX_PHOTO_ID_LENGTH} 个字符")
    if name in {".", ".."}:
        raise InvalidPhotoId("名字不能是 . 或 ..")
    illegal = sorted(set(name) & _ILLEGAL_ID_CHARS)
    if illegal:
        raise InvalidPhotoId(f"名字不能包含这些字符：{' '.join(illegal)}")
    if any(ord(char) < 32 for char in name):
        raise InvalidPhotoId("名字不能包含控制字符")
    if name.upper() in _RESERVED_ID_NAMES:
        raise InvalidPhotoId(f"{name} 是 Windows 保留的设备名，换一个")
    return name


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
