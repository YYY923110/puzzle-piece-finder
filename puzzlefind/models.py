"""索引的数据模型。JSON 是唯一的持久化格式——量级只有几百条记录，
上数据库纯属多余，而 JSON 你可以直接打开看，排查问题快得多。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Piece:
    """一块碎片：几何位置 + 识别结论。"""

    piece_id: int
    contour: list[list[int]]              # [[x, y], ...] 原图坐标系
    bbox: tuple[int, int, int, int]       # (x, y, w, h)
    area: float
    code: str | None
    confidence: float
    raw_text: str | None
    method: str
    angle: int | None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["bbox"] = list(self.bbox)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Piece:
        return cls(
            piece_id=int(data["piece_id"]),
            contour=[[int(x), int(y)] for x, y in data["contour"]],
            bbox=tuple(int(v) for v in data["bbox"]),  # type: ignore[arg-type]
            area=float(data["area"]),
            code=data.get("code"),
            confidence=float(data.get("confidence", 0.0)),
            raw_text=data.get("raw_text"),
            method=str(data.get("method", "none")),
            angle=data.get("angle"),
        )


@dataclass
class PhotoIndex:
    """一张照片的完整索引。"""

    photo_id: str
    image_path: str
    width: int
    height: int
    created_at: str                       # ISO 8601，用于提示索引可能已过期
    pieces: list[Piece] = field(default_factory=list)

    @property
    def recognized(self) -> list[Piece]:
        return [p for p in self.pieces if p.code]

    @property
    def unrecognized(self) -> list[Piece]:
        """未识别的碎片。查询未命中时，答案大概率就在这里面。"""
        return [p for p in self.pieces if not p.code]

    def find(self, code: str) -> Piece | None:
        target = code.strip().upper()
        for piece in self.pieces:
            if piece.code == target:
                return piece
        return None

    def to_dict(self) -> dict:
        return {
            "photo_id": self.photo_id,
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
            "created_at": self.created_at,
            "pieces": [p.to_dict() for p in self.pieces],
        }

    @classmethod
    def from_dict(cls, data: dict) -> PhotoIndex:
        return cls(
            photo_id=str(data["photo_id"]),
            image_path=str(data["image_path"]),
            width=int(data["width"]),
            height=int(data["height"]),
            created_at=str(data["created_at"]),
            pieces=[Piece.from_dict(p) for p in data.get("pieces", [])],
        )
