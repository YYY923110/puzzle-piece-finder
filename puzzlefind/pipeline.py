"""建索引管线：一张照片 → 一个 PhotoIndex。

这是引擎的顶层入口，把分割、识别、消解串起来。它不碰磁盘持久化
（那是 library 的事），也不碰 Web（那是 server 的事）。
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2

from . import resolve, segment
from .models import Piece, PhotoIndex
from .recognize import OcrBackend, recognize_piece


def build_index(
    image_path: Path,
    backend: OcrBackend,
    photo_id: str | None = None,
) -> PhotoIndex:
    """对一张照片建立完整索引。

    流程：读图 → 分割出碎片轮廓 → 逐块裁剪识别 → 全局冲突消解 →
    组装成 PhotoIndex。
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"照片不存在: {path}")

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"无法解码为图像: {path}")

    height, width = image.shape[:2]
    contours = segment.extract_contours(image)

    # 每块碎片的裁剪图里，只把**其他**碎片涂灰，见 segment.crop_piece
    results = [
        recognize_piece(
            backend,
            segment.crop_piece(
                image, contour, contours[:index] + contours[index + 1 :]
            ),
        )
        for index, contour in enumerate(contours)
    ]
    results = resolve.resolve(results)

    pieces = [
        Piece(
            piece_id=index,
            contour=[[int(pt[0][0]), int(pt[0][1])] for pt in contour],
            bbox=segment.contour_bbox(contour),
            area=float(cv2.contourArea(contour)),
            code=result.code,
            confidence=result.confidence,
            raw_text=result.raw_text,
            method=result.method,
            angle=result.angle,
        )
        for index, (contour, result) in enumerate(zip(contours, results))
    ]

    return PhotoIndex(
        photo_id=photo_id or path.stem,
        image_path=str(path),
        width=width,
        height=height,
        created_at=datetime.now().isoformat(timespec="seconds"),
        pieces=pieces,
    )
