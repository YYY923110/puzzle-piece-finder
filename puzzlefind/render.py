"""高亮渲染：把「这块碎片在哪」变成一眼能看到的图。

设计取舍：非目标区压暗但不压黑。全黑会让目标跳得更出来，却会毁掉
用户把屏幕位置映射到桌面真实位置所依赖的参照物（周围碎片的排布、
背景纹理、桌沿）。DIM_FACTOR 默认 0.45 是这两者的平衡点。
"""
from __future__ import annotations

import cv2
import numpy as np

from . import config
from .models import Piece


def _contour_array(piece: Piece) -> np.ndarray:
    return np.array(piece.contour, dtype=np.int32).reshape(-1, 1, 2)


def _outline_thickness(image: np.ndarray) -> int:
    long_edge = max(image.shape[:2])
    return max(2, round(long_edge * config.OUTLINE_THICKNESS_RATIO))


def highlight(
    image: np.ndarray, targets: list[Piece], *, unknown: bool = False
) -> np.ndarray:
    """压暗全图，把目标碎片还原成原亮度并描边。

    unknown=True 时用另一种描边颜色，用于「查询未命中，这些是未识别
    碎片」的场景——视觉上和「找到了」明确区分开。
    """
    dimmed = cv2.convertScaleAbs(image, alpha=config.DIM_FACTOR, beta=0)

    if targets:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for piece in targets:
            cv2.drawContours(mask, [_contour_array(piece)], -1, 255, thickness=-1)
        output = np.where(mask[:, :, None] == 255, image, dimmed)
    else:
        output = dimmed.copy()

    color = config.UNKNOWN_OUTLINE_COLOR if unknown else config.OUTLINE_COLOR
    thickness = _outline_thickness(image)
    for piece in targets:
        cv2.polylines(
            output,
            [_contour_array(piece)],
            isClosed=True,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
    return output


def thumbnail(image: np.ndarray, piece: Piece, size: int = 200) -> np.ndarray:
    """裁出这块碎片的小图，供用户核对系统读的编号对不对。

    这是防「自信地认错」的那道防线——词表吸附和唯一性去重都拦不住
    B-403 被读成 B-408（两个都合法），但用户瞄一眼缩略图就能发现。
    """
    height, width = image.shape[:2]
    x, y, w, h = piece.bbox
    pad = config.CROP_PADDING
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(width, x + w + pad), min(height, y + h + pad)

    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return crop

    scale = size / max(crop.shape[:2])
    return cv2.resize(
        crop,
        (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))),
        interpolation=cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA,
    )
