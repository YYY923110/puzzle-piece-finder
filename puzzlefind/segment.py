"""OpenCV 分割：从深色背景上分离出碎片。

前提：碎片摊在深色纯背景上（见 Global Constraints）。这让前景/背景
分离退化成一次阈值操作，避免了在同色背景上调参的地狱。
"""
from __future__ import annotations

import statistics

import cv2
import numpy as np

from . import config


def build_mask(bgr: np.ndarray) -> np.ndarray:
    """把 BGR 图转成二值掩膜，碎片为 255、背景为 0。

    config.MASK_THRESHOLD 为 None 时用 Otsu 自动求阈值（推荐，
    对光照变化更稳）；给定整数时用固定阈值。
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # 轻度模糊压掉传感器噪点，否则 Otsu 会被噪点拉偏
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    if config.MASK_THRESHOLD is None:
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(gray, config.MASK_THRESHOLD, 255, cv2.THRESH_BINARY)

    # 开运算去毛刺，闭运算补碎片内部的小洞（反光造成的暗斑）
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (config.MORPH_KERNEL_SIZE, config.MORPH_KERNEL_SIZE)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def find_blobs(mask: np.ndarray) -> list[np.ndarray]:
    """从掩膜中提取连通块轮廓，丢弃明显小于碎片的噪点。

    注意：粘连的碎片在这里仍是「一个」连通块。切分由 split_blob 负责。
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    median_area = median_blob_area(list(contours))
    if median_area <= 0.0:
        return []

    min_area = median_area * config.MIN_AREA_RATIO
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def median_blob_area(contours: list[np.ndarray]) -> float:
    """轮廓面积的中位数。用作「一块碎片有多大」的稳健估计。

    中位数而非均值：少数粘连团块的面积是单块的好几倍，会把均值拉飞。
    """
    if not contours:
        return 0.0
    return float(statistics.median(cv2.contourArea(c) for c in contours))
