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


def expected_piece_count(contour: np.ndarray, median_area: float) -> int:
    """这个连通块里大概有几块碎片？

    用「面积 ÷ 中位面积」估计。这是碎片尺寸大致均匀这一先验的直接应用，
    也是我们不需要人工指定分水岭种子数的原因。
    """
    if median_area <= 0.0:
        return 1
    return max(1, round(cv2.contourArea(contour) / median_area))


def _blob_mask(shape: tuple[int, int], contour: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
    return mask


# 山峰扫描的阈值区间（相对于该团块的最大距离）。
# 下界 0.30 是有意的：更低的阈值会让凸角、拼图凸起这类小特征自成一块，
# 把碎片数虚高。上界 0.90 之上则峰顶开始整个消失。
_PEAK_SCAN_RANGE = (0.30, 0.90)
_PEAK_SCAN_STEPS = 13


def peak_count(shape: tuple[int, int], contour: np.ndarray) -> int:
    """这个团块的距离变换里有几个「山峰」——即大概含几块碎片。

    这个估计**与面积先验完全无关**，所以在面积中位数不可用时仍然成立。
    做法是把阈值从 0.30 扫到 0.90，取扫描过程中出现过的最大连通块数：
    粘连的两块碎片，在阈值高到淹没它们之间的那道「峡谷」时就会裂成两个峰。
    """
    blob = _blob_mask(shape, contour)
    distance = cv2.distanceTransform(blob, cv2.DIST_L2, 5)
    max_distance = float(distance.max())
    if max_distance <= 0.0:
        return 1

    low, high = _PEAK_SCAN_RANGE
    step = (high - low) / (_PEAK_SCAN_STEPS - 1)
    best = 1
    for i in range(_PEAK_SCAN_STEPS):
        fraction = low + i * step
        _, peaks = cv2.threshold(
            distance, fraction * max_distance, 255, cv2.THRESH_BINARY
        )
        count, _ = cv2.connectedComponents(peaks.astype(np.uint8))
        best = max(best, count - 1)  # 减去背景标签
    return best


def unit_piece_area(shape: tuple[int, int], contours: list[np.ndarray]) -> float:
    """单块碎片的面积估计。

    为什么不直接用面积中位数：中位数隐含假设「大多数连通块是单块碎片」，
    而这在粘连块占多数的图上不成立。极端情形是整张图只有一个团块——
    此时中位数恒等于它自己的面积，比值恒为 1.00，切分永远不会被触发。

    改法是先用山峰数把每个团块归一化成「其中每块碎片的面积」，再取中位数。
    大多数团块是单块时（真实照片的常态），山峰数为 1，结果与面积中位数一致；
    团块全是粘连时，归一化把它救了回来。
    """
    if not contours:
        return 0.0
    per_piece = [
        cv2.contourArea(c) / max(1, peak_count(shape, c)) for c in contours
    ]
    return float(statistics.median(per_piece))


def split_blob(
    shape: tuple[int, int], contour: np.ndarray, expected: int
) -> list[np.ndarray]:
    """把一个粘连团块切成 expected 块，返回子轮廓列表。

    做法：对团块做距离变换，然后二分搜索一个阈值系数，使得
    「距离 > 系数 × 最大距离」的区域恰好裂成 expected 个连通块。
    这些连通块作为分水岭的种子。

    二分搜索替代了手工调 0.5/0.6/0.7 这类魔法系数——每个团块自己
    找到合适的系数，不同尺寸、不同重叠程度的团块都能自适应。
    """
    if expected <= 1:
        return [contour]

    blob = _blob_mask(shape, contour)
    distance = cv2.distanceTransform(blob, cv2.DIST_L2, 5)
    max_distance = float(distance.max())
    if max_distance <= 0.0:
        return [contour]

    def markers_for(fraction: float) -> tuple[int, np.ndarray]:
        _, peaks = cv2.threshold(distance, fraction * max_distance, 255, cv2.THRESH_BINARY)
        peaks = peaks.astype(np.uint8)
        count, labels = cv2.connectedComponents(peaks)
        return count - 1, labels  # 减去背景标签

    # 二分：系数越大，峰顶区域收得越紧、种子越多。
    # （注意方向：抬高阈值是把碎片之间那道「峡谷」淹掉，从而让粘在一起的
    # 峰顶裂开，所以种子数随阈值单调增加，直到阈值高过峰顶本身。）
    low, high = 0.05, 0.95
    best_labels: np.ndarray | None = None
    for _ in range(config.SPLIT_SEARCH_STEPS):
        mid = (low + high) / 2.0
        count, labels = markers_for(mid)
        if count == expected:
            best_labels = labels
            break
        if count > expected:
            high = mid  # 种子太多 → 降低阈值，让它们重新粘回去
        else:
            low = mid   # 种子太少 → 提高阈值，把峡谷淹掉
    if best_labels is None:
        _, best_labels = markers_for((low + high) / 2.0)

    # 分水岭需要三通道输入，且标签 0 表示「未知区域」
    markers = best_labels.astype(np.int32) + 1
    markers[blob == 0] = 1  # 背景标为 1
    unknown = cv2.subtract(blob, (best_labels > 0).astype(np.uint8) * 255)
    markers[unknown == 255] = 0

    canvas = cv2.cvtColor(blob, cv2.COLOR_GRAY2BGR)
    cv2.watershed(canvas, markers)

    pieces: list[np.ndarray] = []
    for label in range(2, int(markers.max()) + 1):
        region = np.zeros(shape, dtype=np.uint8)
        region[markers == label] = 255
        sub_contours, _ = cv2.findContours(
            region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if sub_contours:
            pieces.append(max(sub_contours, key=cv2.contourArea))

    # 分水岭失败时退回原轮廓——高亮范围偏大，但功能仍在（优雅降级）
    return pieces if pieces else [contour]


def extract_contours(bgr: np.ndarray) -> list[np.ndarray]:
    """端到端：原图 → 已切分的单块碎片轮廓列表。"""
    mask = build_mask(bgr)
    blobs = find_blobs(mask)
    if not blobs:
        return []

    shape = bgr.shape[:2]
    median_area = unit_piece_area(shape, blobs)

    result: list[np.ndarray] = []
    for blob in blobs:
        expected = expected_piece_count(blob, median_area)
        if cv2.contourArea(blob) < median_area * config.SPLIT_AREA_RATIO:
            result.append(blob)
        else:
            result.extend(split_blob(shape, blob, expected))
    return result
