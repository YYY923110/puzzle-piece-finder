"""合成测试图。深色背景上的白色圆形/方形「碎片」。

真实照片不进版本库，所有分割测试跑在这些合成图上。
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest


def make_canvas(width: int = 800, height: int = 600) -> np.ndarray:
    """深色背景画布（不是纯黑——真实照片总有噪点和反光）。"""
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    rng = np.random.default_rng(seed=42)
    noise = rng.integers(0, 12, size=canvas.shape, dtype=np.uint8)
    return cv2.add(canvas, noise)


def draw_piece(canvas: np.ndarray, center: tuple[int, int], radius: int = 40) -> None:
    """在画布上画一块「碎片」——浅色实心圆。"""
    cv2.circle(canvas, center, radius, (235, 233, 228), thickness=-1)


@pytest.fixture
def separated_pieces() -> tuple[np.ndarray, int]:
    """6 块互不接触的碎片。返回 (图像, 碎片数)。"""
    canvas = make_canvas()
    centers = [(120, 120), (320, 120), (520, 120), (120, 380), (320, 380), (520, 380)]
    for center in centers:
        draw_piece(canvas, center)
    return canvas, len(centers)


@pytest.fixture
def touching_pair() -> tuple[np.ndarray, int]:
    """2 块显著重叠的碎片——阈值分割会把它们连成一块。"""
    canvas = make_canvas()
    draw_piece(canvas, (300, 300), radius=60)
    draw_piece(canvas, (390, 300), radius=60)
    return canvas, 2


@pytest.fixture
def touching_triple_with_singles() -> tuple[np.ndarray, int]:
    """3 块粘连 + 3 块独立，共 6 块。用于验证「中位面积」先验。"""
    canvas = make_canvas(1000, 600)
    for center in [(120, 120), (120, 320), (120, 500)]:
        draw_piece(canvas, center, radius=50)
    for center in [(500, 300), (585, 300), (670, 300)]:
        draw_piece(canvas, center, radius=50)
    return canvas, 6


@pytest.fixture
def canvas_with_noise_speck() -> tuple[np.ndarray, int]:
    """3 块正常碎片 + 一个远小于碎片的亮点噪声。应被丢弃。"""
    canvas = make_canvas()
    for center in [(120, 120), (320, 120), (520, 120)]:
        draw_piece(canvas, center, radius=45)
    cv2.circle(canvas, (400, 400), 5, (240, 240, 240), thickness=-1)
    return canvas, 3
