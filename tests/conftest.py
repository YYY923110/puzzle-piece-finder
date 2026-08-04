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


def draw_puzzle_piece(
    canvas: np.ndarray, center: tuple[int, int], size: int = 100
) -> None:
    """画一块**带凸起**的拼图碎片：方形本体 + 若干半圆形 tab。

    为什么必须有 tab：圆形碎片的距离变换只有一个山峰，而真实拼图
    碎片的每个凸起都会形成一个独立山峰。只用圆形做 fixture，
    「单块碎片被误判成多块」这类 bug 完全测不出来——实测一张真实
    照片上，48 块单块碎片的 peak_count 全是 3，没有一个是 1。
    """
    cx, cy = center
    half = size // 2
    cv2.rectangle(
        canvas, (cx - half, cy - half), (cx + half, cy + half),
        (235, 233, 228), thickness=-1,
    )
    tab = max(6, size // 5)
    # 上、右两条边各一个凸起；下、左各一个凹口（凹口用背景色抠掉）
    cv2.circle(canvas, (cx, cy - half), tab, (235, 233, 228), thickness=-1)
    cv2.circle(canvas, (cx + half, cy), tab, (235, 233, 228), thickness=-1)
    cv2.circle(canvas, (cx, cy + half), tab, (20, 20, 20), thickness=-1)
    cv2.circle(canvas, (cx - half, cy), tab, (20, 20, 20), thickness=-1)


@pytest.fixture
def separated_puzzle_pieces() -> tuple[np.ndarray, int]:
    """12 块互不接触的**带凸起**碎片。

    这是对真实照片的最小复现：碎片彼此分开、背景干净，
    分割**必须原样返回 12 块**，一块都不许切开。
    """
    canvas = make_canvas(900, 700)
    centers = [
        (140, 130), (350, 130), (560, 130), (770, 130),
        (140, 350), (350, 350), (560, 350), (770, 350),
        (140, 570), (350, 570), (560, 570), (770, 570),
    ]
    for center in centers:
        draw_puzzle_piece(canvas, center, size=110)
    return canvas, len(centers)


@pytest.fixture
def canvas_with_noise_speck() -> tuple[np.ndarray, int]:
    """3 块正常碎片 + 一个远小于碎片的亮点噪声。应被丢弃。"""
    canvas = make_canvas()
    for center in [(120, 120), (320, 120), (520, 120)]:
        draw_piece(canvas, center, radius=45)
    cv2.circle(canvas, (400, 400), 5, (240, 240, 240), thickness=-1)
    return canvas, 3
