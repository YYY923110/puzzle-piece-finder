import numpy as np
import pytest

from puzzlefind import config, render
from puzzlefind.models import Piece


@pytest.fixture
def bright_image() -> np.ndarray:
    """全白图。压暗效果在白底上最容易断言。"""
    return np.full((300, 400, 3), 255, dtype=np.uint8)


@pytest.fixture
def center_piece() -> Piece:
    return Piece(
        piece_id=0,
        contour=[[150, 100], [150, 200], [250, 200], [250, 100]],
        bbox=(150, 100, 100, 100),
        area=10000.0,
        code="B-403",
        confidence=0.95,
        raw_text="B-403",
        method="direct",
        angle=0,
    )


class TestHighlight:
    def test_output_shape_matches_input(self, bright_image, center_piece):
        out = render.highlight(bright_image, [center_piece])
        assert out.shape == bright_image.shape

    def test_does_not_mutate_the_input_image(self, bright_image, center_piece):
        before = bright_image.copy()
        render.highlight(bright_image, [center_piece])
        assert np.array_equal(bright_image, before)

    def test_background_is_dimmed(self, bright_image, center_piece):
        out = render.highlight(bright_image, [center_piece])
        # (10, 10) 远在轮廓之外
        assert int(out[10, 10].max()) < 255

    def test_background_is_not_fully_black(self, bright_image, center_piece):
        """压暗要克制——全黑会毁掉用户的空间定位参照。"""
        out = render.highlight(bright_image, [center_piece])
        assert int(out[10, 10].max()) > 60

    def test_target_interior_keeps_original_brightness(self, bright_image, center_piece):
        out = render.highlight(bright_image, [center_piece])
        assert int(out[150, 200].min()) == 255

    def test_outline_is_drawn_in_configured_color(self, bright_image, center_piece):
        out = render.highlight(bright_image, [center_piece])
        pixels = out.reshape(-1, 3)
        assert any(tuple(int(v) for v in p) == config.OUTLINE_COLOR for p in pixels)

    def test_unknown_mode_uses_the_other_color(self, bright_image, center_piece):
        out = render.highlight(bright_image, [center_piece], unknown=True)
        pixels = out.reshape(-1, 3)
        assert any(
            tuple(int(v) for v in p) == config.UNKNOWN_OUTLINE_COLOR for p in pixels
        )

    def test_multiple_targets_are_all_highlighted(self, bright_image, center_piece):
        second = Piece(
            piece_id=1,
            contour=[[10, 220], [10, 280], [70, 280], [70, 220]],
            bbox=(10, 220, 60, 60),
            area=3600.0,
            code=None,
            confidence=0.0,
            raw_text=None,
            method="none",
            angle=None,
        )
        out = render.highlight(bright_image, [center_piece, second], unknown=True)
        assert int(out[250, 40].min()) == 255   # 第二块内部保持原亮度
        assert int(out[150, 200].min()) == 255  # 第一块内部也保持

    def test_empty_target_list_dims_everything(self, bright_image):
        out = render.highlight(bright_image, [])
        assert int(out[150, 200].max()) < 255


class TestThumbnail:
    def test_long_edge_matches_requested_size(self, bright_image, center_piece):
        thumb = render.thumbnail(bright_image, center_piece, size=120)
        assert max(thumb.shape[:2]) == 120

    def test_thumbnail_is_three_channel(self, bright_image, center_piece):
        thumb = render.thumbnail(bright_image, center_piece)
        assert thumb.ndim == 3 and thumb.shape[2] == 3

    def test_piece_at_image_edge_does_not_crash(self, bright_image):
        edge = Piece(
            piece_id=0,
            contour=[[0, 0], [0, 30], [30, 30], [30, 0]],
            bbox=(0, 0, 30, 30),
            area=900.0,
            code="A-001",
            confidence=0.9,
            raw_text="A-001",
            method="direct",
            angle=0,
        )
        assert render.thumbnail(bright_image, edge).size > 0
