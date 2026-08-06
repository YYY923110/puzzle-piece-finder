import cv2
import numpy as np
import pytest

from puzzlefind import pipeline
from puzzlefind.recognize import RawDetection


class PieceBackend:
    """按**碎片顺序**（不是调用顺序）吐出预设编号的假后端。

    为什么不能简单地用「第几次调用」当索引：一块识别失败的碎片会触发
    Pass C 的 12 次穷举调用，把后续所有碎片的索引整体错位。那样的替身
    能不能通过测试，取决于 None 恰好排在列表哪个位置——是运气，不是设计。

    这里改成显式记账：每块碎片的调用配额是确定的（命中 1 次；未命中
    则 1 次直接 + len(SWEEP_ANGLES) 次穷举），配额用完才前进到下一块。
    这样测试对穷举次数免疫。
    """

    def __init__(self, codes: list[str | None]):
        self.codes = codes
        self.calls = 0
        self._piece = 0
        self._remaining = self._quota(0)

    def _code_at(self, index: int) -> str | None:
        return self.codes[index] if index < len(self.codes) else None

    def _quota(self, index: int) -> int:
        from puzzlefind import config

        return 1 if self._code_at(index) else 1 + len(config.SWEEP_ANGLES)

    def read(self, image: np.ndarray) -> list[RawDetection]:
        self.calls += 1
        code = self._code_at(self._piece)
        self._remaining -= 1
        if self._remaining <= 0:
            self._piece += 1
            self._remaining = self._quota(self._piece)
        return [RawDetection(code, 0.99)] if code else []


@pytest.fixture
def photo_path(tmp_path, separated_pieces):
    image, _ = separated_pieces
    path = tmp_path / "photo.jpg"
    cv2.imwrite(str(path), image)
    return path


class TestBuildIndex:
    def test_creates_one_piece_per_contour(self, photo_path, separated_pieces):
        _, expected = separated_pieces
        backend = PieceBackend([f"B-{261 + i}" for i in range(expected)])
        index = pipeline.build_index(photo_path, backend)
        assert len(index.pieces) == expected

    def test_records_image_dimensions(self, photo_path, separated_pieces):
        image, _ = separated_pieces
        backend = PieceBackend([])
        index = pipeline.build_index(photo_path, backend)
        assert index.width == image.shape[1]
        assert index.height == image.shape[0]

    def test_assigns_sequential_piece_ids(self, photo_path):
        backend = PieceBackend([])
        index = pipeline.build_index(photo_path, backend)
        assert [p.piece_id for p in index.pieces] == list(range(len(index.pieces)))

    def test_recognized_codes_land_on_pieces(self, photo_path, separated_pieces):
        _, count = separated_pieces
        backend = PieceBackend([f"B-{261 + i}" for i in range(count)])
        index = pipeline.build_index(photo_path, backend)
        assert sorted(p.code for p in index.recognized) == sorted(
            f"B-{261 + i}" for i in range(count)
        )

    def test_unreadable_pieces_become_unrecognized(self, photo_path, separated_pieces):
        _, count = separated_pieces
        codes: list[str | None] = [f"B-{261 + i}" for i in range(count - 2)] + [None, None]
        backend = PieceBackend(codes)
        index = pipeline.build_index(photo_path, backend)
        assert len(index.unrecognized) == 2

    def test_duplicate_reads_are_resolved_to_one(self, photo_path, separated_pieces):
        _, count = separated_pieces
        backend = PieceBackend(["B-403"] * count)
        index = pipeline.build_index(photo_path, backend)
        assert len(index.recognized) == 1

    def test_contour_points_are_within_image_bounds(self, photo_path, separated_pieces):
        image, _ = separated_pieces
        backend = PieceBackend([])
        index = pipeline.build_index(photo_path, backend)
        h, w = image.shape[:2]
        for piece in index.pieces:
            for x, y in piece.contour:
                assert 0 <= x <= w and 0 <= y <= h

    def test_explicit_photo_id_is_honored(self, photo_path):
        backend = PieceBackend([])
        index = pipeline.build_index(photo_path, backend, photo_id="my-id")
        assert index.photo_id == "my-id"

    def test_missing_file_raises(self, tmp_path):
        backend = PieceBackend([])
        with pytest.raises(FileNotFoundError):
            pipeline.build_index(tmp_path / "nope.jpg", backend)
