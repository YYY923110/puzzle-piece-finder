import cv2
import pytest

from puzzlefind import cli
from puzzlefind.library import Library
from puzzlefind.models import Piece, PhotoIndex


@pytest.fixture
def photo_file(tmp_path, separated_pieces):
    image, _ = separated_pieces
    path = tmp_path / "shot.jpg"
    cv2.imwrite(str(path), image)
    return path


@pytest.fixture
def seeded_index_dir(tmp_path):
    index_dir = tmp_path / "index"
    library = Library(index_dir=index_dir)
    library.save_photo(
        PhotoIndex(
            photo_id="p1",
            image_path="x.jpg",
            width=100,
            height=100,
            created_at="2026-08-03T10:00:00",
            pieces=[
                Piece(0, [[0, 0], [0, 9], [9, 9], [9, 0]], (0, 0, 9, 9), 81.0,
                      "B-403", 0.9, "B-403", "direct", 0),
                Piece(1, [[20, 20], [20, 29], [29, 29], [29, 20]], (20, 20, 9, 9), 81.0,
                      None, 0.0, None, "none", None),
            ],
        )
    )
    return index_dir


class TestQueryCommand:
    def test_hit_reports_photo_and_exits_zero(self, seeded_index_dir, capsys):
        code = cli.main(["query", "B-403", "--index-dir", str(seeded_index_dir)])
        assert code == 0
        assert "p1" in capsys.readouterr().out

    def test_miss_exits_nonzero(self, seeded_index_dir):
        assert cli.main(["query", "D-777", "--index-dir", str(seeded_index_dir)]) == 1

    def test_miss_lists_unrecognized_pieces(self, seeded_index_dir, capsys):
        cli.main(["query", "D-777", "--index-dir", str(seeded_index_dir)])
        assert "未识别" in capsys.readouterr().out


class TestStatsCommand:
    def test_reports_recognized_and_unrecognized_counts(self, seeded_index_dir, capsys):
        assert cli.main(["stats", "--index-dir", str(seeded_index_dir)]) == 0
        out = capsys.readouterr().out
        assert "1" in out and "p1" in out


class TestIndexCommand:
    def test_writes_an_index_file(self, photo_file, tmp_path, monkeypatch):
        """用假后端跑，不加载 PaddleOCR。"""
        from puzzlefind.recognize import RawDetection

        class NullBackend:
            def read(self, image):
                return [RawDetection("B-001", 0.99)]

        monkeypatch.setattr(cli, "_make_backend", lambda: NullBackend())
        index_dir = tmp_path / "idx"
        code = cli.main(["index", str(photo_file), "--index-dir", str(index_dir)])
        assert code == 0
        assert (index_dir / "shot.json").exists()

    def test_missing_photo_exits_nonzero(self, tmp_path, monkeypatch):
        class NullBackend:
            def read(self, image):
                return []

        monkeypatch.setattr(cli, "_make_backend", lambda: NullBackend())
        assert cli.main(["index", str(tmp_path / "nope.jpg")]) == 2
