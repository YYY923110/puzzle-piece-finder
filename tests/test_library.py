import pytest

from puzzlefind.library import Library
from puzzlefind.models import Piece, PhotoIndex


def make_piece(piece_id: int, code: str | None) -> Piece:
    return Piece(
        piece_id=piece_id,
        contour=[[0, 0], [0, 10], [10, 10], [10, 0]],
        bbox=(0, 0, 10, 10),
        area=100.0,
        code=code,
        confidence=0.9 if code else 0.0,
        raw_text=code,
        method="direct" if code else "none",
        angle=0,
    )


def make_index(photo_id: str, codes: list[str | None]) -> PhotoIndex:
    return PhotoIndex(
        photo_id=photo_id,
        image_path=f"data/photos/{photo_id}.jpg",
        width=800,
        height=600,
        created_at="2026-08-03T10:00:00",
        pieces=[make_piece(i, c) for i, c in enumerate(codes)],
    )


@pytest.fixture
def library(tmp_path) -> Library:
    lib = Library(index_dir=tmp_path)
    lib.save_photo(make_index("p1", ["A-001", "A-002", None]))
    lib.save_photo(make_index("p2", ["B-403", None, None]))
    return lib


class TestPersistence:
    def test_saved_photo_survives_reload(self, library, tmp_path):
        reloaded = Library.load(tmp_path)
        assert {p.photo_id for p in reloaded.photos} == {"p1", "p2"}

    def test_saved_pieces_survive_reload(self, library, tmp_path):
        reloaded = Library.load(tmp_path)
        photo = next(p for p in reloaded.photos if p.photo_id == "p2")
        assert photo.find("B-403") is not None

    def test_saving_same_id_twice_replaces_not_duplicates(self, library, tmp_path):
        library.save_photo(make_index("p1", ["A-999"]))
        reloaded = Library.load(tmp_path)
        assert len([p for p in reloaded.photos if p.photo_id == "p1"]) == 1
        assert reloaded.query("A-999").found is True

    def test_load_from_empty_dir_yields_empty_library(self, tmp_path):
        assert Library.load(tmp_path / "fresh").photos == []

    def test_delete_removes_photo(self, library, tmp_path):
        assert library.delete_photo("p1") is True
        assert Library.load(tmp_path).photos[0].photo_id == "p2"

    def test_delete_unknown_id_returns_false(self, library):
        assert library.delete_photo("nope") is False


class TestQuery:
    def test_finds_code_in_first_photo(self, library):
        result = library.query("A-002")
        assert result.found is True
        assert result.photo_id == "p1"
        assert result.piece.code == "A-002"

    def test_finds_code_in_second_photo(self, library):
        result = library.query("B-403")
        assert result.found is True
        assert result.photo_id == "p2"

    def test_query_is_case_and_space_insensitive(self, library):
        assert library.query("  b-403 ").found is True

    def test_miss_reports_not_found(self, library):
        assert library.query("D-777").found is False

    def test_miss_returns_unrecognized_pieces_grouped_by_photo(self, library):
        result = library.query("D-777")
        assert result.unrecognized["p1"][0].piece_id == 2
        assert len(result.unrecognized["p2"]) == 2

    def test_miss_omits_photos_with_no_unrecognized_pieces(self, tmp_path):
        lib = Library(index_dir=tmp_path)
        lib.save_photo(make_index("full", ["A-001"]))
        assert lib.query("Z-999").unrecognized == {}

    def test_hit_carries_no_unrecognized_payload(self, library):
        assert library.query("A-002").unrecognized == {}

    def test_malformed_query_is_a_miss_not_a_crash(self, library):
        assert library.query("!!!").found is False
