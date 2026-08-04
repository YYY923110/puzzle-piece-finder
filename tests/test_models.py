import pytest

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


@pytest.fixture
def sample_index() -> PhotoIndex:
    return PhotoIndex(
        photo_id="p1",
        image_path="data/photos/p1.jpg",
        width=800,
        height=600,
        created_at="2026-08-03T10:00:00",
        pieces=[make_piece(0, "B-403"), make_piece(1, None), make_piece(2, "B-404")],
    )


class TestPhotoIndexQueries:
    def test_recognized_returns_only_pieces_with_codes(self, sample_index):
        assert [p.code for p in sample_index.recognized] == ["B-403", "B-404"]

    def test_unrecognized_returns_only_codeless_pieces(self, sample_index):
        assert [p.piece_id for p in sample_index.unrecognized] == [1]

    def test_find_returns_matching_piece(self, sample_index):
        assert sample_index.find("B-404").piece_id == 2

    def test_find_returns_none_for_absent_code(self, sample_index):
        assert sample_index.find("C-100") is None

    def test_find_is_case_insensitive(self, sample_index):
        assert sample_index.find("b-404").piece_id == 2


class TestSerialization:
    def test_round_trip_preserves_all_fields(self, sample_index):
        restored = PhotoIndex.from_dict(sample_index.to_dict())
        assert restored == sample_index

    def test_to_dict_is_json_serializable(self, sample_index):
        import json

        text = json.dumps(sample_index.to_dict(), ensure_ascii=False)
        assert "B-403" in text

    def test_bbox_survives_as_tuple(self, sample_index):
        restored = PhotoIndex.from_dict(sample_index.to_dict())
        assert isinstance(restored.pieces[0].bbox, tuple)
