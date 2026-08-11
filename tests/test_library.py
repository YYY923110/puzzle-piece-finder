import pytest

from puzzlefind.library import InvalidPhotoId, Library, sanitize_photo_id
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
    lib.save_photo(make_index("p1", ["A-1", "A-2", None]))
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
        library.save_photo(make_index("p1", ["A-259"]))
        reloaded = Library.load(tmp_path)
        assert len([p for p in reloaded.photos if p.photo_id == "p1"]) == 1
        assert reloaded.query("A-259").found is True

    def test_load_from_empty_dir_yields_empty_library(self, tmp_path):
        assert Library.load(tmp_path / "fresh").photos == []

    def test_delete_removes_photo(self, library, tmp_path):
        assert library.delete_photo("p1") is True
        assert Library.load(tmp_path).photos[0].photo_id == "p2"

    def test_delete_unknown_id_returns_false(self, library):
        assert library.delete_photo("nope") is False


class TestQuery:
    def test_finds_code_in_first_photo(self, library):
        result = library.query("A-2")
        assert result.found is True
        assert result.photo_id == "p1"
        assert result.piece.code == "A-2"

    def test_finds_code_in_second_photo(self, library):
        result = library.query("B-403")
        assert result.found is True
        assert result.photo_id == "p2"

    def test_query_is_case_and_space_insensitive(self, library):
        assert library.query("  b-403 ").found is True

    def test_query_accepts_the_zero_padded_form(self, library):
        """碎片上印的是 A-1，但补零写法是人手输入时最自然的一种。
        查询走和 OCR 读数同一套归一化，两种写法都该命中同一块。
        """
        assert library.query("A-001").found is True
        assert library.query("A-001").piece.code == "A-1"

    def test_query_accepts_a_missing_hyphen(self, library):
        assert library.query("b403").found is True

    def test_miss_reports_not_found(self, library):
        assert library.query("D-777").found is False

    def test_miss_returns_unrecognized_pieces_grouped_by_photo(self, library):
        result = library.query("D-777")
        assert result.unrecognized["p1"][0].piece_id == 2
        assert len(result.unrecognized["p2"]) == 2

    def test_miss_omits_photos_with_no_unrecognized_pieces(self, tmp_path):
        lib = Library(index_dir=tmp_path)
        lib.save_photo(make_index("full", ["A-1"]))
        assert lib.query("Z-999").unrecognized == {}

    def test_hit_carries_no_unrecognized_payload(self, library):
        assert library.query("A-2").unrecognized == {}

    def test_malformed_query_is_a_miss_not_a_crash(self, library):
        assert library.query("!!!").found is False


class TestSanitizePhotoId:
    """photo_id 直接当文件名用，所以它必须是一个安全的文件名。

    这层保护以前是白捡的——photo_id 来自 Path(filename).stem，而 Path
    顺手剥掉了目录分隔符（Path("../../x.jpg").stem == "x"）。改成读一个
    自由文本字段之后那层意外的保护就没了。
    """

    def test_keeps_an_ordinary_name(self):
        assert sanitize_photo_id("2") == "2"

    def test_keeps_a_chinese_name(self):
        assert sanitize_photo_id("左上角") == "左上角"

    def test_trims_surrounding_whitespace(self):
        assert sanitize_photo_id("  2  ") == "2"

    @pytest.mark.parametrize("raw", ["", "   ", "\t"])
    def test_empty_name_is_rejected(self, raw):
        with pytest.raises(InvalidPhotoId):
            sanitize_photo_id(raw)

    @pytest.mark.parametrize(
        "raw", ["a/b", "a\\b", "a:b", "a*b", "a?b", 'a"b', "a<b", "a>b", "a|b"]
    )
    def test_path_and_wildcard_characters_are_rejected(self, raw):
        with pytest.raises(InvalidPhotoId):
            sanitize_photo_id(raw)

    def test_control_characters_are_rejected(self):
        with pytest.raises(InvalidPhotoId):
            sanitize_photo_id("a\x00b")

    @pytest.mark.parametrize("raw", [".", ".."])
    def test_dot_names_are_rejected(self, raw):
        with pytest.raises(InvalidPhotoId):
            sanitize_photo_id(raw)

    @pytest.mark.parametrize("raw", ["con", "CON", "Com1", "LPT9", "nul", "aux"])
    def test_windows_reserved_device_names_are_rejected(self, raw):
        """这是 Windows 项目，data/index/CON.json 会当场炸。"""
        with pytest.raises(InvalidPhotoId):
            sanitize_photo_id(raw)

    def test_a_name_at_the_length_limit_is_kept(self):
        assert sanitize_photo_id("x" * 40) == "x" * 40

    def test_an_overlong_name_is_rejected(self):
        with pytest.raises(InvalidPhotoId):
            sanitize_photo_id("x" * 41)

    def test_the_error_message_says_why(self):
        """错误直接透给用户看，必须说清违反了哪条。"""
        with pytest.raises(InvalidPhotoId, match="/"):
            sanitize_photo_id("2/3")
