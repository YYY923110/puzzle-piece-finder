import pytest

from puzzlefind import vocabulary as v


class TestIsValidCode:
    @pytest.mark.parametrize("code", ["A-000", "B-403", "C-999", "D-250"])
    def test_accepts_well_formed_codes(self, code):
        assert v.is_valid_code(code) is True

    @pytest.mark.parametrize(
        "code",
        ["E-100", "A-1000", "A-99", "a-100", "A100", "", "A-abc", "AA-100"],
    )
    def test_rejects_malformed_codes(self, code):
        assert v.is_valid_code(code) is False


class TestNormalizeOcrText:
    def test_uppercases_and_strips(self):
        assert v.normalize_ocr_text("  b-403  ") == "B-403"

    def test_removes_internal_whitespace(self):
        assert v.normalize_ocr_text("B - 403") == "B-403"

    def test_inserts_missing_hyphen(self):
        assert v.normalize_ocr_text("B403") == "B-403"

    def test_normalizes_unicode_dashes(self):
        assert v.normalize_ocr_text("B—403") == "B-403"
        assert v.normalize_ocr_text("B–403") == "B-403"

    def test_leaves_unrecoverable_text_alone(self):
        assert v.normalize_ocr_text("???") == "???"


class TestConfusionDistance:
    def test_identical_strings_have_zero_distance(self):
        assert v.confusion_distance("B-403", "B-403") == 0.0

    def test_confusable_pair_costs_less_than_one(self):
        # 8 与 B 形近，代价应低于一次普通替换
        cheap = v.confusion_distance("8-403", "B-403")
        assert 0.0 < cheap < 1.0

    def test_unconfusable_pair_costs_one(self):
        assert v.confusion_distance("A-403", "B-403") == 1.0

    def test_length_difference_counts(self):
        assert v.confusion_distance("B-40", "B-403") == 1.0


class TestSnap:
    def test_exact_valid_code_snaps_to_itself_with_zero_distance(self):
        code, dist = v.snap("B-403")
        assert code == "B-403"
        assert dist == 0.0

    def test_confusable_misread_snaps_to_valid_code(self):
        # 8→B, O→0 都是形近替换，总代价应在阈值内
        code, dist = v.snap("8-4O3")
        assert code == "B-403"
        assert dist > 0.0

    def test_hopeless_garbage_returns_none(self):
        code, dist = v.snap("XYZQWERTY")
        assert code is None

    def test_respects_max_distance(self):
        code, _ = v.snap("8-4O3", max_distance=0.1)
        assert code is None

    @pytest.mark.parametrize(
        "raw", ["8-4O3", "B-4O3", "8-403", "A-1I1", "D-5S0", "B0403", "6-9OO"]
    )
    def test_fast_path_agrees_with_exhaustive_scan(self, raw):
        """长度为 5 的输入走快速路径，结果必须与全量扫描逐字节一致。

        快速路径把 4000 次 DP 压缩成 34 次比较，是靠「等长串的最优对齐
        必为纯替换」这个性质。这个测试就是那个性质的护栏——一旦有人
        改坏了 _snap_aligned，这里立刻红。
        """
        text = v.normalize_ocr_text(raw)
        assert len(text) == 5, "该用例应触发快速路径"
        fast_code, fast_distance = v._snap_aligned(text)
        slow_code, slow_distance = v._snap_exhaustive(text)
        assert fast_distance == pytest.approx(slow_distance)
        assert fast_code == slow_code

    def test_wildly_wrong_length_is_rejected_without_scanning(self):
        code, distance = v.snap("QWERTYUIOP")
        assert code is None
        assert distance >= 5.0


class TestBootstrapRanges:
    def test_derives_range_per_prefix(self):
        codes = ["B-262", "B-300", "B-499", "A-010", "A-050"]
        ranges = v.bootstrap_ranges(codes)
        assert ranges["B"] == (262, 499)
        assert ranges["A"] == (10, 50)

    def test_ignores_prefixes_with_too_few_samples(self):
        codes = ["B-262", "B-300", "B-499", "C-700"]
        ranges = v.bootstrap_ranges(codes, min_samples=3)
        assert "B" in ranges
        assert "C" not in ranges

    def test_empty_input_yields_empty_ranges(self):
        assert v.bootstrap_ranges([]) == {}


class TestRobustRanges:
    def test_fences_out_an_extreme_value(self):
        """离群值不该参与定义它自己所属的区间——这正是 bootstrap_ranges
        抓不到离群值的原因。"""
        codes = ["B-262", "B-300", "B-350", "B-400", "B-901"]
        assert v.robust_ranges(codes)["B"] == (262, 400)

    def test_keeps_every_value_when_they_are_evenly_spread(self):
        codes = ["B-100", "B-200", "B-300", "B-400", "B-500"]
        assert v.robust_ranges(codes)["B"] == (100, 500)

    def test_identical_values_collapse_to_a_point_range(self):
        codes = ["B-403"] * 5
        assert v.robust_ranges(codes)["B"] == (403, 403)

    def test_ignores_prefixes_below_the_sample_floor(self):
        # 四分位数在四个点以下没有意义，宁可不产出区间
        assert v.robust_ranges(["B-262", "B-901", "B-500"]) == {}

    def test_empty_input_yields_empty_ranges(self):
        assert v.robust_ranges([]) == {}


class TestIsOutlier:
    def test_code_inside_range_is_not_outlier(self):
        assert v.is_outlier("B-350", {"B": (262, 499)}) is False

    def test_code_outside_range_is_outlier(self):
        assert v.is_outlier("B-501", {"B": (262, 499)}) is True

    def test_unknown_prefix_is_never_outlier(self):
        assert v.is_outlier("D-501", {"B": (262, 499)}) is False
