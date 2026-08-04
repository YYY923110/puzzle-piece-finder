import pytest

from puzzlefind import resolve
from puzzlefind.recognize import RecogResult


def hit(code: str | None, confidence: float) -> RecogResult:
    return RecogResult(
        code=code,
        confidence=confidence,
        raw_text=code,
        method="direct" if code else "none",
        angle=0,
    )


class TestResolve:
    def test_output_length_always_matches_input(self):
        results = [hit("B-403", 0.9), hit(None, 0.0), hit("B-404", 0.8)]
        assert len(resolve.resolve(results)) == len(results)

    def test_non_conflicting_results_pass_through(self):
        results = [hit("B-403", 0.9), hit("B-404", 0.8)]
        resolved = resolve.resolve(results)
        assert [r.code for r in resolved] == ["B-403", "B-404"]

    def test_duplicate_code_keeps_higher_confidence_and_drops_the_other(self):
        results = [hit("B-403", 0.71), hit("B-403", 0.95)]
        resolved = resolve.resolve(results)
        assert resolved[0].code is None
        assert resolved[1].code == "B-403"

    def test_dropped_duplicate_records_the_reason(self):
        results = [hit("B-403", 0.71), hit("B-403", 0.95)]
        resolved = resolve.resolve(results)
        assert resolved[0].raw_text == "B-403"  # 原始读数保留下来供排查

    def test_three_way_duplicate_keeps_only_the_best(self):
        results = [hit("B-403", 0.60), hit("B-403", 0.95), hit("B-403", 0.80)]
        resolved = resolve.resolve(results)
        assert [r.code for r in resolved] == [None, "B-403", None]

    def test_outlier_code_is_dropped_when_range_is_established(self):
        # B 组自举区间由这些样本确定；B-901 明显越界
        results = [
            hit("B-262", 0.9),
            hit("B-300", 0.9),
            hit("B-350", 0.9),
            hit("B-400", 0.9),
            hit("B-901", 0.9),
        ]
        resolved = resolve.resolve(results)
        assert resolved[-1].code is None

    def test_outlier_is_not_dropped_when_samples_are_too_few(self):
        results = [hit("B-262", 0.9), hit("B-901", 0.9)]
        resolved = resolve.resolve(results)
        assert resolved[-1].code == "B-901"

    def test_none_results_are_left_untouched(self):
        results = [hit(None, 0.0), hit("B-403", 0.9)]
        resolved = resolve.resolve(results)
        assert resolved[0].code is None

    def test_empty_input_yields_empty_output(self):
        assert resolve.resolve([]) == []
