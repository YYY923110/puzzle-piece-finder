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

    def test_a_lone_low_number_is_kept_among_high_ones(self):
        """区间自举那条规则删掉之后，这一条才成立。

        旧实现从**同一批结果**里自举出 D 组区间再剔离群值，于是一张
        全是 D-9xx 的照片里混进一块真实的 D-762，会被四分位围栏当成误读
        丢掉——把对的结果毁掉。越界编号现在在 snap 那一层就进不来了
        （D-762 合法，B-901 根本不是编号），照片内的统计推断不再有存在理由。
        """
        results = [
            hit("D-900", 0.9),
            hit("D-910", 0.9),
            hit("D-920", 0.9),
            hit("D-930", 0.9),
            hit("D-762", 0.9),
        ]
        resolved = resolve.resolve(results)
        assert [r.code for r in resolved] == [
            "D-900", "D-910", "D-920", "D-930", "D-762",
        ]

    def test_none_results_are_left_untouched(self):
        results = [hit(None, 0.0), hit("B-403", 0.9)]
        resolved = resolve.resolve(results)
        assert resolved[0].code is None

    def test_empty_input_yields_empty_output(self):
        assert resolve.resolve([]) == []
