import pytest

from puzzlefind import config
from puzzlefind import vocabulary as v


class TestVocabularySpace:
    def test_holds_exactly_one_code_per_puzzle_piece(self):
        # 四段区间铺满 1–1000，与拼图块数一一对应
        assert len(v.all_codes()) == 1000
        assert len(set(v.all_codes())) == 1000

    def test_number_determines_prefix(self):
        """前缀是冗余的——这是整个新词表的核心性质。

        区间互不重叠，所以每个数字只可能属于一个字母组。前缀因此退化成
        一位校验码：读错了能发现，且能由数字纠正回来。
        """
        for number in range(1, 1001):
            prefixes = [
                p for p in config.CODE_RANGES if v.is_valid_code(f"{p}-{number}")
            ]
            assert prefixes == [v.prefix_for_number(number)]


class TestIsValidCode:
    @pytest.mark.parametrize(
        "code",
        ["A-1", "A-9", "A-42", "A-260", "B-261", "B-403", "C-521", "C-760",
         "D-761", "D-1000"],
    )
    def test_accepts_real_codes(self, code):
        assert v.is_valid_code(code) is True

    @pytest.mark.parametrize(
        "code",
        [
            "A-0",      # 编号从 1 起
            "A-261",    # 越出 A 段上界
            "B-260",    # 越出 B 段下界
            "C-761",
            "D-760",
            "D-1001",   # 越出总上界
            "A-001",    # 碎片上不补零，补零形式不是合法编号
            "A-01",
            "E-100",    # 前缀不存在
            "a-100",
            "A100",
            "",
            "A-abc",
            "AA-100",
        ],
    )
    def test_rejects_everything_else(self, code):
        assert v.is_valid_code(code) is False

    def test_prefix_number_mismatch_is_invalid(self):
        """新增的一整类：格式合法但前缀与数字矛盾。

        旧词表里 A-403 是合法的，四道防线全都拦不住前缀误读。
        区间确定之后它当场不合法——README 里那条已知限制就此消失。
        """
        assert v.is_valid_code("A-403") is False


class TestPrefixForNumber:
    @pytest.mark.parametrize(
        "number,prefix",
        [(1, "A"), (260, "A"), (261, "B"), (520, "B"),
         (521, "C"), (760, "C"), (761, "D"), (1000, "D")],
    )
    def test_maps_each_band_to_its_letter(self, number, prefix):
        assert v.prefix_for_number(number) == prefix

    @pytest.mark.parametrize("number", [0, -1, 1001, 9999])
    def test_returns_none_outside_the_puzzle(self, number):
        assert v.prefix_for_number(number) is None


class TestNormalizeOcrText:
    def test_uppercases_and_strips(self):
        assert v.normalize_ocr_text("  b-403  ") == "B-403"

    def test_removes_internal_whitespace(self):
        assert v.normalize_ocr_text("B - 403") == "B-403"

    def test_normalizes_unicode_dashes(self):
        assert v.normalize_ocr_text("B—403") == "B-403"
        assert v.normalize_ocr_text("B–403") == "B-403"

    @pytest.mark.parametrize(
        "raw,expected",
        [("B403", "B-403"), ("A7", "A-7"), ("A42", "A-42"), ("D1000", "D-1000")],
    )
    def test_inserts_missing_hyphen_for_any_digit_count(self, raw, expected):
        """碎片上不补零，所以数字段是 1–4 位不定长，补连字符不能再假定长度为 4。"""
        assert v.normalize_ocr_text(raw) == expected

    @pytest.mark.parametrize(
        "raw,expected", [("A-001", "A-1"), ("A-042", "A-42"), ("D-0762", "D-762")]
    )
    def test_strips_leading_zeros(self, raw, expected):
        """人手输入和 OCR 都可能补零，但碎片上印的是不补零的形式。"""
        assert v.normalize_ocr_text(raw) == expected

    def test_leaves_unrecoverable_text_alone(self):
        assert v.normalize_ocr_text("???") == "???"

    def test_does_not_insert_hyphen_when_tail_is_not_digits(self):
        assert v.normalize_ocr_text("BA03") == "BA03"


class TestConfusionDistance:
    def test_identical_strings_have_zero_distance(self):
        assert v.confusion_distance("B-403", "B-403") == 0.0

    def test_confusable_pair_costs_less_than_one(self):
        cheap = v.confusion_distance("8-403", "B-403")
        assert 0.0 < cheap < 1.0

    def test_unconfusable_pair_costs_one(self):
        assert v.confusion_distance("A-403", "B-403") == 1.0

    def test_dropped_character_costs_less_than_hallucinated_one(self):
        """不对称是有实测依据的：真实照片里的失败读数是 D-97 / D-89 / D-83，
        全是**漏读**字符，没有一例是凭空多读出字符。

        不做这个区分的话，A-403 到 B-403（一次替换）和到 A-40（删掉尾数）
        代价相同，前缀纠错就永远卡在平局上。
        """
        dropped = v.confusion_distance("B-40", "B-403")     # OCR 漏了一位
        spurious = v.confusion_distance("B-403", "B-40")    # OCR 多读一位
        assert dropped < spurious


class TestSnap:
    def test_exact_valid_code_snaps_to_itself_with_zero_distance(self):
        code, dist = v.snap("B-403")
        assert code == "B-403"
        assert dist == 0.0

    def test_confusable_misread_snaps_to_valid_code(self):
        code, dist = v.snap("8-4O3")
        assert code == "B-403"
        assert dist > 0.0

    def test_prefix_error_is_never_silently_accepted(self):
        """旧词表里 A-403 是合法编号，前缀误读会被当成真编号收下——
        碎片被登记到错误的号上，且占掉一个真编号的位置，四道防线全穿。

        区间确定之后它不再合法。**这是本次改动真正兑现的东西**：
        3000 种可能的前缀误读全部会被检出，一个都不会被静默收下。
        """
        assert v.is_valid_code("A-403") is False
        assert v.snap("A-403")[0] != "A-403"

    def test_prefix_error_is_usually_ambiguous_rather_than_correctable(self):
        """检出 ≠ 可纠正，这一点必须说清楚。

        A-403 有三个等距候选：前缀错了（B-403），或百位错了（A-103 / A-203）。
        没有任何依据能在三者间做选择，所以答案是「不知道」。
        实测 3000 种前缀误读里 2997 种如此——但其中**没有一种**会被纠正成
        错误的编号，全部落进未识别集合。失败模式从「悄悄读错」变成了
        「明确的读不出」，这正是 design.md §7 设计来吃掉的那一类。
        """
        assert v.snap("A-403")[0] is None

    def test_prefix_error_is_corrected_when_no_digit_alternative_exists(self):
        """D-1000 是唯一的四位编号，没有等长的邻居能跟它抢，
        于是前缀怎么错都能纠正回来。"""
        assert v.snap("A-1000")[0] == "D-1000"

    def test_four_digit_code_survives_a_confusable_misread(self):
        code, _ = v.snap("D-l000")
        assert code == "D-1000"

    def test_zero_padded_input_finds_the_unpadded_code(self):
        code, dist = v.snap("A-001")
        assert code == "A-1"
        assert dist == 0.0

    def test_refuses_to_guess_when_a_dropped_digit_is_ambiguous(self):
        """D-97 是实测里真实出现过的读数。补哪一位都说得通
        （D-970…D-979 全部等距），此时唯一诚实的回答是「不知道」。
        猜错会占掉一个真编号，比留空贵得多。
        """
        code, _ = v.snap("D-97")
        assert code is None

    def test_refuses_to_guess_a_one_digit_code_from_an_unreadable_digit(self):
        """A-1…A-9 只有一位数字，读不出来就没有任何信息可依据。"""
        code, _ = v.snap("A-@")
        assert code is None

    def test_one_digit_code_still_recovers_a_confusable_digit(self):
        # I/1 形近，代价低于一次普通替换，仍在短编号的预算内
        code, _ = v.snap("A-I")
        assert code == "A-1"

    def test_budget_scales_down_with_digit_count(self):
        """短编号上一次整字符替换就是全部信息的一大半，不能再按 5 字符
        编号的 2.0 预算放行——否则噪声会被吸附成真编号（假阳性）。
        """
        assert v.snap("R-7")[0] is None       # 1 位数字：预算 0.7，装不下 1.0
        assert v.snap("R-403")[0] == "B-403"  # 3 位数字：预算 2.0，同样的替换放行

    def test_hopeless_garbage_returns_none(self):
        code, _ = v.snap("XYZQWERTY")
        assert code is None

    def test_respects_max_distance(self):
        code, _ = v.snap("8-4O3", max_distance=0.1)
        assert code is None

    def test_wildly_wrong_length_is_rejected_without_scanning(self):
        code, distance = v.snap("QWERTYUIOP")
        assert code is None
        assert distance >= config.SNAP_MAX_DISTANCE

    @pytest.mark.parametrize(
        "raw", ["8-4O3", "B-4O3", "8-403", "D-5S0", "B0403", "6-9OO", "A-1OO",
                "D-l000", "A-I", "D-97", "A-403"],
    )
    def test_result_agrees_with_a_brute_force_scan(self, raw):
        """护栏：snap 内部为了跳过全量 DP 做了剪枝和等长快路径，
        这里用最笨的实现复算一遍，两者必须一致。

        旧版这个位置测的是「4000 候选压成 34 次比较」那条快路径。它的
        正确性依赖「前缀与数字可任意组合」，而区间确定之后该前提不再成立，
        所以那条路径连同它的证明一起删了，换成这个更钝但不会骗人的对照。
        """
        text = v.normalize_ocr_text(raw)
        best_distance = min(
            v.confusion_distance(text, candidate) for candidate in v.all_codes()
        )
        winners = [
            c for c in v.all_codes()
            if v.confusion_distance(text, c) <= best_distance + config.SNAP_AMBIGUITY_MARGIN
        ]

        code, distance = v.snap(text)
        assert distance == pytest.approx(best_distance)
        if len(winners) > 1:
            assert code is None, f"{text} 有 {len(winners)} 个等距候选，不该给出答案"
        elif code is not None:
            assert code == winners[0]

    def test_never_returns_a_code_outside_the_vocabulary(self):
        vocabulary = set(v.all_codes())
        for raw in ["A-403", "8-4O3", "D-l000", "A-I", "C-9O9", "B-1", "E-500"]:
            code, _ = v.snap(raw)
            assert code is None or code in vocabulary
