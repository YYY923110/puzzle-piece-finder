"""编号词表：校验、归一化、混淆感知吸附。

词表是 1000 个编号，四段区间实测确定（见 config.CODE_RANGES），
与拼图块数一一对应。**区间互不重叠，所以数字唯一确定前缀**——
前缀不携带信息，它是一位校验码。这条性质贯穿本模块：

- `is_valid_code` 同时校验格式与区间，A-403 这类前缀/数字矛盾的读数当场出局；
- `snap` 因此能把前缀误读纠正回来（旧词表里这是一个拦不住的洞）。

碎片上**不补零**（A-1 / A-42 / D-1000），所以编号长度在 3–6 之间浮动。
这让漏读一位数字和读错前缀常常等距，`snap` 用不对称的增删代价和一条
歧义判据来处理，两者都在 config 里带着实测依据。
"""
from __future__ import annotations

import re
from functools import lru_cache

from . import config

# 形近字符组。同组内的替换代价低于普通替换。
_CONFUSION_GROUPS: tuple[frozenset[str], ...] = (
    frozenset("0OD"),
    frozenset("1IL"),
    frozenset("8B"),
    frozenset("5S"),
    frozenset("2Z"),
    frozenset("6G"),
    frozenset("9Q"),
)
_CONFUSION_COST = 0.4

_DASH_CHARS = "‐‑‒–—―−_"


def _substitution_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    for group in _CONFUSION_GROUPS:
        if a in group and b in group:
            return _CONFUSION_COST
    return 1.0


@lru_cache(maxsize=1)
def all_codes() -> tuple[str, ...]:
    """全部 1000 个合法编号，按区间顺序。"""
    return tuple(
        f"{prefix}-{number}"
        for prefix, (low, high) in config.CODE_RANGES.items()
        for number in range(low, high + 1)
    )


def prefix_for_number(number: int) -> str | None:
    """数字所属的字母组；不在 1–1000 内则为 None。"""
    for prefix, (low, high) in config.CODE_RANGES.items():
        if low <= number <= high:
            return prefix
    return None


def is_valid_code(code: str) -> bool:
    """是否是**真实存在**的编号——格式合法**且**数字落在该前缀的区间内。

    这比旧版的纯格式校验强一档：A-403 格式没问题，但 403 属于 B 段，
    所以它不是编号，只可能是一次前缀误读。
    """
    match = config.CODE_PATTERN.match(code)
    if match is None:
        return False
    prefix, digits = match.groups()
    low, high = config.CODE_RANGES[prefix]
    return low <= int(digits) <= high


def normalize_ocr_text(raw: str) -> str:
    """把 OCR 原始输出整理成规范形状，尽量凑成 `X-N`。

    做五件事：去空白、转大写、统一各种破折号为 ASCII 连字符、
    在「单字符 + 1–4 位数字」之间补上缺失的连字符、去掉数字段的前导零。
    不做形近字符替换——那是 snap 的职责。

    补连字符时**不要求首字符是字母**：OCR 把 B 读成 8 时吐出的是 `8403`，
    限定成字母就把这条救援路径堵死了。
    去前导零是因为碎片上印的是不补零的形式，而人手查询和 OCR 都可能补零。
    """
    text = raw.strip().upper()
    for dash in _DASH_CHARS:
        text = text.replace(dash, "-")
    text = re.sub(r"\s+", "", text)

    match = re.fullmatch(r"(.)(\d{1,4})", text)
    if match:
        text = f"{match.group(1)}-{match.group(2)}"

    match = re.fullmatch(r"(.)-0*(\d+)", text)
    if match:
        text = f"{match.group(1)}-{match.group(2)}"
    return text


def confusion_distance(text: str, candidate: str) -> float:
    """带形近字符折扣的编辑距离（Levenshtein 变体）。

    **参数不对称，顺序有意义**：`text` 是观测到的读数，`candidate` 是
    假设的真值。于是两种长度差对应两种不同的 OCR 失误：

    - candidate 更长 → OCR **漏读**了字符，代价 SNAP_DROPPED_CHAR_COST；
    - candidate 更短 → OCR **凭空多读**出字符，代价 SNAP_SPURIOUS_CHAR_COST。

    多读比漏读贵，依据是实测：真实照片里的失败读数 D-97 / D-89 / D-83
    全是漏读，没有一例是多读。理由与代价值都记在 config 里。
    """
    if len(text) == len(candidate):
        # 等长时最优对齐必为纯替换：一次替换最多 1.0，而一增一删至少
        # 是 DROPPED + SPURIOUS = 2.5。省掉整个 DP 表。
        return sum(_substitution_cost(a, b) for a, b in zip(text, candidate))

    spurious = config.SNAP_SPURIOUS_CHAR_COST
    dropped = config.SNAP_DROPPED_CHAR_COST
    m, n = len(text), len(candidate)
    prev = [j * dropped for j in range(n + 1)]
    for i in range(1, m + 1):
        cur = [i * spurious] + [0.0] * n
        for j in range(1, n + 1):
            cur[j] = min(
                prev[j] + spurious,                                     # 多读
                cur[j - 1] + dropped,                                   # 漏读
                prev[j - 1] + _substitution_cost(text[i - 1], candidate[j - 1]),
            )
        prev = cur
    return prev[n]


_CANDIDATE_LENGTHS = frozenset(len(code) for code in all_codes())


def _length_cost(text_length: int, candidate_length: int) -> float:
    """仅由长度差决定的距离下界——不看内容就能算，用来剪枝。"""
    if candidate_length >= text_length:
        return (candidate_length - text_length) * config.SNAP_DROPPED_CHAR_COST
    return (text_length - candidate_length) * config.SNAP_SPURIOUS_CHAR_COST


def _budget(code: str, cap: float) -> float:
    """某个候选能接受的最大距离：按它的**数字位数**缩放，再受绝对上限约束。

    理由见 config.SNAP_DISTANCE_PER_DIGIT：五字符编号的 2.0 预算换到
    三字符的 A-1 上就是「三个字符里错两个也认」，那是在制造假阳性。
    """
    digits = len(code) - 2  # 去掉前缀字符和连字符
    return min(cap, digits * config.SNAP_DISTANCE_PER_DIGIT)


def snap(raw: str, max_distance: float | None = None) -> tuple[str | None, float]:
    """把 OCR 输出吸附到最近的合法编号。

    返回 (编号, 距离)。三种情况会返回 (None, 距离)：距离超出该候选的预算、
    最优与次优拉不开差距（歧义），以及长度差得太离谱。

    **歧义时宁可不答。** 不补零的编号里，「漏读一位数字」和「读错前缀」
    经常等距：D-97 补成 D-970…D-979 全都说得通。猜错会占掉一个真编号，
    而查不到时系统本来就会把未识别碎片全部高亮出来（见 design.md §7），
    留空的代价小得多。
    """
    cap = config.SNAP_MAX_DISTANCE if max_distance is None else max_distance
    margin = config.SNAP_AMBIGUITY_MARGIN

    text = normalize_ocr_text(raw)
    if not text:
        return None, float("inf")
    if is_valid_code(text):
        return text, 0.0

    # 长度剪枝：和任何候选都差太多时，连扫都不必扫。
    # 用 cap + margin 而不是 cap，这样被剪掉的候选一定也够不上「次优」。
    floor = min(_length_cost(len(text), n) for n in _CANDIDATE_LENGTHS)
    if floor > cap + margin:
        return None, floor

    best_code: str | None = None
    best = runner_up = float("inf")
    for candidate in all_codes():
        if _length_cost(len(text), len(candidate)) > cap + margin:
            continue
        distance = confusion_distance(text, candidate)
        if distance < best:
            best, runner_up, best_code = distance, best, candidate
        elif distance < runner_up:
            runner_up = distance

    if best_code is None:
        return None, float("inf")
    if runner_up - best < margin:
        return None, best
    if best > _budget(best_code, cap):
        return None, best
    return best_code, best
