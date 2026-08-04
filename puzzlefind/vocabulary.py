"""编号词表：校验、归一化、混淆感知吸附、区间自举。

词表是 {A,B,C,D} × 000..999 共 4000 个候选。各字母组的实际区间未知，
由 bootstrap_ranges 从识别结果自举。
"""
from __future__ import annotations

import re
import statistics
from collections import defaultdict

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


def is_valid_code(code: str) -> bool:
    """是否是格式合法的编号（形如 B-403）。"""
    return bool(config.CODE_PATTERN.match(code))


def normalize_ocr_text(raw: str) -> str:
    """把 OCR 原始输出整理成规范形状，尽量凑成 `X-NNN`。

    做四件事：去空白、转大写、统一各种破折号为 ASCII 连字符、
    在「单字母 + 三位数字」之间补上缺失的连字符。
    不做形近字符替换——那是 snap 的职责。
    """
    text = raw.strip().upper()
    for dash in _DASH_CHARS:
        text = text.replace(dash, "-")
    text = re.sub(r"\s+", "", text)
    # 补连字符：开头一个非连字符字符，紧跟三个非连字符字符
    if "-" not in text and len(text) == 4:
        text = f"{text[0]}-{text[1:]}"
    return text


def confusion_distance(a: str, b: str) -> float:
    """带形近字符折扣的编辑距离（Levenshtein 变体）。"""
    m, n = len(a), len(b)
    prev = [float(j) for j in range(n + 1)]
    for i in range(1, m + 1):
        cur = [float(i)] + [0.0] * n
        for j in range(1, n + 1):
            cur[j] = min(
                prev[j] + 1.0,                                  # 删除
                cur[j - 1] + 1.0,                               # 插入
                prev[j - 1] + _substitution_cost(a[i - 1], b[j - 1]),  # 替换
            )
        prev = cur
    return prev[n]


_CODE_LENGTH = 5  # "X-NNN"
_DIGITS = "0123456789"


def _iter_vocabulary():
    for prefix in config.VALID_PREFIXES:
        for number in range(config.MIN_NUMBER, config.MAX_NUMBER + 1):
            yield f"{prefix}-{number:03d}"


def _snap_aligned(text: str) -> tuple[str, float]:
    """长度恰为 5 时的快速路径：34 次比较取代 4000 次完整 DP。

    为什么这是精确的（不是近似）：两个等长字符串之间，一次替换的代价
    最多 1.0，而一次删除加一次插入固定是 2.0，所以最优对齐一定是纯替换。
    于是总距离等于逐位替换代价之和；而候选空间里前缀和三位数字可以
    任意组合，所以逐位取最小值之和就是全局最小值。

    这条路径覆盖绝大多数调用——Pass C 每块碎片跑 12 个角度，
    全量扫描会让建索引凭空多花一两分钟。
    """
    prefix, total = min(
        ((p, _substitution_cost(text[0], p)) for p in config.VALID_PREFIXES),
        key=lambda item: item[1],
    )
    total += _substitution_cost(text[1], "-")

    digits: list[str] = []
    for char in text[2:5]:
        digit, cost = min(
            ((d, _substitution_cost(char, d)) for d in _DIGITS),
            key=lambda item: item[1],
        )
        digits.append(digit)
        total += cost
    return f"{prefix}-{''.join(digits)}", total


def _snap_exhaustive(text: str) -> tuple[str | None, float]:
    """全量扫描 4000 个候选。仅用于长度不等于 5 的少数情况。"""
    best_code: str | None = None
    best_distance = float("inf")
    for candidate in _iter_vocabulary():
        distance = confusion_distance(text, candidate)
        if distance < best_distance:
            best_distance, best_code = distance, candidate
            if distance == 0.0:
                break
    return best_code, best_distance


def snap(raw: str, max_distance: float | None = None) -> tuple[str | None, float]:
    """把 OCR 输出吸附到最近的合法编号。

    返回 (编号, 距离)。距离超过 max_distance 时返回 (None, 距离)。
    """
    if max_distance is None:
        max_distance = config.SNAP_MAX_DISTANCE

    text = normalize_ocr_text(raw)
    if is_valid_code(text):
        return text, 0.0
    if not text:
        return None, float("inf")

    # 剪枝：合法编号长度恒为 5，每次增删代价 1.0，所以长度差本身
    # 就是编辑距离的下界。差得太多时连算都不必算。
    length_gap = abs(len(text) - _CODE_LENGTH)
    if length_gap > max_distance:
        return None, float(length_gap)

    if len(text) == _CODE_LENGTH:
        best_code, best_distance = _snap_aligned(text)
    else:
        best_code, best_distance = _snap_exhaustive(text)

    if best_distance > max_distance:
        return None, best_distance
    return best_code, best_distance


def bootstrap_ranges(
    codes: list[str], min_samples: int = 2
) -> dict[str, tuple[int, int]]:
    """从已识别的编号自举出每个字母组的实际数字区间。

    样本数不足 min_samples 的字母组不产出区间——样本太少时
    推出来的区间会过窄，反而把正确结果误判为离群值。

    默认门槛只有 2：单个样本推不出区间（min==max，会把该组其他所有
    编号都判成离群），两个就够画出一条线段了。真正防止「区间过窄」的
    是 robust_ranges 的四分位围栏，不是这个门槛。
    """
    buckets: dict[str, list[int]] = defaultdict(list)
    for code in codes:
        if is_valid_code(code):
            buckets[code[0]].append(int(code[2:]))
    return {
        prefix: (min(nums), max(nums))
        for prefix, nums in buckets.items()
        if len(nums) >= min_samples
    }


def robust_ranges(
    codes: list[str], min_samples: int = 4, k: float = 1.5
) -> dict[str, tuple[int, int]]:
    """自举区间的稳健版本：先用四分位距围栏剔掉极端值，再取剩余值的 min/max。

    为什么不能直接用 bootstrap_ranges 来找离群值：那个区间是从**同一批
    数据**里取 min/max 得来的，所以离群值永远是它自己的边界，永远落在
    区间内，永远抓不到。B 组读出 262/300/350/400/901 时，
    bootstrap_ranges 给出 (262, 901)，于是 901 完全合法。

    改法是先画一道围栏：低于 Q1 - k·IQR 或高于 Q3 + k·IQR 的值不参与
    定义区间。上例中围栏是 (150, 550)，901 被挡在外面，区间收成
    (262, 400)，901 这才暴露成离群值。

    样本数不足 min_samples 的字母组不产出区间。四分位数在四个点以下
    没有意义，硬算只会把正常值判成离群。
    """
    buckets: dict[str, list[int]] = defaultdict(list)
    for code in codes:
        if is_valid_code(code):
            buckets[code[0]].append(int(code[2:]))

    ranges: dict[str, tuple[int, int]] = {}
    for prefix, numbers in buckets.items():
        if len(numbers) < min_samples:
            continue
        q1, _, q3 = statistics.quantiles(numbers, n=4, method="inclusive")
        spread = q3 - q1
        low_fence, high_fence = q1 - k * spread, q3 + k * spread
        kept = [n for n in numbers if low_fence <= n <= high_fence]
        if kept:
            ranges[prefix] = (min(kept), max(kept))
    return ranges


def is_outlier(code: str, ranges: dict[str, tuple[int, int]]) -> bool:
    """编号是否落在自举出的区间之外。未知字母组一律不算离群。"""
    if not is_valid_code(code):
        return False
    prefix = code[0]
    if prefix not in ranges:
        return False
    low, high = ranges[prefix]
    return not (low <= int(code[2:]) <= high)
