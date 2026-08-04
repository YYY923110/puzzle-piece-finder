"""全局约束消解：同一张照片内编号必须唯一，且不应落在自举区间之外。

这一层是纯逻辑、零成本，但能吃掉相当一部分识别错误——两块碎片被读成
同一个编号时，至少有一个是错的，我们保留置信度高的那个。
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from . import vocabulary
from .recognize import RecogResult


def _demote(result: RecogResult) -> RecogResult:
    """把一条结果降级为「未识别」，但保留原始读数供排查。"""
    return replace(result, code=None, confidence=0.0, method="conflict")


def resolve(results: list[RecogResult]) -> list[RecogResult]:
    """消解冲突。返回与输入等长的列表，被否决的项 code 变为 None。

    两条规则，依次施加：
    1. 离群值剔除——先从当前结果自举出各字母组的数字区间，
       明显越界的编号（如 B 组集中在 262–499 却读出 B-901）判为误读。
    2. 唯一性——同一编号出现多次时，只保留置信度最高的那一个。
    """
    if not results:
        return []

    resolved = list(results)

    # 规则 1：离群值剔除。
    # 用 robust_ranges 而不是 bootstrap_ranges：后者的 min/max 是从同一批
    # 数据里取的，离群值会成为自己的边界，永远抓不到。详见该函数注释。
    ranges = vocabulary.robust_ranges([r.code for r in resolved if r.code])
    resolved = [
        _demote(r) if r.code and vocabulary.is_outlier(r.code, ranges) else r
        for r in resolved
    ]

    # 规则 2：唯一性
    by_code: dict[str, list[int]] = defaultdict(list)
    for index, result in enumerate(resolved):
        if result.code:
            by_code[result.code].append(index)

    for indices in by_code.values():
        if len(indices) <= 1:
            continue
        winner = max(indices, key=lambda i: resolved[i].confidence)
        for index in indices:
            if index != winner:
                resolved[index] = _demote(resolved[index])

    return resolved
