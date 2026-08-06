"""全局约束消解：同一张照片内编号必须唯一。

这一层是纯逻辑、零成本，但能吃掉相当一部分识别错误——两块碎片被读成
同一个编号时，至少有一个是错的，我们保留置信度高的那个。

**这里曾经还有一条「离群值剔除」规则**：从当前这批结果里自举出各字母组的
数字区间，把明显越界的编号判为误读。编号区间实测确定之后它被删掉了，
两个理由：越界编号（B-901）现在压根不是合法编号，snap 那一层就拦住了，
这条规则永远不会触发；而它真正会触发的场合恰恰是**误伤**——一张全是
D-9xx 的照片里混进一块真实的 D-762，四分位围栏会把这个对的结果丢掉。
一条只会做坏事的规则，留着没有意义。
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from .recognize import RecogResult


def _demote(result: RecogResult) -> RecogResult:
    """把一条结果降级为「未识别」，但保留原始读数供排查。"""
    return replace(result, code=None, confidence=0.0, method="conflict")


def resolve(results: list[RecogResult]) -> list[RecogResult]:
    """消解冲突。返回与输入等长的列表，被否决的项 code 变为 None。

    只剩一条规则：唯一性——同一编号出现多次时，只保留置信度最高的那一个。
    """
    if not results:
        return []

    resolved = list(results)

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
