# 拼图碎片编号查找器 Implementation Plan

> ## ⚠️ 本文档已执行完毕，且其中的代码有 6 处是错的
>
> **状态：2026-08-03 全部 14 个 Task 已实现并提交。这份 plan 现在是历史存档。**
>
> 执行过程中，plan 里给出的代码被它**自己的测试**逼出了 6 处缺陷。
> 下面的代码块保持原样未改，以便对照当时的判断——
> **要看能跑的版本请直接读 `puzzlefind/` 下的源码，不要从这里复制。**
>
> | # | 位置 | 缺陷 | 实际修法 |
> |---|---|---|---|
> | 1 | Task 3 `split_blob` | 二分方向反了。抬高距离变换阈值会**淹掉**碎片间的峡谷、让种子**变多**，代码却往反方向收边界，永远收敛不了 | 交换两个分支 |
> | 2 | Task 3 面积先验 | `median_blob_area` 假设「多数连通块是单块」，只有一个粘连团时中位数恒等于它自己，比值恒为 1.00，切分永不触发 | 新增 `peak_count` / `unit_piece_area` |
> | 3 | 修复 2 引入的回归 | `peak_count` 在**真实**拼图碎片上恒返回 3（每个凸起都是一个距离变换山峰），单块面积被压低 3 倍，**50 块碎片被切成 96 个轮廓**。合成 fixture 全是圆形，圆没有凸起，测不出来 | 连通块 ≥5 时改用面积中位数；补 `separated_puzzle_pieces` 带凸起 fixture |
> | 4 | Task 7 `resolve` | 离群规则**永远不可能触发**：区间用 min/max 从待过滤的同一批数据里取，离群值总是自己的边界 | 新增 `vocabulary.robust_ranges`（四分位围栏） |
> | 5 | Task 1 `bootstrap_ranges` | 默认 `min_samples=3` 与 plan 自己的测试矛盾（该测试要求 2 样本前缀出区间） | 默认改为 2 |
> | 6 | Task 4 `contour_bbox` 测试 | 期望值 off-by-one。`cv2.boundingRect` 在 4.x 和 5.x 上都返回**含两端**的宽高 | 改测试，不改实现 |
>
> 另外还有 3 处**运行时**问题 plan 无从预知，已在源码注释中说明：
> paddlex 的 1 秒模型源探测超时、paddle 3.3.1 的 oneDNN 崩溃、代理 TUN 网卡导致局域网地址误报。
>
> 实测结果与参数决策依据见 [`docs/tuning-log.md`](../../tuning-log.md)，
> 上手运行见 [`README.md`](../../../README.md)。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **先读 spec：** [`../specs/2026-08-03-puzzle-piece-finder.md`](../specs/2026-08-03-puzzle-piece-finder.md)
> 记录了每条设计决策的理由，以及**被明确否决的替代方案**（大模型识别、SAM 分割、
> 任意背景支持、人工回填等）。不要在实现过程中重新提议它们。

**Goal:** 上传若干张拼图碎片照片建立「编号 → 碎片位置」索引，之后输入任意编号（如 `B-403`）即可在照片上高亮出对应碎片。

**Architecture:** 三层。① 纯 Python 引擎：OpenCV 从深色背景分割出碎片轮廓 → 逐块裁剪 → PaddleOCR 识别编号 → 词表吸附 + 同图唯一性去重 → 输出 JSON 索引。② FastAPI 薄服务层：上传建索引、跨照片查询、服务端渲染高亮图。③ 单文件 HTML 前端：手机局域网访问，拍照上传、输编号、看高亮。引擎完全独立于 Web，可用 CLI 单独跑，便于调参迭代。

**Tech Stack:** Python 3.13 / opencv-python-headless 5.0 / paddlepaddle 3.3.1 (CPU) / paddleocr 3.x / FastAPI + uvicorn / 原生 HTML+JS / JSON 文件存储 / pytest

---

## Global Constraints

- **Python 3.13.9**，Anaconda 在 `D:\anaconda3`。所有命令用 `python`（不是 `py`，该启动器不存在）。
- **PaddlePaddle 必须装 CPU 版**：`pip install paddlepaddle==3.3.1`（已确认有 `cp313-win_amd64` 轮子）。**不要**装 `paddlepaddle-gpu`。
- **PaddleOCR 必须用 3.x API**：构造参数是 `use_doc_orientation_classify` / `use_doc_unwarping` / `use_textline_orientation`，推理方法是 `.predict()`。**不要**用 2.x 的 `use_angle_cls` / `.ocr()`——网上大量教程是旧写法。
- **opencv 是 headless 版**（`opencv-python-headless 5.0.0.93`）：`cv2.imshow` / `cv2.waitKey` 不可用。调试一律用 `cv2.imwrite` 写文件到 `debug/`。
- **合法编号词表**：正则 `^[A-D]-\d{3}$`，数字范围 000–999，共 4000 个候选。各字母组的实际区间未知，运行时自举。
- **每张照片内编号唯一**：同一 `PhotoIndex` 中任意编号最多出现一次。
- **拍摄前提**：碎片摊在**深色纯背景**上，每张照片 40–60 块。
- **一切文件路径用 `pathlib.Path`**，不要拼字符串（Windows 反斜杠）。
- **JSON 一律 `ensure_ascii=False, indent=2`**，人要能直接打开看。
- 项目根目录：`d:\ocr_claude`。

---

## File Structure

| 路径 | 职责 |
|---|---|
| `pyproject.toml` | 依赖与 pytest 配置 |
| `puzzlefind/config.py` | 所有可调参数集中一处（阈值、角度表、颜色、路径） |
| `puzzlefind/models.py` | 数据类：`Piece` / `PhotoIndex` / `QueryResult` + JSON 序列化 |
| `puzzlefind/vocabulary.py` | 编号校验、OCR 文本归一化、混淆感知的词表吸附、区间自举 |
| `puzzlefind/segment.py` | 掩膜、轮廓提取、粘连碎片分水岭切分、裁剪归一化 |
| `puzzlefind/recognize.py` | `Recognizer` 协议、PaddleOCR 后端、直接识别 + 旋转穷举 |
| `puzzlefind/resolve.py` | 全局唯一性冲突消解 |
| `puzzlefind/pipeline.py` | 串起分割→识别→消解，产出 `PhotoIndex` |
| `puzzlefind/library.py` | 多照片库：持久化、跨照片查询 |
| `puzzlefind/render.py` | 高亮渲染：压暗 + 描边 + 标签 + 缩略图 |
| `puzzlefind/cli.py` | 命令行入口（脱离 Web 调参用） |
| `puzzlefind/server.py` | FastAPI 应用 |
| `puzzlefind/static/index.html` | 单文件前端 |
| `tests/` | 与上述模块一一对应的测试 + `conftest.py` 合成图 fixture |

**关键设计约束（决定了可测试性）：** `recognize.py` 暴露一个 `Recognizer` 协议。除 Task 5/6 外，所有测试注入 `FakeRecognizer`，**不加载 PaddleOCR**。这让绝大多数测试在毫秒级跑完，且不依赖模型下载。

---

## ⚠️ 需要你在执行前确认的两个偏离

1. **旋转穷举（Pass C）默认非常激进。** `config.SWEEP_CONFIDENCE_THRESHOLD` 默认 `0.90`——意味着直接识别置信度低于 0.90 的碎片全部进入 12 角度穷举。这是根据 PaddleOCR 社区对任意角度文字的负面反馈做的保守设定。跑完第一张真实照片后按 Task 14 的数据下调。
2. **高亮改为服务端渲染。** 原定「Canvas 客户端合成」，改为服务端用 `render.py` 生成高亮 PNG，前端只做 `<img>` + CSS transform 的缩放平移。理由：渲染逻辑因此可被单元测试覆盖，且前端代码量降到几十行。缩放平移体验不受影响。

---

### Task 1: 项目骨架 + 配置 + 词表

**Files:**
- Create: `pyproject.toml`
- Create: `puzzlefind/__init__.py`
- Create: `puzzlefind/config.py`
- Create: `puzzlefind/vocabulary.py`
- Create: `tests/__init__.py`
- Test: `tests/test_vocabulary.py`

**Interfaces:**
- Consumes: 无（首个任务）
- Produces:
  - `config.VALID_PREFIXES: tuple[str, ...]` = `("A","B","C","D")`
  - `config.CODE_PATTERN: re.Pattern`
  - `config.SNAP_MAX_DISTANCE: float` = `2.0`
  - `config.SWEEP_ANGLES: tuple[int, ...]`
  - `config.SWEEP_CONFIDENCE_THRESHOLD: float`
  - `vocabulary.is_valid_code(code: str) -> bool`
  - `vocabulary.normalize_ocr_text(raw: str) -> str`
  - `vocabulary.confusion_distance(a: str, b: str) -> float`
  - `vocabulary.snap(raw: str, max_distance: float = ...) -> tuple[str | None, float]`
  - `vocabulary.bootstrap_ranges(codes: list[str]) -> dict[str, tuple[int, int]]`
  - `vocabulary.is_outlier(code: str, ranges: dict[str, tuple[int, int]]) -> bool`

- [ ] **Step 1: 建立项目骨架文件**

创建 `pyproject.toml`：

```toml
[project]
name = "puzzlefind"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
    "opencv-python-headless>=5.0",
    "numpy>=2.0",
    "paddlepaddle==3.3.1",
    "paddleocr>=3.0",
    "fastapi>=0.141",
    "uvicorn[standard]>=0.30",
    "python-multipart>=0.0.9",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[project.scripts]
puzzlefind = "puzzlefind.cli:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["ocr: 需要加载 PaddleOCR 的慢速测试"]
addopts = "-q"
```

创建空的 `puzzlefind/__init__.py` 和 `tests/__init__.py`。

- [ ] **Step 2: 写配置模块**

创建 `puzzlefind/config.py`：

```python
"""所有可调参数集中在这里。调参只改这一个文件。"""
from __future__ import annotations

import re
from pathlib import Path

# ---------- 路径 ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PHOTOS_DIR = DATA_DIR / "photos"
INDEX_DIR = DATA_DIR / "index"
DEBUG_DIR = PROJECT_ROOT / "debug"

# ---------- 词表 ----------
VALID_PREFIXES: tuple[str, ...] = ("A", "B", "C", "D")
CODE_PATTERN = re.compile(r"^[A-D]-\d{3}$")
MIN_NUMBER = 0
MAX_NUMBER = 999
SNAP_MAX_DISTANCE: float = 2.0

# ---------- 分割 ----------
# 前景（碎片）应显著亮于深色背景。低于此灰度的像素视为背景。
# 设为 None 表示用 Otsu 自动求阈值（推荐）。
MASK_THRESHOLD: int | None = None
# 面积小于「中位面积 × 此系数」的连通块视为噪点丢弃
MIN_AREA_RATIO: float = 0.25
# 面积大于「中位面积 × 此系数」的连通块视为粘连，尝试切分
SPLIT_AREA_RATIO: float = 1.6
# 分水岭种子搜索时，距离变换阈值的二分次数
SPLIT_SEARCH_STEPS: int = 24
# 形态学开运算核尺寸，用于去除掩膜毛刺
MORPH_KERNEL_SIZE: int = 5

# ---------- 裁剪 ----------
# 裁剪时在碎片包围盒外扩的像素数
CROP_PADDING: int = 6
# 裁剪结果的目标长边像素数（放大以喂饱识别模型）
CROP_TARGET_LONG_EDGE: int = 512
# 碎片外区域填充成什么颜色（BGR）。中性灰避免与浅灰文字/白底冲突
CROP_FILL_COLOR: tuple[int, int, int] = (128, 128, 128)

# ---------- 识别 ----------
# 直接识别（Pass A）置信度低于此值的碎片，进入旋转穷举（Pass C）
# 默认激进：宁可多穷举。跑过真实照片后按实测数据下调。
SWEEP_CONFIDENCE_THRESHOLD: float = 0.90
# 穷举的角度表（度）。若方向分类器可靠，可缩短为 (0, 30, 60, 90, 120, 150)
SWEEP_ANGLES: tuple[int, ...] = (0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330)
# 低于此置信度的识别结果直接判为「未识别」
MIN_ACCEPT_CONFIDENCE: float = 0.35

# ---------- 渲染 ----------
# 非目标区域的亮度系数。0.0 全黑，1.0 不变。保留空间定位参照
DIM_FACTOR: float = 0.45
# 目标碎片描边颜色（BGR）：洋红，与黑背景和白碎片都拉得开
OUTLINE_COLOR: tuple[int, int, int] = (255, 0, 255)
# 未识别碎片的描边颜色（BGR）：青色
UNKNOWN_OUTLINE_COLOR: tuple[int, int, int] = (255, 255, 0)
# 描边粗细相对图像长边的比例
OUTLINE_THICKNESS_RATIO: float = 0.0035
```

- [ ] **Step 3: 写词表模块的失败测试**

创建 `tests/test_vocabulary.py`：

```python
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
        assert v.normalize_ocr_text("B\u2014403") == "B-403"
        assert v.normalize_ocr_text("B\u2013403") == "B-403"

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


class TestIsOutlier:
    def test_code_inside_range_is_not_outlier(self):
        assert v.is_outlier("B-350", {"B": (262, 499)}) is False

    def test_code_outside_range_is_outlier(self):
        assert v.is_outlier("B-501", {"B": (262, 499)}) is True

    def test_unknown_prefix_is_never_outlier(self):
        assert v.is_outlier("D-501", {"B": (262, 499)}) is False
```

- [ ] **Step 4: 运行测试确认失败**

Run: `python -m pytest tests/test_vocabulary.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'puzzlefind.vocabulary'`

- [ ] **Step 5: 实现词表模块**

创建 `puzzlefind/vocabulary.py`：

```python
"""编号词表：校验、归一化、混淆感知吸附、区间自举。

词表是 {A,B,C,D} × 000..999 共 4000 个候选。各字母组的实际区间未知，
由 bootstrap_ranges 从识别结果自举。
"""
from __future__ import annotations

import re
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

_DASH_CHARS = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212_"


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
    codes: list[str], min_samples: int = 3
) -> dict[str, tuple[int, int]]:
    """从已识别的编号自举出每个字母组的实际数字区间。

    样本数不足 min_samples 的字母组不产出区间——样本太少时
    推出来的区间会过窄，反而把正确结果误判为离群值。
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


def is_outlier(code: str, ranges: dict[str, tuple[int, int]]) -> bool:
    """编号是否落在自举出的区间之外。未知字母组一律不算离群。"""
    if not is_valid_code(code):
        return False
    prefix = code[0]
    if prefix not in ranges:
        return False
    low, high = ranges[prefix]
    return not (low <= int(code[2:]) <= high)
```

- [ ] **Step 6: 运行测试确认通过**

Run: `python -m pytest tests/test_vocabulary.py -v`
Expected: PASS，全部用例绿

- [ ] **Step 7: 提交**

```bash
cd /d/ocr_claude
git add pyproject.toml puzzlefind/ tests/
git commit -m "feat: project scaffold, config, and vocabulary snapping"
```

---

### Task 2: 分割 — 掩膜与轮廓提取

**Files:**
- Create: `puzzlefind/segment.py`
- Create: `tests/conftest.py`
- Test: `tests/test_segment.py`

**Interfaces:**
- Consumes: `config.MASK_THRESHOLD`, `config.MIN_AREA_RATIO`, `config.MORPH_KERNEL_SIZE`
- Produces:
  - `segment.build_mask(bgr: np.ndarray) -> np.ndarray` — 返回 uint8 二值图，碎片为 255
  - `segment.find_blobs(mask: np.ndarray) -> list[np.ndarray]` — 返回轮廓列表，每个是 `(N,1,2)` int32
  - `segment.median_blob_area(contours: list[np.ndarray]) -> float`

- [ ] **Step 1: 写合成图 fixture**

创建 `tests/conftest.py`。合成图让分割测试完全确定、不依赖真实照片：

```python
"""合成测试图。深色背景上的白色圆形/方形「碎片」。

真实照片不进版本库，所有分割测试跑在这些合成图上。
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest


def make_canvas(width: int = 800, height: int = 600) -> np.ndarray:
    """深色背景画布（不是纯黑——真实照片总有噪点和反光）。"""
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    rng = np.random.default_rng(seed=42)
    noise = rng.integers(0, 12, size=canvas.shape, dtype=np.uint8)
    return cv2.add(canvas, noise)


def draw_piece(canvas: np.ndarray, center: tuple[int, int], radius: int = 40) -> None:
    """在画布上画一块「碎片」——浅色实心圆。"""
    cv2.circle(canvas, center, radius, (235, 233, 228), thickness=-1)


@pytest.fixture
def separated_pieces() -> tuple[np.ndarray, int]:
    """6 块互不接触的碎片。返回 (图像, 碎片数)。"""
    canvas = make_canvas()
    centers = [(120, 120), (320, 120), (520, 120), (120, 380), (320, 380), (520, 380)]
    for center in centers:
        draw_piece(canvas, center)
    return canvas, len(centers)


@pytest.fixture
def touching_pair() -> tuple[np.ndarray, int]:
    """2 块显著重叠的碎片——阈值分割会把它们连成一块。"""
    canvas = make_canvas()
    draw_piece(canvas, (300, 300), radius=60)
    draw_piece(canvas, (390, 300), radius=60)
    return canvas, 2


@pytest.fixture
def touching_triple_with_singles() -> tuple[np.ndarray, int]:
    """3 块粘连 + 3 块独立，共 6 块。用于验证「中位面积」先验。"""
    canvas = make_canvas(1000, 600)
    for center in [(120, 120), (120, 320), (120, 500)]:
        draw_piece(canvas, center, radius=50)
    for center in [(500, 300), (585, 300), (670, 300)]:
        draw_piece(canvas, center, radius=50)
    return canvas, 6


@pytest.fixture
def canvas_with_noise_speck() -> tuple[np.ndarray, int]:
    """3 块正常碎片 + 一个远小于碎片的亮点噪声。应被丢弃。"""
    canvas = make_canvas()
    for center in [(120, 120), (320, 120), (520, 120)]:
        draw_piece(canvas, center, radius=45)
    cv2.circle(canvas, (400, 400), 5, (240, 240, 240), thickness=-1)
    return canvas, 3
```

- [ ] **Step 2: 写分割测试**

创建 `tests/test_segment.py`：

```python
import cv2
import numpy as np

from puzzlefind import segment


class TestBuildMask:
    def test_mask_is_binary_uint8(self, separated_pieces):
        image, _ = separated_pieces
        mask = segment.build_mask(image)
        assert mask.dtype == np.uint8
        assert set(np.unique(mask).tolist()) <= {0, 255}

    def test_mask_has_same_shape_as_input(self, separated_pieces):
        image, _ = separated_pieces
        mask = segment.build_mask(image)
        assert mask.shape == image.shape[:2]

    def test_piece_centers_are_foreground(self, separated_pieces):
        image, _ = separated_pieces
        mask = segment.build_mask(image)
        assert mask[120, 120] == 255

    def test_background_is_zero(self, separated_pieces):
        image, _ = separated_pieces
        mask = segment.build_mask(image)
        assert mask[300, 700] == 0


class TestFindBlobs:
    def test_finds_every_separated_piece(self, separated_pieces):
        image, expected = separated_pieces
        contours = segment.find_blobs(segment.build_mask(image))
        assert len(contours) == expected

    def test_discards_noise_specks(self, canvas_with_noise_speck):
        image, expected = canvas_with_noise_speck
        contours = segment.find_blobs(segment.build_mask(image))
        assert len(contours) == expected

    def test_touching_pieces_come_back_as_one_blob(self, touching_pair):
        # find_blobs 不负责切分——它只找连通块。切分是 Task 3。
        image, _ = touching_pair
        contours = segment.find_blobs(segment.build_mask(image))
        assert len(contours) == 1


class TestMedianBlobArea:
    def test_returns_median_of_contour_areas(self):
        square = np.array([[[0, 0]], [[0, 10]], [[10, 10]], [[10, 0]]], dtype=np.int32)
        big = np.array([[[0, 0]], [[0, 20]], [[20, 20]], [[20, 0]]], dtype=np.int32)
        area = segment.median_blob_area([square, square, big])
        assert abs(area - 100.0) < 1.0

    def test_empty_input_returns_zero(self):
        assert segment.median_blob_area([]) == 0.0
```

- [ ] **Step 3: 运行测试确认失败**

Run: `python -m pytest tests/test_segment.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'puzzlefind.segment'`

- [ ] **Step 4: 实现掩膜与轮廓提取**

创建 `puzzlefind/segment.py`：

```python
"""OpenCV 分割：从深色背景上分离出碎片。

前提：碎片摊在深色纯背景上（见 Global Constraints）。这让前景/背景
分离退化成一次阈值操作，避免了在同色背景上调参的地狱。
"""
from __future__ import annotations

import statistics

import cv2
import numpy as np

from . import config


def build_mask(bgr: np.ndarray) -> np.ndarray:
    """把 BGR 图转成二值掩膜，碎片为 255、背景为 0。

    config.MASK_THRESHOLD 为 None 时用 Otsu 自动求阈值（推荐，
    对光照变化更稳）；给定整数时用固定阈值。
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # 轻度模糊压掉传感器噪点，否则 Otsu 会被噪点拉偏
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    if config.MASK_THRESHOLD is None:
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(gray, config.MASK_THRESHOLD, 255, cv2.THRESH_BINARY)

    # 开运算去毛刺，闭运算补碎片内部的小洞（反光造成的暗斑）
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (config.MORPH_KERNEL_SIZE, config.MORPH_KERNEL_SIZE)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def find_blobs(mask: np.ndarray) -> list[np.ndarray]:
    """从掩膜中提取连通块轮廓，丢弃明显小于碎片的噪点。

    注意：粘连的碎片在这里仍是「一个」连通块。切分由 split_blobs 负责。
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    median_area = median_blob_area(list(contours))
    if median_area <= 0.0:
        return []

    min_area = median_area * config.MIN_AREA_RATIO
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def median_blob_area(contours: list[np.ndarray]) -> float:
    """轮廓面积的中位数。用作「一块碎片有多大」的稳健估计。

    中位数而非均值：少数粘连团块的面积是单块的好几倍，会把均值拉飞。
    """
    if not contours:
        return 0.0
    return float(statistics.median(cv2.contourArea(c) for c in contours))
```

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/test_segment.py -v`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
cd /d/ocr_claude
git add puzzlefind/segment.py tests/
git commit -m "feat: mask building and blob contour extraction"
```

---

### Task 3: 分割 — 粘连碎片的分水岭切分

**Files:**
- Modify: `puzzlefind/segment.py`（追加函数）
- Modify: `tests/test_segment.py`（追加测试类）

**Interfaces:**
- Consumes: `segment.find_blobs`, `segment.median_blob_area`, `config.SPLIT_AREA_RATIO`, `config.SPLIT_SEARCH_STEPS`
- Produces:
  - `segment.expected_piece_count(contour: np.ndarray, median_area: float) -> int`
  - `segment.split_blob(mask: np.ndarray, contour: np.ndarray, expected: int) -> list[np.ndarray]`
  - `segment.extract_contours(bgr: np.ndarray) -> list[np.ndarray]` — 端到端：图 → 已切分的碎片轮廓列表

- [ ] **Step 1: 写切分测试**

在 `tests/test_segment.py` 末尾追加：

```python
class TestExpectedPieceCount:
    def test_single_piece_area_yields_one(self):
        contour = np.array(
            [[[0, 0]], [[0, 10]], [[10, 10]], [[10, 0]]], dtype=np.int32
        )
        assert segment.expected_piece_count(contour, median_area=100.0) == 1

    def test_double_area_yields_two(self):
        contour = np.array(
            [[[0, 0]], [[0, 20]], [[10, 20]], [[10, 0]]], dtype=np.int32
        )
        assert segment.expected_piece_count(contour, median_area=100.0) == 2

    def test_never_returns_less_than_one(self):
        tiny = np.array([[[0, 0]], [[0, 2]], [[2, 2]], [[2, 0]]], dtype=np.int32)
        assert segment.expected_piece_count(tiny, median_area=100.0) == 1


class TestExtractContours:
    def test_separated_pieces_pass_through_unchanged(self, separated_pieces):
        image, expected = separated_pieces
        assert len(segment.extract_contours(image)) == expected

    def test_touching_pair_gets_split(self, touching_pair):
        image, expected = touching_pair
        assert len(segment.extract_contours(image)) == expected

    def test_mixed_scene_resolves_to_correct_total(self, touching_triple_with_singles):
        image, expected = touching_triple_with_singles
        assert len(segment.extract_contours(image)) == expected

    def test_split_contours_are_disjoint_enough(self, touching_pair):
        """切分出的两块，其质心应明显分开——不能是同一块被复制两份。"""
        image, _ = touching_pair
        contours = segment.extract_contours(image)
        centroids = []
        for contour in contours:
            moments = cv2.moments(contour)
            centroids.append((moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]))
        (x1, _), (x2, _) = centroids[0], centroids[1]
        assert abs(x1 - x2) > 40

    def test_empty_image_yields_no_contours(self):
        blank = np.zeros((200, 200, 3), dtype=np.uint8)
        assert segment.extract_contours(blank) == []
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_segment.py -v -k "Expected or ExtractContours"`
Expected: FAIL — `AttributeError: module 'puzzlefind.segment' has no attribute 'expected_piece_count'`

- [ ] **Step 3: 实现切分**

在 `puzzlefind/segment.py` 末尾追加：

```python
def expected_piece_count(contour: np.ndarray, median_area: float) -> int:
    """这个连通块里大概有几块碎片？

    用「面积 ÷ 中位面积」估计。这是碎片尺寸大致均匀这一先验的直接应用，
    也是我们不需要人工指定分水岭种子数的原因。
    """
    if median_area <= 0.0:
        return 1
    return max(1, round(cv2.contourArea(contour) / median_area))


def _blob_mask(shape: tuple[int, int], contour: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
    return mask


def split_blob(
    shape: tuple[int, int], contour: np.ndarray, expected: int
) -> list[np.ndarray]:
    """把一个粘连团块切成 expected 块，返回子轮廓列表。

    做法：对团块做距离变换，然后二分搜索一个阈值系数，使得
    「距离 > 系数 × 最大距离」的区域恰好裂成 expected 个连通块。
    这些连通块作为分水岭的种子。

    二分搜索替代了手工调 0.5/0.6/0.7 这类魔法系数——每个团块自己
    找到合适的系数，不同尺寸、不同重叠程度的团块都能自适应。
    """
    if expected <= 1:
        return [contour]

    blob = _blob_mask(shape, contour)
    distance = cv2.distanceTransform(blob, cv2.DIST_L2, 5)
    max_distance = float(distance.max())
    if max_distance <= 0.0:
        return [contour]

    def markers_for(fraction: float) -> tuple[int, np.ndarray]:
        _, peaks = cv2.threshold(distance, fraction * max_distance, 255, cv2.THRESH_BINARY)
        peaks = peaks.astype(np.uint8)
        count, labels = cv2.connectedComponents(peaks)
        return count - 1, labels  # 减去背景标签

    # 二分：系数越大种子越少、越分散
    low, high = 0.05, 0.95
    best_labels: np.ndarray | None = None
    for _ in range(config.SPLIT_SEARCH_STEPS):
        mid = (low + high) / 2.0
        count, labels = markers_for(mid)
        if count == expected:
            best_labels = labels
            break
        if count > expected:
            low = mid   # 种子太多 → 提高阈值
        else:
            high = mid  # 种子太少 → 降低阈值
    if best_labels is None:
        _, best_labels = markers_for((low + high) / 2.0)

    # 分水岭需要三通道输入，且标签 0 表示「未知区域」
    markers = best_labels.astype(np.int32) + 1
    markers[blob == 0] = 1  # 背景标为 1
    unknown = cv2.subtract(blob, (best_labels > 0).astype(np.uint8) * 255)
    markers[unknown == 255] = 0

    canvas = cv2.cvtColor(blob, cv2.COLOR_GRAY2BGR)
    cv2.watershed(canvas, markers)

    pieces: list[np.ndarray] = []
    for label in range(2, int(markers.max()) + 1):
        region = np.zeros(shape, dtype=np.uint8)
        region[markers == label] = 255
        sub_contours, _ = cv2.findContours(
            region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if sub_contours:
            pieces.append(max(sub_contours, key=cv2.contourArea))

    # 分水岭失败时退回原轮廓——高亮范围偏大，但功能仍在（优雅降级）
    return pieces if pieces else [contour]


def extract_contours(bgr: np.ndarray) -> list[np.ndarray]:
    """端到端：原图 → 已切分的单块碎片轮廓列表。"""
    mask = build_mask(bgr)
    blobs = find_blobs(mask)
    if not blobs:
        return []

    median_area = median_blob_area(blobs)
    shape = bgr.shape[:2]

    result: list[np.ndarray] = []
    for blob in blobs:
        expected = expected_piece_count(blob, median_area)
        if cv2.contourArea(blob) < median_area * config.SPLIT_AREA_RATIO:
            result.append(blob)
        else:
            result.extend(split_blob(shape, blob, expected))
    return result
```

- [ ] **Step 4: 运行全部分割测试**

Run: `python -m pytest tests/test_segment.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
cd /d/ocr_claude
git add puzzlefind/segment.py tests/test_segment.py
git commit -m "feat: watershed splitting of touching pieces with auto seed search"
```

---

### Task 4: 碎片裁剪归一化

**Files:**
- Modify: `puzzlefind/segment.py`（追加函数）
- Modify: `tests/test_segment.py`（追加测试类）

**Interfaces:**
- Consumes: `config.CROP_PADDING`, `config.CROP_TARGET_LONG_EDGE`, `config.CROP_FILL_COLOR`
- Produces:
  - `segment.contour_bbox(contour: np.ndarray) -> tuple[int, int, int, int]` — `(x, y, w, h)`
  - `segment.crop_piece(bgr: np.ndarray, contour: np.ndarray) -> np.ndarray` — 掩膜裁剪并放大到目标尺寸

- [ ] **Step 1: 写裁剪测试**

在 `tests/test_segment.py` 末尾追加：

```python
class TestContourBbox:
    def test_returns_tight_bounding_box(self):
        contour = np.array(
            [[[10, 20]], [[10, 60]], [[50, 60]], [[50, 20]]], dtype=np.int32
        )
        assert segment.contour_bbox(contour) == (10, 20, 40, 40)


class TestCropPiece:
    def test_crop_long_edge_matches_target(self, separated_pieces):
        from puzzlefind import config

        image, _ = separated_pieces
        contour = segment.extract_contours(image)[0]
        crop = segment.crop_piece(image, contour)
        assert max(crop.shape[:2]) == config.CROP_TARGET_LONG_EDGE

    def test_crop_is_three_channel_bgr(self, separated_pieces):
        image, _ = separated_pieces
        contour = segment.extract_contours(image)[0]
        crop = segment.crop_piece(image, contour)
        assert crop.ndim == 3 and crop.shape[2] == 3

    def test_area_outside_contour_is_fill_color(self, separated_pieces):
        from puzzlefind import config

        image, _ = separated_pieces
        contour = segment.extract_contours(image)[0]
        crop = segment.crop_piece(image, contour)
        # 圆形碎片的裁剪图，四角必在轮廓之外
        corner = tuple(int(v) for v in crop[2, 2])
        assert corner == config.CROP_FILL_COLOR

    def test_center_preserves_original_piece_color(self, separated_pieces):
        image, _ = separated_pieces
        contour = segment.extract_contours(image)[0]
        crop = segment.crop_piece(image, contour)
        h, w = crop.shape[:2]
        center = crop[h // 2, w // 2]
        # 合成碎片是浅色 (235,233,228)，中心应仍是浅色
        assert int(center.min()) > 180

    def test_contour_touching_image_edge_does_not_crash(self):
        canvas = np.full((200, 200, 3), 18, dtype=np.uint8)
        cv2.circle(canvas, (5, 5), 40, (235, 233, 228), thickness=-1)
        contours = segment.extract_contours(canvas)
        assert len(contours) == 1
        crop = segment.crop_piece(canvas, contours[0])
        assert crop.size > 0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_segment.py -v -k "ContourBbox or CropPiece"`
Expected: FAIL — `AttributeError: module 'puzzlefind.segment' has no attribute 'contour_bbox'`

- [ ] **Step 3: 实现裁剪**

在 `puzzlefind/segment.py` 末尾追加：

```python
def contour_bbox(contour: np.ndarray) -> tuple[int, int, int, int]:
    """轮廓的紧包围盒 (x, y, w, h)。"""
    x, y, w, h = cv2.boundingRect(contour)
    return int(x), int(y), int(w), int(h)


def crop_piece(bgr: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """裁出单块碎片并放大到识别友好的尺寸。

    三个动作，每个都有目的：
    1. 用轮廓做掩膜，把邻块的像素替换成中性灰——否则相邻碎片上的
       编号会混进这块的裁剪图，OCR 会读出两个编号。
    2. 按包围盒裁剪并外扩少量边距，给检测器留出上下文。
    3. 放大到 CROP_TARGET_LONG_EDGE。这是整条管线里对识别率影响
       最大的一步：PP-OCR 会把文字行缩放到固定高度 48px，源图字符
       太小就等于喂给模型一张糊图。
    """
    height, width = bgr.shape[:2]
    x, y, w, h = contour_bbox(contour)

    pad = config.CROP_PADDING
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(width, x + w + pad), min(height, y + h + pad)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=-1)

    filled = np.full_like(bgr, config.CROP_FILL_COLOR, dtype=np.uint8)
    composited = np.where(mask[:, :, None] == 255, bgr, filled)
    crop = composited[y0:y1, x0:x1]

    if crop.size == 0:
        return crop

    long_edge = max(crop.shape[:2])
    scale = config.CROP_TARGET_LONG_EDGE / long_edge
    # 放大用 INTER_CUBIC（保边缘锐度），缩小用 INTER_AREA（抗锯齿）
    interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(
        crop,
        (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))),
        interpolation=interpolation,
    )
```

- [ ] **Step 4: 运行全部分割测试**

Run: `python -m pytest tests/test_segment.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
cd /d/ocr_claude
git add puzzlefind/segment.py tests/test_segment.py
git commit -m "feat: masked piece cropping with upscale for OCR"
```

---

### Task 5: 识别器协议 + PaddleOCR 直接识别（Pass A）

**Files:**
- Create: `puzzlefind/recognize.py`
- Create: `scripts/probe_paddleocr.py`
- Test: `tests/test_recognize.py`

**Interfaces:**
- Consumes: `vocabulary.snap`, `config.MIN_ACCEPT_CONFIDENCE`
- Produces:
  - `recognize.RawDetection` — dataclass，字段 `text: str`, `score: float`
  - `recognize.OcrBackend` — Protocol，方法 `read(image: np.ndarray) -> list[RawDetection]`
  - `recognize.PaddleBackend` — 实现类，惰性加载模型
  - `recognize.RecogResult` — dataclass，字段 `code: str | None`, `confidence: float`, `raw_text: str | None`, `method: str`, `angle: int | None`
  - `recognize.recognize_direct(backend: OcrBackend, crop: np.ndarray) -> RecogResult`

- [ ] **Step 1: 先探明 PaddleOCR 3.x 的真实输出结构**

**不要凭记忆写适配器。** 先跑一个探针把实际结构打出来。

创建 `scripts/probe_paddleocr.py`：

```python
"""一次性探针：打印 PaddleOCR 3.x predict() 的真实返回结构。

写 PaddleBackend 的适配层之前先跑这个。3.x 的返回对象结构与 2.x
完全不同，网上教程大多过时，凭记忆写必错。
"""
from __future__ import annotations

import cv2
import numpy as np
from paddleocr import PaddleOCR


def make_sample() -> np.ndarray:
    image = np.full((160, 480, 3), 245, dtype=np.uint8)
    cv2.putText(image, "B-403", (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (90, 90, 90), 6)
    return image


def main() -> None:
    ocr = PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
    )
    results = ocr.predict(make_sample())
    print(f"type(results) = {type(results)}, len = {len(results)}")
    for i, res in enumerate(results):
        print(f"\n--- result[{i}] type = {type(res)} ---")
        print("dir():", [a for a in dir(res) if not a.startswith('_')])
        payload = getattr(res, "json", None)
        print("res.json =", payload)


if __name__ == "__main__":
    main()
```

先装依赖再跑：

```bash
cd /d/ocr_claude
python -m pip install paddlepaddle==3.3.1 paddleocr
python scripts/probe_paddleocr.py
```

**把打印出的键名记下来**（预期能看到 `rec_texts` 和 `rec_scores`，但以实际输出为准）。下一步的适配器要照着实际结构写。首次运行会下载模型，需要几分钟。

- [ ] **Step 2: 写识别测试（用假后端，不加载 PaddleOCR）**

创建 `tests/test_recognize.py`：

```python
import numpy as np
import pytest

from puzzlefind import recognize
from puzzlefind.recognize import RawDetection


class FakeBackend:
    """按调用顺序吐出预设结果的假 OCR 后端。

    存在的意义：让识别逻辑的测试与 PaddleOCR 完全解耦，毫秒级跑完，
    不需要下载模型。真正的 PaddleBackend 只在带 @pytest.mark.ocr
    的测试里被碰。
    """

    def __init__(self, responses: list[list[RawDetection]]):
        self.responses = responses
        self.calls = 0

    def read(self, image: np.ndarray) -> list[RawDetection]:
        result = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return result


@pytest.fixture
def blank_crop() -> np.ndarray:
    return np.full((100, 100, 3), 200, dtype=np.uint8)


class TestRecognizeDirect:
    def test_clean_read_produces_code(self, blank_crop):
        backend = FakeBackend([[RawDetection("B-403", 0.97)]])
        result = recognize.recognize_direct(backend, blank_crop)
        assert result.code == "B-403"
        assert result.confidence == pytest.approx(0.97)
        assert result.method == "direct"

    def test_confusable_read_is_snapped_to_vocabulary(self, blank_crop):
        backend = FakeBackend([[RawDetection("8-4O3", 0.88)]])
        result = recognize.recognize_direct(backend, blank_crop)
        assert result.code == "B-403"
        assert result.raw_text == "8-4O3"

    def test_unsnappable_text_yields_no_code(self, blank_crop):
        backend = FakeBackend([[RawDetection("QWERTY", 0.99)]])
        result = recognize.recognize_direct(backend, blank_crop)
        assert result.code is None

    def test_empty_detection_yields_no_code(self, blank_crop):
        backend = FakeBackend([[]])
        result = recognize.recognize_direct(backend, blank_crop)
        assert result.code is None
        assert result.confidence == 0.0

    def test_low_confidence_read_is_rejected(self, blank_crop):
        backend = FakeBackend([[RawDetection("B-403", 0.10)]])
        result = recognize.recognize_direct(backend, blank_crop)
        assert result.code is None

    def test_picks_highest_scoring_detection_among_several(self, blank_crop):
        backend = FakeBackend(
            [[RawDetection("A-111", 0.55), RawDetection("B-403", 0.93)]]
        )
        result = recognize.recognize_direct(backend, blank_crop)
        assert result.code == "B-403"


@pytest.mark.ocr
class TestPaddleBackendIntegration:
    def test_reads_rendered_code_from_synthetic_image(self):
        import cv2

        image = np.full((160, 480, 3), 245, dtype=np.uint8)
        cv2.putText(
            image, "B-403", (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (90, 90, 90), 6
        )
        backend = recognize.PaddleBackend()
        result = recognize.recognize_direct(backend, image)
        assert result.code == "B-403"
```

- [ ] **Step 3: 运行测试确认失败**

Run: `python -m pytest tests/test_recognize.py -v -m "not ocr"`
Expected: FAIL — `ModuleNotFoundError: No module named 'puzzlefind.recognize'`

- [ ] **Step 4: 实现识别模块**

创建 `puzzlefind/recognize.py`。**若 Step 1 探针打印出的键名与下面 `_extract` 里假设的不同，以探针结果为准修改 `_extract`。**

```python
"""识别层：把碎片裁剪图变成编号。

设计要点：OcrBackend 是一个 Protocol，PaddleOCR 只是它的一个实现。
这让绝大多数测试可以注入假后端，不必加载模型。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from . import config, vocabulary


@dataclass(frozen=True)
class RawDetection:
    """OCR 后端返回的一条原始检测结果。"""

    text: str
    score: float


@dataclass(frozen=True)
class RecogResult:
    """一块碎片的识别结论。"""

    code: str | None
    confidence: float
    raw_text: str | None
    method: str          # "direct" | "sweep" | "none"
    angle: int | None    # 命中时的旋转角度（仅 sweep 有值）


class OcrBackend(Protocol):
    def read(self, image: np.ndarray) -> list[RawDetection]:
        """对整张图做 OCR，返回全部检测结果。"""
        ...


class PaddleBackend:
    """PaddleOCR 3.x 后端。模型惰性加载——首次 read 时才初始化。"""

    def __init__(self, lang: str = "en") -> None:
        self._lang = lang
        self._ocr = None

    def _ensure_loaded(self) -> None:
        if self._ocr is None:
            from paddleocr import PaddleOCR

            # 关掉文档级方向分类和去扭曲：我们的输入是单块碎片的小图，
            # 不是扫描文档，那两个模块只会拖慢速度并引入误判。
            # 保留 textline 方向分类——它负责 180° 正反歧义。
            self._ocr = PaddleOCR(
                lang=self._lang,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=True,
            )

    def read(self, image: np.ndarray) -> list[RawDetection]:
        self._ensure_loaded()
        assert self._ocr is not None
        try:
            results = self._ocr.predict(image)
        except Exception:
            # 单块碎片识别失败不应该炸掉整轮建索引
            return []
        detections: list[RawDetection] = []
        for res in results:
            detections.extend(_extract(res))
        return detections


def _extract(res: object) -> list[RawDetection]:
    """从 PaddleOCR 3.x 的结果对象里挖出文本和分数。

    结构在版本间会变，所以这里做防御式提取：先试 res.json，
    再试字典下标。若 scripts/probe_paddleocr.py 打印出的键名与
    这里不同，改这个函数即可，上层无需变动。
    """
    payload = getattr(res, "json", None)
    if isinstance(payload, dict):
        # 3.x 通常把内容包在 "res" 键下
        payload = payload.get("res", payload)
    if not isinstance(payload, dict):
        try:
            payload = dict(res)  # type: ignore[call-overload]
        except Exception:
            return []

    texts = payload.get("rec_texts") or []
    scores = payload.get("rec_scores") or []
    return [
        RawDetection(str(t), float(s))
        for t, s in zip(texts, scores)
        if isinstance(t, str)
    ]


_NO_RESULT = RecogResult(code=None, confidence=0.0, raw_text=None, method="none", angle=None)


def _best_snapped(
    detections: list[RawDetection], method: str, angle: int | None
) -> RecogResult:
    """在一组检测结果里，挑出能吸附到合法编号且分数最高的那条。"""
    best: RecogResult | None = None
    for detection in detections:
        if detection.score < config.MIN_ACCEPT_CONFIDENCE:
            continue
        code, _ = vocabulary.snap(detection.text)
        if code is None:
            continue
        if best is None or detection.score > best.confidence:
            best = RecogResult(
                code=code,
                confidence=detection.score,
                raw_text=detection.text,
                method=method,
                angle=angle,
            )
    return best if best is not None else _NO_RESULT


def recognize_direct(backend: OcrBackend, crop: np.ndarray) -> RecogResult:
    """Pass A：整块裁剪图直接喂给 OCR，靠检测器自己处理旋转框。"""
    return _best_snapped(backend.read(crop), method="direct", angle=0)
```

- [ ] **Step 5: 运行非 OCR 测试确认通过**

Run: `python -m pytest tests/test_recognize.py -v -m "not ocr"`
Expected: PASS

- [ ] **Step 6: 运行 OCR 集成测试**

Run: `python -m pytest tests/test_recognize.py -v -m ocr`
Expected: PASS。若失败，多半是 `_extract` 的键名与实际不符——回看 Step 1 探针的输出并修正。

- [ ] **Step 7: 提交**

```bash
cd /d/ocr_claude
git add puzzlefind/recognize.py scripts/ tests/test_recognize.py
git commit -m "feat: OCR backend protocol and direct recognition pass"
```

---

### Task 6: 旋转穷举兜底（Pass C）

**Files:**
- Modify: `puzzlefind/recognize.py`（追加函数）
- Modify: `tests/test_recognize.py`（追加测试类）

**Interfaces:**
- Consumes: `recognize.recognize_direct`, `config.SWEEP_ANGLES`, `config.SWEEP_CONFIDENCE_THRESHOLD`
- Produces:
  - `recognize.rotate_expand(image: np.ndarray, angle: int) -> np.ndarray`
  - `recognize.recognize_sweep(backend: OcrBackend, crop: np.ndarray) -> RecogResult`
  - `recognize.recognize_piece(backend: OcrBackend, crop: np.ndarray) -> RecogResult`

- [ ] **Step 1: 写穷举测试**

在 `tests/test_recognize.py` 末尾追加：

```python
class TestRotateExpand:
    def test_zero_degrees_returns_same_shape(self):
        image = np.zeros((60, 100, 3), dtype=np.uint8)
        assert recognize.rotate_expand(image, 0).shape == image.shape

    def test_ninety_degrees_swaps_dimensions(self):
        image = np.zeros((60, 100, 3), dtype=np.uint8)
        rotated = recognize.rotate_expand(image, 90)
        assert rotated.shape[0] == 100
        assert rotated.shape[1] == 60

    def test_forty_five_degrees_expands_canvas_without_clipping(self):
        image = np.zeros((60, 100, 3), dtype=np.uint8)
        rotated = recognize.rotate_expand(image, 45)
        assert rotated.shape[0] > 60
        assert rotated.shape[1] > 100


class TestRecognizeSweep:
    def test_finds_code_at_a_later_angle(self, blank_crop):
        from puzzlefind import config

        # 前两个角度读不出，第三个角度读出——模拟只有摆正后才认得出
        responses = [[], [RawDetection("???", 0.9)], [RawDetection("B-403", 0.95)]]
        backend = FakeBackend(responses)
        result = recognize.recognize_sweep(backend, blank_crop)
        assert result.code == "B-403"
        assert result.method == "sweep"
        assert result.angle == config.SWEEP_ANGLES[2]

    def test_tries_every_angle_when_nothing_hits(self, blank_crop):
        from puzzlefind import config

        backend = FakeBackend([[]])
        result = recognize.recognize_sweep(backend, blank_crop)
        assert result.code is None
        assert backend.calls == len(config.SWEEP_ANGLES)

    def test_keeps_highest_confidence_across_angles(self, blank_crop):
        responses = [
            [RawDetection("B-403", 0.60)],
            [RawDetection("B-403", 0.98)],
            [RawDetection("B-403", 0.71)],
        ]
        backend = FakeBackend(responses)
        result = recognize.recognize_sweep(backend, blank_crop)
        assert result.confidence == pytest.approx(0.98)


class TestRecognizePiece:
    def test_high_confidence_direct_hit_skips_the_sweep(self, blank_crop):
        backend = FakeBackend([[RawDetection("B-403", 0.99)]])
        result = recognize.recognize_piece(backend, blank_crop)
        assert result.method == "direct"
        assert backend.calls == 1

    def test_low_confidence_direct_hit_escalates_to_sweep(self, blank_crop):
        # 首次调用置信度低于阈值 → 进入穷举，穷举里读出高分结果
        responses = [
            [RawDetection("B-403", 0.50)],   # direct
            [RawDetection("B-403", 0.50)],   # sweep angle 0
            [RawDetection("B-403", 0.97)],   # sweep angle 1
        ]
        backend = FakeBackend(responses)
        result = recognize.recognize_piece(backend, blank_crop)
        assert result.method == "sweep"
        assert backend.calls > 1

    def test_returns_no_result_when_every_pass_fails(self, blank_crop):
        backend = FakeBackend([[]])
        result = recognize.recognize_piece(backend, blank_crop)
        assert result.code is None
        assert result.method == "none"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_recognize.py -v -m "not ocr" -k "RotateExpand or Sweep or RecognizePiece"`
Expected: FAIL — `AttributeError: module 'puzzlefind.recognize' has no attribute 'rotate_expand'`

- [ ] **Step 3: 实现穷举**

在 `puzzlefind/recognize.py` 末尾追加：

```python
def rotate_expand(image: np.ndarray, angle: int) -> np.ndarray:
    """绕中心旋转，并扩大画布以免四角被裁掉。

    普通的 warpAffine 会把旋转后超出原尺寸的部分切掉——对我们是致命的，
    因为编号常印在碎片边缘，一裁就没了。
    """
    import cv2

    if angle % 360 == 0:
        return image

    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_width = int(height * sin + width * cos)
    new_height = int(height * cos + width * sin)

    matrix[0, 2] += new_width / 2.0 - center[0]
    matrix[1, 2] += new_height / 2.0 - center[1]

    return cv2.warpAffine(
        image,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=config.CROP_FILL_COLOR,
    )


def recognize_sweep(backend: OcrBackend, crop: np.ndarray) -> RecogResult:
    """Pass C：把裁剪图旋转一圈，每个角度都识别一次，取最优。

    「必须能吸附到合法词表」这个约束让穷举的判据非常硬——错误的角度
    几乎不可能凑出一个合法编号，所以误报率很低。代价只是 CPU 时间，
    而建索引是一次性的。
    """
    best = _NO_RESULT
    for angle in config.SWEEP_ANGLES:
        rotated = rotate_expand(crop, angle)
        candidate = _best_snapped(backend.read(rotated), method="sweep", angle=angle)
        if candidate.code is not None and candidate.confidence > best.confidence:
            best = candidate
    return best


def recognize_piece(backend: OcrBackend, crop: np.ndarray) -> RecogResult:
    """完整识别流程：先 Pass A，置信度不够就升级到 Pass C。

    SWEEP_CONFIDENCE_THRESHOLD 默认设得很激进（宁可多穷举）。跑过
    真实照片后按 Task 14 的实测数据下调，能大幅缩短建索引时间。
    """
    direct = recognize_direct(backend, crop)
    if direct.code is not None and direct.confidence >= config.SWEEP_CONFIDENCE_THRESHOLD:
        return direct

    sweep = recognize_sweep(backend, crop)
    if sweep.code is not None and sweep.confidence > direct.confidence:
        return sweep
    return direct
```

- [ ] **Step 4: 运行全部识别测试**

Run: `python -m pytest tests/test_recognize.py -v -m "not ocr"`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
cd /d/ocr_claude
git add puzzlefind/recognize.py tests/test_recognize.py
git commit -m "feat: rotation sweep fallback for arbitrary-angle text"
```

---

### Task 7: 全局唯一性冲突消解

**Files:**
- Create: `puzzlefind/resolve.py`
- Test: `tests/test_resolve.py`

**Interfaces:**
- Consumes: `recognize.RecogResult`, `vocabulary.bootstrap_ranges`, `vocabulary.is_outlier`
- Produces:
  - `resolve.resolve(results: list[RecogResult]) -> list[RecogResult]` — 输入输出等长，冲突方被降级为 `code=None`

- [ ] **Step 1: 写消解测试**

创建 `tests/test_resolve.py`：

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_resolve.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'puzzlefind.resolve'`

- [ ] **Step 3: 实现消解**

创建 `puzzlefind/resolve.py`：

```python
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

    # 规则 1：离群值剔除
    ranges = vocabulary.bootstrap_ranges([r.code for r in resolved if r.code])
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_resolve.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
cd /d/ocr_claude
git add puzzlefind/resolve.py tests/test_resolve.py
git commit -m "feat: uniqueness and outlier conflict resolution"
```

---

### Task 8: 数据模型 + 建索引管线

**Files:**
- Create: `puzzlefind/models.py`
- Create: `puzzlefind/pipeline.py`
- Test: `tests/test_models.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `segment.extract_contours`, `segment.crop_piece`, `segment.contour_bbox`, `recognize.recognize_piece`, `resolve.resolve`
- Produces:
  - `models.Piece` — dataclass：`piece_id: int`, `contour: list[list[int]]`, `bbox: tuple[int,int,int,int]`, `area: float`, `code: str | None`, `confidence: float`, `raw_text: str | None`, `method: str`, `angle: int | None`
  - `models.PhotoIndex` — dataclass：`photo_id: str`, `image_path: str`, `width: int`, `height: int`, `created_at: str`, `pieces: list[Piece]`
  - `models.PhotoIndex.to_dict() / from_dict(data) -> PhotoIndex`
  - `models.PhotoIndex.recognized / unrecognized -> list[Piece]`
  - `models.PhotoIndex.find(code) -> Piece | None`
  - `pipeline.build_index(image_path: Path, backend, photo_id: str | None = None) -> PhotoIndex`

- [ ] **Step 1: 写数据模型测试**

创建 `tests/test_models.py`：

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'puzzlefind.models'`

- [ ] **Step 3: 实现数据模型**

创建 `puzzlefind/models.py`：

```python
"""索引的数据模型。JSON 是唯一的持久化格式——量级只有几百条记录，
上数据库纯属多余，而 JSON 你可以直接打开看，排查问题快得多。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Piece:
    """一块碎片：几何位置 + 识别结论。"""

    piece_id: int
    contour: list[list[int]]              # [[x, y], ...] 原图坐标系
    bbox: tuple[int, int, int, int]       # (x, y, w, h)
    area: float
    code: str | None
    confidence: float
    raw_text: str | None
    method: str
    angle: int | None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["bbox"] = list(self.bbox)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Piece:
        return cls(
            piece_id=int(data["piece_id"]),
            contour=[[int(x), int(y)] for x, y in data["contour"]],
            bbox=tuple(int(v) for v in data["bbox"]),  # type: ignore[arg-type]
            area=float(data["area"]),
            code=data.get("code"),
            confidence=float(data.get("confidence", 0.0)),
            raw_text=data.get("raw_text"),
            method=str(data.get("method", "none")),
            angle=data.get("angle"),
        )


@dataclass
class PhotoIndex:
    """一张照片的完整索引。"""

    photo_id: str
    image_path: str
    width: int
    height: int
    created_at: str                       # ISO 8601，用于提示索引可能已过期
    pieces: list[Piece] = field(default_factory=list)

    @property
    def recognized(self) -> list[Piece]:
        return [p for p in self.pieces if p.code]

    @property
    def unrecognized(self) -> list[Piece]:
        """未识别的碎片。查询未命中时，答案大概率就在这里面。"""
        return [p for p in self.pieces if not p.code]

    def find(self, code: str) -> Piece | None:
        target = code.strip().upper()
        for piece in self.pieces:
            if piece.code == target:
                return piece
        return None

    def to_dict(self) -> dict:
        return {
            "photo_id": self.photo_id,
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
            "created_at": self.created_at,
            "pieces": [p.to_dict() for p in self.pieces],
        }

    @classmethod
    def from_dict(cls, data: dict) -> PhotoIndex:
        return cls(
            photo_id=str(data["photo_id"]),
            image_path=str(data["image_path"]),
            width=int(data["width"]),
            height=int(data["height"]),
            created_at=str(data["created_at"]),
            pieces=[Piece.from_dict(p) for p in data.get("pieces", [])],
        )
```

- [ ] **Step 4: 运行模型测试确认通过**

Run: `python -m pytest tests/test_models.py -v`
Expected: PASS

- [ ] **Step 5: 写管线测试**

创建 `tests/test_pipeline.py`：

```python
import cv2
import numpy as np
import pytest

from puzzlefind import pipeline
from puzzlefind.recognize import RawDetection


class PieceBackend:
    """按**碎片顺序**（不是调用顺序）吐出预设编号的假后端。

    为什么不能简单地用「第几次调用」当索引：一块识别失败的碎片会触发
    Pass C 的 12 次穷举调用，把后续所有碎片的索引整体错位。那样的替身
    能不能通过测试，取决于 None 恰好排在列表哪个位置——是运气，不是设计。

    这里改成显式记账：每块碎片的调用配额是确定的（命中 1 次；未命中
    则 1 次直接 + len(SWEEP_ANGLES) 次穷举），配额用完才前进到下一块。
    这样测试对穷举次数免疫。
    """

    def __init__(self, codes: list[str | None]):
        self.codes = codes
        self.calls = 0
        self._piece = 0
        self._remaining = self._quota(0)

    def _code_at(self, index: int) -> str | None:
        return self.codes[index] if index < len(self.codes) else None

    def _quota(self, index: int) -> int:
        from puzzlefind import config

        return 1 if self._code_at(index) else 1 + len(config.SWEEP_ANGLES)

    def read(self, image: np.ndarray) -> list[RawDetection]:
        self.calls += 1
        code = self._code_at(self._piece)
        self._remaining -= 1
        if self._remaining <= 0:
            self._piece += 1
            self._remaining = self._quota(self._piece)
        return [RawDetection(code, 0.99)] if code else []


@pytest.fixture
def photo_path(tmp_path, separated_pieces):
    image, _ = separated_pieces
    path = tmp_path / "photo.jpg"
    cv2.imwrite(str(path), image)
    return path


class TestBuildIndex:
    def test_creates_one_piece_per_contour(self, photo_path, separated_pieces):
        _, expected = separated_pieces
        backend = PieceBackend([f"B-{i:03d}" for i in range(expected)])
        index = pipeline.build_index(photo_path, backend)
        assert len(index.pieces) == expected

    def test_records_image_dimensions(self, photo_path, separated_pieces):
        image, _ = separated_pieces
        backend = PieceBackend([])
        index = pipeline.build_index(photo_path, backend)
        assert index.width == image.shape[1]
        assert index.height == image.shape[0]

    def test_assigns_sequential_piece_ids(self, photo_path):
        backend = PieceBackend([])
        index = pipeline.build_index(photo_path, backend)
        assert [p.piece_id for p in index.pieces] == list(range(len(index.pieces)))

    def test_recognized_codes_land_on_pieces(self, photo_path, separated_pieces):
        _, count = separated_pieces
        backend = PieceBackend([f"B-{i:03d}" for i in range(count)])
        index = pipeline.build_index(photo_path, backend)
        assert sorted(p.code for p in index.recognized) == sorted(
            f"B-{i:03d}" for i in range(count)
        )

    def test_unreadable_pieces_become_unrecognized(self, photo_path, separated_pieces):
        _, count = separated_pieces
        codes: list[str | None] = [f"B-{i:03d}" for i in range(count - 2)] + [None, None]
        backend = PieceBackend(codes)
        index = pipeline.build_index(photo_path, backend)
        assert len(index.unrecognized) == 2

    def test_duplicate_reads_are_resolved_to_one(self, photo_path, separated_pieces):
        _, count = separated_pieces
        backend = PieceBackend(["B-403"] * count)
        index = pipeline.build_index(photo_path, backend)
        assert len(index.recognized) == 1

    def test_contour_points_are_within_image_bounds(self, photo_path, separated_pieces):
        image, _ = separated_pieces
        backend = PieceBackend([])
        index = pipeline.build_index(photo_path, backend)
        h, w = image.shape[:2]
        for piece in index.pieces:
            for x, y in piece.contour:
                assert 0 <= x <= w and 0 <= y <= h

    def test_explicit_photo_id_is_honored(self, photo_path):
        backend = PieceBackend([])
        index = pipeline.build_index(photo_path, backend, photo_id="my-id")
        assert index.photo_id == "my-id"

    def test_missing_file_raises(self, tmp_path):
        backend = PieceBackend([])
        with pytest.raises(FileNotFoundError):
            pipeline.build_index(tmp_path / "nope.jpg", backend)
```

注意 `PieceBackend` 与生产代码之间存在一条**刻意的耦合**：它需要知道每块碎片的调用配额，因此依赖 `SWEEP_ANGLES` 的长度。这是替身对被测行为的合法建模，但如果将来改了 `recognize_piece` 的重试策略，这个替身也要同步改——已在类注释里写明。

- [ ] **Step 6: 运行管线测试确认失败**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'puzzlefind.pipeline'`

- [ ] **Step 7: 实现管线**

创建 `puzzlefind/pipeline.py`：

```python
"""建索引管线：一张照片 → 一个 PhotoIndex。

这是引擎的顶层入口，把分割、识别、消解串起来。它不碰磁盘持久化
（那是 library 的事），也不碰 Web（那是 server 的事）。
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2

from . import resolve, segment
from .models import Piece, PhotoIndex
from .recognize import OcrBackend, recognize_piece


def build_index(
    image_path: Path,
    backend: OcrBackend,
    photo_id: str | None = None,
) -> PhotoIndex:
    """对一张照片建立完整索引。

    流程：读图 → 分割出碎片轮廓 → 逐块裁剪识别 → 全局冲突消解 →
    组装成 PhotoIndex。
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"照片不存在: {path}")

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"无法解码为图像: {path}")

    height, width = image.shape[:2]
    contours = segment.extract_contours(image)

    results = [
        recognize_piece(backend, segment.crop_piece(image, contour))
        for contour in contours
    ]
    results = resolve.resolve(results)

    pieces = [
        Piece(
            piece_id=index,
            contour=[[int(pt[0][0]), int(pt[0][1])] for pt in contour],
            bbox=segment.contour_bbox(contour),
            area=float(cv2.contourArea(contour)),
            code=result.code,
            confidence=result.confidence,
            raw_text=result.raw_text,
            method=result.method,
            angle=result.angle,
        )
        for index, (contour, result) in enumerate(zip(contours, results))
    ]

    return PhotoIndex(
        photo_id=photo_id or path.stem,
        image_path=str(path),
        width=width,
        height=height,
        created_at=datetime.now().isoformat(timespec="seconds"),
        pieces=pieces,
    )
```

- [ ] **Step 8: 运行全部测试**

Run: `python -m pytest tests/ -v -m "not ocr"`
Expected: PASS

- [ ] **Step 9: 提交**

```bash
cd /d/ocr_claude
git add puzzlefind/models.py puzzlefind/pipeline.py tests/test_models.py tests/test_pipeline.py
git commit -m "feat: index data model and build pipeline"
```

---

### Task 9: 多照片库与跨照片查询

**Files:**
- Create: `puzzlefind/library.py`
- Test: `tests/test_library.py`

**Interfaces:**
- Consumes: `models.PhotoIndex`, `config.INDEX_DIR`
- Produces:
  - `library.QueryResult` — dataclass：`found: bool`, `code: str`, `photo_id: str | None`, `piece: Piece | None`, `unrecognized: dict[str, list[Piece]]`
  - `library.Library.load(index_dir: Path | None = None) -> Library`
  - `library.Library.save_photo(index: PhotoIndex) -> None`
  - `library.Library.delete_photo(photo_id: str) -> bool`
  - `library.Library.query(code: str) -> QueryResult`
  - `library.Library.photos -> list[PhotoIndex]`

- [ ] **Step 1: 写库测试**

创建 `tests/test_library.py`：

```python
import pytest

from puzzlefind.library import Library
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
    lib.save_photo(make_index("p1", ["A-001", "A-002", None]))
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
        library.save_photo(make_index("p1", ["A-999"]))
        reloaded = Library.load(tmp_path)
        assert len([p for p in reloaded.photos if p.photo_id == "p1"]) == 1
        assert reloaded.query("A-999").found is True

    def test_load_from_empty_dir_yields_empty_library(self, tmp_path):
        assert Library.load(tmp_path / "fresh").photos == []

    def test_delete_removes_photo(self, library, tmp_path):
        assert library.delete_photo("p1") is True
        assert Library.load(tmp_path).photos[0].photo_id == "p2"

    def test_delete_unknown_id_returns_false(self, library):
        assert library.delete_photo("nope") is False


class TestQuery:
    def test_finds_code_in_first_photo(self, library):
        result = library.query("A-002")
        assert result.found is True
        assert result.photo_id == "p1"
        assert result.piece.code == "A-002"

    def test_finds_code_in_second_photo(self, library):
        result = library.query("B-403")
        assert result.found is True
        assert result.photo_id == "p2"

    def test_query_is_case_and_space_insensitive(self, library):
        assert library.query("  b-403 ").found is True

    def test_miss_reports_not_found(self, library):
        assert library.query("D-777").found is False

    def test_miss_returns_unrecognized_pieces_grouped_by_photo(self, library):
        result = library.query("D-777")
        assert result.unrecognized["p1"][0].piece_id == 2
        assert len(result.unrecognized["p2"]) == 2

    def test_miss_omits_photos_with_no_unrecognized_pieces(self, tmp_path):
        lib = Library(index_dir=tmp_path)
        lib.save_photo(make_index("full", ["A-001"]))
        assert lib.query("Z-999").unrecognized == {}

    def test_hit_carries_no_unrecognized_payload(self, library):
        assert library.query("A-002").unrecognized == {}

    def test_malformed_query_is_a_miss_not_a_crash(self, library):
        assert library.query("!!!").found is False
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_library.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'puzzlefind.library'`

- [ ] **Step 3: 实现库**

创建 `puzzlefind/library.py`：

```python
"""多照片索引库。每张照片一个 JSON 文件，落在 config.INDEX_DIR。

查询跨所有照片进行——1000 块碎片分散在十几张照片里，用户不该需要
记得某个编号在哪张图上。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import config
from .models import Piece, PhotoIndex


@dataclass
class QueryResult:
    """一次查询的结果。

    未命中时 unrecognized 会被填上——这是本工具的关键设计：
    「没找到」这句话毫无信息量，但「没找到，而这 5 块是未识别的」
    能把搜索范围从几百块塌缩到个位数。
    """

    found: bool
    code: str
    photo_id: str | None = None
    piece: Piece | None = None
    unrecognized: dict[str, list[Piece]] = field(default_factory=dict)


class Library:
    def __init__(self, index_dir: Path | None = None) -> None:
        self.index_dir = Path(index_dir) if index_dir else config.INDEX_DIR
        self._photos: dict[str, PhotoIndex] = {}

    @property
    def photos(self) -> list[PhotoIndex]:
        return list(self._photos.values())

    @classmethod
    def load(cls, index_dir: Path | None = None) -> Library:
        library = cls(index_dir)
        if not library.index_dir.exists():
            return library
        for path in sorted(library.index_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                index = PhotoIndex.from_dict(data)
            except (json.JSONDecodeError, KeyError, ValueError):
                # 单个损坏的索引文件不该让整个库加载失败
                continue
            library._photos[index.photo_id] = index
        return library

    def save_photo(self, index: PhotoIndex) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        path = self.index_dir / f"{index.photo_id}.json"
        path.write_text(
            json.dumps(index.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._photos[index.photo_id] = index

    def delete_photo(self, photo_id: str) -> bool:
        if photo_id not in self._photos:
            return False
        del self._photos[photo_id]
        path = self.index_dir / f"{photo_id}.json"
        path.unlink(missing_ok=True)
        return True

    def query(self, code: str) -> QueryResult:
        target = code.strip().upper()
        for photo in self._photos.values():
            piece = photo.find(target)
            if piece is not None:
                return QueryResult(
                    found=True, code=target, photo_id=photo.photo_id, piece=piece
                )

        unrecognized = {
            photo.photo_id: photo.unrecognized
            for photo in self._photos.values()
            if photo.unrecognized
        }
        return QueryResult(found=False, code=target, unrecognized=unrecognized)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_library.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
cd /d/ocr_claude
git add puzzlefind/library.py tests/test_library.py
git commit -m "feat: multi-photo library with cross-photo query"
```

---

### Task 10: 高亮渲染

**Files:**
- Create: `puzzlefind/render.py`
- Test: `tests/test_render.py`

**Interfaces:**
- Consumes: `models.Piece`, `config.DIM_FACTOR`, `config.OUTLINE_COLOR`, `config.UNKNOWN_OUTLINE_COLOR`, `config.OUTLINE_THICKNESS_RATIO`
- Produces:
  - `render.highlight(image: np.ndarray, targets: list[Piece], *, unknown: bool = False) -> np.ndarray`
  - `render.thumbnail(image: np.ndarray, piece: Piece, size: int = 200) -> np.ndarray`

- [ ] **Step 1: 写渲染测试**

创建 `tests/test_render.py`：

```python
import numpy as np
import pytest

from puzzlefind import config, render
from puzzlefind.models import Piece


@pytest.fixture
def bright_image() -> np.ndarray:
    """全白图。压暗效果在白底上最容易断言。"""
    return np.full((300, 400, 3), 255, dtype=np.uint8)


@pytest.fixture
def center_piece() -> Piece:
    return Piece(
        piece_id=0,
        contour=[[150, 100], [150, 200], [250, 200], [250, 100]],
        bbox=(150, 100, 100, 100),
        area=10000.0,
        code="B-403",
        confidence=0.95,
        raw_text="B-403",
        method="direct",
        angle=0,
    )


class TestHighlight:
    def test_output_shape_matches_input(self, bright_image, center_piece):
        out = render.highlight(bright_image, [center_piece])
        assert out.shape == bright_image.shape

    def test_does_not_mutate_the_input_image(self, bright_image, center_piece):
        before = bright_image.copy()
        render.highlight(bright_image, [center_piece])
        assert np.array_equal(bright_image, before)

    def test_background_is_dimmed(self, bright_image, center_piece):
        out = render.highlight(bright_image, [center_piece])
        # (10, 10) 远在轮廓之外
        assert int(out[10, 10].max()) < 255

    def test_background_is_not_fully_black(self, bright_image, center_piece):
        """压暗要克制——全黑会毁掉用户的空间定位参照。"""
        out = render.highlight(bright_image, [center_piece])
        assert int(out[10, 10].max()) > 60

    def test_target_interior_keeps_original_brightness(self, bright_image, center_piece):
        out = render.highlight(bright_image, [center_piece])
        assert int(out[150, 200].min()) == 255

    def test_outline_is_drawn_in_configured_color(self, bright_image, center_piece):
        out = render.highlight(bright_image, [center_piece])
        pixels = out.reshape(-1, 3)
        assert any(tuple(int(v) for v in p) == config.OUTLINE_COLOR for p in pixels)

    def test_unknown_mode_uses_the_other_color(self, bright_image, center_piece):
        out = render.highlight(bright_image, [center_piece], unknown=True)
        pixels = out.reshape(-1, 3)
        assert any(
            tuple(int(v) for v in p) == config.UNKNOWN_OUTLINE_COLOR for p in pixels
        )

    def test_multiple_targets_are_all_highlighted(self, bright_image, center_piece):
        second = Piece(
            piece_id=1,
            contour=[[10, 220], [10, 280], [70, 280], [70, 220]],
            bbox=(10, 220, 60, 60),
            area=3600.0,
            code=None,
            confidence=0.0,
            raw_text=None,
            method="none",
            angle=None,
        )
        out = render.highlight(bright_image, [center_piece, second], unknown=True)
        assert int(out[250, 40].min()) == 255   # 第二块内部保持原亮度
        assert int(out[150, 200].min()) == 255  # 第一块内部也保持

    def test_empty_target_list_dims_everything(self, bright_image):
        out = render.highlight(bright_image, [])
        assert int(out[150, 200].max()) < 255


class TestThumbnail:
    def test_long_edge_matches_requested_size(self, bright_image, center_piece):
        thumb = render.thumbnail(bright_image, center_piece, size=120)
        assert max(thumb.shape[:2]) == 120

    def test_thumbnail_is_three_channel(self, bright_image, center_piece):
        thumb = render.thumbnail(bright_image, center_piece)
        assert thumb.ndim == 3 and thumb.shape[2] == 3

    def test_piece_at_image_edge_does_not_crash(self, bright_image):
        edge = Piece(
            piece_id=0,
            contour=[[0, 0], [0, 30], [30, 30], [30, 0]],
            bbox=(0, 0, 30, 30),
            area=900.0,
            code="A-001",
            confidence=0.9,
            raw_text="A-001",
            method="direct",
            angle=0,
        )
        assert render.thumbnail(bright_image, edge).size > 0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_render.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'puzzlefind.render'`

- [ ] **Step 3: 实现渲染**

创建 `puzzlefind/render.py`：

```python
"""高亮渲染：把「这块碎片在哪」变成一眼能看到的图。

设计取舍：非目标区压暗但不压黑。全黑会让目标跳得更出来，却会毁掉
用户把屏幕位置映射到桌面真实位置所依赖的参照物（周围碎片的排布、
背景纹理、桌沿）。DIM_FACTOR 默认 0.45 是这两者的平衡点。
"""
from __future__ import annotations

import cv2
import numpy as np

from . import config
from .models import Piece


def _contour_array(piece: Piece) -> np.ndarray:
    return np.array(piece.contour, dtype=np.int32).reshape(-1, 1, 2)


def _outline_thickness(image: np.ndarray) -> int:
    long_edge = max(image.shape[:2])
    return max(2, round(long_edge * config.OUTLINE_THICKNESS_RATIO))


def highlight(
    image: np.ndarray, targets: list[Piece], *, unknown: bool = False
) -> np.ndarray:
    """压暗全图，把目标碎片还原成原亮度并描边。

    unknown=True 时用另一种描边颜色，用于「查询未命中，这些是未识别
    碎片」的场景——视觉上和「找到了」明确区分开。
    """
    dimmed = cv2.convertScaleAbs(image, alpha=config.DIM_FACTOR, beta=0)

    if targets:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for piece in targets:
            cv2.drawContours(mask, [_contour_array(piece)], -1, 255, thickness=-1)
        output = np.where(mask[:, :, None] == 255, image, dimmed)
    else:
        output = dimmed.copy()

    color = config.UNKNOWN_OUTLINE_COLOR if unknown else config.OUTLINE_COLOR
    thickness = _outline_thickness(image)
    for piece in targets:
        cv2.polylines(
            output,
            [_contour_array(piece)],
            isClosed=True,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
    return output


def thumbnail(image: np.ndarray, piece: Piece, size: int = 200) -> np.ndarray:
    """裁出这块碎片的小图，供用户核对系统读的编号对不对。

    这是防「自信地认错」的那道防线——词表吸附和唯一性去重都拦不住
    B-403 被读成 B-408（两个都合法），但用户瞄一眼缩略图就能发现。
    """
    height, width = image.shape[:2]
    x, y, w, h = piece.bbox
    pad = config.CROP_PADDING
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(width, x + w + pad), min(height, y + h + pad)

    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return crop

    scale = size / max(crop.shape[:2])
    return cv2.resize(
        crop,
        (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))),
        interpolation=cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA,
    )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_render.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
cd /d/ocr_claude
git add puzzlefind/render.py tests/test_render.py
git commit -m "feat: highlight rendering with restrained dimming"
```

---

### Task 11: 命令行入口

**Files:**
- Create: `puzzlefind/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `pipeline.build_index`, `library.Library`, `render.highlight`, `recognize.PaddleBackend`
- Produces:
  - `cli.main(argv: list[str] | None = None) -> int`
  - 三个子命令：`index`、`query`、`stats`

- [ ] **Step 1: 写 CLI 测试**

创建 `tests/test_cli.py`：

```python
import cv2
import pytest

from puzzlefind import cli
from puzzlefind.library import Library
from puzzlefind.models import Piece, PhotoIndex


@pytest.fixture
def photo_file(tmp_path, separated_pieces):
    image, _ = separated_pieces
    path = tmp_path / "shot.jpg"
    cv2.imwrite(str(path), image)
    return path


@pytest.fixture
def seeded_index_dir(tmp_path):
    index_dir = tmp_path / "index"
    library = Library(index_dir=index_dir)
    library.save_photo(
        PhotoIndex(
            photo_id="p1",
            image_path="x.jpg",
            width=100,
            height=100,
            created_at="2026-08-03T10:00:00",
            pieces=[
                Piece(0, [[0, 0], [0, 9], [9, 9], [9, 0]], (0, 0, 9, 9), 81.0,
                      "B-403", 0.9, "B-403", "direct", 0),
                Piece(1, [[20, 20], [20, 29], [29, 29], [29, 20]], (20, 20, 9, 9), 81.0,
                      None, 0.0, None, "none", None),
            ],
        )
    )
    return index_dir


class TestQueryCommand:
    def test_hit_reports_photo_and_exits_zero(self, seeded_index_dir, capsys):
        code = cli.main(["query", "B-403", "--index-dir", str(seeded_index_dir)])
        assert code == 0
        assert "p1" in capsys.readouterr().out

    def test_miss_exits_nonzero(self, seeded_index_dir):
        assert cli.main(["query", "D-777", "--index-dir", str(seeded_index_dir)]) == 1

    def test_miss_lists_unrecognized_pieces(self, seeded_index_dir, capsys):
        cli.main(["query", "D-777", "--index-dir", str(seeded_index_dir)])
        assert "未识别" in capsys.readouterr().out


class TestStatsCommand:
    def test_reports_recognized_and_unrecognized_counts(self, seeded_index_dir, capsys):
        assert cli.main(["stats", "--index-dir", str(seeded_index_dir)]) == 0
        out = capsys.readouterr().out
        assert "1" in out and "p1" in out


class TestIndexCommand:
    def test_writes_an_index_file(self, photo_file, tmp_path, monkeypatch):
        """用假后端跑，不加载 PaddleOCR。"""
        from puzzlefind.recognize import RawDetection

        class NullBackend:
            def read(self, image):
                return [RawDetection("B-001", 0.99)]

        monkeypatch.setattr(cli, "_make_backend", lambda: NullBackend())
        index_dir = tmp_path / "idx"
        code = cli.main(["index", str(photo_file), "--index-dir", str(index_dir)])
        assert code == 0
        assert (index_dir / "shot.json").exists()

    def test_missing_photo_exits_nonzero(self, tmp_path, monkeypatch):
        class NullBackend:
            def read(self, image):
                return []

        monkeypatch.setattr(cli, "_make_backend", lambda: NullBackend())
        assert cli.main(["index", str(tmp_path / "nope.jpg")]) == 2
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'puzzlefind.cli'`

- [ ] **Step 3: 实现 CLI**

创建 `puzzlefind/cli.py`：

```python
"""命令行入口。存在的意义是让引擎能脱离 Web 单独跑——调分割和识别
参数时，改一个数字然后 `puzzlefind index photo.jpg` 看结果，比每次
都起服务器传图快得多。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from . import config, render
from .library import Library
from .pipeline import build_index


def _make_backend():
    """惰性构造 PaddleOCR 后端。测试里被 monkeypatch 掉。"""
    from .recognize import PaddleBackend

    return PaddleBackend()


def _cmd_index(args: argparse.Namespace) -> int:
    photo = Path(args.photo)
    try:
        index = build_index(photo, _make_backend(), photo_id=args.photo_id)
    except (FileNotFoundError, ValueError) as error:
        print(f"错误: {error}", file=sys.stderr)
        return 2

    library = Library.load(Path(args.index_dir) if args.index_dir else None)
    library.save_photo(index)

    total = len(index.pieces)
    hit = len(index.recognized)
    sweeps = sum(1 for p in index.pieces if p.method == "sweep")
    print(f"照片 {index.photo_id}: 分割出 {total} 块，识别 {hit} 块，未识别 {total - hit} 块")
    print(f"  其中靠旋转穷举救回: {sweeps} 块")
    if total:
        print(f"  识别率: {hit / total:.1%}")
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    library = Library.load(Path(args.index_dir) if args.index_dir else None)
    result = library.query(args.code)

    if result.found:
        assert result.piece is not None
        print(f"{result.code} → 照片 {result.photo_id}，碎片 #{result.piece.piece_id}，"
              f"包围盒 {result.piece.bbox}，置信度 {result.piece.confidence:.2f}")
        if args.out:
            _write_highlight(library, result.photo_id, [result.piece], args.out, False)
        return 0

    print(f"{result.code} 未找到。")
    for photo_id, pieces in result.unrecognized.items():
        print(f"  照片 {photo_id} 有 {len(pieces)} 块未识别碎片: "
              f"{[p.piece_id for p in pieces]}")
    if not result.unrecognized:
        print("  所有碎片均已识别——这个编号确实不在任何一张照片里。")
    return 1


def _cmd_stats(args: argparse.Namespace) -> int:
    library = Library.load(Path(args.index_dir) if args.index_dir else None)
    if not library.photos:
        print("索引库是空的。")
        return 0
    for photo in library.photos:
        total = len(photo.pieces)
        hit = len(photo.recognized)
        print(f"{photo.photo_id}: {hit}/{total} 已识别  (建于 {photo.created_at})")
    return 0


def _write_highlight(library, photo_id, pieces, out_path, unknown) -> None:
    photo = next(p for p in library.photos if p.photo_id == photo_id)
    image = cv2.imread(photo.image_path)
    if image is None:
        print(f"警告: 无法读取原图 {photo.image_path}，跳过高亮输出", file=sys.stderr)
        return
    cv2.imwrite(str(out_path), render.highlight(image, pieces, unknown=unknown))
    print(f"  高亮图已写入 {out_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="puzzlefind", description="拼图碎片编号查找器")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_index = subparsers.add_parser("index", help="对一张照片建立索引")
    p_index.add_argument("photo")
    p_index.add_argument("--photo-id", default=None)
    p_index.set_defaults(func=_cmd_index)

    p_query = subparsers.add_parser("query", help="查询一个编号")
    p_query.add_argument("code")
    p_query.add_argument("--out", default=None, help="把高亮图写到这个路径")
    p_query.set_defaults(func=_cmd_query)

    p_stats = subparsers.add_parser("stats", help="打印索引库概况")
    p_stats.set_defaults(func=_cmd_stats)

    # --index-dir 只挂在子命令上，不挂全局。
    #
    # 陷阱：argparse 里同名参数同时出现在主解析器和子解析器上时，
    # 子解析器后解析，会用它的默认值 None **覆盖掉**主解析器已经解析到的
    # 值。于是 `puzzlefind --index-dir X query CODE` 会静默失效——不报错，
    # 只是悄悄用了默认索引目录。只在一处定义就没有这个问题。
    for sub in (p_index, p_query, p_stats):
        sub.add_argument(
            "--index-dir", default=None, help=f"索引目录，默认 {config.INDEX_DIR}"
        )

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
```

用法一律是 `puzzlefind <子命令> ... --index-dir X`，`--index-dir` 放在子命令**之后**。

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_cli.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
cd /d/ocr_claude
git add puzzlefind/cli.py tests/test_cli.py
git commit -m "feat: CLI for index, query, and stats"
```

---

### Task 12: FastAPI 服务

**Files:**
- Create: `puzzlefind/server.py`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `pipeline.build_index`, `library.Library`, `render.highlight`, `render.thumbnail`, `config.PHOTOS_DIR`
- Produces:
  - `server.create_app(index_dir=None, photos_dir=None, backend_factory=None) -> FastAPI`
  - 路由：`GET /`、`POST /api/photos`、`GET /api/photos`、`DELETE /api/photos/{photo_id}`、`GET /api/query`、`GET /api/highlight`、`GET /api/thumbnail`
  - `server.run(host: str = "0.0.0.0", port: int = 8000) -> None`

- [ ] **Step 1: 写服务测试**

创建 `tests/test_server.py`：

```python
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from puzzlefind import server
from puzzlefind.recognize import RawDetection


class CountingBackend:
    """每次调用返回一个递增编号，让每块碎片拿到不同的 code。"""

    def __init__(self):
        self.n = 0

    def read(self, image: np.ndarray) -> list[RawDetection]:
        self.n += 1
        return [RawDetection(f"B-{self.n:03d}", 0.99)]


@pytest.fixture
def client(tmp_path):
    app = server.create_app(
        index_dir=tmp_path / "index",
        photos_dir=tmp_path / "photos",
        backend_factory=CountingBackend,
    )
    return TestClient(app)


@pytest.fixture
def photo_bytes(separated_pieces) -> bytes:
    image, _ = separated_pieces
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    return buffer.tobytes()


class TestUpload:
    def test_upload_returns_summary(self, client, photo_bytes, separated_pieces):
        _, count = separated_pieces
        response = client.post(
            "/api/photos", files={"file": ("shot.jpg", photo_bytes, "image/jpeg")}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == count
        assert body["recognized"] == count

    def test_upload_makes_photo_listable(self, client, photo_bytes):
        client.post("/api/photos", files={"file": ("shot.jpg", photo_bytes, "image/jpeg")})
        listing = client.get("/api/photos").json()
        assert len(listing["photos"]) == 1

    def test_non_image_upload_is_rejected(self, client):
        response = client.post(
            "/api/photos", files={"file": ("x.txt", b"not an image", "text/plain")}
        )
        assert response.status_code == 400


class TestQuery:
    def test_hit_returns_piece_geometry(self, client, photo_bytes):
        client.post("/api/photos", files={"file": ("shot.jpg", photo_bytes, "image/jpeg")})
        body = client.get("/api/query", params={"code": "B-001"}).json()
        assert body["found"] is True
        assert "bbox" in body["piece"]
        assert body["photo_id"]

    def test_miss_returns_unrecognized_summary(self, client, photo_bytes):
        client.post("/api/photos", files={"file": ("shot.jpg", photo_bytes, "image/jpeg")})
        body = client.get("/api/query", params={"code": "D-999"}).json()
        assert body["found"] is False
        assert "unrecognized" in body

    def test_query_without_any_photo_is_a_clean_miss(self, client):
        body = client.get("/api/query", params={"code": "B-001"}).json()
        assert body["found"] is False


class TestHighlight:
    def test_returns_png_for_a_hit(self, client, photo_bytes):
        client.post("/api/photos", files={"file": ("shot.jpg", photo_bytes, "image/jpeg")})
        response = client.get("/api/highlight", params={"code": "B-001"})
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

    def test_returns_png_for_a_miss_showing_unrecognized(self, client, photo_bytes):
        client.post("/api/photos", files={"file": ("shot.jpg", photo_bytes, "image/jpeg")})
        response = client.get(
            "/api/highlight", params={"code": "D-999", "photo_id": "shot"}
        )
        assert response.status_code == 200

    def test_unknown_photo_id_returns_404(self, client):
        response = client.get(
            "/api/highlight", params={"code": "B-001", "photo_id": "nope"}
        )
        assert response.status_code == 404


class TestThumbnail:
    def test_returns_png(self, client, photo_bytes):
        client.post("/api/photos", files={"file": ("shot.jpg", photo_bytes, "image/jpeg")})
        response = client.get("/api/thumbnail", params={"code": "B-001"})
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

    def test_unknown_code_returns_404(self, client, photo_bytes):
        client.post("/api/photos", files={"file": ("shot.jpg", photo_bytes, "image/jpeg")})
        assert client.get("/api/thumbnail", params={"code": "Z-999"}).status_code == 404


class TestDelete:
    def test_deleting_removes_from_listing(self, client, photo_bytes):
        client.post("/api/photos", files={"file": ("shot.jpg", photo_bytes, "image/jpeg")})
        assert client.delete("/api/photos/shot").status_code == 200
        assert client.get("/api/photos").json()["photos"] == []


class TestFrontend:
    def test_root_serves_html(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_server.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'puzzlefind.server'`

- [ ] **Step 3: 实现服务**

创建 `puzzlefind/server.py`：

```python
"""FastAPI 薄服务层。所有真正的逻辑都在引擎模块里，这里只做
HTTP 编解码和文件落盘。

高亮图在服务端渲染（复用已被单元测试覆盖的 render.py），前端只
负责显示和缩放——这让渲染逻辑可测，也让前端代码降到几十行。
"""
from __future__ import annotations

import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response

from . import config, render
from .library import Library
from .pipeline import build_index

_STATIC_DIR = Path(__file__).resolve().parent / "static"


def _default_backend_factory():
    from .recognize import PaddleBackend

    return PaddleBackend()


def _png_response(image: np.ndarray) -> Response:
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise HTTPException(status_code=500, detail="PNG 编码失败")
    return Response(content=buffer.tobytes(), media_type="image/png")


def create_app(
    index_dir: Path | None = None,
    photos_dir: Path | None = None,
    backend_factory=None,
) -> FastAPI:
    app = FastAPI(title="拼图碎片编号查找器")

    resolved_index_dir = Path(index_dir) if index_dir else config.INDEX_DIR
    resolved_photos_dir = Path(photos_dir) if photos_dir else config.PHOTOS_DIR
    resolved_photos_dir.mkdir(parents=True, exist_ok=True)
    make_backend = backend_factory or _default_backend_factory

    # 后端惰性单例：PaddleOCR 模型加载很慢，不能每次请求都建一个
    state: dict = {"backend": None}

    def backend():
        if state["backend"] is None:
            state["backend"] = make_backend()
        return state["backend"]

    def library() -> Library:
        return Library.load(resolved_index_dir)

    def load_photo_image(photo_id: str) -> np.ndarray:
        photo = next(
            (p for p in library().photos if p.photo_id == photo_id), None
        )
        if photo is None:
            raise HTTPException(status_code=404, detail=f"照片不存在: {photo_id}")
        image = cv2.imread(photo.image_path)
        if image is None:
            raise HTTPException(status_code=404, detail=f"原图不可读: {photo.image_path}")
        return image

    @app.get("/")
    def index_page() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    @app.post("/api/photos")
    async def upload_photo(file: UploadFile = File(...)) -> dict:
        raw = await file.read()
        buffer = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="无法解码为图像")

        stem = Path(file.filename or "").stem or uuid.uuid4().hex[:8]
        photo_path = resolved_photos_dir / f"{stem}.jpg"
        cv2.imwrite(str(photo_path), image)

        photo_index = build_index(photo_path, backend(), photo_id=stem)
        lib = library()
        lib.save_photo(photo_index)

        total = len(photo_index.pieces)
        return {
            "photo_id": photo_index.photo_id,
            "total": total,
            "recognized": len(photo_index.recognized),
            "unrecognized": len(photo_index.unrecognized),
            "created_at": photo_index.created_at,
        }

    @app.get("/api/photos")
    def list_photos() -> dict:
        return {
            "photos": [
                {
                    "photo_id": p.photo_id,
                    "total": len(p.pieces),
                    "recognized": len(p.recognized),
                    "unrecognized": len(p.unrecognized),
                    "created_at": p.created_at,
                }
                for p in library().photos
            ]
        }

    @app.delete("/api/photos/{photo_id}")
    def delete_photo(photo_id: str) -> dict:
        if not library().delete_photo(photo_id):
            raise HTTPException(status_code=404, detail=f"照片不存在: {photo_id}")
        return {"deleted": photo_id}

    @app.get("/api/query")
    def query(code: str = Query(...)) -> dict:
        result = library().query(code)
        if result.found:
            assert result.piece is not None
            return {
                "found": True,
                "code": result.code,
                "photo_id": result.photo_id,
                "piece": result.piece.to_dict(),
            }
        return {
            "found": False,
            "code": result.code,
            "unrecognized": {
                photo_id: [p.piece_id for p in pieces]
                for photo_id, pieces in result.unrecognized.items()
            },
        }

    @app.get("/api/highlight")
    def highlight(code: str = Query(...), photo_id: str | None = None) -> Response:
        """命中时高亮目标碎片；未命中时高亮指定照片的全部未识别碎片。"""
        lib = library()
        result = lib.query(code)

        if result.found:
            assert result.piece is not None and result.photo_id is not None
            image = load_photo_image(result.photo_id)
            return _png_response(render.highlight(image, [result.piece]))

        if photo_id is None:
            raise HTTPException(
                status_code=404, detail=f"{code} 未找到，且未指定要查看哪张照片"
            )
        image = load_photo_image(photo_id)
        photo = next(p for p in lib.photos if p.photo_id == photo_id)
        return _png_response(render.highlight(image, photo.unrecognized, unknown=True))

    @app.get("/api/thumbnail")
    def thumbnail(code: str = Query(...), size: int = 200) -> Response:
        result = library().query(code)
        if not result.found:
            raise HTTPException(status_code=404, detail=f"{code} 未找到")
        assert result.piece is not None and result.photo_id is not None
        image = load_photo_image(result.photo_id)
        return _png_response(render.thumbnail(image, result.piece, size=size))

    return app


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    """起服务并打印局域网访问地址。"""
    import socket

    import uvicorn

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        lan_ip = sock.getsockname()[0]
    except OSError:
        lan_ip = "127.0.0.1"
    finally:
        sock.close()

    print(f"\n手机浏览器打开: http://{lan_ip}:{port}\n")
    print("若手机连不上，检查 Windows 防火墙是否放行了该端口。\n")
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    run()
```

- [ ] **Step 4: 建占位前端文件让 `GET /` 的测试能过**

创建 `puzzlefind/static/index.html`，暂时只放一行（Task 13 替换成完整实现）：

```html
<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>拼图碎片查找器</title></head><body>placeholder</body></html>
```

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/test_server.py -v`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
cd /d/ocr_claude
git add puzzlefind/server.py puzzlefind/static/ tests/test_server.py
git commit -m "feat: FastAPI service with upload, query, and highlight rendering"
```

---

### Task 13: 单文件前端

**Files:**
- Modify: `puzzlefind/static/index.html`（替换占位内容）
- Modify: `tests/test_server.py`（追加内容断言）

**Interfaces:**
- Consumes: `POST /api/photos`, `GET /api/photos`, `DELETE /api/photos/{id}`, `GET /api/query`, `GET /api/highlight`, `GET /api/thumbnail`
- Produces: 无 Python 接口（纯静态资源）

- [ ] **Step 1: 追加前端内容断言**

在 `tests/test_server.py` 的 `TestFrontend` 类里追加：

```python
    def test_html_wires_up_the_query_endpoint(self, client):
        body = client.get("/").text
        assert "/api/query" in body

    def test_html_wires_up_the_upload_endpoint(self, client):
        body = client.get("/").text
        assert "/api/photos" in body

    def test_html_uses_camera_capture_input(self, client):
        body = client.get("/").text
        assert 'capture="environment"' in body
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_server.py -v -k Frontend`
Expected: FAIL — 占位 HTML 里没有 `/api/query`

- [ ] **Step 3: 写完整前端**

用以下内容**完整替换** `puzzlefind/static/index.html`：

```html
<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>拼图碎片查找器</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: system-ui, -apple-system, "PingFang SC", sans-serif;
    background: #14161a; color: #e8e8ea;
  }
  header { padding: 12px 14px; background: #1c1f25; border-bottom: 1px solid #2a2e36; }
  h1 { margin: 0 0 10px; font-size: 16px; font-weight: 600; }
  .row { display: flex; gap: 8px; align-items: center; }
  .row + .row { margin-top: 8px; }
  input[type=text] {
    flex: 1; padding: 11px 12px; font-size: 18px; letter-spacing: 1px;
    border: 1px solid #363b45; border-radius: 8px; background: #0f1114; color: #e8e8ea;
  }
  button {
    padding: 11px 16px; font-size: 15px; border: 0; border-radius: 8px;
    background: #3d6fe0; color: #fff; font-weight: 600;
  }
  button.ghost { background: #2a2e36; color: #b9bcc4; font-weight: 500; }
  button:disabled { opacity: .5; }
  #status { padding: 9px 14px; font-size: 13px; min-height: 34px; line-height: 1.5; }
  #status.err { color: #ff8a8a; }
  #status.ok { color: #8ae6a8; }
  #stage {
    position: relative; overflow: hidden; background: #000;
    height: calc(100vh - 210px); touch-action: none;
  }
  #view { position: absolute; transform-origin: 0 0; }
  #thumb {
    position: absolute; right: 10px; bottom: 10px; max-width: 128px;
    border: 2px solid #ff00ff; border-radius: 6px; display: none; background: #000;
  }
  #photos { padding: 8px 14px 14px; font-size: 13px; color: #9aa0ab; }
  .photo { display: flex; justify-content: space-between; padding: 5px 0; }
  .photo button { padding: 3px 9px; font-size: 12px; }
</style>
</head>
<body>

<header>
  <h1>拼图碎片查找器</h1>
  <div class="row">
    <input type="text" id="code" placeholder="输入编号，如 B-403" autocomplete="off"
           autocapitalize="characters" inputmode="text">
    <button id="find">查找</button>
  </div>
  <div class="row">
    <input type="file" id="file" accept="image/*" capture="environment" hidden>
    <button class="ghost" id="upload">拍照 / 选图建索引</button>
    <button class="ghost" id="reset">复位视图</button>
  </div>
</header>

<div id="status">准备就绪。先拍一张碎片照片建立索引。</div>

<div id="stage">
  <img id="view" alt="">
  <img id="thumb" alt="碎片缩略图">
</div>

<div id="photos"></div>

<script>
const $ = (id) => document.getElementById(id);
const statusEl = $("status"), viewEl = $("view"), thumbEl = $("thumb"), stageEl = $("stage");

let tx = 0, ty = 0, scale = 1;

function say(text, kind) {
  statusEl.textContent = text;
  statusEl.className = kind || "";
}

function applyTransform() {
  viewEl.style.transform = `translate(${tx}px, ${ty}px) scale(${scale})`;
}

function resetView() {
  tx = 0; ty = 0;
  scale = viewEl.naturalWidth ? stageEl.clientWidth / viewEl.naturalWidth : 1;
  applyTransform();
}

// ---- 缩放平移：单指拖动，双指捏合 ----
let pointers = new Map(), lastMid = null, lastDist = 0;

stageEl.addEventListener("pointerdown", (e) => {
  stageEl.setPointerCapture(e.pointerId);
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  lastMid = null; lastDist = 0;
});

stageEl.addEventListener("pointermove", (e) => {
  if (!pointers.has(e.pointerId)) return;
  const prev = pointers.get(e.pointerId);
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });

  const pts = [...pointers.values()];
  if (pts.length === 1) {
    tx += e.clientX - prev.x;
    ty += e.clientY - prev.y;
  } else if (pts.length >= 2) {
    const [a, b] = pts;
    const dist = Math.hypot(a.x - b.x, a.y - b.y);
    const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
    if (lastDist > 0) {
      const factor = dist / lastDist;
      // 以两指中点为锚点缩放，避免图像跳走
      tx = mid.x - (mid.x - tx) * factor;
      ty = mid.y - (mid.y - ty) * factor;
      scale *= factor;
    }
    if (lastMid) { tx += mid.x - lastMid.x; ty += mid.y - lastMid.y; }
    lastDist = dist; lastMid = mid;
  }
  applyTransform();
});

function endPointer(e) {
  pointers.delete(e.pointerId);
  if (pointers.size < 2) { lastDist = 0; lastMid = null; }
}
stageEl.addEventListener("pointerup", endPointer);
stageEl.addEventListener("pointercancel", endPointer);

stageEl.addEventListener("wheel", (e) => {
  e.preventDefault();
  const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
  tx = e.clientX - (e.clientX - tx) * factor;
  ty = e.clientY - (e.clientY - ty) * factor;
  scale *= factor;
  applyTransform();
}, { passive: false });

$("reset").onclick = resetView;
viewEl.onload = resetView;

// ---- 照片列表 ----
async function refreshPhotos() {
  const data = await (await fetch("/api/photos")).json();
  $("photos").innerHTML = data.photos.length
    ? data.photos.map((p) =>
        `<div class="photo"><span>${p.photo_id} — 已识别 ${p.recognized}/${p.total}
         （建于 ${p.created_at.replace("T", " ")}）</span>
         <button class="ghost" onclick="removePhoto('${p.photo_id}')">删除</button></div>`
      ).join("")
    : "<div>还没有任何照片。</div>";
}

window.removePhoto = async (photoId) => {
  await fetch(`/api/photos/${photoId}`, { method: "DELETE" });
  await refreshPhotos();
  say(`已删除 ${photoId}。`, "ok");
};

// ---- 上传建索引 ----
$("upload").onclick = () => $("file").click();

$("file").onchange = async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  say("正在建索引，可能需要一两分钟，别关页面…");
  $("upload").disabled = true;
  try {
    const form = new FormData();
    form.append("file", file);
    const response = await fetch("/api/photos", { method: "POST", body: form });
    if (!response.ok) throw new Error((await response.json()).detail);
    const r = await response.json();
    say(`照片 ${r.photo_id}：分割 ${r.total} 块，识别 ${r.recognized} 块，`
        + `未识别 ${r.unrecognized} 块。`, "ok");
    await refreshPhotos();
  } catch (error) {
    say("建索引失败：" + error.message, "err");
  } finally {
    $("upload").disabled = false;
    e.target.value = "";
  }
};

// ---- 查询 ----
async function find() {
  const code = $("code").value.trim().toUpperCase();
  if (!code) return;
  thumbEl.style.display = "none";
  say("查询中…");
  try {
    const result = await (await fetch(`/api/query?code=${encodeURIComponent(code)}`)).json();
    if (result.found) {
      viewEl.src = `/api/highlight?code=${encodeURIComponent(code)}&t=${Date.now()}`;
      thumbEl.src = `/api/thumbnail?code=${encodeURIComponent(code)}&t=${Date.now()}`;
      thumbEl.style.display = "block";
      say(`${code} 在照片 ${result.photo_id}，碎片 #${result.piece.piece_id}，`
          + `置信度 ${result.piece.confidence.toFixed(2)}。对照右下角缩略图确认。`, "ok");
    } else {
      const entries = Object.entries(result.unrecognized);
      if (!entries.length) {
        say(`${code} 未找到，且所有碎片都已识别——它确实不在任何一张照片里。`, "err");
        return;
      }
      const [photoId, ids] = entries[0];
      viewEl.src = `/api/highlight?code=${encodeURIComponent(code)}`
                 + `&photo_id=${encodeURIComponent(photoId)}&t=${Date.now()}`;
      const others = entries.slice(1)
        .map(([id, list]) => `${id}(${list.length})`).join("、");
      say(`${code} 没认出来。照片 ${photoId} 里这 ${ids.length} 块是未识别的（青色圈出），`
          + `大概率在里面。` + (others ? ` 其他照片还有：${others}` : ""), "err");
    }
  } catch (error) {
    say("查询失败：" + error.message, "err");
  }
}

$("find").onclick = find;
$("code").addEventListener("keydown", (e) => { if (e.key === "Enter") find(); });

refreshPhotos();
</script>
</body>
</html>
```

- [ ] **Step 4: 运行前端测试确认通过**

Run: `python -m pytest tests/test_server.py -v -k Frontend`
Expected: PASS

- [ ] **Step 5: 手工验收**

```bash
cd /d/ocr_claude
python -m puzzlefind.server
```

用手机浏览器打开打印出的局域网地址，确认：页面能加载、能唤起相机、能输入编号。此时还没有索引，查询应显示「未找到」而不是白屏或报错。

- [ ] **Step 6: 提交**

```bash
cd /d/ocr_claude
git add puzzlefind/static/index.html tests/test_server.py
git commit -m "feat: single-file frontend with pan/zoom and query flow"
```

---

### Task 14: 真实照片标定与调参

**Files:**
- Create: `scripts/calibrate.py`
- Create: `docs/tuning-log.md`
- Modify: `puzzlefind/config.py`（按实测数据调整默认值）

**Interfaces:**
- Consumes: 全部引擎模块
- Produces: `scripts/calibrate.py` 的命令行入口；`debug/` 下的中间产物图

**这是唯一一个不写单元测试的任务**——它的产出是标定数据和调好的参数，不是代码功能。

- [ ] **Step 1: 写标定脚本**

创建 `scripts/calibrate.py`：

```python
"""对一张真实照片做全流程标定，把每一步的中间结果落到 debug/。

opencv 是 headless 版，没有 imshow，所以一切靠写文件观察。

用法:
    python scripts/calibrate.py data/photos/real1.jpg
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import cv2

from puzzlefind import config, render, segment
from puzzlefind.pipeline import build_index
from puzzlefind.recognize import PaddleBackend


def main() -> int:
    if len(sys.argv) < 2:
        print("用法: python scripts/calibrate.py <照片路径>", file=sys.stderr)
        return 2

    photo = Path(sys.argv[1])
    image = cv2.imread(str(photo))
    if image is None:
        print(f"无法读取: {photo}", file=sys.stderr)
        return 2

    debug = config.DEBUG_DIR / photo.stem
    debug.mkdir(parents=True, exist_ok=True)

    # --- 阶段 1：掩膜 ---
    mask = segment.build_mask(image)
    cv2.imwrite(str(debug / "01_mask.png"), mask)
    foreground_ratio = float((mask > 0).mean())
    print(f"[1] 掩膜前景占比 {foreground_ratio:.1%}")
    print("    合理范围大约 15%–50%。太高说明背景没压住（背景不够深或曝光过度），")
    print("    太低说明碎片被吃掉了。异常时调 config.MASK_THRESHOLD。")

    # --- 阶段 2：连通块 ---
    blobs = segment.find_blobs(mask)
    median_area = segment.median_blob_area(blobs)
    overlay = image.copy()
    cv2.drawContours(overlay, blobs, -1, (0, 255, 255), 2)
    cv2.imwrite(str(debug / "02_blobs.png"), overlay)
    print(f"[2] 连通块 {len(blobs)} 个，中位面积 {median_area:.0f} px²")

    # --- 阶段 3：切分 ---
    contours = segment.extract_contours(image)
    split_overlay = image.copy()
    cv2.drawContours(split_overlay, contours, -1, (255, 0, 255), 2)
    cv2.imwrite(str(debug / "03_split.png"), split_overlay)
    print(f"[3] 切分后 {len(contours)} 块（比连通块多出的就是被拆开的粘连团）")
    print("    打开 03_split.png 数一下和实际碎片数差多少。差得多就调")
    print("    config.SPLIT_AREA_RATIO 和 MIN_AREA_RATIO。")

    # --- 阶段 4：裁剪采样 ---
    crops_dir = debug / "crops"
    crops_dir.mkdir(exist_ok=True)
    for i, contour in enumerate(contours[:12]):
        cv2.imwrite(str(crops_dir / f"{i:03d}.png"), segment.crop_piece(image, contour))
    print(f"[4] 前 12 块裁剪图已写入 {crops_dir}")
    print("    肉眼检查：编号是否清晰可读？有没有混进邻块的编号？")
    print("    字太小就调大 config.CROP_TARGET_LONG_EDGE，或者下次少拍几块。")

    # --- 阶段 5：完整识别 ---
    print("[5] 开始识别（首次会下载模型）…")
    started = time.time()
    index = build_index(photo, PaddleBackend())
    elapsed = time.time() - started

    total = len(index.pieces)
    methods = Counter(p.method for p in index.pieces)
    hit = len(index.recognized)
    print(f"\n=== 结果 ===")
    print(f"总块数        {total}")
    print(f"已识别        {hit}  ({hit / total:.1%})" if total else "已识别 0")
    print(f"  Pass A 直接 {methods.get('direct', 0)}")
    print(f"  Pass C 穷举 {methods.get('sweep', 0)}")
    print(f"  冲突降级    {methods.get('conflict', 0)}")
    print(f"未识别        {len(index.unrecognized)}")
    print(f"耗时          {elapsed:.1f}s  ({elapsed / max(1, total):.2f}s/块)")

    # --- 阶段 6：可视化 ---
    cv2.imwrite(
        str(debug / "04_unrecognized.png"),
        render.highlight(image, index.unrecognized, unknown=True),
    )
    print(f"\n未识别碎片可视化: {debug / '04_unrecognized.png'}")

    print("\n=== 调参建议 ===")
    direct = methods.get("direct", 0)
    if total and direct / total > 0.85:
        print(f"Pass A 覆盖率 {direct / total:.0%}，很高。可以把")
        print(f"config.SWEEP_CONFIDENCE_THRESHOLD 从 {config.SWEEP_CONFIDENCE_THRESHOLD}")
        print("下调到 0.75 左右，建索引会快一倍以上。")
    elif total and methods.get("sweep", 0) > direct:
        print("穷举承担了主要工作量——说明 PaddleOCR 的检测器在这批图上")
        print("确实处理不好任意角度。保持高阈值，并考虑把 SWEEP_ANGLES")
        print("加密到每 15 度一档以进一步提升覆盖率。")
    if total and hit / total < 0.7:
        print("识别率偏低。优先排查顺序：")
        print("  a) 看 crops/ 里的字够不够大 → 下次每张少拍点碎片")
        print("  b) 看 01_mask.png 背景压住没有 → 换更深的背景布")
        print("  c) 看 03_split.png 切分对不对 → 调 SPLIT_AREA_RATIO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: 拍第一张真实照片并跑标定**

在深色纯背景上摊开 **40–60 块**碎片，拍照，存到 `data/photos/real1.jpg`，然后：

```bash
cd /d/ocr_claude
python scripts/calibrate.py data/photos/real1.jpg
```

- [ ] **Step 3: 逐张检查 debug 产物**

按顺序打开 `debug/real1/` 下的文件：

1. `01_mask.png` — 碎片应是干净的白色团块，背景应是纯黑。若背景有大片白色，背景不够深或曝光过度，重拍。
2. `02_blobs.png` — 黄色轮廓应贴合碎片。若多块被圈成一个，正常（下一步切分）。
3. `03_split.png` — 洋红轮廓的数量应接近实际碎片数。**数一下差多少，这是分割质量的硬指标。**
4. `crops/*.png` — **最关键**。编号必须清晰可读，且每张图里只有一个编号。
5. `04_unrecognized.png` — 青色圈出的就是没认出来的。看看它们有什么共同点（都在边缘？都被压住了？）。

- [ ] **Step 4: 把观察记录进调参日志**

创建 `docs/tuning-log.md`：

```markdown
# 调参日志

每次标定跑完在这里追加一条。目的是让参数调整有据可依，
而不是凭感觉来回改。

## 模板

### YYYY-MM-DD — <照片名>

**拍摄条件**：背景 / 光源 / 碎片数 / 手机型号与分辨率

**分割**
- 掩膜前景占比：
- 连通块数 / 切分后块数 / 实际碎片数：
- 观察：

**识别**
- 总数 / 已识别 / 识别率：
- Pass A / Pass C / 冲突降级：
- 耗时（总 / 每块）：
- 观察：

**本次调整的参数**
- `config.XXX`: 旧值 → 新值，理由

---

### 2026-08-03 — real1.jpg（待填）
```

把 Step 2/3 的实际数字填进去。

- [ ] **Step 5: 按数据调整 config 默认值**

根据标定脚本 Step 5 打印的建议，修改 `puzzlefind/config.py` 中的参数。**每改一个就重跑一次标定，确认识别率没有倒退。** 常见调整：

- `SWEEP_CONFIDENCE_THRESHOLD` — Pass A 覆盖率高就调低，能大幅提速
- `CROP_TARGET_LONG_EDGE` — 裁剪图里的字看着糊就调大（但超过 768 收益递减）
- `SPLIT_AREA_RATIO` — 该切的没切就调低，不该切的被切了就调高
- `MIN_AREA_RATIO` — 有小碎片被当噪点丢掉就调低
- `MASK_THRESHOLD` — Otsu 在这批照片上不稳时改成固定值

- [ ] **Step 6: 跑全套测试确认调参没破坏任何东西**

Run: `python -m pytest tests/ -v -m "not ocr"`
Expected: PASS

若某个测试因为 config 改动而挂了，说明测试硬编码了某个参数值——把它改成从 `config` 读取，而不是改回参数。

- [ ] **Step 7: 提交**

```bash
cd /d/ocr_claude
git add scripts/calibrate.py docs/tuning-log.md puzzlefind/config.py
git commit -m "chore: calibration harness and first-pass parameter tuning"
```

- [ ] **Step 8: 端到端验收**

```bash
cd /d/ocr_claude
python -m puzzlefind.server
```

手机打开局域网地址，完整走一遍：拍照上传 → 等待建索引 → 查一个你知道确实在桌上的编号 → 确认高亮的碎片和缩略图对得上 → 查一个你知道不在的编号 → 确认给出了未识别碎片列表。

---

## Self-Review

**1. 需求覆盖检查**

| 共识条目 | 落实位置 |
|---|---|
| 建索引 + 查表 + 重拍刷新 | Task 8/9；`Library.save_photo` 覆盖同 ID 即刷新，`created_at` 提示过期 |
| 本地引擎 + 局域网 Web | Task 11（CLI）/ Task 12（`run()` 打印局域网 IP） |
| 词表 `[A-D]-NNN` + 区间自举 | Task 1（`bootstrap_ranges`）/ Task 7（离群剔除） |
| OpenCV + PaddleOCR，不用大模型 | Task 2–6，无任何 LLM 调用 |
| Pass A 打底 + Pass C 兜底 | Task 5/6，`recognize_piece` |
| 深色背景 + 分水岭切分 | Task 2/3，`split_blob` 二分搜种子 |
| 压暗 40–50% + 描边 + 编号 + 缩略图 | Task 10 / Task 13（缩略图在右下角，编号在状态栏） |
| 失败时高亮未识别碎片（方案 B） | Task 9 `QueryResult.unrecognized` / Task 12 `/api/highlight` miss 分支 / Task 13 前端 |
| 分块拍多张 + 跨照片索引 | Task 9 `Library.query` 遍历全部照片 |
| CPU 版 / FastAPI / 单 HTML / JSON / 引擎独立 | Global Constraints + Task 11/12/13 |

**2. 占位符扫描**：无 TBD、无「适当处理错误」、无「参考 Task N」。每个代码步骤都含完整可运行代码。Task 14 的 config 数值调整依赖真实照片，这是标定任务的本质，已在任务说明中标注为「产出是数据不是代码」。

**3. 类型一致性检查**：`RecogResult` 的五个字段在 Task 5 定义、Task 6/7/8 使用，命名一致；`Piece` 的九个字段在 Task 8 定义，Task 10/11/12 使用一致；`segment.extract_contours` 返回 `list[np.ndarray]`，Task 8 中按 `pt[0][0]` 解包 OpenCV 轮廓格式，与 Task 2/3 产出格式一致；`Library.query` 返回的 `QueryResult.unrecognized` 是 `dict[str, list[Piece]]`，Task 12 把它降维成 `dict[str, list[int]]` 后返回 JSON——已在测试中断言。

**执行前复审时发现并已修正的三处缺陷：**

1. **Task 11 argparse 静默失效。** `--index-dir` 原本同时挂在主解析器和子解析器上。argparse 的解析顺序会让子解析器的默认值 `None` 覆盖主解析器已解析到的值，导致 `puzzlefind --index-dir X query CODE` 不报错但用错目录。已改为只挂在子解析器上，并在代码注释里写明这个陷阱。

2. **Task 1 `snap()` 的性能问题。** 原实现对 4000 个候选逐一跑完整 DP，约 10 万次单元格计算。Pass C 每块碎片跑 12 个角度，250 块碎片会因此凭空多花一两分钟。已加入两级优化：长度差下界剪枝，以及等长输入的可分解快速路径（34 次比较取代 4000 次 DP）。快速路径是**精确**而非近似的——等长串的最优对齐必为纯替换，且候选空间中前缀与数字可自由组合，故逐位取最小值之和即全局最小值。已加参数化测试锁死快慢两条路径的一致性。

3. **Task 8 测试替身的隐性错位。** 原 `SequenceBackend` 用「第几次调用」当索引，但识别失败的碎片会触发 12 次穷举调用，使后续碎片索引整体错位。原测试恰好因为 `None` 排在列表末尾而通过——那是运气不是设计。已改为 `PieceBackend`，按碎片配额显式记账，对穷举次数免疫；替身与 `SWEEP_ANGLES` 之间的耦合已在类注释中写明。

---

## 执行入口

**依赖安装（Task 1 之后、Task 5 之前必须完成）：**

```bash
cd /d/ocr_claude
python -m pip install -e ".[dev]"
```

`paddlepaddle` 与 `paddleocr` 体积较大，首次安装及首次推理时的模型下载都需要时间，建议在 Task 5 之前提前跑一次 `scripts/probe_paddleocr.py` 把模型拉下来。
