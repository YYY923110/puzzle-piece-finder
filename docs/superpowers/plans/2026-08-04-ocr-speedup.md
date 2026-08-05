# OCR 提速实施方案（杠杆 1 + 杠杆 2）

> ## ⚠️ 本文档已执行完毕，是历史存档
>
> 2026-08-04 全部实现并提交（`f7bcc68`..`a51313a`）。实测 **2.94x**（385 s → 131 s，
> 识别率不变），收益全部来自 `SWEEP_EARLY_EXIT_CONFIDENCE`。
> 量过但无效的杠杆（批量推理、关方向分类、oneDNN、GPU…）记在
> [调参日志](../../tuning-log.md)，**别再试一遍**。
>
> 注意：这里的耗时数字是 2026-08-04 裁剪掩膜改动**之前**测的。当前实测见 README。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把建索引耗时降到目前的 1/4 以下，且**识别率一块不掉**。

**Architecture:** 实测发现每次 `PaddleOCR.predict()` 的 1.71 秒里，**检测模型（det）占 92%，识别模型（rec）只占 8%**。而现在的 Pass C 把整张裁剪图旋转 12 次、每次重跑一遍完整管线——等于把最贵的 det 白跑了 12 遍。本方案让 Pass A 顺手把检测框（`dt_polys`）带出来，Pass C 改成「按这个四边形透视裁出文字行 → 只喂 rec 模型试正立/翻转两种」。det 无框时回退到现有的全量穷举，因此**识别率有下限保证，只可能持平或变好**。

**Tech Stack:** Python 3.13 / PaddleOCR 3.7.0 / paddlepaddle 3.3.1 (CPU) / OpenCV / pytest

## Global Constraints

- 所有命令走项目自带的 `.\.venv\Scripts\python.exe`，**不要装进 base Anaconda**（PaddleOCR 会拉进 `opencv-contrib-python`，会顶掉 base 里的 headless OpenCV 5.0）。
- `enable_mkldnn` **必须保持 `False`**。paddle 3.3.1 的 oneDNN 算子在 PIR 执行器下抛 `ConvertPirAttribute2RuntimeAttribute not support`。已实测 `FLAGS_enable_pir_api=0` / `FLAGS_enable_pir_in_executor=0` **都绕不过**，别再试。
- `recognize.OcrBackend` 这个 `Protocol` 是架构支点（spec §6）。**`read(image) -> list[RawDetection]` 必须保持为唯一的必需方法**，新增的按行识别能力只能是**可选**协议，缺席时自动降级。
- 除标了 `@pytest.mark.ocr` 的测试外，**任何测试都不许加载 PaddleOCR**。验证方式：跑完非 ocr 测试后 `sys.modules` 里不得出现 paddle 系模块。
- 所有可调参数集中在 `puzzlefind/config.py` 一个文件里，不许散落到别处。
- 基准照片固定用 `data/photos/real6.jpg`（1086×1448，50 块，深色纯背景，连通块 50 = 切分后 50，面积中位 7929 px²）。
- 每个任务结束时 `.\.venv\Scripts\python.exe -m pytest tests\ -m "not ocr"` 必须全绿。**实测基线：非 ocr 168 个、ocr 1 个、合计 169 个**（README 第 149 行写的「两个 `@pytest.mark.ocr` 的测试」是笔误，实际只有 `TestPaddleBackendIntegration` 一个类一条，Task 8 顺手改掉）。

---

## 已量到的事实（不要重新验证，直接用）

| 事实 | 数值 |
|---|---|
| 分割耗时 | 1 ms/块（50 块共 0.03s）——**不是瓶颈，不要动 segment.py** |
| 单次 `PaddleOCR.predict()` | 1.71 s |
| 其中 det / rec | **1.485 s / 0.129 s（det 占 92%）** |
| 批量 `predict([...])` | **1.00x，无效** |
| 关 `use_textline_orientation` | 1.02x，无效 |
| 调 `text_det_limit_side_len` | 0.95x，反而略慢 |
| GPU | `paddle.is_compiled_with_cuda() == False`，装的是 CPU 版轮子 |
| 单进程 CPU 占用 | 7.9 / 16 逻辑核 |
| `TextRecognition.predict()` 返回键 | `rec_text`（单数）、`rec_score` |
| `PaddleOCR.predict()` 返回键 | `dt_polys`、`rec_texts`、`rec_scores`、`rec_polys`、`rec_boxes`、`textline_orientation_angles` |
| 本方案 Pass C 提速 | **77x**（12 次完整 predict 20.5s → 自裁 + 2 次 rec 0.265s） |
| 机制准确率 | 61 个有框样本，与完整管线一致 **58**；3 处不一致中 2 处是完整管线自己读成垃圾，1 处新路径**读得更对** |

**关键实现细节（第一轮验证踩到的坑）：** 透视裁剪出的文字行若 `高/宽 >= 1.5`，说明检测框的点序把长短边判反了（文字竖排时必然发生），**必须补一次 `np.rot90` 转正**。不补这一步，90° 和 270° 的碎片会读出 `89` / `169` / `382` 这类垃圾。这一步是 PaddleOCR 内部 `get_rotate_crop_image` 有而我们最初漏掉的。

---

## 文件结构

| 文件 | 改动 | 职责 |
|---|---|---|
| `puzzlefind/config.py` | 修改 | 新增 3 个参数：提前退出阈值、rec 模型名、转正比例阈值 |
| `puzzlefind/recognize.py` | 修改 | 全部新逻辑都在这里：`RawDetection.poly`、`best_poly`、`deskew_quad`、`LineOcrBackend`、`recognize_line_sweep`、`recognize_sweep` 提前退出、`recognize_piece` 编排、`PaddleBackend.read_line` |
| `scripts/benchmark.py` | 新建 | 可复跑的耗时基准，前后对比用 |
| `tests/test_recognize.py` | 修改 | 新增假后端与 5 组测试 |
| `docs/tuning-log.md` | 修改 | 追加 real6.jpg 改造前/后两条记录 |
| `README.md`、`docs/superpowers/specs/2026-08-03-puzzle-piece-finder.md` | 修改 | 同步 `method` 取值与耗时数字 |

新逻辑全部落在 `recognize.py`（现 230 行，完工后约 340 行）。这是刻意的：识别策略是一个整体，拆成多个文件反而要在模块间传递 `dt_polys` 这种中间态。`segment.py` 一行不动。

---

### Task 1: 建立可复跑的耗时基准（改代码前先量）

**Files:**
- Create: `scripts/benchmark.py`

**Interfaces:**
- Produces: `scripts/benchmark.py`，命令行 `python scripts/benchmark.py <photo>`，把结果写进 `debug/<照片名>/benchmark.json`

**为什么必须先做：** 改完再量就没有对照了。这一步产出的数字是后面所有任务的验收依据。

- [ ] **Step 1: 写基准脚本**

创建 `scripts/benchmark.py`：

```python
"""建索引耗时基准。改识别策略前后各跑一次，用于对照。

跑法:
    .\\.venv\\Scripts\\python.exe scripts\\benchmark.py data\\photos\\real6.jpg
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from puzzlefind import config
from puzzlefind.pipeline import build_index
from puzzlefind.recognize import PaddleBackend


def main() -> int:
    if len(sys.argv) < 2:
        print("用法: python scripts/benchmark.py <照片路径>", file=sys.stderr)
        return 2

    photo = Path(sys.argv[1])
    if not photo.exists():
        print(f"照片不存在: {photo}", file=sys.stderr)
        return 2

    backend = PaddleBackend()
    # 预热：把模型加载的时间排除在计时之外，否则首次跑会虚高几秒
    import numpy as np

    backend.read(np.full((64, 200, 3), 220, dtype=np.uint8))

    start = time.perf_counter()
    index = build_index(photo, backend)
    elapsed = time.perf_counter() - start

    total = len(index.pieces)
    hit = len(index.recognized)
    methods = Counter(p.method for p in index.pieces)

    report = {
        "photo": str(photo),
        "total": total,
        "recognized": hit,
        "rate": round(hit / total, 4) if total else 0.0,
        "seconds": round(elapsed, 1),
        "seconds_per_piece": round(elapsed / total, 3) if total else 0.0,
        "methods": dict(methods),
        "codes": sorted(p.code for p in index.recognized),
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    out_dir = config.DEBUG_DIR / photo.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "benchmark.json"
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n已写入 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: 跑基准，记下数字**

Run: `.\.venv\Scripts\python.exe scripts\benchmark.py data\photos\real6.jpg`

预期：约 9 分钟跑完（50 块 × 约 10 s/块）。输出形如：

```json
{
  "total": 50,
  "recognized": 42,
  "rate": 0.84,
  "seconds": 485.4,
  "seconds_per_piece": 9.71,
  "methods": {"direct": 32, "sweep": 10, "conflict": 8}
}
```

**把实际输出完整抄下来**，Task 8 要用它做前后对照。注意实际数字大概率与上面示例不同（上面是 real3.jpg 的），照抄示例数字是错的。

- [ ] **Step 3: 确认基准文件已落盘**

Run: `.\.venv\Scripts\python.exe -c "import json,pathlib; print(pathlib.Path('debug/real6.jpg/benchmark.json').read_text(encoding='utf-8'))"`
Expected: 打印出刚才那份 JSON

- [ ] **Step 4: 提交**

```bash
git add scripts/benchmark.py
git commit -m "chore: add reusable indexing benchmark script

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: `RawDetection` 带上检测框

**Files:**
- Modify: `puzzlefind/recognize.py:17-23`（`RawDetection`）、`puzzlefind/recognize.py:109-136`（`_extract`）
- Test: `tests/test_recognize.py`

**Interfaces:**
- Produces:
  - `RawDetection(text: str, score: float, poly: list[list[int]] | None = None)` —— 新增第三个字段，**带默认值 None**，因此所有现存构造点（测试里的 `RawDetection("B-403", 0.97)`）不受影响
  - `recognize.best_poly(detections: list[RawDetection]) -> list[list[int]] | None` —— 取分数最高且带框的那条检测的四边形

**设计要点：** 取「分数最高」而不是「吸附成功」的那条。rec 读错时 det 的框往往仍是对的——这正是本方案要利用的。

- [ ] **Step 1: 写失败的测试**

在 `tests/test_recognize.py` 里，`TestRotateExpand` 类**之前**插入：

```python
class TestBestPoly:
    def test_returns_none_when_no_detection_has_a_poly(self):
        detections = [RawDetection("B-403", 0.9), RawDetection("A-111", 0.5)]
        assert recognize.best_poly(detections) is None

    def test_returns_none_for_empty_detections(self):
        assert recognize.best_poly([]) is None

    def test_picks_poly_of_highest_scoring_detection(self):
        low = [[0, 0], [10, 0], [10, 5], [0, 5]]
        high = [[20, 20], [40, 20], [40, 30], [20, 30]]
        detections = [
            RawDetection("A-111", 0.40, low),
            RawDetection("B-403", 0.93, high),
        ]
        assert recognize.best_poly(detections) == high

    def test_ignores_high_scoring_detection_that_has_no_poly(self):
        poly = [[0, 0], [10, 0], [10, 5], [0, 5]]
        detections = [
            RawDetection("QWERTY", 0.99),          # 分最高但没有框
            RawDetection("B-403", 0.42, poly),
        ]
        assert recognize.best_poly(detections) == poly

    def test_poly_defaults_to_none_so_existing_call_sites_keep_working(self):
        assert RawDetection("B-403", 0.9).poly is None
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_recognize.py::TestBestPoly -v`
Expected: FAIL，`AttributeError: module 'puzzlefind.recognize' has no attribute 'best_poly'`（以及 `RawDetection` 只接受 2 个位置参数）

- [ ] **Step 3: 实现**

把 `puzzlefind/recognize.py` 里的 `RawDetection` 改成：

```python
@dataclass(frozen=True)
class RawDetection:
    """OCR 后端返回的一条原始检测结果。

    poly 是检测器给出的文字四边形（原图坐标，4 个点）。它是本工具提速的
    关键：det 模型占了单次识别 92% 的耗时，而它在 rec 读错时往往**框仍是对的**，
    所以拿到框之后就能只重跑便宜的 rec 模型，不必再跑一遍 det。
    后端没有框（或不是基于检测的后端）时为 None，此时自动降级为全量角度穷举。
    """

    text: str
    score: float
    poly: list[list[int]] | None = None
```

在 `_extract` 之后新增：

```python
def best_poly(detections: list[RawDetection]) -> list[list[int]] | None:
    """分数最高的那条检测的四边形；都没有框则返回 None。

    刻意按「分数」而不是「能否吸附到合法编号」来挑：rec 把 B-296 读成
    38929 的时候，det 的框仍然精确地圈着那行字。本方案就是靠这一点，
    用一个对的框换掉 12 次昂贵的重复检测。
    """
    best: RawDetection | None = None
    for detection in detections:
        if detection.poly is None:
            continue
        if best is None or detection.score > best.score:
            best = detection
    return best.poly if best is not None else None
```

把 `_extract` 的返回部分改成同时取出 `dt_polys`：

```python
    texts = payload.get("rec_texts") or []
    scores = payload.get("rec_scores") or []
    polys = payload.get("dt_polys") or []

    detections: list[RawDetection] = []
    for index, (text, score) in enumerate(zip(texts, scores)):
        if not isinstance(text, str):
            continue
        poly = None
        if index < len(polys):
            # dt_polys 的元素可能是 numpy 数组也可能是嵌套 list，
            # 统一成 list[list[int]]，免得下游还要判类型
            raw = polys[index]
            points = raw.tolist() if hasattr(raw, "tolist") else raw
            poly = [[int(x), int(y)] for x, y in points]
        detections.append(RawDetection(text, float(score), poly))
    return detections
```

同时把 `_extract` 的 docstring 里的实测结构补上 `dt_polys`：

```python
    本机 paddleocr 3.7.0 / PP-OCRv6 的实测结构（探针输出）：
        res.json == {"res": {..., "rec_texts": ["B-403"],
                             "rec_scores": [0.9999],
                             "dt_polys": [[[145,142],[348,142],[348,231],[145,231]]],
                             ...}}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_recognize.py -v -m "not ocr"`
Expected: 全部 PASS，含新增的 5 条

- [ ] **Step 5: 跑全量测试确认没打破别处**

Run: `.\.venv\Scripts\python.exe -m pytest tests\ -m "not ocr"`
Expected: 173 passed（原 168 + 新增 5）

- [ ] **Step 6: 提交**

```bash
git add puzzlefind/recognize.py tests/test_recognize.py
git commit -m "feat(recognize): carry the detection quad on RawDetection

det 占单次识别 92% 的耗时，而 rec 读错时它的框往往仍是对的。
把框带出来，后续就能只重跑便宜的 rec 模型。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: 按四边形裁出并摆正文字行

**Files:**
- Modify: `puzzlefind/recognize.py`（在 `rotate_expand` 之后新增 `deskew_quad`）
- Modify: `puzzlefind/config.py`（新增 `LINE_DESKEW_ROTATE_RATIO`）
- Test: `tests/test_recognize.py`

**Interfaces:**
- Consumes: Task 2 的 `RawDetection.poly`
- Produces: `recognize.deskew_quad(image: np.ndarray, quad: list[list[int]]) -> np.ndarray` —— 输入原图与四边形，输出摆正的水平文字行小图

**这是整个方案里最容易写错的一步。** 第一轮验证时漏掉 `rot90` 补偿，90°/270° 的碎片全部读成垃圾。测试里专门有一条钉住它。

- [ ] **Step 1: 写失败的测试**

在 `tests/test_recognize.py` 的 `TestRotateExpand` 之后插入：

```python
class TestDeskewQuad:
    def test_axis_aligned_quad_crops_exactly_that_rectangle(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        image[20:50, 30:130] = 255
        quad = [[30, 20], [130, 20], [130, 50], [30, 50]]

        line = recognize.deskew_quad(image, quad)

        assert line.shape[1] == 100      # 宽
        assert line.shape[0] == 30       # 高
        assert line.mean() > 250         # 裁出来的确实是那块白区

    def test_vertical_quad_is_rotated_back_to_horizontal(self):
        """竖排的框必须被转正——否则 rec 模型读不出来。

        这条钉的是第一轮验证踩到的真实缺陷：碎片旋转 90°/270° 时
        检测框的点序会把长短边判反，不补 rot90 就会读出 89/169/382 这类垃圾。
        """
        image = np.zeros((200, 100, 3), dtype=np.uint8)
        image[30:130, 20:50] = 255
        quad = [[20, 30], [50, 30], [50, 130], [20, 130]]   # 高 100 / 宽 30 = 3.3

        line = recognize.deskew_quad(image, quad)

        assert line.shape[1] > line.shape[0], "竖排文字行没有被转正"

    def test_wide_quad_is_left_alone(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        quad = [[10, 10], [150, 10], [150, 40], [10, 40]]   # 高 30 / 宽 140，很扁
        line = recognize.deskew_quad(image, quad)
        assert line.shape[1] > line.shape[0]

    def test_rotated_quad_is_straightened(self):
        """把四边形按 45° 给出，裁出来的应该是一条水平的条。"""
        import math

        image = np.zeros((300, 300, 3), dtype=np.uint8)
        cx, cy = 150.0, 150.0
        half_w, half_h = 60.0, 12.0
        corners = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
        angle = math.radians(45)
        quad = [
            [round(cx + x * math.cos(angle) - y * math.sin(angle)),
             round(cy + x * math.sin(angle) + y * math.cos(angle))]
            for x, y in corners
        ]

        line = recognize.deskew_quad(image, quad)

        assert line.shape[1] == pytest.approx(2 * half_w, abs=3)
        assert line.shape[0] == pytest.approx(2 * half_h, abs=3)

    def test_degenerate_quad_does_not_crash(self):
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        quad = [[10, 10], [10, 10], [10, 10], [10, 10]]
        line = recognize.deskew_quad(image, quad)
        assert line.size >= 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_recognize.py::TestDeskewQuad -v`
Expected: FAIL，`AttributeError: module 'puzzlefind.recognize' has no attribute 'deskew_quad'`

- [ ] **Step 3: 加配置项**

在 `puzzlefind/config.py` 的「识别」一节（`MIN_ACCEPT_CONFIDENCE` 之后）追加：

```python
# 透视裁剪出的文字行，高/宽 达到此值时判定检测框的点序把长短边判反了
# （文字竖排时必然发生），转 90° 摆正。编号 "B-299" 本身宽远大于高，
# 正常裁剪不可能触发。不补这一步，旋转 90°/270° 的碎片会读出 89/169/382 这类垃圾。
LINE_DESKEW_ROTATE_RATIO: float = 1.5
```

- [ ] **Step 4: 实现**

在 `puzzlefind/recognize.py` 的 `rotate_expand` 之后新增（注意文件顶部已有 `import numpy as np`；`cv2` 在本模块是函数内局部 import 的风格，这里沿用）：

```python
def deskew_quad(image: np.ndarray, quad: list[list[int]]) -> np.ndarray:
    """按检测器给出的四边形透视裁剪，输出一条摆正的水平文字行。

    这是 PaddleOCR 内部 `get_rotate_crop_image` 的等价实现。自己实现而不是
    从 paddlex 内部 import，是因为那是私有路径，版本间会挪窝；而这段几何
    只有十几行，还能脱离 PaddleOCR 单独测。

    末尾那次 rot90 是**必须的**，不是保险：文字竖排时（碎片旋转 90°/270°）
    DB 检测器给出的点序会把长短边判反，裁出来是一条竖着的图，rec 模型
    读它只会吐出 89/169/382 这类垃圾。实测漏掉这一步会让 90° 和 270°
    两个角度全线失败。
    """
    import cv2

    points = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    width = int(max(
        np.linalg.norm(points[0] - points[1]),
        np.linalg.norm(points[2] - points[3]),
    ))
    height = int(max(
        np.linalg.norm(points[0] - points[3]),
        np.linalg.norm(points[1] - points[2]),
    ))
    width, height = max(1, width), max(1, height)

    destination = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
    )
    line = cv2.warpPerspective(
        image,
        cv2.getPerspectiveTransform(points, destination),
        (width, height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )

    if line.shape[0] / max(1, line.shape[1]) >= config.LINE_DESKEW_ROTATE_RATIO:
        line = np.rot90(line)

    # warpPerspective / rot90 之后可能不是连续内存，Paddle 要求连续
    return np.ascontiguousarray(line)
```

- [ ] **Step 5: 跑测试确认通过**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_recognize.py::TestDeskewQuad -v`
Expected: 5 passed

- [ ] **Step 6: 跑全量测试**

Run: `.\.venv\Scripts\python.exe -m pytest tests\ -m "not ocr"`
Expected: 178 passed

- [ ] **Step 7: 提交**

```bash
git add puzzlefind/recognize.py puzzlefind/config.py tests/test_recognize.py
git commit -m "feat(recognize): deskew a text line from the detection quad

含竖排转正补偿——漏掉它会让旋转 90/270 度的碎片全线读成垃圾。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: 按行识别的可选协议与 Pass C 新路径

**Files:**
- Modify: `puzzlefind/recognize.py`（新增 `LineOcrBackend`、`recognize_line_sweep`）
- Modify: `puzzlefind/config.py`（新增 `LINE_ORIENTATIONS`）
- Test: `tests/test_recognize.py`

**Interfaces:**
- Consumes: Task 2 的 `best_poly`、Task 3 的 `deskew_quad`
- Produces:
  - `LineOcrBackend` —— `@runtime_checkable` 的**可选**协议，唯一方法 `read_line(image: np.ndarray) -> RawDetection`
  - `recognize.recognize_line_sweep(backend: LineOcrBackend, crop: np.ndarray, quad: list[list[int]]) -> RecogResult` —— 返回 `method="line"`，`angle` 为命中的朝向（0 或 180）

**架构约束（务必遵守）：** `LineOcrBackend` 是**独立的第二个 Protocol**，不是往 `OcrBackend` 里加方法。spec §6 承诺「换识别后端只需实现 `read()`」，加必需方法会毁掉这个承诺。缺席时 Task 6 会自动走全量穷举。

- [ ] **Step 1: 写失败的测试**

在 `tests/test_recognize.py` 顶部的 `FakeBackend` 之后新增假后端：

```python
class FakeLineBackend(FakeBackend):
    """额外实现可选的 read_line 的假后端。

    read 与 read_line 各自独立计数，测试才能分辨「跑的是便宜的 rec 路径
    还是昂贵的全量穷举」——这正是本次改造要证明的事。
    """

    def __init__(
        self,
        responses: list[list[RawDetection]],
        line_responses: list[RawDetection],
    ):
        super().__init__(responses)
        self.line_responses = line_responses
        self.line_calls = 0

    def read_line(self, image: np.ndarray) -> RawDetection:
        result = self.line_responses[
            min(self.line_calls, len(self.line_responses) - 1)
        ]
        self.line_calls += 1
        return result
```

在 `TestRecognizeSweep` 之后插入：

```python
QUAD = [[10, 10], [90, 10], [90, 40], [10, 40]]


class TestRecognizeLineSweep:
    def test_reads_code_from_the_upright_orientation(self, blank_crop):
        backend = FakeLineBackend([[]], [RawDetection("B-403", 0.98)])
        result = recognize.recognize_line_sweep(backend, blank_crop, QUAD)
        assert result.code == "B-403"
        assert result.method == "line"
        assert result.angle == 0

    def test_falls_through_to_the_flipped_orientation(self, blank_crop):
        backend = FakeLineBackend(
            [[]],
            [RawDetection("EOP-8", 0.40), RawDetection("B-403", 0.97)],
        )
        result = recognize.recognize_line_sweep(backend, blank_crop, QUAD)
        assert result.code == "B-403"
        assert result.angle == 180

    def test_tries_both_orientations_when_nothing_snaps(self, blank_crop):
        backend = FakeLineBackend([[]], [RawDetection("QWERTY", 0.99)])
        result = recognize.recognize_line_sweep(backend, blank_crop, QUAD)
        assert result.code is None
        assert backend.line_calls == 2

    def test_stops_early_on_a_very_confident_read(self, blank_crop):
        from puzzlefind import config

        backend = FakeLineBackend(
            [[]],
            [
                RawDetection("B-403", config.SWEEP_EARLY_EXIT_CONFIDENCE),
                RawDetection("A-111", 1.0),
            ],
        )
        result = recognize.recognize_line_sweep(backend, blank_crop, QUAD)
        assert result.code == "B-403"
        assert backend.line_calls == 1, "拿到高置信度结果后不该再试翻转"

    def test_never_calls_the_expensive_full_read(self, blank_crop):
        backend = FakeLineBackend([[]], [RawDetection("B-403", 0.98)])
        recognize.recognize_line_sweep(backend, blank_crop, QUAD)
        assert backend.calls == 0, "按行识别不该触碰检测模型"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_recognize.py::TestRecognizeLineSweep -v`
Expected: FAIL，`AttributeError: ... has no attribute 'recognize_line_sweep'`

- [ ] **Step 3: 加配置项**

在 `puzzlefind/config.py` 里 `SWEEP_ANGLES` 之后追加：

```python
# 拿到检测框后，摆正的文字行只剩正反歧义，试这两个朝向就够。
# 实测：补了竖排转正之后，61 个有框样本里 58 个与完整管线读数一致，
# 再多试 90/270 两个朝向准确率**一模一样**，只是白花一倍时间。
LINE_ORIENTATIONS: tuple[int, ...] = (0, 180)
```

- [ ] **Step 4: 实现**

在 `puzzlefind/recognize.py` 的 `OcrBackend` 协议之后新增：

```python
@runtime_checkable
class LineOcrBackend(Protocol):
    """**可选**能力：只跑识别模型，不跑检测模型。

    刻意做成独立于 OcrBackend 的第二个协议，而不是往 OcrBackend 里加方法。
    spec §6 承诺「换识别后端只需实现 read()」，把这个方法设成必需会毁掉
    那个承诺。不实现它的后端一样能用，只是 Pass C 会走昂贵的全量角度穷举。
    """

    def read_line(self, image: np.ndarray) -> RawDetection:
        """对一条**已摆正**的文字行做识别，跳过检测模型。"""
        ...
```

顶部 import 相应改为：

```python
from typing import Protocol, runtime_checkable
```

在 `recognize_sweep` 之前新增：

```python
def recognize_line_sweep(
    backend: LineOcrBackend, crop: np.ndarray, quad: list[list[int]]
) -> RecogResult:
    """Pass C 的快路径：按检测框裁出文字行，只重跑识别模型。

    为什么这条路快得这么离谱：单次完整 predict 的 1.71 秒里，检测模型占
    1.485 秒（92%），识别模型只占 0.129 秒。而检测框在 Pass A 就已经拿到了，
    重跑 12 遍检测纯属浪费。实测 20.5 秒 → 0.265 秒，**77 倍**。

    摆正之后只剩正反歧义，所以只试两个朝向，不是 12 个角度。
    """
    import cv2

    line = deskew_quad(crop, quad)
    best = _NO_RESULT

    for orientation in config.LINE_ORIENTATIONS:
        image = cv2.rotate(line, cv2.ROTATE_180) if orientation == 180 else line
        detection = backend.read_line(np.ascontiguousarray(image))
        candidate = _best_snapped([detection], method="line", angle=orientation)
        if candidate.code is not None and candidate.confidence > best.confidence:
            best = candidate
        if best.code is not None and best.confidence >= config.SWEEP_EARLY_EXIT_CONFIDENCE:
            break

    return best
```

> `SWEEP_EARLY_EXIT_CONFIDENCE` 由 Task 5 加入 config。若按顺序执行到这里它还不存在，先在 config 的「识别」一节加上
> `SWEEP_EARLY_EXIT_CONFIDENCE: float = 0.99`（Task 5 会补它的完整注释与测试）。

- [ ] **Step 5: 跑测试确认通过**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_recognize.py::TestRecognizeLineSweep -v`
Expected: 5 passed

- [ ] **Step 6: 跑全量测试**

Run: `.\.venv\Scripts\python.exe -m pytest tests\ -m "not ocr"`
Expected: 183 passed

- [ ] **Step 7: 提交**

```bash
git add puzzlefind/recognize.py puzzlefind/config.py tests/test_recognize.py
git commit -m "feat(recognize): add rec-only line sweep as an optional backend capability

det 占单次识别 92% 的耗时。拿到检测框后只重跑 rec，Pass C 提速 77 倍。
做成独立的可选 Protocol，OcrBackend 仍只要求 read()（spec 6）。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: 全量穷举的提前退出（杠杆 2）

**Files:**
- Modify: `puzzlefind/recognize.py:201-214`（`recognize_sweep`）
- Modify: `puzzlefind/config.py`
- Test: `tests/test_recognize.py`

**Interfaces:**
- Produces: `config.SWEEP_EARLY_EXIT_CONFIDENCE: float = 0.99`；`recognize_sweep` 行为不变，只是提前收工

**依据：** 调参日志 real1.jpg 那条记着「9 次穷举命中里 5 次置信度 ≥0.99，但代码仍把剩余角度跑完」。做完 Task 4 之后这条杠杆的价值已经缩水（多数碎片走不到全量穷举了），但对 det 无框的那几块仍有效，且改动只有三行。

- [ ] **Step 1: 写失败的测试**

在 `tests/test_recognize.py` 的 `TestRecognizeSweep` 类里追加：

```python
    def test_stops_early_on_a_very_confident_hit(self, blank_crop):
        from puzzlefind import config

        responses = [
            [],
            [RawDetection("B-403", config.SWEEP_EARLY_EXIT_CONFIDENCE)],
            [RawDetection("A-111", 1.0)],       # 不该跑到这里
        ]
        backend = FakeBackend(responses)
        result = recognize.recognize_sweep(backend, blank_crop)
        assert result.code == "B-403"
        assert backend.calls == 2, "拿到 0.99 之后仍把剩余角度跑完了"

    def test_keeps_sweeping_when_confidence_stays_below_the_exit_bar(self, blank_crop):
        from puzzlefind import config

        responses = [[RawDetection("B-403", 0.95)]]
        backend = FakeBackend(responses)
        recognize.recognize_sweep(backend, blank_crop)
        assert backend.calls == len(config.SWEEP_ANGLES)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_recognize.py::TestRecognizeSweep -v`
Expected: `test_stops_early_on_a_very_confident_hit` FAIL，实际 `backend.calls == 12`

- [ ] **Step 3: 补全配置项注释**

把 Task 4 临时加的那行替换为（放在 `SWEEP_CONFIDENCE_THRESHOLD` 之后）：

```python
# 穷举时拿到不低于此置信度的合法编号就收工，不再试剩余角度。
# 依据 real1.jpg 实测：9 次穷举命中里 5 次置信度 ≥0.99，剩余角度是白跑的。
# 设成 0.99 而不是更低，是因为「命中合法词表」这个判据虽硬但不是绝对——
# 留一点余量，让明显更好的读数还有机会翻盘。
SWEEP_EARLY_EXIT_CONFIDENCE: float = 0.99
```

- [ ] **Step 4: 实现**

把 `recognize_sweep` 的循环体改成：

```python
    best = _NO_RESULT
    for angle in config.SWEEP_ANGLES:
        rotated = rotate_expand(crop, angle)
        candidate = _best_snapped(backend.read(rotated), method="sweep", angle=angle)
        if candidate.code is not None and candidate.confidence > best.confidence:
            best = candidate
        # 提前退出：这么高的置信度再试下去也翻不了盘，而每个角度都要跑一遍
        # 最贵的检测模型
        if best.code is not None and best.confidence >= config.SWEEP_EARLY_EXIT_CONFIDENCE:
            break
    return best
```

- [ ] **Step 5: 跑测试确认通过**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_recognize.py -v -m "not ocr"`
Expected: 全 PASS。特别确认原有的 `test_tries_every_angle_when_nothing_hits` 与 `test_keeps_highest_confidence_across_angles` 仍绿（后者最高分 0.98 < 0.99，不触发提前退出）

- [ ] **Step 6: 跑全量测试**

Run: `.\.venv\Scripts\python.exe -m pytest tests\ -m "not ocr"`
Expected: 185 passed

- [ ] **Step 7: 提交**

```bash
git add puzzlefind/recognize.py puzzlefind/config.py tests/test_recognize.py
git commit -m "perf(recognize): stop the angle sweep once a read is confident enough

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: 把新路径接进 `recognize_piece`

**Files:**
- Modify: `puzzlefind/recognize.py:217-230`（`recognize_piece`）
- Test: `tests/test_recognize.py`

**Interfaces:**
- Consumes: Task 2 `best_poly`、Task 4 `recognize_line_sweep` / `LineOcrBackend`、Task 5 的提前退出
- Produces: `recognize_piece` 的四级降级链，签名不变

**降级链（顺序不能变）：**

1. Pass A 直读；置信度够 → 收工
2. 有检测框 **且** 后端支持 `read_line` → 按行识别；够了 → 收工
3. 否则 → 现有的全量角度穷举（已带提前退出）
4. 取三者中最好的

**第 3 步是识别率的下限保证**：新路径失败的碎片一个不少地回到老路上，所以识别率**只可能持平或变好**，不可能变差。这是本方案敢承诺「零精度损失」的唯一依据，不许为了提速把它砍掉。

- [ ] **Step 1: 写失败的测试**

在 `tests/test_recognize.py` 的 `TestRecognizePiece` 类里追加：

```python
    def test_low_confidence_direct_uses_the_line_path_when_a_quad_is_available(
        self, blank_crop
    ):
        backend = FakeLineBackend(
            [[RawDetection("B-403", 0.50, QUAD)]],      # Pass A：分低，但有框
            [RawDetection("B-403", 0.98)],              # 按行识别：读准了
        )
        result = recognize.recognize_piece(backend, blank_crop)
        assert result.method == "line"
        assert result.code == "B-403"
        assert backend.calls == 1, "不该再跑一遍昂贵的全量穷举"
        # 0.98 低于 SWEEP_EARLY_EXIT_CONFIDENCE(0.99)，所以两个朝向都试了——
        # 这是对的。够格收工的门槛是 SWEEP_CONFIDENCE_THRESHOLD(0.90)，
        # 它决定的是「要不要回退到全量穷举」，与提前退出是两回事。
        assert backend.line_calls == 2

    def test_falls_back_to_the_full_sweep_when_the_line_path_fails(self, blank_crop):
        backend = FakeLineBackend(
            [
                [RawDetection("B-403", 0.50, QUAD)],   # Pass A
                [RawDetection("B-403", 0.96)],         # 全量穷举的第一个角度
            ],
            [RawDetection("QWERTY", 0.99)],            # 按行识别：吸附不上
        )
        result = recognize.recognize_piece(backend, blank_crop)
        assert result.method == "sweep"
        assert result.code == "B-403"
        assert backend.line_calls == 2, "回退前应把两个朝向都试过"
        assert backend.calls > 1

    def test_falls_back_to_the_full_sweep_when_there_is_no_quad(self, blank_crop):
        backend = FakeLineBackend(
            [
                [RawDetection("B-403", 0.50)],         # Pass A：分低且**没有框**
                [RawDetection("B-403", 0.96)],
            ],
            [RawDetection("B-403", 0.99)],
        )
        result = recognize.recognize_piece(backend, blank_crop)
        assert result.method == "sweep"
        assert backend.line_calls == 0, "没有框就不该走按行识别"

    def test_backend_without_read_line_still_works(self, blank_crop):
        """不实现可选协议的后端必须照常工作——这是 spec 6 的架构承诺。"""
        backend = FakeBackend(
            [
                [RawDetection("B-403", 0.50, QUAD)],
                [RawDetection("B-403", 0.96)],
            ]
        )
        result = recognize.recognize_piece(backend, blank_crop)
        assert result.method == "sweep"
        assert result.code == "B-403"

    def test_high_confidence_direct_still_skips_everything(self, blank_crop):
        backend = FakeLineBackend(
            [[RawDetection("B-403", 0.99, QUAD)]], [RawDetection("A-111", 1.0)]
        )
        result = recognize.recognize_piece(backend, blank_crop)
        assert result.method == "direct"
        assert backend.calls == 1
        assert backend.line_calls == 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_recognize.py::TestRecognizePiece -v`
Expected: 新增的前三条 FAIL（`method` 是 `"sweep"` 而非 `"line"`，`line_calls == 0`）

- [ ] **Step 3: 实现**

把 `recognize_piece` 整个替换为：

```python
def recognize_piece(backend: OcrBackend, crop: np.ndarray) -> RecogResult:
    """完整识别流程，四级降级。

    1. Pass A 直读。置信度够就收工——合格照片上这条吃下约 2/3 的碎片。
    2. Pass A 拿到了检测框、且后端支持按行识别 → 只重跑 rec 模型试正反两个朝向。
       这是本管线最大的提速来源：det 占单次识别 92% 的耗时，而框在第 1 步
       就已经有了，没必要为 12 个角度重跑 12 遍检测。
    3. 上面都不行 → 回退到全量角度穷举（det 完全没给出框时唯一的出路）。
    4. 取三者里最好的。

    第 3 步是识别率的下限保证：走不通快路径的碎片一个不少地回到老路上，
    所以这次改造**只可能持平或变好**。不要为了提速把它砍掉。
    """
    detections = backend.read(crop)
    best = _best_snapped(detections, method="direct", angle=0)
    if best.code is not None and best.confidence >= config.SWEEP_CONFIDENCE_THRESHOLD:
        return best

    quad = best_poly(detections)
    if quad is not None and isinstance(backend, LineOcrBackend):
        line = recognize_line_sweep(backend, crop, quad)
        if line.code is not None and line.confidence > best.confidence:
            best = line
        if best.code is not None and best.confidence >= config.SWEEP_CONFIDENCE_THRESHOLD:
            return best

    sweep = recognize_sweep(backend, crop)
    if sweep.code is not None and sweep.confidence > best.confidence:
        best = sweep
    return best
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_recognize.py -v -m "not ocr"`
Expected: 全 PASS，含原有的三条 `TestRecognizePiece` 测试

- [ ] **Step 5: 跑全量测试**

Run: `.\.venv\Scripts\python.exe -m pytest tests\ -m "not ocr"`
Expected: 190 passed

- [ ] **Step 6: 确认没把 PaddleOCR 拽进非 ocr 测试**

Run: `.\.venv\Scripts\python.exe -m pytest tests\ -m "not ocr" -p no:cacheprovider --tb=no -q`
然后：
```bash
.\.venv\Scripts\python.exe -c "import subprocess,sys; subprocess.run([sys.executable,'-m','pytest','tests/','-m','not ocr','-q'],check=True); import sys as s; print('paddle 系模块:', [m for m in s.modules if 'paddle' in m.lower()] or 'NONE')"
```
Expected: `paddle 系模块: NONE`

- [ ] **Step 7: 提交**

```bash
git add puzzlefind/recognize.py tests/test_recognize.py
git commit -m "feat(recognize): route Pass C through the rec-only path when a quad exists

四级降级：直读 -> 按行识别 -> 全量穷举。第三级是识别率的下限保证。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: `PaddleBackend` 实现 `read_line`

**Files:**
- Modify: `puzzlefind/recognize.py`（`PaddleBackend`）
- Modify: `puzzlefind/config.py`
- Test: `tests/test_recognize.py`（`@pytest.mark.ocr`）

**Interfaces:**
- Consumes: Task 4 的 `LineOcrBackend` 协议
- Produces: `PaddleBackend.read_line(image) -> RawDetection`；`config.PADDLE_REC_MODEL_NAME: str = "PP-OCRv6_medium_rec"`

**两个必须注意的点：**
1. rec 模型名**必须与主管线选用的一致**（本机实测是 `PP-OCRv6_medium_rec`），否则按行识别的读数会与 Pass A 系统性不一致。
2. `TextRecognition` 构造失败（模型名对不上、版本变动）时**不许炸**——把 `read_line` 变成不可用并降级到全量穷举即可。这是一个纯提速特性，不该有能力让整轮建索引失败。

- [ ] **Step 1: 写失败的测试**

在 `tests/test_recognize.py` 的 `TestPaddleBackendIntegration` 类里追加：

```python
    def test_read_line_reads_an_already_upright_line(self):
        import cv2

        line = np.full((60, 300, 3), 245, dtype=np.uint8)
        cv2.putText(
            line, "B-403", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (90, 90, 90), 3
        )
        backend = recognize.PaddleBackend()
        detection = backend.read_line(line)
        assert detection.text.replace(" ", "") == "B-403"
        assert detection.score > 0.5

    def test_paddle_backend_satisfies_the_optional_line_protocol(self):
        backend = recognize.PaddleBackend()
        assert isinstance(backend, recognize.LineOcrBackend)

    def test_full_read_carries_the_detection_quad(self):
        import cv2

        image = np.full((160, 480, 3), 245, dtype=np.uint8)
        cv2.putText(
            image, "B-403", (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (90, 90, 90), 6
        )
        backend = recognize.PaddleBackend()
        detections = backend.read(image)
        assert detections
        assert detections[0].poly is not None
        assert len(detections[0].poly) == 4
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_recognize.py::TestPaddleBackendIntegration -v`
Expected: 前两条 FAIL（`AttributeError: 'PaddleBackend' object has no attribute 'read_line'`）。第三条应当已经通过——Task 2 就做完了。

- [ ] **Step 3: 加配置项**

在 `puzzlefind/config.py` 的「PaddleOCR 运行时」一节追加：

```python
# 按行识别（跳过检测模型）用的 rec 模型名。**必须与 PaddleOCR 主管线实际
# 选用的一致**，否则按行识别的读数会与 Pass A 系统性不一致。
# 本机 paddleocr 3.7.0 实测主管线加载的是 PP-OCRv6_medium_rec。
# 换 PaddleOCR 版本后先跑 scripts/probe_paddleocr.py 确认这个名字还对。
PADDLE_REC_MODEL_NAME: str = "PP-OCRv6_medium_rec"
```

- [ ] **Step 4: 实现**

把 `PaddleBackend` 改成：

```python
class PaddleBackend:
    """PaddleOCR 3.x 后端。模型惰性加载——首次 read 时才初始化。

    同时实现可选的 LineOcrBackend：单独持有一个 rec 模型，用于跳过检测
    直接识别已摆正的文字行。
    """

    def __init__(self, lang: str = "en") -> None:
        self._lang = lang
        self._ocr = None
        self._rec = None
        self._rec_unavailable = False

    def _ensure_loaded(self) -> None:
        ...   # 原样保留，不动

    def read(self, image: np.ndarray) -> list[RawDetection]:
        ...   # 原样保留，不动

    def _ensure_rec_loaded(self) -> None:
        """惰性加载单独的 rec 模型。加载失败就永久标记不可用。

        故意不让它抛出去：按行识别是纯提速特性，模型名对不上时应当安静地
        降级回全量角度穷举，而不是让整轮建索引崩掉。
        """
        if self._rec is not None or self._rec_unavailable:
            return
        prepare_paddle_env()
        try:
            from paddleocr import TextRecognition

            self._rec = TextRecognition(
                model_name=config.PADDLE_REC_MODEL_NAME,
                enable_mkldnn=config.PADDLE_ENABLE_MKLDNN,
            )
        except Exception:
            self._rec_unavailable = True

    def read_line(self, image: np.ndarray) -> RawDetection:
        """只跑识别模型。输入必须是已摆正的水平文字行。

        返回结构与 read 不同：这里只有一条结果，键名也是单数
        （实测 TextRecognition 返回 `rec_text` / `rec_score`，
        而完整管线返回 `rec_texts` / `rec_scores`）。
        """
        self._ensure_rec_loaded()
        if self._rec is None:
            return RawDetection("", 0.0)
        try:
            results = self._rec.predict(image)
        except Exception:
            return RawDetection("", 0.0)
        for res in results:
            payload = getattr(res, "json", None)
            if isinstance(payload, dict):
                payload = payload.get("res", payload)
            if not isinstance(payload, dict):
                continue
            text = payload.get("rec_text")
            score = payload.get("rec_score")
            if isinstance(text, str):
                return RawDetection(text, float(score or 0.0))
        return RawDetection("", 0.0)
```

- [ ] **Step 5: 跑 OCR 集成测试确认通过**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_recognize.py -v`
Expected: 全 PASS（含 ocr 标记的 4 条）

- [ ] **Step 6: 跑全量测试**

Run: `.\.venv\Scripts\python.exe -m pytest tests\`
Expected: 194 passed（非 ocr 190 + ocr 4）

- [ ] **Step 7: 提交**

```bash
git add puzzlefind/recognize.py puzzlefind/config.py tests/test_recognize.py
git commit -m "feat(recognize): implement read_line on PaddleBackend

rec 模型单独持有。加载失败时安静降级到全量穷举，不让提速特性有能力
炸掉整轮建索引。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: 复测、对照、更新文档

**Files:**
- Modify: `docs/tuning-log.md`
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-08-03-puzzle-piece-finder.md`
- Modify: `scripts/calibrate.py`

**Interfaces:**
- Consumes: Task 1 的基线数字、Task 7 完工的代码

- [ ] **Step 1: 复跑基准**

Run: `.\.venv\Scripts\python.exe scripts\benchmark.py data\photos\real6.jpg`

**验收门槛（两条都必须满足）：**
- `recognized` **不低于 Task 1 记下的基线值**。低了就是识别率回归，必须停下来查，不许「先合了再说」。
- `seconds` 不高于基线的 **40%**。

新的 `methods` 里应当出现 `"line"` 这一项。

- [ ] **Step 2: 逐块比对读数，确认没有静默的错读**

创建 `scratchpad` 下的临时脚本（**不要提交**）：

```python
import json
from pathlib import Path

before = json.loads(Path("debug/real6.jpg/benchmark-before.json").read_text(encoding="utf-8"))
after = json.loads(Path("debug/real6.jpg/benchmark.json").read_text(encoding="utf-8"))

lost = sorted(set(before["codes"]) - set(after["codes"]))
gained = sorted(set(after["codes"]) - set(before["codes"]))
print(f"改造前 {before['recognized']}/{before['total']}  {before['seconds']}s")
print(f"改造后 {after['recognized']}/{after['total']}  {after['seconds']}s")
print(f"提速 {before['seconds'] / after['seconds']:.2f}x")
print(f"丢失的编号: {lost or '无'}")
print(f"新增的编号: {gained or '无'}")
```

跑之前先把 Task 1 那份基准复制成 `debug/real6.jpg/benchmark-before.json`。

**「丢失的编号」非空时必须逐个查清楚**：把该碎片的裁剪图和检测框画出来看。可能是真的回归，也可能是它换了个更高置信度的读数从而在唯一性消解里换了赢家——后者无害，但必须确认是哪一种，不许猜。

- [ ] **Step 3: 让标定脚本认识新的 method**

在 `scripts/calibrate.py` 里找到统计 `method` 的地方，把 `"line"` 与 `direct` / `sweep` / `conflict` 并列打印出来。改完跑一次确认输出正常：

Run: `.\.venv\Scripts\python.exe scripts\calibrate.py data\photos\real6.jpg`

- [ ] **Step 4: 往调参日志追加一条**

在 `docs/tuning-log.md` 末尾追加（**用 Step 1/2 的真实数字填，不要照抄示例**）：

```markdown
### 2026-08-04 — real6.jpg（OCR 提速改造，前后对照）

**拍摄条件**：1086×1448，深色纯背景，50 块摊开留缝。

**分割**：前景占比 24.1%，连通块 50，切分后 50，面积中位 7929 px²。**分割耗时 12 ms**——
整条管线里可以忽略不计的那一部分。

**改造前 / 改造后**
| | 耗时 | s/块 | 识别 | direct | line | sweep | conflict |
|---|---|---|---|---|---|---|---|
| 前 | <填> | <填> | <填> | <填> | — | <填> | <填> |
| 后 | <填> | <填> | <填> | <填> | <填> | <填> | <填> |

**改了什么**：Pass C 从「整图旋转 12 次 × 完整 predict」改为「按 Pass A 的
检测框裁出文字行 → 只重跑 rec 模型试正反两个朝向」。

**为什么这么快**：单次完整 predict 的 1.71 s 里，**检测模型占 1.485 s（92%）**，
识别模型只占 0.129 s。检测框在 Pass A 就已经拿到了，重跑 12 遍检测纯属浪费。
实测 Pass C 从 20.5 s 降到 0.265 s，**77 倍**。

**实现上的坑**：透视裁剪出的文字行若 高/宽 ≥ 1.5，说明检测框点序把长短边判反了
（文字竖排时必然发生），**必须补一次 rot90**。第一轮漏掉这步，90°/270° 的碎片
全部读出 89/169/382 这类垃圾。这一步 PaddleOCR 内部的 get_rotate_crop_image 有。

**量过但无效的杠杆（别再试）**
- 批量 `predict([...])`：**1.00x**，Paddle 在这里没做真批处理
- 关 `use_textline_orientation`：1.02x
- 调 `text_det_limit_side_len`（320 / max-736）：0.95x，反而略慢
- 重开 oneDNN：`FLAGS_enable_pir_api=0` 和 `FLAGS_enable_pir_in_executor=0` 都**绕不过**
  `ConvertPirAttribute2RuntimeAttribute`，config.py 里那条注释是对的
- GPU：`paddle.is_compiled_with_cuda() == False`，装的是 CPU 版轮子
- PP-OCRv5_mobile：确实快 3.54x，但**有精度代价**（6 块里读出 B53 / B-55 / 1 块读不出），
  未采用

**还没动的杠杆**
- 多进程：单进程实测占 7.9 / 16 逻辑核，还剩一半空闲，估计还有 ~1.7x
- 彻底绕开 det（用 OpenCV 自己定位那行浅灰字）：理论上 13x，但是新的调参战场。
  spec §5.4 当初否决过「自行估计文字角度」，理由是调参成本高——**那个否决的前提是
  「det 免费」，而现在知道 det 占 92%，前提已经变了**，值得重新评估。

**本次调整的参数**
- 新增 `LINE_DESKEW_ROTATE_RATIO = 1.5`
- 新增 `LINE_ORIENTATIONS = (0, 180)`（实测再加 90/270 两个朝向准确率一模一样，纯浪费）
- 新增 `SWEEP_EARLY_EXIT_CONFIDENCE = 0.99`
- 新增 `PADDLE_REC_MODEL_NAME = "PP-OCRv6_medium_rec"`
```

- [ ] **Step 5: 更新 README**

`README.md` 需要改四处：
1. 开头的实测段：把「建索引耗时约 9 分钟」换成新数字。
2. 「拍照要求」表里「编号尽量朝正」那一行的理由：现在歪的碎片走的是**便宜的**按行识别，不再是 12 次昂贵穷举，所以摆正的收益变小了。据实改写，别留旧说法。
3. 「调参」一节末尾那条「想提速的杠杆是拍照时把碎片摆正」已经过时——改成指向本次改造，并把「量过但无效的杠杆」清单摘要过去。
4. 第 149 行「除**两个**标了 `@pytest.mark.ocr` 的测试外」是笔误——实测全仓只有 `TestPaddleBackendIntegration` 一个 ocr 类。本方案的 Task 7 给它加了 3 条，改完是 4 条，据实写成「除 `TestPaddleBackendIntegration` 里那几条标了 `@pytest.mark.ocr` 的测试外」。

- [ ] **Step 6: 更新 spec**

`docs/superpowers/specs/2026-08-03-puzzle-piece-finder.md` 改三处：
1. §7 数据模型契约里 `method` 的取值加上 `"line"`：
   `method  "direct" | "line" | "sweep" | "conflict" | "none"`
2. §5.4 追加一段执行注记，说明 Pass C 的实现已经改成「复用检测框 + 只重跑 rec」，
   并记下 det 占 92% 这个测量结果。
3. §11 决策日志追加一行：
   `| 2026-08-04 | OCR 提速：实测 det 占单次识别 92% 耗时，Pass C 改为复用 Pass A 的检测框、只重跑 rec 模型。<填实测倍数>x，识别率<持平/提升>。批量推理、关方向分类、调检测输入尺寸、重开 oneDNN、GPU 均已实测无效 |`

- [ ] **Step 7: 跑全量测试**

Run: `.\.venv\Scripts\python.exe -m pytest tests\`
Expected: 194 passed

- [ ] **Step 8: 提交**

```bash
git add docs/tuning-log.md README.md docs/superpowers/specs/2026-08-03-puzzle-piece-finder.md scripts/calibrate.py
git commit -m "docs: record the OCR speedup measurements on real6.jpg

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## 完工验收清单

- [ ] `.\.venv\Scripts\python.exe -m pytest tests\` 全绿（预期 194 passed = 非 ocr 190 + ocr 4）
- [ ] 非 ocr 测试跑完后 `sys.modules` 里没有 paddle 系模块
- [ ] `real6.jpg` 上识别块数 **不低于** Task 1 的基线
- [ ] `real6.jpg` 上耗时 **不高于** 基线的 40%
- [ ] `benchmark.json` 的 `methods` 里出现 `"line"`
- [ ] 不实现 `read_line` 的后端仍能跑通（由 `test_backend_without_read_line_still_works` 覆盖）
- [ ] `docs/tuning-log.md` 里的数字是实测粘贴的，不是从方案里抄的示例值

---

## Self-Review

**1. 覆盖检查**

| 目标 | 落在哪个任务 |
|---|---|
| 杠杆 1：复用检测框，只重跑 rec | Task 2（带出框）+ Task 3（裁剪摆正）+ Task 4（按行识别）+ Task 6（接线）+ Task 7（真后端） |
| 杠杆 2：穷举提前退出 | Task 5 |
| 识别率不回归 | Task 6 第 3 级降级 + Task 8 Step 1/2 的逐块比对 |
| 前后有对照数字 | Task 1（改前量）+ Task 8（改后量） |
| spec §6 架构约束不被破坏 | Task 4 独立 Protocol + Task 6 `test_backend_without_read_line_still_works` |
| 参数集中在 config.py | Task 3/4/5/7 各自的「加配置项」步骤 |

**2. 类型一致性**

- `RawDetection(text, score, poly=None)` —— Task 2 定义，Task 4/6/7 使用，字段名一致
- `best_poly(list[RawDetection]) -> list[list[int]] | None` —— Task 2 定义，Task 6 使用
- `deskew_quad(image, quad) -> np.ndarray` —— Task 3 定义，Task 4 使用
- `read_line(image) -> RawDetection`（单数，非列表）—— Task 4 协议定义，Task 7 实现，测试里的 `FakeLineBackend` 也返回单个 `RawDetection`，一致
- `method="line"` —— Task 4 产出，Task 6 断言，Task 8 更新 spec 契约与 calibrate.py，四处一致
- `SWEEP_EARLY_EXIT_CONFIDENCE` —— Task 4 先临时加入、Task 5 补全注释与测试；Task 4 的步骤里已显式提示这一点，不会出现引用未定义常量

**3. 顺序依赖**

Task 1 必须最先（基线一旦被改动污染就取不回来了）。Task 2 → 3 → 4 → 6 是硬依赖链。Task 5 与 Task 4 无依赖，可互换。Task 7 必须在 Task 4 之后（要实现那个协议）。Task 8 最后。

**4. 一处刻意的取舍**

Task 6 在按行识别失败后仍会跑完整的 12 角度穷举，所以**最坏情况比改造前还慢一点**（多了两次 rec 调用，约 0.26 s）。这是有意的：它换来「识别率有下限保证」这个承诺。真实照片上走到这一步的碎片是少数，整体仍然大幅净赚。
