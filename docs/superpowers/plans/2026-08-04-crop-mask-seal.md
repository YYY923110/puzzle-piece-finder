# 裁剪掩膜补缝 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> ## ⚠️ 执行结果：本方案的做法被实测否决，已改用别的实现
>
> Task 1 的「闭运算补缝」在真实照片上**造成回归**：`real6` 从 50/50 掉到 49/50，
> 且 Pass A 命中率整体塌陷（`real6` direct 40→25、`real3` direct 32→12），
> `real3` 耗时 131 s → 644 s。根因是量错了几何——缝的口子就是**凹口**的口子，
> 核大到够得着字符时，凹口连同深色背景一起被灌进裁剪图。
>
> **最终采用**：`crop_piece` 只把**邻块**涂灰，不再遮自己（无参数，正确性由构造保证）。
> 实测 `IMG` 22/23 → **23/23**、`real3` 42/50 持平、`real6` 50/50 持平。
>
> 完整数据、被否决的四条路（闭运算、凸包、改 `build_mask`、膨胀 0.06）见
> [docs/tuning-log.md](../../tuning-log.md) 的 2026-08-04 条目。
> **下面的 Task 1 步骤保留作历史记录，不要照着执行。**

**Goal:** 让 `crop_piece` 不再把碎片自己的编号抹掉——填充掩膜前先封住轮廓上从边缘钻进内部的细缝。

**Architecture:** `crop_piece` 用轮廓填充出的掩膜，把轮廓外的像素替换成中性灰。当外轮廓存在「内凹缝隙」时，缝隙内部（可能正压着编号）会被当成轮廓外而填灰。修法是在填充前对该掩膜做一次闭运算，核尺寸按碎片包围盒长边等比缩放。改动范围严格限制在 `segment.crop_piece` 及其新辅助函数，`build_mask` / `find_blobs` / `split_blob` 一律不动。

**Tech Stack:** Python 3.13、OpenCV (`opencv-python-headless>=5.0`)、numpy、pytest。

## Global Constraints

- 全本地运行，不联网、不调大模型。
- 所有可调参数集中在 `puzzlefind/config.py`，不得散落到其他模块。
- `recognize.OcrBackend` 协议不变；本次改动完全不碰识别层。
- 除 `@pytest.mark.ocr` 标记的测试外，测试**不得加载 PaddleOCR**；分割测试只跑 `tests/conftest.py` 里的合成图。
- 真实照片不进版本库。
- 所有命令走项目自带解释器：`D:\ocr_claude\.venv\Scripts\python.exe`。

## 根因（本次修的是什么）

实测 `IMG_20260805_082927.jpg`（3072×4096，23 块）：

- 分割**完全正确**——23 个连通块 → 23 个轮廓，分水岭切分一次都没触发。
- 其中一块的**外轮廓**从底部凹口钻进碎片内部、绕着印刷字符走了一圈再出来。成因是深色印刷字符低于 Otsu 阈值、在掩膜上是洞，而该字符离凹口阴影足够近，5×5 形态学运算把洞和背景连通了，于是 `RETR_EXTERNAL` 把它当成了外部入口。
- `crop_piece` 按轮廓填灰，最后一位数字被抹掉一半 → OCR 读出 `D-79` → 吸附成 `D-079` → 被 `resolve` 的离群规则剔除 → 该块显示为未识别。

用真实 PaddleOCR 后端验证过修法有效：

```
块10 [现状 crop_piece] -> code=D-079 raw='D-79'  conf=0.983  (随后被离群剔除)
块10 [闭运算修复后]     -> code=D-797 raw='D-797' conf=1.000  method=direct
```

## 被否决的替代方案（勿重新提议）

| 方案 | 否决理由 |
|---|---|
| 用 `cv2.convexHull(contour)` 代替轮廓做填充掩膜 | 实测凸包把掩膜面积撑大 **+34%**（闭运算只 +0.3~1.8%）。凸包会填平全部凹口，把邻块像素连同其编号一起放进裁剪图——正是掩膜设计初衷要防的事。碎片摆密时必然误读。 |
| 改 `build_mask`，在源头阻止字符与背景连通（如更大的闭运算核、或先做 hole filling） | 影响面过大。`build_mask` 的输出直接决定连通块面积，而 `MIN_AREA_RATIO` / `SPLIT_AREA_RATIO` / `unit_piece_area` 全部依赖面积分布，这套参数是拿三张实拍照片标定出来的（见 `docs/tuning-log.md`）。为一个只在填充阶段显现的缺陷去动它，回归风险远大于收益。缝隙是从背景连通进来的，hole filling 本来也补不上。 |
| 定值闭运算核（如 41 px） | 实测碎片长边跨度极大：`IMG_20260805` 是 608 px，`real3`/`real6` 只有 120~132 px。41 px 的核在 `real3` 上会把拼图凹口（约占长边 15%，即 ~18 px）整个填平，等于退化成凸包。必须按比例。 |
| 在整图掩膜上做闭运算（本方案初稿的写法） | 实测 12 MP 照片上 `crop_piece` 从 130 ms/块拖到 **922 ms/块（7.1x）**——单块碎片的开销不该随图像总面积增长。改为在包围盒外扩一个核宽的 ROI 内计算，结果完全一致（形态学只看核半径内的邻域），实测降到 66.6 ms/块，比改动前还快一倍。 |
| 放宽 `resolve` 的离群规则，让 `D-079` 不被剔除 | 治标且有害。离群规则这次**判对了**——`D-079` 确实是误读。放宽它只会让错误编号混进索引，比未识别更糟。 |

---

### Task 1: 填充掩膜补缝

**Files:**
- Modify: `puzzlefind/config.py`（在「裁剪」段末尾追加参数）
- Modify: `puzzlefind/segment.py:247-283`（新增 `seal_mask_inlets`，`crop_piece` 调用它）
- Test: `tests/conftest.py`（新增 fixture）、`tests/test_segment.py`（`TestCropPiece` 内新增两条测试）

**Interfaces:**
- Consumes: `config.CROP_MASK_SEAL_RATIO: float`、`segment.contour_bbox(contour) -> tuple[int,int,int,int]`
- Produces: `segment.seal_mask_inlets(mask: np.ndarray, long_edge: int) -> np.ndarray`（Task 2 不调用它，只跑整条管线）

- [ ] **Step 1: 加 fixture——复现「编号紧挨凹口」的碎片**

在 `tests/conftest.py` 末尾追加：

```python
@pytest.fixture
def piece_with_code_beside_notch() -> tuple[np.ndarray, int]:
    """一块碎片，深色编号紧挨底边凹口——掩膜上会形成一条从背景钻进内部的缝。

    这是 IMG_20260805_082927.jpg 上 D-797 那一块的最小复现：印刷字符低于
    Otsu 阈值、在掩膜上是洞，而它离凹口阴影足够近，形态学运算把洞和背景
    连通了，外轮廓于是从凹口钻进碎片内部绕字符一圈。crop_piece 按轮廓
    填灰时，正好把半个编号抹掉。

    返回 (图像, 编号的深色像素数)。
    """
    canvas = make_canvas(400, 400)
    cv2.rectangle(canvas, (100, 100), (300, 300), (235, 233, 228), thickness=-1)
    # 底边凹口：宽 10 px，从碎片底边一直伸到「编号」下沿
    cv2.rectangle(canvas, (195, 240), (205, 301), (20, 20, 20), thickness=-1)
    # 「编号」：紧贴凹口顶端的深色横条
    cv2.rectangle(canvas, (180, 225), (230, 239), (20, 20, 20), thickness=-1)
    return canvas, 50 * 15
```

- [ ] **Step 2: 写失败的测试**

在 `tests/test_segment.py` 的 `class TestCropPiece` 内追加两条：

```python
    def test_code_beside_a_notch_is_not_erased_by_the_mask(
        self, piece_with_code_beside_notch
    ):
        """轮廓从凹口钻进内部时，缝隙里的编号不能被填成灰色。

        实测回归：IMG_20260805_082927.jpg 的 D-797 被抹掉最后一位，
        读成 D-79，吸附成 D-079 后被离群规则剔除，整块显示为未识别。
        """
        image, _ = piece_with_code_beside_notch
        contour = segment.extract_contours(image)[0]
        crop = segment.crop_piece(image, contour)
        # 轮廓外一律是中性灰(128)，所以裁剪图里的深色像素只可能来自编号本身
        dark = int((crop.reshape(-1, 3).max(axis=1) < 90).sum())
        assert dark > 200

    def test_sealing_does_not_fill_puzzle_notches(self, separated_puzzle_pieces):
        """补缝的核不能大到把拼图凹口一并填平——那等于退化成凸包。

        实测凸包会把掩膜面积撑大 34%，把邻块的编号一起吃进裁剪图。
        """
        image, _ = separated_puzzle_pieces
        contour = segment.extract_contours(image)[0]
        _, _, width, height = segment.contour_bbox(contour)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
        sealed = segment.seal_mask_inlets(mask, max(width, height))
        before = int(mask.sum())
        gain = (int(sealed.sum()) - before) / before
        assert gain < 0.05
```

- [ ] **Step 3: 跑测试，确认它们失败**

Run: `D:\ocr_claude\.venv\Scripts\python.exe -m pytest tests\test_segment.py -k "notch" -v`
Expected: 两条都 FAIL——第一条断言深色像素数为 0（编号被填灰），第二条报 `AttributeError: module 'puzzlefind.segment' has no attribute 'seal_mask_inlets'`。

若第一条意外 PASS，说明合成图上的洞与背景没连通、缝隙没形成，fixture 不成立：把凹口矩形加宽 2 px 再试，直到 `segment.extract_contours` 出来的轮廓确实带内凹（可用 `cv2.contourArea(contour)` 明显小于 `cv2.contourArea(cv2.convexHull(contour))` 判断）。

- [ ] **Step 4: 加参数**

在 `puzzlefind/config.py` 的「裁剪」段末尾（`CROP_FILL_COLOR` 之后）追加：

```python
# 填充掩膜前「补缝」用的闭运算核尺寸，取碎片包围盒长边的这个比例。
# 深色印刷字符低于 Otsu 阈值，在掩膜上是洞；字符离碎片凹口够近时，
# 形态学运算会把洞和背景连通，外轮廓便从凹口钻进碎片内部绕字符一圈，
# 填灰时正好抹掉半个编号。实测 IMG_20260805_082927.jpg 的 D-797 被读成
# D-79，吸附成 D-079 后被离群规则剔除。
# 闭运算只填「窄于核」的缺口，所以这个值必须夹在两个实测边界之间：
# **下界**——缝的开口比想象中宽。IMG_20260805 那道缝要 k=37 才封住，
#   即 5.9% 长边（k=29 时只补回 0.68%，k=37 跳到 1.69% 并饱和）。
# **上界**——0.15 起拼图凹口开始被填平：三张照片上的面积增益从
#   最大 4.4%（0.12）跳到最大 14.1%（0.15），0.20 时已达 20%，
#   而凸包（把凹口全填平）是 30%。凹口一旦填平，邻块的编号就会
#   混进裁剪图，正是掩膜要防的事。
# 取 0.12：是最大实测缝宽的两倍，面积增益中位 2.8~3.3%、最大 4.4%。
# **必须按比例而非定值**：real3 的碎片长边只有 120 px，IMG_20260805 是
# 608 px，定值 75 会把前者整个抹平。
CROP_MASK_SEAL_RATIO: float = 0.12
```

> 初稿定的 0.07 是拍脑袋的，执行时量了才发现只比实测缝宽（5.9%）高一点点，
> 合成图上就没封住（`assert 0 > 200` 变成 `assert 67 > 200`）。定参数前先量边界。

- [ ] **Step 5: 实现补缝函数并接进 crop_piece**

在 `puzzlefind/segment.py` 中 `crop_piece` **之前**插入：

```python
def seal_kernel_size(long_edge: int) -> int:
    """补缝闭运算的核尺寸（奇数）。见 config.CROP_MASK_SEAL_RATIO。"""
    return max(3, int(long_edge * config.CROP_MASK_SEAL_RATIO) | 1)


def seal_mask_inlets(mask: np.ndarray, long_edge: int) -> np.ndarray:
    """封住轮廓掩膜上从边缘钻进内部的细缝。

    闭运算只填「窄于核」的缺口：实测那道缝的开口约占碎片长边 6%，
    而拼图凹口约占 15~30%，中间留得下一个安全窗口，所以一次闭运算
    能补掉缝隙又保住凹口。核尺寸按 long_edge 等比缩放，见
    config.CROP_MASK_SEAL_RATIO 的注释。
    """
    kernel_size = seal_kernel_size(long_edge)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
```

在 `crop_piece` 里，把整图掩膜那三行

```python
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=-1)

    filled = np.full_like(bgr, config.CROP_FILL_COLOR, dtype=np.uint8)
    composited = np.where(mask[:, :, None] == 255, bgr, filled)
    crop = composited[y0:y1, x0:x1]
```

改成在 ROI 内计算（**不要**在整图上做闭运算，见「被否决的替代方案」）：

```python
    # 掩膜只在包围盒的一个小邻域里算。对整张 12 MP 的图做大核闭运算，
    # 实测把 crop_piece 从 130 ms/块拖到 922 ms/块（7.1x）——单块碎片
    # 的开销必须和图像总面积无关。外扩一个核宽，保证补缝结果与在整图
    # 上做的完全一致（形态学只看核半径内的邻域）。
    margin = seal_kernel_size(max(w, h))
    rx0, ry0 = max(0, x0 - margin), max(0, y0 - margin)
    rx1, ry1 = min(width, x1 + margin), min(height, y1 + margin)

    mask = np.zeros((ry1 - ry0, rx1 - rx0), dtype=np.uint8)
    shifted = contour - np.array([rx0, ry0], dtype=contour.dtype)
    cv2.drawContours(mask, [shifted], -1, 255, thickness=-1)
    mask = seal_mask_inlets(mask, max(w, h))
    mask = mask[y0 - ry0 : y1 - ry0, x0 - rx0 : x1 - rx0]

    region = bgr[y0:y1, x0:x1]
    filled = np.full_like(region, config.CROP_FILL_COLOR, dtype=np.uint8)
    crop = np.where(mask[:, :, None] == 255, region, filled)
```

并把 `crop_piece` docstring 里的第 1 条改为：

```
    1. 用轮廓做掩膜（填充前先补缝，见 seal_mask_inlets），把邻块的
       像素替换成中性灰——否则相邻碎片上的编号会混进这块的裁剪图，
       OCR 会读出两个编号。
```

- [ ] **Step 6: 跑测试，确认通过**

Run: `D:\ocr_claude\.venv\Scripts\python.exe -m pytest tests\test_segment.py -v`
Expected: 全部 PASS，含新增两条。

- [ ] **Step 7: 跑全量快速测试，确认没有回归**

Run: `D:\ocr_claude\.venv\Scripts\python.exe -m pytest tests\ -m "not ocr" -q`
Expected: 190 + 2 = 192 passed, 4 deselected。

顺带跑一次性能对照，确认补缝没有把 `crop_piece` 拖慢（把
`config.CROP_MASK_SEAL_RATIO` 临时设为 0.0 作对照）：12 MP 照片上
应在 70 ms/块以内，若到了几百毫秒，说明闭运算做在整图上了。

- [ ] **Step 8: 提交**

```bash
git add puzzlefind/config.py puzzlefind/segment.py tests/conftest.py tests/test_segment.py
git commit -m "fix(segment): seal contour inlets before masking the crop"
```

---

### Task 2: 真实照片验收与文档

**Files:**
- Create: 无
- Modify: `docs/tuning-log.md`（追加一条）、`README.md`（识别率数字与已知限制）
- Test: 无新增自动化测试——本任务的产出是实拍数据

**Interfaces:**
- Consumes: Task 1 改好的 `segment.crop_piece`
- Produces: 无代码接口

- [ ] **Step 1: 在三张实拍照片上重跑，与基线对比**

基线（改动前，已存档）：

| 照片 | 块数 | 识别 | 备注 |
|---|---|---|---|
| `IMG_20260805_082927.jpg` | 23 | 22（96%）| 1 次 conflict |
| `real3.jpg` | 50 | 42（84%）| 8 次 conflict |
| `real6.jpg` | 50 | 50（100%）| — |

依次运行（每张约 1~2 分钟）：

```powershell
D:\ocr_claude\.venv\Scripts\python.exe scripts\calibrate.py data\photos\IMG_20260805_082927.jpg
D:\ocr_claude\.venv\Scripts\python.exe scripts\calibrate.py data\photos\real3.jpg
D:\ocr_claude\.venv\Scripts\python.exe scripts\calibrate.py data\photos\real6.jpg
```

用这段脚本读出每张的识别率与方法分布：

```powershell
D:\ocr_claude\.venv\Scripts\python.exe -c "import json,glob;from collections import Counter;[print(d['photo_id'], len(d['pieces']), sum(1 for p in d['pieces'] if p['code']), dict(Counter(p['method'] for p in d['pieces']))) for f in glob.glob('debug/*/index.json') for d in [json.load(open(f,encoding='utf-8'))]]"
```

**验收标准：**
- `IMG_20260805_082927.jpg` 达到 23/23。
- `real6.jpg` 保持 50/50（**不允许下降**——这张是干净基线，一旦下降说明补缝伤到了正常块，回到 Task 1 把 `CROP_MASK_SEAL_RATIO` 调小到 0.05 重测）。
- `real3.jpg` 不低于 42/50。这张的 8 次 conflict 是否同源尚未确认，**提升是预期不是要求**；若无变化，照实记录，不要为了让数字变好看去动别的参数。

- [ ] **Step 2: 追加调参日志**

在 `docs/tuning-log.md` 末尾按模板追加一条 `### 2026-08-04 — 裁剪掩膜补缝`，必须写清：根因（轮廓内凹缝隙抹掉编号）、三张照片改动前后的识别率与 conflict 数、`CROP_MASK_SEAL_RATIO` 定为 0.07 的依据（面积增益中位 0.3~0.5%、最大 1.8%，凹口占长边 ~15%），以及被否决的三个替代方案（凸包 +34%、改 `build_mask` 回归风险、定值核在小图上填平凹口）。

- [ ] **Step 3: 更新 README**

把 README 开头「识别率 **84%**」一行按 Step 1 的实测结果改写；若 `real3` 数字有变化，同步「已知限制」里与之相关的描述。不要写没量过的数字。

- [ ] **Step 4: 提交**

```bash
git add docs/tuning-log.md README.md
git commit -m "docs: record the crop-mask seal measurements"
```
