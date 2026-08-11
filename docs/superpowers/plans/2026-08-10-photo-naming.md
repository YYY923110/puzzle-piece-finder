# 照片命名 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `photo_id` 由用户在上传时指定并跨重拍稳定，从而「照片 2」能对应桌上一片区域，且重拍同一区域是替换而非新增。

**Architecture:** 三层各改一处，互不耦合。`library.py` 新增一个纯函数校验 photo_id 能否安全当文件名；`server.py` 的上传接口收一个**可选**表单字段并调用它；前端在拉起文件选择器**之前**先问「这是哪片区域」，把名字随 FormData 发出。替换语义不需要新代码——`cv2.imwrite` 和 `write_text` 本来就是覆盖写，photo_id 一稳定就自动成立。

**Tech Stack:** Python 3.13 / FastAPI / pytest / 单文件原生 JS 前端（无框架、无构建）

## Global Constraints

- 识别管线（`pipeline.py` `segment.py` `recognize.py` `vocabulary.py` `resolve.py` `render.py` `cli.py`）**一行都不改**——这件事与识别无关。
- `photo_id` 校验**违反即抛错，绝不静默改写**。把 `2/3` 悄悄存成 `2_3` 会让用户以为存成了自己输入的名字，而名字正是这个功能的全部意义。
- `POST /api/photos` 的 `photo_id` 字段是**可选**的。不给时保持现有的 `Path(filename).stem or uuid` 行为，`test_server.py` 现有 8 条用例与 curl 上传都不受影响。
- 前端把 `photo_id` 插进 DOM 一律用 `textContent`，绝不用 `innerHTML`——它是用户可控字符串。页面里已有这条约定（见 `buildPhotoSwitch` 的注释）。
- 新增的 CSS 复用既有变量（`--table` `--ridge` `--chipboard` `--muted` `--target`），不引入新配色。页面配色刻意与引擎的 `OUTLINE_COLOR` / `UNKNOWN_OUTLINE_COLOR` 对齐。
- 运行测试一律用项目自带的解释器：`.\.venv\Scripts\python.exe -m pytest tests\ -m "not ocr"`。
- 提交信息用英文祈使句、小写 type 前缀（`feat:` / `fix:` / `docs:`），与现有 git log 一致。

## File Structure

| 文件 | 职责 | 本次改动 |
| --- | --- | --- |
| `puzzlefind/library.py` | 索引持久化；文件名约束的归属地 | 新增 `InvalidPhotoId` 异常与 `sanitize_photo_id()` |
| `puzzlefind/server.py` | HTTP 编解码与落盘 | `upload_photo` 收可选 `photo_id`，校验后使用 |
| `puzzlefind/static/index.html` | 单文件前端 | 上传前的区域选择面板 + 发送 `photo_id` |
| `tests/test_library.py` | library 单测 | 新增 `TestSanitizePhotoId` |
| `tests/test_server.py` | HTTP 层单测 | `TestUpload` 加 4 条；`TestFrontend` 加 1 条 |
| `README.md` | 上手用法 | 「建索引」一节的 `photo_id` 注记重写 |
| `docs/design.md` | 为什么这么做 | §2 补一段：刷新靠稳定的 photo_id |

校验函数放 `library.py` 而不是 `server.py`：「photo_id 得能当文件名」是这个模块的持久化约束（它写 `{id}.json`），`server.py` 只是恰好也要用它给 jpg 命名。

---

### Task 1: `sanitize_photo_id` —— photo_id 的文件名安全校验

**Files:**
- Modify: `puzzlefind/library.py`（在 `QueryResult` 之前、import 之后插入）
- Test: `tests/test_library.py`

**Interfaces:**
- Consumes: 无（纯函数，不依赖任何既有代码）
- Produces:
  - `class InvalidPhotoId(ValueError)` —— 校验失败时抛出，`str(error)` 是给用户看的中文原因
  - `def sanitize_photo_id(raw: str) -> str` —— 通过则返回 `raw.strip()`，否则抛 `InvalidPhotoId`
  - `MAX_PHOTO_ID_LENGTH: int = 40`

- [ ] **Step 1: 写失败的测试**

在 `tests/test_library.py` 顶部把 import 改成：

```python
import pytest

from puzzlefind.library import InvalidPhotoId, Library, sanitize_photo_id
from puzzlefind.models import Piece, PhotoIndex
```

在文件**末尾**追加：

```python
class TestSanitizePhotoId:
    """photo_id 直接当文件名用，所以它必须是一个安全的文件名。

    这层保护以前是白捡的——photo_id 来自 Path(filename).stem，而 Path
    顺手剥掉了目录分隔符（Path("../../x.jpg").stem == "x"）。改成读一个
    自由文本字段之后那层意外的保护就没了。
    """

    def test_keeps_an_ordinary_name(self):
        assert sanitize_photo_id("2") == "2"

    def test_keeps_a_chinese_name(self):
        assert sanitize_photo_id("左上角") == "左上角"

    def test_trims_surrounding_whitespace(self):
        assert sanitize_photo_id("  2  ") == "2"

    @pytest.mark.parametrize("raw", ["", "   ", "\t"])
    def test_empty_name_is_rejected(self, raw):
        with pytest.raises(InvalidPhotoId):
            sanitize_photo_id(raw)

    @pytest.mark.parametrize(
        "raw", ["a/b", "a\\b", "a:b", "a*b", "a?b", 'a"b', "a<b", "a>b", "a|b"]
    )
    def test_path_and_wildcard_characters_are_rejected(self, raw):
        with pytest.raises(InvalidPhotoId):
            sanitize_photo_id(raw)

    def test_control_characters_are_rejected(self):
        with pytest.raises(InvalidPhotoId):
            sanitize_photo_id("a\x00b")

    @pytest.mark.parametrize("raw", [".", ".."])
    def test_dot_names_are_rejected(self, raw):
        with pytest.raises(InvalidPhotoId):
            sanitize_photo_id(raw)

    @pytest.mark.parametrize("raw", ["con", "CON", "Com1", "LPT9", "nul", "aux"])
    def test_windows_reserved_device_names_are_rejected(self, raw):
        """这是 Windows 项目，data/index/CON.json 会当场炸。"""
        with pytest.raises(InvalidPhotoId):
            sanitize_photo_id(raw)

    def test_a_name_at_the_length_limit_is_kept(self):
        assert sanitize_photo_id("x" * 40) == "x" * 40

    def test_an_overlong_name_is_rejected(self):
        with pytest.raises(InvalidPhotoId):
            sanitize_photo_id("x" * 41)

    def test_the_error_message_says_why(self):
        """错误直接透给用户看，必须说清违反了哪条。"""
        with pytest.raises(InvalidPhotoId, match="/"):
            sanitize_photo_id("2/3")
```

- [ ] **Step 2: 跑测试确认它失败**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_library.py::TestSanitizePhotoId -v`
Expected: 收集阶段就 FAIL —— `ImportError: cannot import name 'InvalidPhotoId' from 'puzzlefind.library'`

- [ ] **Step 3: 写实现**

在 `puzzlefind/library.py` 里，`from .models import Piece, PhotoIndex` 那行**之后**、`@dataclass class QueryResult` 之前插入：

```python
# photo_id 直接当文件名用：data/index/{id}.json、data/photos/{id}.jpg。
MAX_PHOTO_ID_LENGTH = 40
_ILLEGAL_ID_CHARS = frozenset('/\\:*?"<>|')
_RESERVED_ID_NAMES = frozenset(
    ["CON", "PRN", "AUX", "NUL"]
    + [f"COM{i}" for i in range(1, 10)]
    + [f"LPT{i}" for i in range(1, 10)]
)


class InvalidPhotoId(ValueError):
    """photo_id 不能安全地当文件名用。消息直接透给用户看。"""


def sanitize_photo_id(raw: str) -> str:
    """校验用户给的 photo_id，通过则返回去掉首尾空白的结果。

    这层校验以前是白捡的：photo_id 来自 `Path(filename).stem`，而 Path
    顺手剥掉了目录分隔符（`Path("../../x.jpg").stem == "x"`）。改成读一个
    用户自由输入的字段之后，那层意外的保护就没了，得自己做。

    **违反规则一律抛错，绝不静默改写。** 把 `2/3` 悄悄存成 `2_3` 会让用户
    以为存成了自己输入的名字——而名字正是这个功能的全部意义，改写它等于
    把功能悄悄做坏。
    """
    name = raw.strip()
    if not name:
        raise InvalidPhotoId("名字不能为空")
    if len(name) > MAX_PHOTO_ID_LENGTH:
        raise InvalidPhotoId(f"名字不能超过 {MAX_PHOTO_ID_LENGTH} 个字符")
    if name in {".", ".."}:
        raise InvalidPhotoId("名字不能是 . 或 ..")
    illegal = sorted(set(name) & _ILLEGAL_ID_CHARS)
    if illegal:
        raise InvalidPhotoId(f"名字不能包含这些字符：{' '.join(illegal)}")
    if any(ord(char) < 32 for char in name):
        raise InvalidPhotoId("名字不能包含控制字符")
    if name.upper() in _RESERVED_ID_NAMES:
        raise InvalidPhotoId(f"{name} 是 Windows 保留的设备名，换一个")
    return name
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_library.py -v`
Expected: PASS，包括 `TestPersistence` / `TestQuery` 原有全部用例

- [ ] **Step 5: 提交**

```bash
git add puzzlefind/library.py tests/test_library.py
git commit -m "feat: validate photo_id before it becomes a filename"
```

---

### Task 2: 上传接口接受用户指定的 `photo_id`

**Files:**
- Modify: `puzzlefind/server.py:15`（import）、`puzzlefind/server.py:18`（import）、`puzzlefind/server.py:76-99`（`upload_photo`）
- Modify: `docs/design.md` §2 末尾
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: Task 1 的 `sanitize_photo_id(raw: str) -> str` 与 `InvalidPhotoId`
- Produces: `POST /api/photos` 接受可选表单字段 `photo_id`；同名上传即替换。前端（Task 3）依赖这个字段名。

- [ ] **Step 1: 写失败的测试**

在 `tests/test_server.py` 的 `class TestUpload` **末尾**追加四条：

```python
    def test_explicit_photo_id_wins_over_the_filename(self, client, photo_bytes):
        """手机传上来的文件名是浏览器现造的时间戳，用户给的名字必须压过它。"""
        client.post(
            "/api/photos",
            files={"file": ("1786177906346.jpg", photo_bytes, "image/jpeg")},
            data={"photo_id": "2"},
        )
        listing = client.get("/api/photos").json()
        assert [p["photo_id"] for p in listing["photos"]] == ["2"]

    def test_reshooting_the_same_region_replaces_the_index(self, client, photo_bytes):
        """重拍同一片区域必须**替换**，不是再堆一份。

        这是本功能的核心断言。photo_id 不稳定时，design.md §2 承诺的
        「重拍刷新」实际上在积累过期数据：旧索引原地留下来，而查询跨所有
        照片扫，可能命中那份陈旧的。两次上传的文件名故意取不同，正是为了
        证明替换只由 photo_id 决定。
        """
        for filename in ("shot-a.jpg", "shot-b.jpg"):
            response = client.post(
                "/api/photos",
                files={"file": (filename, photo_bytes, "image/jpeg")},
                data={"photo_id": "2"},
            )
            assert response.status_code == 200
        assert len(client.get("/api/photos").json()["photos"]) == 1

    def test_illegal_photo_id_is_rejected(self, client, photo_bytes):
        response = client.post(
            "/api/photos",
            files={"file": ("shot.jpg", photo_bytes, "image/jpeg")},
            data={"photo_id": "a/b"},
        )
        assert response.status_code == 400

    def test_photo_id_may_be_omitted(self, client, photo_bytes):
        """不给这个字段时行为与从前一致——curl 上传不受影响。"""
        client.post(
            "/api/photos", files={"file": ("shot.jpg", photo_bytes, "image/jpeg")}
        )
        listing = client.get("/api/photos").json()
        assert [p["photo_id"] for p in listing["photos"]] == ["shot"]
```

- [ ] **Step 2: 跑测试确认它失败**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_server.py::TestUpload -v`
Expected: `test_explicit_photo_id_wins_over_the_filename` 与 `test_reshooting_the_same_region_replaces_the_index` FAIL（photo_id 被忽略，落成 `1786177906346` / 两条记录）；`test_illegal_photo_id_is_rejected` FAIL（返回 200 而非 400）。`test_photo_id_may_be_omitted` 本来就该 PASS。

- [ ] **Step 3: 写实现**

`puzzlefind/server.py` 第 15 行的 fastapi import 加上 `Form`：

```python
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
```

第 18-19 行的两条本地 import 改成：

```python
from . import config, render
from .library import InvalidPhotoId, Library, sanitize_photo_id
```

把 `upload_photo`（原 76-99 行）整个替换为：

```python
    @app.post("/api/photos")
    async def upload_photo(
        file: UploadFile = File(...),
        photo_id: str | None = Form(None),
    ) -> dict:
        # 先校验名字再解码图像：名字错是用户的笔误，不必先花几十毫秒解一张
        # 四千万像素的 jpg 再告诉他打错了。
        if photo_id is None:
            # 没给名字时保持老行为。命令行和 curl 走这条路。
            stem = Path(file.filename or "").stem or uuid.uuid4().hex[:8]
        else:
            try:
                stem = sanitize_photo_id(photo_id)
            except InvalidPhotoId as error:
                raise HTTPException(status_code=400, detail=str(error)) from error

        raw = await file.read()
        buffer = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="无法解码为图像")

        # 同名即替换：imwrite 与 save_photo 都是覆盖写，所以「重拍刷新」
        # 不需要任何额外逻辑——它只需要一个跨重拍稳定的 photo_id。
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.\.venv\Scripts\python.exe -m pytest tests\ -m "not ocr" -v`
Expected: 全部 PASS

- [ ] **Step 5: 更新 design.md §2**

在 `docs/design.md` §2 末尾那段（`……索引带时间戳供界面提示。`）**之后**追加：

```markdown
**刷新靠稳定的 `photo_id` 覆盖同名索引，所以照片的名字必须由用户指定。**
浏览器给出的文件名是手机相册在点选那一刻现造的时间戳，每次重拍都不一样
——旧索引会原地留下来继续参与查询，刷新反而变成了在积累过期数据。名字
由用户在上传时选定之后，重拍同一区域就是一次覆盖写，不需要额外的替换逻辑。
```

- [ ] **Step 6: 提交**

```bash
git add puzzlefind/server.py tests/test_server.py docs/design.md
git commit -m "feat: let the uploader name the region it shot"
```

---

### Task 3: 上传前先问「这张拍的是哪片区域」

**Files:**
- Modify: `puzzlefind/static/index.html`（CSS 插在 223 行前、HTML 插在 354 行后、JS 替换 586-609 行）
- Modify: `README.md`「建索引」一节
- Test: `tests/test_server.py::TestFrontend`

**Interfaces:**
- Consumes: Task 2 的 `POST /api/photos` 表单字段 `photo_id`；既有的 `GET /api/photos`（返回 `{"photos": [{"photo_id": ..., ...}]}`）
- Produces: 无下游

- [ ] **Step 1: 写失败的测试**

在 `tests/test_server.py` 的 `class TestFrontend` **末尾**追加：

```python
    def test_html_asks_which_region_before_uploading(self, client):
        """上传前必须先问「这张拍的是哪片区域」，并把答案发出去。

        用户在手机上把照片改名成 1/2/3/4，但那个名字从来没进过 HTTP 请求
        ——安卓相册交给浏览器的是一个不带显示名的句柄，浏览器用点选时刻的
        毫秒时间戳兜底。名字只能在这里问，没法从文件名里抢救。
        """
        body = client.get("/").text
        assert 'id="regionPicker"' in body
        assert 'form.append("photo_id"' in body
```

- [ ] **Step 2: 跑测试确认它失败**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_server.py::TestFrontend -v`
Expected: `test_html_asks_which_region_before_uploading` FAIL —— `assert 'id="regionPicker"' in body`

- [ ] **Step 3: 加 CSS**

在 `puzzlefind/static/index.html` 的**第 223 行**（CSS 里那句 `/* ── 未识别照片切换器 ─… */`，注意第 396 行 JS 里还有一句同名注释，别插错地方）**之前**插入：

```css
/* ── 上传前的区域选择器 ─────────────────────────────────────── */
/* 桌面区域是固定的，所以常态是「重拍已有的某一张」，不是「新建」。
   四选一比在手机上打字友好得多，而按钮上直接写「重拍替换」把语义前置，
   就不必再弹一个事后确认框。 */
#regionPicker { display: none; flex-direction: column; gap: 6px; margin-top: 10px; }
#regionPicker.is-shown { display: flex; }
.picker-title { margin: 0; font-size: 12px; color: var(--muted); }
#regionChoices { display: flex; flex-direction: column; gap: 6px; }
.picker-new { display: flex; gap: 6px; }
#regionName {
  flex: 1;
  min-width: 0;
  padding: 9px 11px;
  background: var(--table);
  border: 1px solid var(--ridge);
  border-radius: 7px;
  color: var(--chipboard);
  font-family: var(--mono);
  font-size: 13px;
}
#regionName:focus { outline: 2px solid var(--target); outline-offset: 1px; }
.picker-new .btn, #regionAdd { padding: 9px 11px; font-size: 12px; }
```

- [ ] **Step 4: 加 HTML**

`<div class="rail-actions">` 那一段占 348-354 行，在它的闭合 `</div>`（第 354 行）**之后**插入：

```html
    <!-- 上传前先问「这张拍的是哪片区域」。这里选的名字就是 photo_id，
         选一个已有的名字即为重拍替换。名字没法从文件名里取：手机相册
         交给浏览器的文件名是点选那一刻现造的时间戳。 -->
    <div id="regionPicker">
      <p class="picker-title">这张拍的是哪片区域？</p>
      <div id="regionChoices"></div>
      <button class="btn btn-quiet" id="regionAdd" type="button">＋ 新的区域</button>
      <div class="picker-new" id="regionNew" hidden>
        <input id="regionName" type="text" placeholder="比如 1" autocomplete="off"
               maxlength="40">
        <button class="btn" id="regionGo" type="button">确定</button>
      </div>
      <button class="btn btn-quiet" id="regionCancel" type="button">取消</button>
    </div>
```

- [ ] **Step 5: 加 JS，并改上传处理**

把 **586-609 行整段**（从 `/* ── 上传建索引 ─… */` 到 `$("file").onchange` 那个函数的收尾 `};`）**整体替换**为：

```js
/* ── 上传建索引 ─────────────────────────────────────────────────
   点「上传照片」不直接拉起文件选择器，先问一句「这张拍的是哪片区域」。
   这里选的名字就是 photo_id，也是查询结果里显示的那个名字。

   为什么名字只能在这里问：用户在手机上把照片改成 1/2/3/4，但那个名字
   从来没进过上传请求——安卓相册交给浏览器的是一个不带显示名的句柄，
   浏览器只好用点选那一刻的毫秒时间戳兜底。文件名里没有可抢救的东西。 */
const picker = $("regionPicker");
let pendingRegion = null;   // 已选定、等着文件选择器返回的区域名

function openPicker(names) {
  const choices = $("regionChoices");
  choices.replaceChildren();
  for (const name of names) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "switch-btn";

    // photo_id 是用户可控字符串，只能用 textContent
    const label = document.createElement("span");
    label.className = "switch-name";
    label.textContent = name;
    const hint = document.createElement("span");
    hint.className = "switch-count";
    hint.textContent = "重拍替换";

    button.append(label, hint);
    button.onclick = () => pickRegion(name);
    choices.appendChild(button);
  }

  // 库为空时没什么可选的，直接把输入框摆出来，省掉一次点击
  const empty = names.length === 0;
  $("regionAdd").hidden = empty;
  $("regionNew").hidden = !empty;
  $("regionName").value = "";
  picker.classList.add("is-shown");
  if (empty) $("regionName").focus();
}

function closePicker() {
  picker.classList.remove("is-shown");
}

function pickRegion(name) {
  pendingRegion = name;
  closePicker();
  $("file").click();
}

function confirmNewRegion() {
  const name = $("regionName").value.trim();
  if (!name) { $("regionName").focus(); return; }
  // 输入一个已存在的名字与点那个区域的按钮完全等价：都是替换。
  // 不报重名——重拍本来就是常态操作。
  pickRegion(name);
}

$("upload").onclick = async () => {
  let names = [];
  try {
    const data = await (await fetch("/api/photos")).json();
    names = data.photos.map((p) => p.photo_id);
  } catch { /* 列表拿不到就当空库处理，仍然可以新建一个区域 */ }
  openPicker(names);
};

$("regionAdd").onclick = () => {
  $("regionAdd").hidden = true;
  $("regionNew").hidden = false;
  $("regionName").focus();
};
$("regionGo").onclick = confirmNewRegion;
$("regionCancel").onclick = () => { pendingRegion = null; closePicker(); };
$("regionName").addEventListener("keydown", (e) => {
  if (e.key === "Enter") confirmNewRegion();
});

$("file").onchange = async (e) => {
  const file = e.target.files[0];
  const region = pendingRegion;
  pendingRegion = null;
  if (!file) return;   // 用户在系统选择器里点了取消

  say("正在建索引，别关页面。");
  $("upload").disabled = true;
  try {
    const form = new FormData();
    form.append("file", file);
    if (region) form.append("photo_id", region);
    const response = await fetch("/api/photos", { method: "POST", body: form });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || response.status);
    say(`照片 ${body.photo_id} 已建索引：分割 ${body.total} 块，`
      + `读出编号 ${body.recognized} 块，未识别 ${body.unrecognized} 块。`, "hit");
    await refreshPhotos();
  } catch (error) {
    say(`建索引失败：${error.message}。换一张深色纯背景、碎片摊开不重叠的照片再试。`, "bad");
  } finally {
    $("upload").disabled = false;
    e.target.value = "";
  }
};
```

- [ ] **Step 6: 跑测试确认通过**

Run: `.\.venv\Scripts\python.exe -m pytest tests\ -m "not ocr" -v`
Expected: 全部 PASS，含 `test_html_accepts_images_without_forcing_the_camera`（那条正则只看 `<input type="file">` 标签，新增的 `#regionName` 是 `type="text"`，不受影响）

- [ ] **Step 7: 人工验一遍**

Run: `.\.venv\Scripts\python.exe main.py`，浏览器打开它打印的地址，依次确认：

1. 空库时点「上传照片建索引」→ 直接出现输入框和「确定」
2. 输入 `1`，点确定 → 系统文件选择器弹出；选一张照片 → 状态栏出现「照片 1 已建索引」，右侧列表出现一行 `1`
3. 再点「上传照片建索引」→ 出现 `1 · 重拍替换` 按钮和「＋ 新的区域」
4. 点 `1 · 重拍替换`，选另一张照片 → 建索引后列表**仍然只有一行** `1`
5. 点「取消」→ 面板收起，不弹文件选择器
6. 「＋ 新的区域」输入 `a/b` 点确定 → 状态栏红字显示「名字不能包含这些字符：/」

- [ ] **Step 8: 更新 README**

在 `README.md`「### 建索引」一节里，把这段引用块：

```markdown
> `photo_id` 取自文件名去掉最后一个扩展名，所以 `桌面.jpg.jpg` 会得到
> `photo_id` = `桌面.jpg`。传之前把文件名理干净。
```

**整段替换**为：

```markdown
> **网页上传会先问「这张拍的是哪片区域」**，你填的名字就是 `photo_id`，
> 也是查询结果里显示的那个名字。选一个**已有的名字**即为重拍替换——那份
> 索引被整个覆盖，这就是碎片被拿走之后刷新索引的方式。
>
> **名字只能在网页上问，没法从文件名里取。** 手机相册交给浏览器的文件名是
> 点选那一刻现造的毫秒时间戳（`1786177906346.jpg`），你在手机上改的名字
> 从来没进过上传请求。
>
> 命令行用 `--photo-id` 指定；不给时取自文件名去掉最后一个扩展名，
> 所以 `桌面.jpg.jpg` 会得到 `photo_id` = `桌面.jpg`。
```

- [ ] **Step 9: 提交**

```bash
git add puzzlefind/static/index.html tests/test_server.py README.md
git commit -m "feat: ask which region a photo shows before uploading it"
```

---

## 收尾

存量那 4 份时间戳命名的索引**不写迁移代码**（spec §4.4）。功能就位后由用户在网页上删掉、按 `1`/`2`/`3`/`4` 重新上传，四张各约 10 秒 OCR。
