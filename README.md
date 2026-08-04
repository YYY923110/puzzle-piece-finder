# 拼图碎片编号查找器

一套 1000 块的拼图，每块背面印着编号（`B-506`）。把碎片摊在深色背景上拍照建索引，
之后输入编号，照片上就会高亮出那一块——**目的是让你的手能在桌上找到它，屏幕只是中介。**

全本地运行：OpenCV 分割 + PaddleOCR 识别，不联网、不调大模型、无使用成本。

**实测**（`real3.jpg`，深色纯背景 50 块）：分割 50/50 完全正确，识别率 **84%**。
建索引耗时在 2026-08-04 的提速改造后降到 **约 2.6 s/块**（50 块约 2 分钟，
改造前是 7.7 s/块）。详见 [docs/tuning-log.md](docs/tuning-log.md)。

---

## 装好之后怎么跑

依赖装在项目自带的 `.venv` 里，**没有装进 base Anaconda**（PaddleOCR 会拉进
`opencv-contrib-python`，装进 base 会顶掉你原有的 headless OpenCV 5.0）。
所以所有命令都要走 `.venv` 里的解释器：

```powershell
cd d:\ocr_claude
.\.venv\Scripts\python.exe -m puzzlefind.server 8791
```

打开：

- 电脑 <http://127.0.0.1:8791>
- 手机（同一 Wi-Fi）用它启动时打印的那个地址

> **端口参数别省。** 不带参数默认 8000，而本机 8000 已被另一个 python 进程占用。
> Windows 允许两个进程绑同一端口且不报错，请求会随机落到其中一个，很难排查。
> 也可以用环境变量 `$env:PUZZLEFIND_PORT=8791`。

首次全新安装：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

---

## 拍照要求（决定成败的一步）

工具的识别率几乎完全由照片质量决定，调参救不回来。三次实拍的教训：

| 要求 | 为什么 | 踩过的坑 |
|---|---|---|
| **纯深色、无花纹背景** | 前景分离退化成一次 Otsu 阈值 | 蓝白格子毯：米白格子和碎片同色，2/3 的碎片直接消失，斜纹还炸出几百个假轮廓 |
| **一张 40–60 块** | 决定每个字符占多少像素 | 一张塞 85+ 块，字太小 |
| **摊开留缝，别互相压着** | 粘连的碎片会被当成一块，只读出一个编号 | 40 块粘成 14 个连通块 |
| **导出别缩图** | 同上 | 用了裁剪缩小到 905×933 的图，字符只剩 12 px |
| **编号尽量朝正** | 朝正的走 Pass A（1 次识别），歪的要进角度穷举 | 朝向随机时 35 秒/块，朝正后 11 秒/块（均为提速改造前的数字） |

自查：跑完标定看 `debug/<照片名>/crops/` 里的图，**编号必须清晰可读，且一张图里只有一个编号**。

---

## 建索引

三条路，引擎完全一样：

```powershell
# 1. Web 界面上传（手机拍完直接传）
#    注意 photo_id 取自文件名，"real4.jpg.jpg" 会得到 photo_id "real4.jpg"

# 2. 命令行
.\.venv\Scripts\python.exe -m puzzlefind.cli index data\photos\real5.jpg --index-dir data\index

# 3. 标定脚本：额外输出每一步的中间图，调参时用这个
.\.venv\Scripts\python.exe scripts\calibrate.py data\photos\real5.jpg
```

`calibrate.py` 会把中间产物写进 `debug/<照片名>/`：

| 文件 | 看什么 |
|---|---|
| `01_mask.png` | 碎片是干净白团、背景是纯黑？背景有大片白说明背景不够深 |
| `02_blobs.png` | 黄轮廓贴合碎片（多块被圈成一个是正常的，下一步会切） |
| `03_split.png` | **洋红轮廓的数量应等于实际碎片数**，这是分割质量的硬指标 |
| `crops/*.png` | **最关键**：编号清晰吗？有没有混进邻块的编号？ |
| `04_unrecognized.png` | 青色圈出没认出来的，看它们有什么共同点 |
| `index.json` | 每块碎片读成了什么，可直接打开看 |

---

## 查询

```powershell
.\.venv\Scripts\python.exe -m puzzlefind.cli query B-794 --index-dir data\index
.\.venv\Scripts\python.exe -m puzzlefind.cli stats --index-dir data\index
```

`--index-dir` 要放在**子命令之后**。查询命中退出码 0，未命中 1。

未命中时不会只说「没找到」——它会列出该照片里所有**未识别**的碎片，
把搜索范围从几百块塌缩到个位数。Web 界面上这些块会用青色圈出。

---

## 调参

所有可调参数集中在 [`puzzlefind/config.py`](puzzlefind/config.py) 一个文件里。
改之前先读 [docs/tuning-log.md](docs/tuning-log.md)——里面记着三次实拍的完整数据，
以及每个参数**为什么保持现值**。

排查顺序（来自 spec §9）：**裁剪字号 → 背景 → 切分参数**。别跳步。

一条实测结论：`SWEEP_CONFIDENCE_THRESHOLD` 在合格照片上完全不起作用
（Pass A 命中时置信度最低 0.997，全部远高于阈值 0.90），调它是白费力气。

**关于提速**（2026-08-04 改造后的现状）：

- 瓶颈是**检测模型**，不是识别模型——单次 `predict()` 的 1.71 s 里
  det 占 1.485 s（**92%**），rec 只占 0.129 s。分割则完全不是问题（1 ms/块）。
- 已经拿到的收益来自 `SWEEP_EARLY_EXIT_CONFIDENCE`：穷举时拿到 ≥0.99
  就收工，实测 **2.94x**（385s → 131s，识别率不变）。
- **量过且确认无效的，别再试**：批量推理（1.00x）、关方向分类（1.02x）、
  调 `text_det_limit_side_len`（0.95x）、重开 oneDNN（绕不过 PIR 的算子缺口）、
  GPU（装的是 CPU 版 paddle）。换 PP-OCRv5_mobile 确实快 3.54x，
  但会把编号读残，未采用。
- 下一个杠杆是多进程（还剩约一半 CPU 空闲）或彻底绕开检测模型。

---

## 环境上的坑（都已在代码里处理，此处备查）

| 现象 | 真因 |
|---|---|
| `No available model hosting platforms detected. Please check your network connection.` | paddlex 探测四个下载源只给 **1 秒**超时。四个站点其实都通，只是没那么快。`recognize.prepare_paddle_env()` 已跳过该探测并把 BOS 排到第一位 |
| `ConvertPirAttribute2RuntimeAttribute not support` | paddle 3.3.1 的 oneDNN 算子在 PIR 执行器下有缺口。必须 `enable_mkldnn=False`，代价是 CPU 推理慢一些 |
| 手机连不上打印出来的地址 | 代理软件（Clash/Mihomo）的 TUN 网卡接管默认路由，经典的「连 8.8.8.8 看本地端点」会返回 `198.18.0.0`。已改为遍历所有网卡并按网段排序 |
| 服务起来了但请求落到别的应用 | 本机 8000 被占，而 Windows 允许重复绑定不报错。显式指定端口 |

模型缓存在 `C:\Users\<用户>\.paddlex\official_models`（约 139 MB），只下载一次。

---

## 项目结构

```
puzzlefind/
  config.py      所有可调参数集中在这里
  vocabulary.py  编号校验、归一化、混淆感知吸附、区间自举
  segment.py     掩膜、轮廓提取、粘连切分、裁剪归一化
  recognize.py   OcrBackend 协议、PaddleOCR 后端、直接识别 + 旋转穷举
  resolve.py     同图唯一性与离群值消解
  models.py      Piece / PhotoIndex 及 JSON 序列化
  pipeline.py    串起分割→识别→消解
  library.py     多照片库、跨照片查询
  render.py      高亮渲染（压暗 + 描边）与缩略图
  cli.py         命令行入口
  server.py      FastAPI 服务
  static/        单文件前端
scripts/
  calibrate.py       标定与调参
  probe_paddleocr.py 探明 PaddleOCR 3.x 的返回结构
```

**关键架构约束：** `recognize.OcrBackend` 是一个 `Protocol`，PaddleOCR 只是它的一个实现。
除 `TestPaddleBackendIntegration` 里那几条标了 `@pytest.mark.ocr` 的测试外，
所有测试注入假后端，**完全不加载 PaddleOCR**——所以测试几秒钟跑完，
也不依赖模型下载。换识别后端只需实现 `read(image) -> list[RawDetection]`。

`recognize.LineOcrBackend` 是**可选**的第二个协议（`read_line`，跳过检测模型
只跑识别）。不实现它的后端一样能用，只会走全量角度穷举——所以上面那条
「只需实现 `read()`」的承诺仍然成立。

---

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest tests\ -m "not ocr"   # 快，不碰模型
.\.venv\Scripts\python.exe -m pytest tests\                # 含 OCR 集成测试
```

---

## 已知限制

- **编号不唯一。** 实测同一个编号会印在多块碎片上（`B-506`、`B-652` 各出现在 3 块上）。
  当前保留「同图编号唯一」规则，重复时只保留置信度最高的一块——
  **查 `B-506` 只会高亮 3 块中的 1 块，且不提示还有其他同号碎片**。
  被丢弃的那些会落进「未识别」集合。这是权衡后的决定，见 spec §4 与决策日志。
- **编号不按字母分段。** B 组实测跨 187–796，所以基于数字区间的离群剔除几乎空转。
- 索引是静态的：拿走碎片后需要重拍那张照片刷新。
- 只认背面编号，不识别正面图案，不自动解拼图。

---

## 文档

- [spec](docs/superpowers/specs/2026-08-03-puzzle-piece-finder.md) —— **为什么**这么做，以及被明确否决的替代方案
- [plan](docs/superpowers/plans/2026-08-03-puzzle-piece-finder.md) —— 原始实施方案（**已有 6 处被修正，勿照抄代码**）
- [调参日志](docs/tuning-log.md) —— 三次实拍的完整数据与参数决策依据
