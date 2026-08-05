# 拼图碎片编号查找器

一套 1000 块的拼图，每块背面印着编号（`B-506`）。把碎片摊在深色背景上拍照建索引，
之后输入编号，照片上就会高亮出那一块——**目的是让你的手能在桌上找到它，屏幕只是中介。**

全本地运行：OpenCV 分割 + PaddleOCR 识别，不联网、不调大模型、无使用成本。

**实测能力**（2026-08-05，`real7.jpg`，50 块密排 1086×1448，同一张照片上的 CPU/GPU 对照）：

| 跑在哪 | 分割 | 识别率 | 总耗时 | 每块 |
| ------ | ----- | -------- | ------- | --------- |
| **GPU**（RTX 3050 4GB） | 50/50 | **100%** | **7.2 s** | **0.144 s** |
| CPU | 50/50 | **100%** | 246.4 s | 4.928 s |

**GPU 快 34.2 倍，50 个编号逐条相同**，`direct 39 / sweep 11` 的走向也一致——
纯提速，没有精度代价。装法见下面「首次全新安装」。

更早几轮标定（2026-08-04，CPU，照片已清空）：23 块摊开留缝 100% / 2.65 s 块，
50 块密排 100% / 6.72 s 块，50 块密排 84% / 5.34 s 块。那 8 块损失全部来自
**同图编号重复**（见「已知限制」），不是读不出来。摊开拍不只是识别率的事，
也是速度的事——摊得开的那张 Pass A 命中率高得多，几乎不用跑角度穷举。

> 照片和中间产物都不入库（见 `.gitignore`）。上面 GPU/CPU 那两行可以在留有
> `real7.jpg` 时用 `scripts/benchmark.py` 复现；更早那三张已清空，是历史记录。
> 结论和参数依据留在 [docs/tuning-log.md](docs/tuning-log.md)。

---

## 装好之后怎么跑

**在 VS Code 里打开 [`main.py`](main.py) 点运行**，或者：

```powershell
python main.py
```

它自己会处理两件容易踩的事：**切到 `.venv` 里的解释器**（VS Code 选中哪个都行），
**挑一个没人在听的端口**（默认 8791，被占就往上顺延并打印实际用的那个）。

打开：

- 电脑 [http://127.0.0.1:8791](http://127.0.0.1:8791)
- 手机（同一 Wi-Fi）用它启动时打印的那个地址

依赖装在项目自带的 `.venv` 里，**没有装进 base Anaconda**（PaddleOCR 会拉进
`opencv-contrib-python`，装进 base 会顶掉你原有的 headless OpenCV 5.0）。
`main.py` 之所以要自己切解释器就是为了这个。手动起服务时也要走那个解释器：

```powershell
cd d:\Puzzle-Solver\ocr_claude
.\.venv\Scripts\python.exe -m puzzlefind.server 8791
```

> **手动起时端口参数别省。** 不带参数默认 8000，而本机 8000 已被另一个 python
> 进程占用。Windows 允许两个进程绑同一端口且不报错，请求会随机落到其中一个，
> 很难排查。也可以用环境变量 `$env:PUZZLEFIND_PORT=8791`。
> 走 `main.py` 则不必操心——它是靠「能不能连上」而不是「能不能绑上」来判断的，
> 正好能识破这个坑。

首次全新安装（**两步，paddlepaddle 要单独装**）：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

# 有 NVIDIA 显卡就装 GPU 版，实测快 34 倍。CUDA 运行时和 cuDNN 是它的
# pip 依赖，会一并拉下来，不必自己装 CUDA Toolkit。约 1.9 GB。
.\.venv\Scripts\python.exe -m pip install paddlepaddle-gpu==3.3.1 `
  -i https://www.paddlepaddle.org.cn/packages/stable/cu130/
```

没有显卡（或驱动低于 CUDA 13）就换成 CPU 版：`pip install -e ".[dev,cpu]"`。
驱动支持的 CUDA 版本用 `nvidia-smi` 右上角那个数字看；13.x 用 `cu130`，
12.x 换成 `cu129` / `cu126` / `cu118`（这几个通道都有 cp313 的 Windows 轮子）。

> **`paddlepaddle` 故意没写进 `pyproject.toml` 的主依赖。** 它有 CPU / GPU 两个
> 发行名，抢同一个 `paddle` 包目录。写在主依赖里的后果实测过：GPU 版装好后
> 再跑一次 `pip install -e .`，pip 会不声不响地把 CPU 版装回来，**34 倍就这么
> 没了，还不报错**。拆成 extra 之后，装漏了会当场 ImportError——响亮的失败
> 好过安静的退化。

---

## 拍照要求（决定成败的一步）

工具的识别率几乎完全由照片质量决定，调参救不回来。三次实拍的教训：

| 要求                           | 为什么                                        | 踩过的坑                                                                   |
| ------------------------------ | --------------------------------------------- | -------------------------------------------------------------------------- |
| **纯深色、无花纹背景**   | 前景分离退化成一次 Otsu 阈值                  | 蓝白格子毯：米白格子和碎片同色，2/3 的碎片直接消失，斜纹还炸出几百个假轮廓 |
| **一张 40–60 块**       | 决定每个字符占多少像素                        | 一张塞 85+ 块，字太小                                                      |
| **摊开留缝，别互相压着** | 粘连的碎片会被当成一块，只读出一个编号        | 40 块粘成 14 个连通块                                                      |
| **导出别缩图**           | 同上                                          | 用了裁剪缩小到 905×933 的图，字符只剩 12 px                               |
| **编号尽量朝正**         | 朝正的走 Pass A（1 次识别），歪的要进角度穷举 | 朝向随机时 35 秒/块，朝正后 11 秒/块（均为提速改造前的数字）               |

自查：跑完标定看 `debug/<照片名>/crops/` 里的图，**编号必须清晰可读，且一张图里只有一个编号**。

---

## 建索引

三条路，引擎完全一样：

```powershell
# 1. Web 界面上传（手机上点「上传照片」会弹出「拍照 / 相册 / 文件」，
#    现拍和相册里已有的照片都能传）
#    注意 photo_id 取自文件名去掉最后一个扩展名，
#    所以 "桌面.jpg.jpg" 会得到 photo_id "桌面.jpg"——传之前把名字理干净

# 2. 命令行
.\.venv\Scripts\python.exe -m puzzlefind.cli index data\photos\桌面1.jpg --index-dir data\index

# 3. 标定脚本：额外输出每一步的中间图，调参时用这个
.\.venv\Scripts\python.exe scripts\calibrate.py data\photos\桌面1.jpg
```

`calibrate.py` 会把中间产物写进 `debug/<照片名>/`：

| 文件                    | 看什么                                                         |
| ----------------------- | -------------------------------------------------------------- |
| `01_mask.png`         | 碎片是干净白团、背景是纯黑？背景有大片白说明背景不够深         |
| `02_blobs.png`        | 黄轮廓贴合碎片（多块被圈成一个是正常的，下一步会切）           |
| `03_split.png`        | **洋红轮廓的数量应等于实际碎片数**，这是分割质量的硬指标 |
| `crops/*.png`         | **最关键**：编号清晰吗？有没有混进邻块的编号？           |
| `04_unrecognized.png` | 青色圈出没认出来的，看它们有什么共同点                         |
| `index.json`          | 每块碎片读成了什么，可直接打开看                               |

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

**关于提速**（2026-08-05 换 GPU 之后的现状）：

- **最大的一笔收益是 GPU：34.2x**（同一张 real7，246.4 s → 7.2 s，
  50 个编号逐条相同）。**代码一行没改**——`paddleocr` 的 `DEFAULT_DEVICE`
  是 `None`，检测到显卡自动就走 GPU 了。
- 瓶颈是**检测模型**，不是识别模型——CPU 上单次 `predict()` 的 1.71 s 里
  det 占 1.485 s（**92%**），rec 只占 0.129 s。分割则完全不是问题（1 ms/块）。
  GPU 打的正是这一刀。
- 上一笔收益来自 `SWEEP_EARLY_EXIT_CONFIDENCE`：穷举时拿到 ≥0.99
  就收工，实测 **2.94x**（385s → 131s，识别率不变）。
- **量过且确认无效的，别再试**：批量推理（1.00x）、关方向分类（1.02x）、
  调 `text_det_limit_side_len`（0.95x）、重开 oneDNN（绕不过 PIR 的算子缺口）。
  换 PP-OCRv5_mobile 确实快 3.54x，但会把编号读残，未采用。
- 2026-08-04 的裁剪掩膜改动（只遮邻块，见下）在密排照片上要回吐约 **55%**：
  多两块掉进角度穷举。这是为修正「编号被自己的轮廓抹掉」付的价，
  **有一个能把速度全拿回来的候选方案（膨胀 0.06）但被否决了**——
  它离一道 3 倍的悬崖只有 2 px，理由记在调参日志里。
- **在 GPU 上，剩下的提速杠杆基本都不值得做了。** 50 块 7.2 秒，
  多进程（估 1.7x）和「绕开 det 自己定位文字行」（理论 13x）都是在
  7 秒上做文章，却各自带来一堆复杂度和新的调参战场。没有 GPU 时它们
  才重新有意义。

**关于裁剪掩膜**（2026-08-04）：`crop_piece` 只把**邻块**涂灰，绝不遮自己。
曾经按碎片自己的轮廓填灰，结果紧挨凹口的编号被连轮廓上的缝一起抹掉
（`D-797` 读成 `D-79`）。**别改回去，也别试图用闭运算补那道缝**——
缝的口子就是凹口的口子，实测会让 Pass A 命中率塌一半。调参日志里有完整数据。

---

## 环境上的坑（都已在代码里处理，此处备查）

| 现象                                                                                     | 真因                                                                                                                                             |
| ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `No available model hosting platforms detected. Please check your network connection.` | paddlex 探测四个下载源只给**1 秒**超时。四个站点其实都通，只是没那么快。`recognize.prepare_paddle_env()` 已跳过该探测并把 BOS 排到第一位 |
| `ConvertPirAttribute2RuntimeAttribute not support`                                     | paddle 3.3.1 的 oneDNN 算子在 PIR 执行器下有缺口。必须`enable_mkldnn=False`，代价是 CPU 推理慢一些                                             |
| 手机连不上打印出来的地址                                                                 | 代理软件（Clash/Mihomo）的 TUN 网卡接管默认路由，经典的「连 8.8.8.8 看本地端点」会返回`198.18.0.0`。已改为遍历所有网卡并按网段排序             |
| 服务起来了但请求落到别的应用                                                             | 本机 8000 被占，而 Windows 允许重复绑定不报错。显式指定端口                                                                                      |

模型缓存在 `C:\Users\<用户>\.paddlex\official_models`（约 139 MB），只下载一次。

---

## 项目结构

```
main.py          一键启动（切解释器 + 挑端口 + 起服务）
puzzlefind/
  config.py      所有可调参数集中在这里
  vocabulary.py  编号校验、归一化、混淆感知吸附、区间自举
  segment.py     掩膜、轮廓提取、粘连切分、裁剪（只遮邻块）
  recognize.py   OcrBackend 协议、PaddleOCR 后端、四级降级识别
  resolve.py     同图唯一性与离群值消解
  models.py      Piece / PhotoIndex 及 JSON 序列化
  pipeline.py    串起分割→识别→消解
  library.py     多照片库、跨照片查询
  render.py      高亮渲染（压暗 + 描边）与缩略图
  cli.py         命令行入口
  server.py      FastAPI 服务
  static/        单文件前端
scripts/
  calibrate.py       标定与调参（输出每一步的中间图）
  benchmark.py       建索引耗时基准，改识别策略前后各跑一次
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
- [调参日志](docs/tuning-log.md) —— **改参数前先读这个**。每次实拍的完整数据、
  每个参数为什么保持现值，以及**被实测否决的方案**（省得再试一遍）

三份实施方案（原始实现、OCR 提速、裁剪掩膜修正）已执行完毕并从文档里移除——
它们描述的是**当时**的代码，与现状已有多处出入，留着只会误导。
其中经得起时间检验的结论都已并入上面两份文档，原文在 git 历史里。
