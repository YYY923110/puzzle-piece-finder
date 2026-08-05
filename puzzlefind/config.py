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
# 连通块少于这个数时，面积分布没有统计支撑，改用形状信号（山峰计数）
# 估计单块面积。真实照片动辄几十个连通块，走的一定是面积分支。
MIN_BLOBS_FOR_AREA_PRIOR: int = 5
# 形态学开运算核尺寸，用于去除掩膜毛刺
MORPH_KERNEL_SIZE: int = 5

# ---------- 裁剪 ----------
# 裁剪时在碎片包围盒外扩的像素数
CROP_PADDING: int = 6
# 裁剪结果的目标长边像素数（放大以喂饱识别模型）
CROP_TARGET_LONG_EDGE: int = 512
# 碎片外区域填充成什么颜色（BGR）。中性灰避免与浅灰文字/白底冲突
CROP_FILL_COLOR: tuple[int, int, int] = (128, 128, 128)
# 填充掩膜前「补缝」用的闭运算核尺寸，取碎片包围盒长边的这个比例。
# 深色印刷字符低于 Otsu 阈值，在掩膜上是洞；字符离碎片凹口够近时，
# 形态学运算会把洞和背景连通，外轮廓便从凹口钻进碎片内部绕字符一圈，
# 填灰时正好抹掉半个编号。实测 IMG_20260805_082927.jpg 的 D-797 被读成
# D-79，吸附成 D-079 后被离群规则剔除。
# 注：这里曾有一个 CROP_MASK_SEAL_RATIO（用闭运算补轮廓上的缝）。
# 2026-08-04 实测证明那条路走不通，已删除，理由记在 crop_piece 的
# docstring 与 docs/tuning-log.md 里。别再加回来。

# ---------- 识别 ----------
# 直接识别（Pass A）置信度低于此值的碎片，进入旋转穷举（Pass C）
# 默认激进：宁可多穷举。跑过真实照片后按实测数据下调。
SWEEP_CONFIDENCE_THRESHOLD: float = 0.90
# 穷举的角度表（度）。若方向分类器可靠，可缩短为 (0, 30, 60, 90, 120, 150)
SWEEP_ANGLES: tuple[int, ...] = (0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330)
# 穷举时拿到不低于此置信度的合法编号就收工，不再试剩余角度。
# 依据 real1.jpg 实测：9 次穷举命中里 5 次置信度 ≥0.99，剩余角度是白跑的。
# 设成 0.99 而不是更低，是因为「命中合法词表」这个判据虽硬但不是绝对——
# 留一点余量，让明显更好的读数还有机会翻盘。
SWEEP_EARLY_EXIT_CONFIDENCE: float = 0.99
# 拿到检测框后，摆正的文字行只剩正反歧义，试这两个朝向就够。
# 实测：补了竖排转正之后，61 个有框样本里 58 个与完整管线读数一致，
# 再多试 90/270 两个朝向准确率**一模一样**，只是白花一倍时间。
LINE_ORIENTATIONS: tuple[int, ...] = (0, 180)
# 低于此置信度的识别结果直接判为「未识别」
MIN_ACCEPT_CONFIDENCE: float = 0.35
# 透视裁剪出的文字行，高/宽 达到此值时判定检测框的点序把长短边判反了
# （文字竖排时必然发生），转 90° 摆正。编号 "B-299" 本身宽远大于高，
# 正常裁剪不可能触发。不补这一步，旋转 90°/270° 的碎片会读出 89/169/382 这类垃圾。
LINE_DESKEW_ROTATE_RATIO: float = 1.5

# ---------- PaddleOCR 运行时 ----------
# 模型下载源。国内 bos（百度自家）最快；可选 modelscope / aistudio / huggingface
PADDLE_MODEL_SOURCE: str = "bos"
# oneDNN(MKLDNN) 加速。**必须保持 False**：paddle 3.3.1 开着它跑 PP-OCR 检测
# 模型会抛 ConvertPirAttribute2RuntimeAttribute 未实现。详见 recognize.py。
PADDLE_ENABLE_MKLDNN: bool = False
# 按行识别（跳过检测模型）用的 rec 模型名。**必须与 PaddleOCR 主管线实际
# 选用的一致**，否则按行识别的读数会与 Pass A 系统性不一致。
# 本机 paddleocr 3.7.0 实测主管线加载的是 PP-OCRv6_medium_rec。
# 换 PaddleOCR 版本后先跑 scripts/probe_paddleocr.py 确认这个名字还对。
PADDLE_REC_MODEL_NAME: str = "PP-OCRv6_medium_rec"

# ---------- 渲染 ----------
# 非目标区域的亮度系数。0.0 全黑，1.0 不变。保留空间定位参照
DIM_FACTOR: float = 0.45
# 目标碎片描边颜色（BGR）：洋红，与黑背景和白碎片都拉得开
OUTLINE_COLOR: tuple[int, int, int] = (255, 0, 255)
# 未识别碎片的描边颜色（BGR）：青色
UNKNOWN_OUTLINE_COLOR: tuple[int, int, int] = (255, 255, 0)
# 描边粗细相对图像长边的比例
OUTLINE_THICKNESS_RATIO: float = 0.0035
