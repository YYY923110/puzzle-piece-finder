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
# 编号区间已实测确定，不再从识别结果自举。四段互不重叠且铺满 1–1000，
# 与拼图块数一一对应，于是**数字唯一确定前缀**——前缀退化成一位校验码。
# 前缀误读因此必定暴露（A-403 不是合法编号）并可由数字纠正回来。
CODE_RANGES: dict[str, tuple[int, int]] = {
    "A": (1, 260),
    "B": (261, 520),
    "C": (521, 760),
    "D": (761, 1000),
}
# 碎片上**不补零**：A-1 / A-42 / A-100 / D-1000。数字段因此是 1–4 位不定长，
# 全码长度在 3–6 之间浮动，代码里不能再假定「长度恒为 5」。
CODE_PATTERN = re.compile(r"^([A-D])-([1-9]\d{0,3})$")

# 吸附预算的绝对上限。
SNAP_MAX_DISTANCE: float = 2.0
# 每个数字位允许的编辑预算，实际上限取 min(上限, 位数 × 本值)。
# 为什么要随位数缩放：2.0 对五字符的 B-403 是「允许 40% 的字符出错」，
# 对三字符的 A-1 却是 67%——噪声会被吸附成真编号，产生假阳性。
# 0.7 使三位数字仍是 2.0（现有行为一字不变），两位 1.4，一位 0.7
# ——一位数字读不出来时只放行形近替换，不做无依据的猜测。
SNAP_DISTANCE_PER_DIGIT: float = 0.7
# 次优候选必须比最优差这么多，否则判为歧义、拒绝作答。
# 不补零的代价：漏读一位数字和读错前缀常常等距（D-97 补哪一位都说得通），
# 这种时候猜错会占掉一个真编号，比留空贵得多。
SNAP_AMBIGUITY_MARGIN: float = 0.25
# OCR 凭空多读出一个字符的代价。定得比漏读贵，依据是实测：真实照片里的
# 失败读数 D-97 / D-89 / D-83 全是**漏读**，没有一例是多读。
# 少了这个不对称，A-403 到 B-403（换前缀）与到 A-40（删尾数）永远平局。
SNAP_SPURIOUS_CHAR_COST: float = 1.5
# OCR 漏读一个字符的代价。
SNAP_DROPPED_CHAR_COST: float = 1.0

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
# 注：这里曾有一个 CROP_MASK_SEAL_RATIO——用闭运算去补碎片轮廓上的那道缝。
# 实测证明那条路走不通（缝的口子就是拼图凹口的口子，核大到够得着字符时，
# 凹口连同深色背景一起被灌进裁剪图，识别率反而塌一半），已删除。
# 完整理由在 segment.crop_piece 的 docstring 里。别再加回来。

# ---------- 识别 ----------
# 直接识别（Pass A）置信度低于此值的碎片，进入旋转穷举（Pass C）
# 默认激进：宁可多穷举。跑过真实照片后按实测数据下调。
SWEEP_CONFIDENCE_THRESHOLD: float = 0.90
# 穷举的角度表（度）。若方向分类器可靠，可缩短为 (0, 30, 60, 90, 120, 150)
SWEEP_ANGLES: tuple[int, ...] = (0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330)
# 穷举时拿到不低于此置信度的合法编号就收工，不再试剩余角度。
# 实测依据：一轮标定里 9 次穷举命中有 5 次置信度 ≥0.99，剩余角度是白跑的。
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
