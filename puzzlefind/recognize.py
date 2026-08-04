"""识别层：把碎片裁剪图变成编号。

设计要点：OcrBackend 是一个 Protocol，PaddleOCR 只是它的一个实现。
这让绝大多数测试可以注入假后端，不必加载模型。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from . import config, vocabulary


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


def prepare_paddle_env() -> None:
    """在 import paddleocr **之前**把两个环境变量摆正。

    这两条都是本机实测踩出来的，不是照抄文档：

    1. `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK` — paddlex 在下载模型前会
       逐个 HEAD 探测 HuggingFace / AIStudio / ModelScope / BOS，超时只给
       **1 秒**。这四个站点在本机都能正常访问，但没有一个能在 1 秒内应答，
       于是 paddlex 报「No available model hosting platforms detected.
       Please check your network connection」——一句完全误导人的错误。
       跳过探测即可，反正真正下载失败时还会报错。
    2. `PADDLE_PDX_MODEL_SOURCE=bos` — 把百度自家的 BOS 排到第一位，
       国内下载最快。

    paddlex 在 import 时就把这些读进模块级常量，所以必须赶在 import 前设置。
    用 setdefault：用户显式设过就不覆盖。
    """
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", config.PADDLE_MODEL_SOURCE)


class PaddleBackend:
    """PaddleOCR 3.x 后端。模型惰性加载——首次 read 时才初始化。"""

    def __init__(self, lang: str = "en") -> None:
        self._lang = lang
        self._ocr = None

    def _ensure_loaded(self) -> None:
        if self._ocr is None:
            prepare_paddle_env()
            from paddleocr import PaddleOCR

            # 关掉文档级方向分类和去扭曲：我们的输入是单块碎片的小图，
            # 不是扫描文档，那两个模块只会拖慢速度并引入误判。
            # 保留 textline 方向分类——它负责 180° 正反歧义。
            #
            # enable_mkldnn 必须显式关掉（默认是开的）。开着时 paddle 3.3.1
            # 在本机 CPU 上跑检测模型会直接抛：
            #   NotImplementedError: (Unimplemented)
            #   ConvertPirAttribute2RuntimeAttribute not support
            #   [pir::ArrayAttribute<pir::DoubleAttribute>]
            # 这是 oneDNN 算子在 PIR 执行器下的缺口，不是我们能绕的，
            # 只能关掉 oneDNN 加速。代价是 CPU 推理慢一些——而建索引是
            # 一次性开销，这个代价可以接受。
            self._ocr = PaddleOCR(
                lang=self._lang,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=True,
                enable_mkldnn=config.PADDLE_ENABLE_MKLDNN,
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

    本机 paddleocr 3.7.0 / PP-OCRv6 的实测结构（探针输出）：
        res.json == {"res": {..., "rec_texts": ["B-403"],
                             "rec_scores": [0.9999],
                             "dt_polys": [[[145,142],[348,142],[348,231],[145,231]]],
                             ...}}
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
