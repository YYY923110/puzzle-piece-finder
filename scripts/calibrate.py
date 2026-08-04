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

from puzzlefind import config, render, segment, vocabulary
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
    unit_area = segment.unit_piece_area(image.shape[:2], blobs)
    overlay = image.copy()
    cv2.drawContours(overlay, blobs, -1, (0, 255, 255), 2)
    cv2.imwrite(str(debug / "02_blobs.png"), overlay)
    print(f"[2] 连通块 {len(blobs)} 个，中位面积 {median_area:.0f} px²")
    print(f"    单块碎片面积估计 {unit_area:.0f} px²（山峰数归一化后的中位数）")
    print("    两者差很多说明粘连团块占比高——正常，切分那一步会处理。")

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

    # 自举区间：这是 spec §4「各字母组的数字区间未知」那个悬案的答案
    codes = [p.code for p in index.recognized if p.code]
    print("\n=== 自举出的编号区间 ===")
    plain = vocabulary.bootstrap_ranges(codes)
    fenced = vocabulary.robust_ranges(codes)
    if not plain:
        print("样本太少，还推不出区间。")
    for prefix in sorted(plain):
        low, high = plain[prefix]
        extra = f"，围栏收紧后 {fenced[prefix]}" if prefix in fenced else ""
        print(f"  {prefix} 组: {low:03d}–{high:03d}{extra}")
    print("把这个区间记到 spec 的 §4 里，那条悬案就结了。")

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
