"""对一张真实照片做全流程标定，把每一步的中间结果落到 debug/。

opencv 是 headless 版，没有 imshow，所以一切靠写文件观察。

用法:
    python scripts/calibrate.py data/photos/桌面1.jpg
"""
from __future__ import annotations

import json
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
        # 与 pipeline 一致：把其他碎片当邻块涂灰
        neighbours = contours[:i] + contours[i + 1 :]
        cv2.imwrite(
            str(crops_dir / f"{i:03d}.png"),
            segment.crop_piece(image, contour, neighbours),
        )
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
    print(f"  Pass C 按行 {methods.get('line', 0)}   (复用检测框、只重跑 rec 的快路径)")
    print(f"  Pass C 穷举 {methods.get('sweep', 0)}   (det 没给出框时的回退)")
    print(f"  冲突降级    {methods.get('conflict', 0)}")
    print(f"未识别        {len(index.unrecognized)}")
    print(f"耗时          {elapsed:.1f}s  ({elapsed / max(1, total):.2f}s/块)")

    # 把索引落盘。一次标定要跑好几分钟，结果不留下来就只能重跑。
    index_path = debug / "index.json"
    index_path.write_text(
        json.dumps(index.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n索引已存到 {index_path}（可直接打开看每块碎片读成了什么）")

    # 冲突降级的明细。这些碎片 OCR 其实读出来了，是被唯一性规则否掉的，
    # 跟「压根没读出来」是完全不同的失败模式，排查方向也不同。
    demoted = [p for p in index.pieces if p.method == "conflict"]
    if demoted:
        print(f"\n=== 被冲突消解降级的 {len(demoted)} 块 ===")
        kept = {p.code: p for p in index.recognized}
        for piece in demoted:
            raw = piece.raw_text or ""
            snapped, _ = vocabulary.snap(raw) if raw else (None, 0.0)
            winner = kept.get(snapped) if snapped else None
            detail = (
                f"被 #{winner.piece_id}(置信度 {winner.confidence:.3f}) 挤掉"
                if winner else "原因见 raw_text"
            )
            print(f"  #{piece.piece_id:>3} 原始读数 {raw!r} → {snapped} : {detail}")
        print("  这些块要么是误读撞了车，要么是粘连导致一张裁剪图里有两个编号。")
        print("  对照 debug/<照片>/crops/ 里对应编号的图肉眼判一下是哪种。")

    # 置信度分布：直接决定 SWEEP_CONFIDENCE_THRESHOLD 该定在哪
    direct_confs = sorted(p.confidence for p in index.pieces if p.method == "direct")
    if direct_confs:
        n = len(direct_confs)
        print(f"\n=== Pass A 命中时的置信度分布（{n} 块）===")
        print(f"  最低 {direct_confs[0]:.3f}  25% {direct_confs[n//4]:.3f}  "
              f"中位 {direct_confs[n//2]:.3f}  最高 {direct_confs[-1]:.3f}")
        print(f"  当前阈值 {config.SWEEP_CONFIDENCE_THRESHOLD} 之下的有 "
              f"{sum(1 for c in direct_confs if c < config.SWEEP_CONFIDENCE_THRESHOLD)} 块"
              f"——它们本来直接读对了，却仍被迫跑完整轮穷举。")

    # 编号区间已确定（config.CODE_RANGES），不再自举。改报这张照片覆盖了哪一段
    # ——碎片按区域摊开，同一张照片的编号通常挤在一起，跨度异常大就是误读的信号。
    codes = [p.code for p in index.recognized if p.code]
    print("\n=== 编号覆盖 ===")
    if not codes:
        print("一个编号都没读出来。")
    for prefix, (low, high) in config.CODE_RANGES.items():
        numbers = sorted(int(c.split("-")[1]) for c in codes if c.startswith(prefix))
        if not numbers:
            continue
        print(
            f"  {prefix} 组({low}–{high}): {len(numbers)} 块，"
            f"实际跨 {numbers[0]}–{numbers[-1]}"
        )
    if len(set(c[0] for c in codes)) > 1:
        print("  这张照片跨了多个字母组。碎片若是按区域摊开的，这多半是前缀误读。")

    # --- 阶段 6：可视化 ---
    cv2.imwrite(
        str(debug / "04_unrecognized.png"),
        render.highlight(image, index.unrecognized, unknown=True),
    )
    print(f"\n未识别碎片可视化: {debug / '04_unrecognized.png'}")

    print("\n=== 调参建议 ===")
    direct = methods.get("direct", 0)
    if not total:
        print("一块碎片都没分割出来，先看 01_mask.png。")
    elif demoted and len(demoted) >= max(2, total * 0.1):
        # 这一条要排在覆盖率建议前面：降级块数多时，它是最大的损失来源，
        # 而且调 SWEEP_CONFIDENCE_THRESHOLD 对它一点用都没有。
        print(f"最大的损失来源是冲突降级（{len(demoted)}/{total} 块），不是识别不出来。")
        print("这些碎片 OCR 都读出编号了，是撞了车才被丢掉。优先查上面那份明细：")
        print("  · 若是两块碎片粘在一张裁剪图里 → 调分割，不是调识别")
        print("  · 若是形近误读（B-529 读成 B-520 之类）→ 提高 CROP_TARGET_LONG_EDGE，")
        print("    或者下次每张少拍几块，让字更大")
    elif methods.get("sweep", 0) >= max(2, total * 0.1):
        # 耗时几乎全在这里：全量穷举要跑 12 遍最贵的检测模型，而按行识别
        # 只重跑 rec，便宜约 77 倍。穷举块数多 = 检测器在这些碎片上没给出框。
        sweep_n = methods.get("sweep", 0)
        print(f"有 {sweep_n}/{total} 块走了**全量角度穷举**，这是目前的耗时大头——")
        print("每块要跑 12 遍检测模型，比按行识别贵约 77 倍。")
        print("穷举只在「检测器压根没给出文字框」时才会触发，所以要查的是检测失败：")
        print("  · 看 crops/ 里对应的图，字是不是太小或太糊 → 每张少拍几块")
        print("  · 编号是不是被画幅或邻块切掉了一截")
    else:
        line_n = methods.get("line", 0)
        print(f"Pass A 直读 {direct / total:.0%}，按行识别 {line_n / total:.0%}，"
              f"全量穷举 {methods.get('sweep', 0) / total:.0%}，配比正常。")
        print("耗时已经由检测模型主导（占单次识别 92%），而每块只跑一次检测，")
        print("没有明显的浪费了。SWEEP_CONFIDENCE_THRESHOLD 在合格照片上基本空转，")
        print("调它不会有可观收益——下一个杠杆是多进程，或彻底绕开检测模型。")
    if total and hit / total < 0.7:
        print("识别率偏低。优先排查顺序：")
        print("  a) 看 crops/ 里的字够不够大 → 下次每张少拍点碎片")
        print("  b) 看 01_mask.png 背景压住没有 → 换更深的背景布")
        print("  c) 看 03_split.png 切分对不对 → 调 SPLIT_AREA_RATIO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
