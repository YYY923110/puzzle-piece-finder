"""命令行入口。存在的意义是让引擎能脱离 Web 单独跑——调分割和识别
参数时，改一个数字然后 `puzzlefind index photo.jpg` 看结果，比每次
都起服务器传图快得多。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from . import config, render
from .library import Library
from .pipeline import build_index


def _make_backend():
    """惰性构造 PaddleOCR 后端。测试里被 monkeypatch 掉。"""
    from .recognize import PaddleBackend

    return PaddleBackend()


def _cmd_index(args: argparse.Namespace) -> int:
    photo = Path(args.photo)
    try:
        index = build_index(photo, _make_backend(), photo_id=args.photo_id)
    except (FileNotFoundError, ValueError) as error:
        print(f"错误: {error}", file=sys.stderr)
        return 2

    library = Library.load(Path(args.index_dir) if args.index_dir else None)
    library.save_photo(index)

    total = len(index.pieces)
    hit = len(index.recognized)
    sweeps = sum(1 for p in index.pieces if p.method == "sweep")
    print(f"照片 {index.photo_id}: 分割出 {total} 块，识别 {hit} 块，未识别 {total - hit} 块")
    print(f"  其中靠旋转穷举救回: {sweeps} 块")
    if total:
        print(f"  识别率: {hit / total:.1%}")
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    library = Library.load(Path(args.index_dir) if args.index_dir else None)
    result = library.query(args.code)

    if result.found:
        assert result.piece is not None
        print(f"{result.code} → 照片 {result.photo_id}，碎片 #{result.piece.piece_id}，"
              f"包围盒 {result.piece.bbox}，置信度 {result.piece.confidence:.2f}")
        if args.out:
            _write_highlight(library, result.photo_id, [result.piece], args.out, False)
        return 0

    print(f"{result.code} 未找到。")
    for photo_id, pieces in result.unrecognized.items():
        print(f"  照片 {photo_id} 有 {len(pieces)} 块未识别碎片: "
              f"{[p.piece_id for p in pieces]}")
    if not result.unrecognized:
        print("  所有碎片均已识别——这个编号确实不在任何一张照片里。")
    return 1


def _cmd_stats(args: argparse.Namespace) -> int:
    library = Library.load(Path(args.index_dir) if args.index_dir else None)
    if not library.photos:
        print("索引库是空的。")
        return 0
    for photo in library.photos:
        total = len(photo.pieces)
        hit = len(photo.recognized)
        print(f"{photo.photo_id}: {hit}/{total} 已识别  (建于 {photo.created_at})")
    return 0


def _write_highlight(library, photo_id, pieces, out_path, unknown) -> None:
    photo = next(p for p in library.photos if p.photo_id == photo_id)
    image = cv2.imread(photo.image_path)
    if image is None:
        print(f"警告: 无法读取原图 {photo.image_path}，跳过高亮输出", file=sys.stderr)
        return
    cv2.imwrite(str(out_path), render.highlight(image, pieces, unknown=unknown))
    print(f"  高亮图已写入 {out_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="puzzlefind", description="拼图碎片编号查找器")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_index = subparsers.add_parser("index", help="对一张照片建立索引")
    p_index.add_argument("photo")
    p_index.add_argument("--photo-id", default=None)
    p_index.set_defaults(func=_cmd_index)

    p_query = subparsers.add_parser("query", help="查询一个编号")
    p_query.add_argument("code")
    p_query.add_argument("--out", default=None, help="把高亮图写到这个路径")
    p_query.set_defaults(func=_cmd_query)

    p_stats = subparsers.add_parser("stats", help="打印索引库概况")
    p_stats.set_defaults(func=_cmd_stats)

    # --index-dir 只挂在子命令上，不挂全局。
    #
    # 陷阱：argparse 里同名参数同时出现在主解析器和子解析器上时，
    # 子解析器后解析，会用它的默认值 None **覆盖掉**主解析器已经解析到的
    # 值。于是 `puzzlefind --index-dir X query CODE` 会静默失效——不报错，
    # 只是悄悄用了默认索引目录。只在一处定义就没有这个问题。
    for sub in (p_index, p_query, p_stats):
        sub.add_argument(
            "--index-dir", default=None, help=f"索引目录，默认 {config.INDEX_DIR}"
        )

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
