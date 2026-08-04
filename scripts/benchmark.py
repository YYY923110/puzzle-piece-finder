"""建索引耗时基准。改识别策略前后各跑一次，用于对照。

跑法:
    .\\.venv\\Scripts\\python.exe scripts\\benchmark.py data\\photos\\real6.jpg
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from puzzlefind import config
from puzzlefind.pipeline import build_index
from puzzlefind.recognize import PaddleBackend


def main() -> int:
    if len(sys.argv) < 2:
        print("用法: python scripts/benchmark.py <照片路径>", file=sys.stderr)
        return 2

    photo = Path(sys.argv[1])
    if not photo.exists():
        print(f"照片不存在: {photo}", file=sys.stderr)
        return 2

    backend = PaddleBackend()
    # 预热：把模型加载的时间排除在计时之外，否则首次跑会虚高几秒
    import numpy as np

    backend.read(np.full((64, 200, 3), 220, dtype=np.uint8))

    start = time.perf_counter()
    index = build_index(photo, backend)
    elapsed = time.perf_counter() - start

    total = len(index.pieces)
    hit = len(index.recognized)
    methods = Counter(p.method for p in index.pieces)

    report = {
        "photo": str(photo),
        "total": total,
        "recognized": hit,
        "rate": round(hit / total, 4) if total else 0.0,
        "seconds": round(elapsed, 1),
        "seconds_per_piece": round(elapsed / total, 3) if total else 0.0,
        "methods": dict(methods),
        "codes": sorted(p.code for p in index.recognized),
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    out_dir = config.DEBUG_DIR / photo.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "benchmark.json"
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n已写入 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
