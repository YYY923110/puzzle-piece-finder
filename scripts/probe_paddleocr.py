"""一次性探针：打印 PaddleOCR 3.x predict() 的真实返回结构。

写 PaddleBackend 的适配层之前先跑这个。3.x 的返回对象结构与 2.x
完全不同，网上教程大多过时，凭记忆写必错。
"""
from __future__ import annotations

import cv2
import numpy as np

from puzzlefind.recognize import prepare_paddle_env

prepare_paddle_env()  # 必须在 import paddleocr 之前，见该函数注释

from paddleocr import PaddleOCR  # noqa: E402


def make_sample() -> np.ndarray:
    image = np.full((160, 480, 3), 245, dtype=np.uint8)
    cv2.putText(image, "B-403", (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (90, 90, 90), 6)
    return image


def main() -> None:
    ocr = PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
    )
    results = ocr.predict(make_sample())
    print(f"type(results) = {type(results)}, len = {len(results)}")
    for i, res in enumerate(results):
        print(f"\n--- result[{i}] type = {type(res)} ---")
        print("dir():", [a for a in dir(res) if not a.startswith('_')])
        payload = getattr(res, "json", None)
        print("res.json =", payload)


if __name__ == "__main__":
    main()
