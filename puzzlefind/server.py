"""FastAPI 薄服务层。所有真正的逻辑都在引擎模块里，这里只做
HTTP 编解码和文件落盘。

高亮图在服务端渲染（复用已被单元测试覆盖的 render.py），前端只
负责显示和缩放——这让渲染逻辑可测，也让前端代码降到几十行。
"""
from __future__ import annotations

import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response

from . import config, render
from .library import Library
from .pipeline import build_index

_STATIC_DIR = Path(__file__).resolve().parent / "static"


def _default_backend_factory():
    from .recognize import PaddleBackend

    return PaddleBackend()


def _png_response(image: np.ndarray) -> Response:
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise HTTPException(status_code=500, detail="PNG 编码失败")
    return Response(content=buffer.tobytes(), media_type="image/png")


def create_app(
    index_dir: Path | None = None,
    photos_dir: Path | None = None,
    backend_factory=None,
) -> FastAPI:
    app = FastAPI(title="拼图碎片编号查找器")

    resolved_index_dir = Path(index_dir) if index_dir else config.INDEX_DIR
    resolved_photos_dir = Path(photos_dir) if photos_dir else config.PHOTOS_DIR
    resolved_photos_dir.mkdir(parents=True, exist_ok=True)
    make_backend = backend_factory or _default_backend_factory

    # 后端惰性单例：PaddleOCR 模型加载很慢，不能每次请求都建一个
    state: dict = {"backend": None}

    def backend():
        if state["backend"] is None:
            state["backend"] = make_backend()
        return state["backend"]

    def library() -> Library:
        return Library.load(resolved_index_dir)

    def load_photo_image(photo_id: str) -> np.ndarray:
        photo = next(
            (p for p in library().photos if p.photo_id == photo_id), None
        )
        if photo is None:
            raise HTTPException(status_code=404, detail=f"照片不存在: {photo_id}")
        image = cv2.imread(photo.image_path)
        if image is None:
            raise HTTPException(status_code=404, detail=f"原图不可读: {photo.image_path}")
        return image

    @app.get("/")
    def index_page() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    @app.post("/api/photos")
    async def upload_photo(file: UploadFile = File(...)) -> dict:
        raw = await file.read()
        buffer = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="无法解码为图像")

        stem = Path(file.filename or "").stem or uuid.uuid4().hex[:8]
        photo_path = resolved_photos_dir / f"{stem}.jpg"
        cv2.imwrite(str(photo_path), image)

        photo_index = build_index(photo_path, backend(), photo_id=stem)
        lib = library()
        lib.save_photo(photo_index)

        total = len(photo_index.pieces)
        return {
            "photo_id": photo_index.photo_id,
            "total": total,
            "recognized": len(photo_index.recognized),
            "unrecognized": len(photo_index.unrecognized),
            "created_at": photo_index.created_at,
        }

    @app.get("/api/photos")
    def list_photos() -> dict:
        return {
            "photos": [
                {
                    "photo_id": p.photo_id,
                    "total": len(p.pieces),
                    "recognized": len(p.recognized),
                    "unrecognized": len(p.unrecognized),
                    "created_at": p.created_at,
                }
                for p in library().photos
            ]
        }

    @app.delete("/api/photos/{photo_id}")
    def delete_photo(photo_id: str) -> dict:
        if not library().delete_photo(photo_id):
            raise HTTPException(status_code=404, detail=f"照片不存在: {photo_id}")
        return {"deleted": photo_id}

    @app.get("/api/query")
    def query(code: str = Query(...)) -> dict:
        result = library().query(code)
        if result.found:
            assert result.piece is not None
            return {
                "found": True,
                "code": result.code,
                "photo_id": result.photo_id,
                "piece": result.piece.to_dict(),
            }
        return {
            "found": False,
            "code": result.code,
            "unrecognized": {
                photo_id: [p.piece_id for p in pieces]
                for photo_id, pieces in result.unrecognized.items()
            },
        }

    @app.get("/api/highlight")
    def highlight(code: str = Query(...), photo_id: str | None = None) -> Response:
        """命中时高亮目标碎片；未命中时高亮指定照片的全部未识别碎片。"""
        lib = library()
        result = lib.query(code)

        if result.found:
            assert result.piece is not None and result.photo_id is not None
            image = load_photo_image(result.photo_id)
            return _png_response(render.highlight(image, [result.piece]))

        if photo_id is None:
            raise HTTPException(
                status_code=404, detail=f"{code} 未找到，且未指定要查看哪张照片"
            )
        image = load_photo_image(photo_id)
        photo = next(p for p in lib.photos if p.photo_id == photo_id)
        return _png_response(render.highlight(image, photo.unrecognized, unknown=True))

    @app.get("/api/thumbnail")
    def thumbnail(code: str = Query(...), size: int = 200) -> Response:
        result = library().query(code)
        if not result.found:
            raise HTTPException(status_code=404, detail=f"{code} 未找到")
        assert result.piece is not None and result.photo_id is not None
        image = load_photo_image(result.photo_id)
        return _png_response(render.thumbnail(image, result.piece, size=size))

    return app


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    """起服务并打印局域网访问地址。"""
    import socket

    import uvicorn

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        lan_ip = sock.getsockname()[0]
    except OSError:
        lan_ip = "127.0.0.1"
    finally:
        sock.close()

    print(f"\n手机浏览器打开: http://{lan_ip}:{port}\n")
    print("若手机连不上，检查 Windows 防火墙是否放行了该端口。\n")
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    run()
