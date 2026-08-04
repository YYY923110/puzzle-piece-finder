"""FastAPI 薄服务层。所有真正的逻辑都在引擎模块里，这里只做
HTTP 编解码和文件落盘。

高亮图在服务端渲染（复用已被单元测试覆盖的 render.py），前端只
负责显示和缩放——这让渲染逻辑可测，也让前端代码降到几十行。
"""
from __future__ import annotations

import sys
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


def _pick_lan_ip(candidates: list[str]) -> str:
    """从本机所有 IPv4 里挑一个手机真能连上的。

    不能只用「连一下 8.8.8.8 看本地端点」那个经典技巧：本机装了代理
    （Clash/Mihomo 之类）时，它的 TUN 网卡会接管默认路由，那个技巧
    返回的是 198.18.0.0——RFC2544 基准测试段，手机连不上，而且报错
    形式是「连接超时」，跟防火墙拦截长得一模一样，排查起来很费劲。

    所以改成按网段优先级挑：192.168 最优（家用路由器几乎都发这个），
    其次 10.x，最后 172.16–31。虚拟网卡（WSL 常占 172.22）因此自然
    排在真实无线网卡后面。
    """
    def rank(ip: str) -> int:
        if ip.startswith("192.168."):
            return 0
        if ip.startswith("10."):
            return 1
        if ip.startswith("172."):
            second = int(ip.split(".")[1]) if ip.count(".") == 3 else 0
            return 2 if 16 <= second <= 31 else 99
        return 99

    usable = sorted((ip for ip in candidates if rank(ip) < 99), key=rank)
    return usable[0] if usable else "127.0.0.1"


def _local_ipv4_addresses() -> list[str]:
    """本机所有 IPv4 地址，尽量把每块网卡都枚举出来。

    三个来源依次尝试，因为没有哪一个在所有情况下都够用：

    1. psutil —— 唯一能真正遍历所有网卡的。本机实测：装了代理时，
       只有它能看见 WLAN 的 192.168.2.119，另外两种方法全都只返回
       代理 TUN 网卡的 198.18.0.0。psutil 是 paddlex 的依赖，装好
       PaddleOCR 就一定在，所以不额外声明为直接依赖，但它缺席时也不能崩。
    2. 主机名解析 —— psutil 不在时的退路。
    3. 连 8.8.8.8 的 UDP 技巧 —— 没有代理时它是对的，留着做兜底。
    """
    import socket

    addresses: list[str] = []

    def add(address: str) -> None:
        if address and address not in addresses:
            addresses.append(address)

    try:
        import psutil

        for interface_addrs in psutil.net_if_addrs().values():
            for addr in interface_addrs:
                if addr.family == socket.AF_INET:
                    add(addr.address)
    except Exception:
        pass

    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            add(info[4][0])
    except OSError:
        pass

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        add(sock.getsockname()[0])
    except OSError:
        pass
    finally:
        sock.close()

    return addresses


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    """起服务并打印局域网访问地址。"""
    import uvicorn

    lan_ip = _pick_lan_ip(_local_ipv4_addresses())

    print(f"\n手机浏览器打开: http://{lan_ip}:{port}\n")
    if lan_ip == "127.0.0.1":
        print("没找到局域网地址。确认电脑连着 Wi-Fi，和手机在同一个网络下。\n")
    print("若手机连不上，检查 Windows 防火墙是否放行了该端口。\n")
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    # 端口可覆盖：本机 8000 已被另一个 python 进程占着，而 Windows 允许
    # 两个进程绑同一端口却不报错，请求会随机落到其中一个——排查起来很费劲。
    #   python -m puzzlefind.server 8765
    import os as _os

    _port = int(sys.argv[1]) if len(sys.argv) > 1 else int(_os.environ.get("PUZZLEFIND_PORT", 8000))
    run(port=_port)
