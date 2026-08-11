import re

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from puzzlefind import server
from puzzlefind.recognize import RawDetection


class CountingBackend:
    """每次调用返回一个递增编号，让每块碎片拿到不同的 code。"""

    def __init__(self):
        self.n = 0

    def read(self, image: np.ndarray) -> list[RawDetection]:
        self.n += 1
        return [RawDetection(f"B-{260 + self.n}", 0.99)]


@pytest.fixture
def client(tmp_path):
    app = server.create_app(
        index_dir=tmp_path / "index",
        photos_dir=tmp_path / "photos",
        backend_factory=CountingBackend,
    )
    return TestClient(app)


@pytest.fixture
def photo_bytes(separated_pieces) -> bytes:
    image, _ = separated_pieces
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    return buffer.tobytes()


class TestUpload:
    def test_upload_returns_summary(self, client, photo_bytes, separated_pieces):
        _, count = separated_pieces
        response = client.post(
            "/api/photos", files={"file": ("shot.jpg", photo_bytes, "image/jpeg")}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == count
        assert body["recognized"] == count

    def test_upload_makes_photo_listable(self, client, photo_bytes):
        client.post("/api/photos", files={"file": ("shot.jpg", photo_bytes, "image/jpeg")})
        listing = client.get("/api/photos").json()
        assert len(listing["photos"]) == 1

    def test_non_image_upload_is_rejected(self, client):
        response = client.post(
            "/api/photos", files={"file": ("x.txt", b"not an image", "text/plain")}
        )
        assert response.status_code == 400

    def test_explicit_photo_id_wins_over_the_filename(self, client, photo_bytes):
        """手机传上来的文件名是浏览器现造的时间戳，用户给的名字必须压过它。"""
        client.post(
            "/api/photos",
            files={"file": ("1786177906346.jpg", photo_bytes, "image/jpeg")},
            data={"photo_id": "2"},
        )
        listing = client.get("/api/photos").json()
        assert [p["photo_id"] for p in listing["photos"]] == ["2"]

    def test_reshooting_the_same_region_replaces_the_index(self, client, photo_bytes):
        """重拍同一片区域必须**替换**，不是再堆一份。

        这是本功能的核心断言。photo_id 不稳定时，design.md §2 承诺的
        「重拍刷新」实际上在积累过期数据：旧索引原地留下来，而查询跨所有
        照片扫，可能命中那份陈旧的。两次上传的文件名故意取不同，正是为了
        证明替换只由 photo_id 决定。
        """
        for filename in ("shot-a.jpg", "shot-b.jpg"):
            response = client.post(
                "/api/photos",
                files={"file": (filename, photo_bytes, "image/jpeg")},
                data={"photo_id": "2"},
            )
            assert response.status_code == 200
        assert len(client.get("/api/photos").json()["photos"]) == 1

    def test_illegal_photo_id_is_rejected(self, client, photo_bytes):
        response = client.post(
            "/api/photos",
            files={"file": ("shot.jpg", photo_bytes, "image/jpeg")},
            data={"photo_id": "a/b"},
        )
        assert response.status_code == 400

    def test_photo_id_may_be_omitted(self, client, photo_bytes):
        """不给这个字段时行为与从前一致——curl 上传不受影响。"""
        client.post(
            "/api/photos", files={"file": ("shot.jpg", photo_bytes, "image/jpeg")}
        )
        listing = client.get("/api/photos").json()
        assert [p["photo_id"] for p in listing["photos"]] == ["shot"]

    def test_an_empty_photo_id_counts_as_omitted_not_as_an_error(
        self, client, photo_bytes
    ):
        """空的 photo_id 等同于没给，**不会**被 sanitize_photo_id 拒绝。

        钉住这条是因为它看上去像一道防线，其实不是：FastAPI 对 `Form(None)`
        的空字符串直接套用默认值，handler 拿到的是 None，和「字段压根不存在」
        不可区分（实测过——手工构造一个带空值字段的 multipart 请求，handler
        同样只看到 None，与客户端无关）。所以 sanitize_photo_id 里那条
        「名字不能为空」在 HTTP 路径上永远不会被触发。

        真正拦住空名字的是前端：confirmNewRegion 遇到空输入直接 return，
        而 onchange 里的 `if (region)` 决定这个字段发不发。
        """
        client.post(
            "/api/photos",
            files={"file": ("shot.jpg", photo_bytes, "image/jpeg")},
            data={"photo_id": ""},
        )
        listing = client.get("/api/photos").json()
        assert [p["photo_id"] for p in listing["photos"]] == ["shot"]


class TestQuery:
    def test_hit_returns_piece_geometry(self, client, photo_bytes):
        client.post("/api/photos", files={"file": ("shot.jpg", photo_bytes, "image/jpeg")})
        body = client.get("/api/query", params={"code": "B-261"}).json()
        assert body["found"] is True
        assert "bbox" in body["piece"]
        assert body["photo_id"]

    def test_miss_returns_unrecognized_summary(self, client, photo_bytes):
        client.post("/api/photos", files={"file": ("shot.jpg", photo_bytes, "image/jpeg")})
        body = client.get("/api/query", params={"code": "D-999"}).json()
        assert body["found"] is False
        assert "unrecognized" in body

    def test_query_without_any_photo_is_a_clean_miss(self, client):
        body = client.get("/api/query", params={"code": "B-261"}).json()
        assert body["found"] is False


class TestHighlight:
    def test_returns_png_for_a_hit(self, client, photo_bytes):
        client.post("/api/photos", files={"file": ("shot.jpg", photo_bytes, "image/jpeg")})
        response = client.get("/api/highlight", params={"code": "B-261"})
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

    def test_returns_png_for_a_miss_showing_unrecognized(self, client, photo_bytes):
        client.post("/api/photos", files={"file": ("shot.jpg", photo_bytes, "image/jpeg")})
        response = client.get(
            "/api/highlight", params={"code": "D-999", "photo_id": "shot"}
        )
        assert response.status_code == 200

    def test_unknown_photo_id_returns_404(self, client):
        response = client.get(
            "/api/highlight", params={"code": "B-261", "photo_id": "nope"}
        )
        assert response.status_code == 404


class TestThumbnail:
    def test_returns_png(self, client, photo_bytes):
        client.post("/api/photos", files={"file": ("shot.jpg", photo_bytes, "image/jpeg")})
        response = client.get("/api/thumbnail", params={"code": "B-261"})
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

    def test_unknown_code_returns_404(self, client, photo_bytes):
        client.post("/api/photos", files={"file": ("shot.jpg", photo_bytes, "image/jpeg")})
        assert client.get("/api/thumbnail", params={"code": "Z-999"}).status_code == 404


class TestDelete:
    def test_deleting_removes_from_listing(self, client, photo_bytes):
        client.post("/api/photos", files={"file": ("shot.jpg", photo_bytes, "image/jpeg")})
        assert client.delete("/api/photos/shot").status_code == 200
        assert client.get("/api/photos").json()["photos"] == []


class TestFrontend:
    def test_root_serves_html(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_html_wires_up_the_query_endpoint(self, client):
        body = client.get("/").text
        assert "/api/query" in body

    def test_html_wires_up_the_upload_endpoint(self, client):
        body = client.get("/").text
        assert "/api/photos" in body

    def test_html_accepts_images_without_forcing_the_camera(self, client):
        """上传入口必须收图片，但**不能**锁死成现拍。

        原来这里带着 capture="environment"，手机上点「上传照片」会直接
        跳进相机，相册里已经拍好的照片根本选不了。去掉之后手机弹出
        「拍照 / 相册 / 文件」三选一，两条路都在。
        """
        body = client.get("/").text
        # 只看 input 标签本身——页面上那条解释为什么不加 capture 的注释里
        # 也有这个词，整页搜字符串会误判
        tag = re.search(r"<input[^>]*type=\"file\"[^>]*>", body)
        assert tag is not None, "页面里没有文件上传 input"
        assert 'accept="image/*"' in tag.group(0)
        assert "capture" not in tag.group(0)

    def test_html_asks_which_region_before_uploading(self, client):
        """上传前必须先问「这张拍的是哪片区域」，并把答案发出去。

        用户在手机上把照片改名成 1/2/3/4，但那个名字从来没进过 HTTP 请求
        ——安卓相册交给浏览器的是一个不带显示名的句柄，浏览器用点选时刻的
        毫秒时间戳兜底。名字只能在这里问，没法从文件名里抢救。
        """
        body = client.get("/").text
        assert 'id="regionPicker"' in body
        assert 'form.append("photo_id"' in body
