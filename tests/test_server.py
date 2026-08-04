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
        return [RawDetection(f"B-{self.n:03d}", 0.99)]


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


class TestQuery:
    def test_hit_returns_piece_geometry(self, client, photo_bytes):
        client.post("/api/photos", files={"file": ("shot.jpg", photo_bytes, "image/jpeg")})
        body = client.get("/api/query", params={"code": "B-001"}).json()
        assert body["found"] is True
        assert "bbox" in body["piece"]
        assert body["photo_id"]

    def test_miss_returns_unrecognized_summary(self, client, photo_bytes):
        client.post("/api/photos", files={"file": ("shot.jpg", photo_bytes, "image/jpeg")})
        body = client.get("/api/query", params={"code": "D-999"}).json()
        assert body["found"] is False
        assert "unrecognized" in body

    def test_query_without_any_photo_is_a_clean_miss(self, client):
        body = client.get("/api/query", params={"code": "B-001"}).json()
        assert body["found"] is False


class TestHighlight:
    def test_returns_png_for_a_hit(self, client, photo_bytes):
        client.post("/api/photos", files={"file": ("shot.jpg", photo_bytes, "image/jpeg")})
        response = client.get("/api/highlight", params={"code": "B-001"})
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
            "/api/highlight", params={"code": "B-001", "photo_id": "nope"}
        )
        assert response.status_code == 404


class TestThumbnail:
    def test_returns_png(self, client, photo_bytes):
        client.post("/api/photos", files={"file": ("shot.jpg", photo_bytes, "image/jpeg")})
        response = client.get("/api/thumbnail", params={"code": "B-001"})
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
