import numpy as np
import pytest

from puzzlefind import recognize
from puzzlefind.recognize import RawDetection


class FakeBackend:
    """按调用顺序吐出预设结果的假 OCR 后端。

    存在的意义：让识别逻辑的测试与 PaddleOCR 完全解耦，毫秒级跑完，
    不需要下载模型。真正的 PaddleBackend 只在带 @pytest.mark.ocr
    的测试里被碰。
    """

    def __init__(self, responses: list[list[RawDetection]]):
        self.responses = responses
        self.calls = 0

    def read(self, image: np.ndarray) -> list[RawDetection]:
        result = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return result


@pytest.fixture
def blank_crop() -> np.ndarray:
    return np.full((100, 100, 3), 200, dtype=np.uint8)


class TestRecognizeDirect:
    def test_clean_read_produces_code(self, blank_crop):
        backend = FakeBackend([[RawDetection("B-403", 0.97)]])
        result = recognize.recognize_direct(backend, blank_crop)
        assert result.code == "B-403"
        assert result.confidence == pytest.approx(0.97)
        assert result.method == "direct"

    def test_confusable_read_is_snapped_to_vocabulary(self, blank_crop):
        backend = FakeBackend([[RawDetection("8-4O3", 0.88)]])
        result = recognize.recognize_direct(backend, blank_crop)
        assert result.code == "B-403"
        assert result.raw_text == "8-4O3"

    def test_unsnappable_text_yields_no_code(self, blank_crop):
        backend = FakeBackend([[RawDetection("QWERTY", 0.99)]])
        result = recognize.recognize_direct(backend, blank_crop)
        assert result.code is None

    def test_empty_detection_yields_no_code(self, blank_crop):
        backend = FakeBackend([[]])
        result = recognize.recognize_direct(backend, blank_crop)
        assert result.code is None
        assert result.confidence == 0.0

    def test_low_confidence_read_is_rejected(self, blank_crop):
        backend = FakeBackend([[RawDetection("B-403", 0.10)]])
        result = recognize.recognize_direct(backend, blank_crop)
        assert result.code is None

    def test_picks_highest_scoring_detection_among_several(self, blank_crop):
        backend = FakeBackend(
            [[RawDetection("A-111", 0.55), RawDetection("B-403", 0.93)]]
        )
        result = recognize.recognize_direct(backend, blank_crop)
        assert result.code == "B-403"


@pytest.mark.ocr
class TestPaddleBackendIntegration:
    def test_reads_rendered_code_from_synthetic_image(self):
        import cv2

        image = np.full((160, 480, 3), 245, dtype=np.uint8)
        cv2.putText(
            image, "B-403", (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (90, 90, 90), 6
        )
        backend = recognize.PaddleBackend()
        result = recognize.recognize_direct(backend, image)
        assert result.code == "B-403"
