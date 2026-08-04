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


class TestBestPoly:
    def test_returns_none_when_no_detection_has_a_poly(self):
        detections = [RawDetection("B-403", 0.9), RawDetection("A-111", 0.5)]
        assert recognize.best_poly(detections) is None

    def test_returns_none_for_empty_detections(self):
        assert recognize.best_poly([]) is None

    def test_picks_poly_of_highest_scoring_detection(self):
        low = [[0, 0], [10, 0], [10, 5], [0, 5]]
        high = [[20, 20], [40, 20], [40, 30], [20, 30]]
        detections = [
            RawDetection("A-111", 0.40, low),
            RawDetection("B-403", 0.93, high),
        ]
        assert recognize.best_poly(detections) == high

    def test_ignores_high_scoring_detection_that_has_no_poly(self):
        poly = [[0, 0], [10, 0], [10, 5], [0, 5]]
        detections = [
            RawDetection("QWERTY", 0.99),          # 分最高但没有框
            RawDetection("B-403", 0.42, poly),
        ]
        assert recognize.best_poly(detections) == poly

    def test_poly_defaults_to_none_so_existing_call_sites_keep_working(self):
        assert RawDetection("B-403", 0.9).poly is None


class TestRotateExpand:
    def test_zero_degrees_returns_same_shape(self):
        image = np.zeros((60, 100, 3), dtype=np.uint8)
        assert recognize.rotate_expand(image, 0).shape == image.shape

    def test_ninety_degrees_swaps_dimensions(self):
        image = np.zeros((60, 100, 3), dtype=np.uint8)
        rotated = recognize.rotate_expand(image, 90)
        assert rotated.shape[0] == 100
        assert rotated.shape[1] == 60

    def test_forty_five_degrees_expands_canvas_without_clipping(self):
        image = np.zeros((60, 100, 3), dtype=np.uint8)
        rotated = recognize.rotate_expand(image, 45)
        assert rotated.shape[0] > 60
        assert rotated.shape[1] > 100


class TestRecognizeSweep:
    def test_finds_code_at_a_later_angle(self, blank_crop):
        from puzzlefind import config

        # 前两个角度读不出，第三个角度读出——模拟只有摆正后才认得出
        responses = [[], [RawDetection("???", 0.9)], [RawDetection("B-403", 0.95)]]
        backend = FakeBackend(responses)
        result = recognize.recognize_sweep(backend, blank_crop)
        assert result.code == "B-403"
        assert result.method == "sweep"
        assert result.angle == config.SWEEP_ANGLES[2]

    def test_tries_every_angle_when_nothing_hits(self, blank_crop):
        from puzzlefind import config

        backend = FakeBackend([[]])
        result = recognize.recognize_sweep(backend, blank_crop)
        assert result.code is None
        assert backend.calls == len(config.SWEEP_ANGLES)

    def test_keeps_highest_confidence_across_angles(self, blank_crop):
        responses = [
            [RawDetection("B-403", 0.60)],
            [RawDetection("B-403", 0.98)],
            [RawDetection("B-403", 0.71)],
        ]
        backend = FakeBackend(responses)
        result = recognize.recognize_sweep(backend, blank_crop)
        assert result.confidence == pytest.approx(0.98)


class TestRecognizePiece:
    def test_high_confidence_direct_hit_skips_the_sweep(self, blank_crop):
        backend = FakeBackend([[RawDetection("B-403", 0.99)]])
        result = recognize.recognize_piece(backend, blank_crop)
        assert result.method == "direct"
        assert backend.calls == 1

    def test_low_confidence_direct_hit_escalates_to_sweep(self, blank_crop):
        # 首次调用置信度低于阈值 → 进入穷举，穷举里读出高分结果
        responses = [
            [RawDetection("B-403", 0.50)],   # direct
            [RawDetection("B-403", 0.50)],   # sweep angle 0
            [RawDetection("B-403", 0.97)],   # sweep angle 1
        ]
        backend = FakeBackend(responses)
        result = recognize.recognize_piece(backend, blank_crop)
        assert result.method == "sweep"
        assert backend.calls > 1

    def test_returns_no_result_when_every_pass_fails(self, blank_crop):
        backend = FakeBackend([[]])
        result = recognize.recognize_piece(backend, blank_crop)
        assert result.code is None
        assert result.method == "none"


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
