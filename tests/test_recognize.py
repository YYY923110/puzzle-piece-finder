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


class FakeLineBackend(FakeBackend):
    """额外实现可选的 read_line 的假后端。

    read 与 read_line 各自独立计数，测试才能分辨「跑的是便宜的 rec 路径
    还是昂贵的全量穷举」——这正是本次改造要证明的事。
    """

    def __init__(
        self,
        responses: list[list[RawDetection]],
        line_responses: list[RawDetection],
    ):
        super().__init__(responses)
        self.line_responses = line_responses
        self.line_calls = 0

    def read_line(self, image: np.ndarray) -> RawDetection:
        result = self.line_responses[
            min(self.line_calls, len(self.line_responses) - 1)
        ]
        self.line_calls += 1
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


class TestDeskewQuad:
    def test_axis_aligned_quad_crops_exactly_that_rectangle(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        image[20:50, 30:130] = 255
        quad = [[30, 20], [130, 20], [130, 50], [30, 50]]

        line = recognize.deskew_quad(image, quad)

        assert line.shape[1] == 100      # 宽
        assert line.shape[0] == 30       # 高
        assert line.mean() > 250         # 裁出来的确实是那块白区

    def test_vertical_quad_is_rotated_back_to_horizontal(self):
        """竖排的框必须被转正——否则 rec 模型读不出来。

        这条钉的是第一轮验证踩到的真实缺陷：碎片旋转 90°/270° 时
        检测框的点序会把长短边判反，不补 rot90 就会读出 89/169/382 这类垃圾。
        """
        image = np.zeros((200, 100, 3), dtype=np.uint8)
        image[30:130, 20:50] = 255
        quad = [[20, 30], [50, 30], [50, 130], [20, 130]]   # 高 100 / 宽 30 = 3.3

        line = recognize.deskew_quad(image, quad)

        assert line.shape[1] > line.shape[0], "竖排文字行没有被转正"

    def test_wide_quad_is_left_alone(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        quad = [[10, 10], [150, 10], [150, 40], [10, 40]]   # 高 30 / 宽 140，很扁
        line = recognize.deskew_quad(image, quad)
        assert line.shape[1] > line.shape[0]

    def test_rotated_quad_is_straightened(self):
        """把四边形按 45° 给出，裁出来的应该是一条水平的条。"""
        import math

        image = np.zeros((300, 300, 3), dtype=np.uint8)
        cx, cy = 150.0, 150.0
        half_w, half_h = 60.0, 12.0
        corners = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
        angle = math.radians(45)
        quad = [
            [round(cx + x * math.cos(angle) - y * math.sin(angle)),
             round(cy + x * math.sin(angle) + y * math.cos(angle))]
            for x, y in corners
        ]

        line = recognize.deskew_quad(image, quad)

        assert line.shape[1] == pytest.approx(2 * half_w, abs=3)
        assert line.shape[0] == pytest.approx(2 * half_h, abs=3)

    def test_degenerate_quad_does_not_crash(self):
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        quad = [[10, 10], [10, 10], [10, 10], [10, 10]]
        line = recognize.deskew_quad(image, quad)
        assert line.size >= 0


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

    def test_stops_early_on_a_very_confident_hit(self, blank_crop):
        from puzzlefind import config

        responses = [
            [],
            [RawDetection("B-403", config.SWEEP_EARLY_EXIT_CONFIDENCE)],
            [RawDetection("A-111", 1.0)],       # 不该跑到这里
        ]
        backend = FakeBackend(responses)
        result = recognize.recognize_sweep(backend, blank_crop)
        assert result.code == "B-403"
        assert backend.calls == 2, "拿到 0.99 之后仍把剩余角度跑完了"

    def test_keeps_sweeping_when_confidence_stays_below_the_exit_bar(self, blank_crop):
        from puzzlefind import config

        responses = [[RawDetection("B-403", 0.95)]]
        backend = FakeBackend(responses)
        recognize.recognize_sweep(backend, blank_crop)
        assert backend.calls == len(config.SWEEP_ANGLES)


QUAD = [[10, 10], [90, 10], [90, 40], [10, 40]]


class TestRecognizeLineSweep:
    def test_reads_code_from_the_upright_orientation(self, blank_crop):
        backend = FakeLineBackend([[]], [RawDetection("B-403", 0.98)])
        result = recognize.recognize_line_sweep(backend, blank_crop, QUAD)
        assert result.code == "B-403"
        assert result.method == "line"
        assert result.angle == 0

    def test_falls_through_to_the_flipped_orientation(self, blank_crop):
        backend = FakeLineBackend(
            [[]],
            [RawDetection("EOP-8", 0.40), RawDetection("B-403", 0.97)],
        )
        result = recognize.recognize_line_sweep(backend, blank_crop, QUAD)
        assert result.code == "B-403"
        assert result.angle == 180

    def test_tries_both_orientations_when_nothing_snaps(self, blank_crop):
        backend = FakeLineBackend([[]], [RawDetection("QWERTY", 0.99)])
        result = recognize.recognize_line_sweep(backend, blank_crop, QUAD)
        assert result.code is None
        assert backend.line_calls == 2

    def test_stops_early_on_a_very_confident_read(self, blank_crop):
        from puzzlefind import config

        backend = FakeLineBackend(
            [[]],
            [
                RawDetection("B-403", config.SWEEP_EARLY_EXIT_CONFIDENCE),
                RawDetection("A-111", 1.0),
            ],
        )
        result = recognize.recognize_line_sweep(backend, blank_crop, QUAD)
        assert result.code == "B-403"
        assert backend.line_calls == 1, "拿到高置信度结果后不该再试翻转"

    def test_never_calls_the_expensive_full_read(self, blank_crop):
        backend = FakeLineBackend([[]], [RawDetection("B-403", 0.98)])
        recognize.recognize_line_sweep(backend, blank_crop, QUAD)
        assert backend.calls == 0, "按行识别不该触碰检测模型"


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

    def test_low_confidence_direct_uses_the_line_path_when_a_quad_is_available(
        self, blank_crop
    ):
        backend = FakeLineBackend(
            [[RawDetection("B-403", 0.50, QUAD)]],      # Pass A：分低，但有框
            [RawDetection("B-403", 0.98)],              # 按行识别：读准了
        )
        result = recognize.recognize_piece(backend, blank_crop)
        assert result.method == "line"
        assert result.code == "B-403"
        assert backend.calls == 1, "不该再跑一遍昂贵的全量穷举"
        # 0.98 低于 SWEEP_EARLY_EXIT_CONFIDENCE(0.99)，所以两个朝向都试了——
        # 这是对的。够格收工的门槛是 SWEEP_CONFIDENCE_THRESHOLD(0.90)，
        # 它决定的是「要不要回退到全量穷举」，与提前退出是两回事。
        assert backend.line_calls == 2

    def test_falls_back_to_the_full_sweep_when_the_line_path_fails(self, blank_crop):
        backend = FakeLineBackend(
            [
                [RawDetection("B-403", 0.50, QUAD)],   # Pass A
                [RawDetection("B-403", 0.96)],         # 全量穷举的第一个角度
            ],
            [RawDetection("QWERTY", 0.99)],            # 按行识别：吸附不上
        )
        result = recognize.recognize_piece(backend, blank_crop)
        assert result.method == "sweep"
        assert result.code == "B-403"
        assert backend.line_calls == 2, "回退前应把两个朝向都试过"
        assert backend.calls > 1

    def test_falls_back_to_the_full_sweep_when_there_is_no_quad(self, blank_crop):
        backend = FakeLineBackend(
            [
                [RawDetection("B-403", 0.50)],         # Pass A：分低且**没有框**
                [RawDetection("B-403", 0.96)],
            ],
            [RawDetection("B-403", 0.99)],
        )
        result = recognize.recognize_piece(backend, blank_crop)
        assert result.method == "sweep"
        assert backend.line_calls == 0, "没有框就不该走按行识别"

    def test_backend_without_read_line_still_works(self, blank_crop):
        """不实现可选协议的后端必须照常工作——这是 spec §6 的架构承诺。"""
        backend = FakeBackend(
            [
                [RawDetection("B-403", 0.50, QUAD)],
                [RawDetection("B-403", 0.96)],
            ]
        )
        result = recognize.recognize_piece(backend, blank_crop)
        assert result.method == "sweep"
        assert result.code == "B-403"

    def test_high_confidence_direct_still_skips_everything(self, blank_crop):
        backend = FakeLineBackend(
            [[RawDetection("B-403", 0.99, QUAD)]], [RawDetection("A-111", 1.0)]
        )
        result = recognize.recognize_piece(backend, blank_crop)
        assert result.method == "direct"
        assert backend.calls == 1
        assert backend.line_calls == 0


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
