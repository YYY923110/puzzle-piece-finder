import cv2
import numpy as np

from puzzlefind import segment


class TestBuildMask:
    def test_mask_is_binary_uint8(self, separated_pieces):
        image, _ = separated_pieces
        mask = segment.build_mask(image)
        assert mask.dtype == np.uint8
        assert set(np.unique(mask).tolist()) <= {0, 255}

    def test_mask_has_same_shape_as_input(self, separated_pieces):
        image, _ = separated_pieces
        mask = segment.build_mask(image)
        assert mask.shape == image.shape[:2]

    def test_piece_centers_are_foreground(self, separated_pieces):
        image, _ = separated_pieces
        mask = segment.build_mask(image)
        assert mask[120, 120] == 255

    def test_background_is_zero(self, separated_pieces):
        image, _ = separated_pieces
        mask = segment.build_mask(image)
        assert mask[300, 700] == 0


class TestFindBlobs:
    def test_finds_every_separated_piece(self, separated_pieces):
        image, expected = separated_pieces
        contours = segment.find_blobs(segment.build_mask(image))
        assert len(contours) == expected

    def test_discards_noise_specks(self, canvas_with_noise_speck):
        image, expected = canvas_with_noise_speck
        contours = segment.find_blobs(segment.build_mask(image))
        assert len(contours) == expected

    def test_touching_pieces_come_back_as_one_blob(self, touching_pair):
        # find_blobs 不负责切分——它只找连通块。切分是 Task 3。
        image, _ = touching_pair
        contours = segment.find_blobs(segment.build_mask(image))
        assert len(contours) == 1


class TestMedianBlobArea:
    def test_returns_median_of_contour_areas(self):
        square = np.array([[[0, 0]], [[0, 10]], [[10, 10]], [[10, 0]]], dtype=np.int32)
        big = np.array([[[0, 0]], [[0, 20]], [[20, 20]], [[20, 0]]], dtype=np.int32)
        area = segment.median_blob_area([square, square, big])
        assert abs(area - 100.0) < 1.0

    def test_empty_input_returns_zero(self):
        assert segment.median_blob_area([]) == 0.0


class TestExpectedPieceCount:
    def test_single_piece_area_yields_one(self):
        contour = np.array(
            [[[0, 0]], [[0, 10]], [[10, 10]], [[10, 0]]], dtype=np.int32
        )
        assert segment.expected_piece_count(contour, median_area=100.0) == 1

    def test_double_area_yields_two(self):
        contour = np.array(
            [[[0, 0]], [[0, 20]], [[10, 20]], [[10, 0]]], dtype=np.int32
        )
        assert segment.expected_piece_count(contour, median_area=100.0) == 2

    def test_never_returns_less_than_one(self):
        tiny = np.array([[[0, 0]], [[0, 2]], [[2, 2]], [[2, 0]]], dtype=np.int32)
        assert segment.expected_piece_count(tiny, median_area=100.0) == 1


class TestExtractContours:
    def test_separated_pieces_pass_through_unchanged(self, separated_pieces):
        image, expected = separated_pieces
        assert len(segment.extract_contours(image)) == expected

    def test_touching_pair_gets_split(self, touching_pair):
        image, expected = touching_pair
        assert len(segment.extract_contours(image)) == expected

    def test_mixed_scene_resolves_to_correct_total(self, touching_triple_with_singles):
        image, expected = touching_triple_with_singles
        assert len(segment.extract_contours(image)) == expected

    def test_split_contours_are_disjoint_enough(self, touching_pair):
        """切分出的两块，其质心应明显分开——不能是同一块被复制两份。"""
        image, _ = touching_pair
        contours = segment.extract_contours(image)
        centroids = []
        for contour in contours:
            moments = cv2.moments(contour)
            centroids.append((moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]))
        (x1, _), (x2, _) = centroids[0], centroids[1]
        assert abs(x1 - x2) > 40

    def test_empty_image_yields_no_contours(self):
        blank = np.zeros((200, 200, 3), dtype=np.uint8)
        assert segment.extract_contours(blank) == []

    def test_separated_puzzle_shaped_pieces_are_never_split(
        self, separated_puzzle_pieces
    ):
        """带凸起的单块碎片必须原样通过，一块都不许切开。

        这条是真实照片的回归测试。实测 real2.jpg：50 块碎片被
        find_blobs 正确切成 50 个连通块，却因为每块的 peak_count 都是 3
        （凸起各自形成一个山峰），unit_piece_area 被压低 3 倍，
        48/48 被判定需要切分，最终产出 96 个轮廓——每块碎片被劈成两半，
        裁剪图里只剩半个编号。
        """
        image, expected = separated_puzzle_pieces
        assert len(segment.extract_contours(image)) == expected

    def test_puzzle_piece_blobs_are_counted_as_one_each(
        self, separated_puzzle_pieces
    ):
        """把根因单独钉死：单块碎片的期望块数必须是 1。"""
        image, _ = separated_puzzle_pieces
        blobs = segment.find_blobs(segment.build_mask(image))
        unit = segment.unit_piece_area(image.shape[:2], blobs)
        assert all(segment.expected_piece_count(b, unit) == 1 for b in blobs)


class TestContourBbox:
    def test_returns_tight_bounding_box(self):
        contour = np.array(
            [[[10, 20]], [[10, 60]], [[50, 60]], [[50, 20]]], dtype=np.int32
        )
        # 宽高是「像素含两端」的：列 10..50 共 41 列，不是 50-10。
        # cv2.boundingRect 在 4.x 和 5.x 上都是这个语义，crop_piece 的
        # x+w 切片也依赖它才能把最后一列裁进去。
        assert segment.contour_bbox(contour) == (10, 20, 41, 41)


class TestCropPiece:
    def test_crop_long_edge_matches_target(self, separated_pieces):
        from puzzlefind import config

        image, _ = separated_pieces
        contour = segment.extract_contours(image)[0]
        crop = segment.crop_piece(image, contour)
        assert max(crop.shape[:2]) == config.CROP_TARGET_LONG_EDGE

    def test_crop_is_three_channel_bgr(self, separated_pieces):
        image, _ = separated_pieces
        contour = segment.extract_contours(image)[0]
        crop = segment.crop_piece(image, contour)
        assert crop.ndim == 3 and crop.shape[2] == 3

    def test_area_outside_contour_is_fill_color(self, separated_pieces):
        from puzzlefind import config

        image, _ = separated_pieces
        contour = segment.extract_contours(image)[0]
        crop = segment.crop_piece(image, contour)
        # 圆形碎片的裁剪图，四角必在轮廓之外
        corner = tuple(int(v) for v in crop[2, 2])
        assert corner == config.CROP_FILL_COLOR

    def test_center_preserves_original_piece_color(self, separated_pieces):
        image, _ = separated_pieces
        contour = segment.extract_contours(image)[0]
        crop = segment.crop_piece(image, contour)
        h, w = crop.shape[:2]
        center = crop[h // 2, w // 2]
        # 合成碎片是浅色 (235,233,228)，中心应仍是浅色
        assert int(center.min()) > 180

    def test_contour_touching_image_edge_does_not_crash(self):
        canvas = np.full((200, 200, 3), 18, dtype=np.uint8)
        cv2.circle(canvas, (5, 5), 40, (235, 233, 228), thickness=-1)
        contours = segment.extract_contours(canvas)
        assert len(contours) == 1
        crop = segment.crop_piece(canvas, contours[0])
        assert crop.size > 0
