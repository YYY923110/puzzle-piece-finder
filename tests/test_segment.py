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
