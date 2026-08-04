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
