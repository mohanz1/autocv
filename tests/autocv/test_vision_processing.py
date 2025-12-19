import numpy as np
import pytest
from mock import MagicMock, patch
from PIL import Image

from autocv import AutoCV
from autocv.core.vision import Vision


@pytest.fixture
def autocv():
    with patch("autocv.autocv.antigcp", create=True), patch("autocv.autocv.logging.getLogger"):
        obj = AutoCV(hwnd=1234)
        obj.opencv_image = np.ones((10, 10, 3), dtype=np.uint8) * 255
        return obj


def test_calculate_median_difference_handles_mismatch_and_mask():
    img1 = np.zeros((2, 2, 3), dtype=np.uint8)
    img2 = np.ones((2, 2, 3), dtype=np.uint8) * 10

    assert Vision._calculate_median_difference(img1, img2) == 10
    assert Vision._calculate_median_difference(img1, img2[:, :, :2]) == -1

    mask_none = np.zeros((2, 2), dtype=np.uint8)
    assert Vision._calculate_median_difference(img1, img2, mask=mask_none) == -1

    mask_one = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    assert Vision._calculate_median_difference(img1, img2, mask=mask_one) == 10


def test_erode_and_dilate_modify_backbuffer(autocv):
    autocv.opencv_image = np.zeros((5, 5, 3), dtype=np.uint8)
    autocv.opencv_image[2, 2] = [255, 255, 255]
    before_sum = int(autocv.opencv_image.sum())

    autocv.dilate_image()
    assert int(autocv.opencv_image.sum()) > before_sum

    autocv.opencv_image = np.zeros((5, 5, 3), dtype=np.uint8)
    autocv.opencv_image[2, 2] = [255, 255, 255]
    autocv.erode_image()
    assert int(autocv.opencv_image.sum()) == 0


def test_prepare_sub_image_accepts_rgba_and_pil():
    rgba = np.array([[[1, 2, 3, 4]]], dtype=np.uint8)
    sub_bgr, mask = Vision._prepare_sub_image(rgba)
    assert sub_bgr.shape == (1, 1, 3)
    assert mask is not None
    assert mask.shape == (1, 1)
    assert sub_bgr[0, 0].tolist() == [3, 2, 1]

    pil = Image.new("RGBA", (1, 1), (1, 2, 3, 4))
    sub_bgr, mask = Vision._prepare_sub_image(pil)
    assert sub_bgr.shape == (1, 1, 3)
    assert mask is not None


def test_prepare_sub_image_rejects_invalid_shape():
    with pytest.raises(ValueError):
        Vision._prepare_sub_image(np.zeros((2, 2, 2), dtype=np.uint8))


def test_perform_template_matching_ignores_infinite_results():
    main = np.zeros((1, 1), dtype=np.uint8)
    sub = np.zeros((1, 1), dtype=np.uint8)

    with patch("autocv.core.vision.cv.matchTemplate", return_value=np.array([[np.inf]], dtype=np.float32)):
        mask = Vision._perform_template_matching(main, sub, mask=None, confidence=0.5)
        assert mask.shape == (1, 1)
        assert int(mask[0, 0]) == 0

    with patch("autocv.core.vision.cv.matchTemplate", return_value=np.array([[0.9]], dtype=np.float32)):
        mask = Vision._perform_template_matching(main, sub, mask=None, confidence=0.5)
        assert int(mask[0, 0]) == 1


def test_process_matching_results_applies_offsets_and_tolerance(autocv):
    main = np.zeros((3, 3, 3), dtype=np.uint8)
    sub = np.zeros((1, 1, 3), dtype=np.uint8)
    res = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.uint8)

    rects = autocv._process_matching_results(res, main, sub, mask=None, rect=(10, 20, 3, 3), median_tolerance=None)
    assert rects == [(10, 20, 1, 1)]

    with patch.object(AutoCV, "_calculate_median_difference", return_value=10):
        rects = autocv._process_matching_results(res, main, sub, mask=None, rect=None, median_tolerance=5)
        assert rects == []


def test_group_and_convert_to_shape_list_handles_empty_and_grouped():
    assert Vision._group_and_convert_to_shape_list([]) == []

    with patch("autocv.core.vision.cv.groupRectangles", return_value=(np.array([[1, 2, 3, 4]]), None)):
        grouped = Vision._group_and_convert_to_shape_list([(1, 2, 3, 4)])
        assert grouped == [(1, 2, 3, 4)]


def test_find_contours_offsets_results(autocv):
    autocv.opencv_image = np.zeros((20, 20, 3), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:8, 2:8] = 255

    with patch("autocv.core.vision.filter_colors", return_value=mask):
        contours = autocv.find_contours((0, 0, 0), rect=(5, 5, 10, 10), min_area=1)
        assert contours
        assert int(contours[0].min()) >= 5


def test_find_contours_close_and_dilate_merges_nearby_shapes(autocv):
    autocv.opencv_image = np.zeros((20, 20, 3), dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:10, 5:8] = 255
    mask[5:10, 9:12] = 255

    with patch("autocv.core.vision.filter_colors", return_value=mask):
        contours = autocv.find_contours((0, 0, 0), min_area=1)
        assert len(contours) == 2

        contours = autocv.find_contours((0, 0, 0), min_area=1, close_and_dilate=True)
        assert len(contours) == 1


def test_draw_helpers_and_filter_colors(autocv):
    autocv.opencv_image = np.zeros((10, 10, 3), dtype=np.uint8)

    autocv.draw_points([(1, 2)], color=(255, 0, 0))
    assert autocv.opencv_image[2, 1].tolist() == [0, 0, 255]

    autocv.draw_circle((5, 5, 2))
    autocv.draw_rectangle((1, 1, 3, 3))

    contour = np.array([[[1, 1]], [[4, 1]], [[4, 4]], [[1, 4]]], dtype=np.int32)
    autocv.draw_contours(contour)

    autocv.filter_colors((0, 0, 0))
    assert autocv.opencv_image.ndim == 2
