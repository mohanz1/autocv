import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from autocv import AutoCV
from autocv.core.vision import Vision
from PIL import Image


@pytest.fixture
def autocv():
    with patch("autocv.autocv.antigcp", create=True), patch("autocv.autocv.logging.getLogger"):
        obj = AutoCV(hwnd=1234)
        obj.opencv_image = np.ones((10, 10, 3), dtype=np.uint8) * 255
        return obj


class TestAutoCVVision:
    @patch("win32gui.DeleteObject")
    @patch("win32gui.GetWindowRect", return_value=(0, 0, 2, 2))
    @patch("win32gui.GetWindowDC")
    @patch("win32ui.CreateDCFromHandle")
    @patch("win32ui.CreateBitmap")
    def test_refresh(self, mock_bmp, mock_dc, mock_getdc, mock_rect, mock_delete_obj, autocv):
        mem_dc = MagicMock()
        bmp_dc = MagicMock()
        bmp = MagicMock()
        bmp.GetBitmapBits.return_value = bytearray(2 * 2 * 4)
        bmp.GetHandle.return_value = 1

        mock_dc.return_value = mem_dc
        mem_dc.CreateCompatibleDC.return_value = bmp_dc
        mock_bmp.return_value = bmp

        result = autocv.refresh(set_backbuffer=False)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2, 3)

    def test_set_backbuffer_from_numpy(self, autocv):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        autocv.set_backbuffer(image)
        assert autocv.opencv_image.shape == (10, 10, 3)

    @patch("autocv.core.vision.filter_colors", return_value=np.ones((10, 10), dtype=np.uint8))
    def test_find_color(self, mock_filter, autocv):
        result = autocv.find_color((255, 255, 255))
        assert all(isinstance(x, tuple) and len(x) == 2 for x in result)

    def test_get_color(self, autocv):
        color = autocv.get_color((0, 0))
        assert color == (255, 255, 255)

    def test_get_average_color(self, autocv):
        avg = autocv.get_average_color()
        assert avg == (255, 255, 255)

    def test_get_median_color(self, autocv):
        median = autocv.get_median_color()
        assert median == (255, 255, 255)

    @patch("autocv.core.filter_colors", return_value=np.ones((10, 10), dtype=np.uint8))
    def test_get_count_of_color(self, mock_filter, autocv):
        count = autocv.get_count_of_color((255, 255, 255))
        assert count == 100

    def test_get_all_colors_with_counts(self, autocv):
        results = autocv.get_all_colors_with_counts()
        assert isinstance(results[0], tuple)
        assert isinstance(results[0][0], tuple)  # RGB
        assert isinstance(results[0][1], int)  # Count

    @patch("autocv.core.Vision._ensure_ocr")
    def test_get_text_filters_confidence(self, mock_ensure_ocr, autocv):
        mock_ensure_ocr.return_value.predict.return_value = {
            "rec_texts": ["foo"],
            "rec_scores": [0.9],
            "rec_boxes": [[0, 0, 2, 2]],
        }
        assert autocv.get_text(confidence=0.8) == [{"text": "foo", "rect": [0, 0, 1, 1], "confidence": 0.9}]

    def test_crop_image_clamps_rect(self, autocv):
        cropped = autocv._crop_image((8, 8, 10, 10))
        assert cropped.shape == (2, 2, 3)

    def test_crop_image_rejects_invalid_rect(self, autocv):
        with pytest.raises(ValueError):
            autocv._crop_image((0, 0, 0, 1))
        with pytest.raises(ValueError):
            autocv._crop_image((-1, 0, 1, 1))

    def test_crop_image_rejects_out_of_bounds_rect(self, autocv):
        with pytest.raises(ValueError):
            autocv._crop_image((100, 100, 1, 1))

    def test_extract_ocr_payload_variants(self):
        assert Vision._extract_ocr_payload([]) is None
        assert Vision._extract_ocr_payload([{"ok": True}]) == {"ok": True}
        assert Vision._extract_ocr_payload({"ok": True}) == {"ok": True}

        class DummyPrediction:
            json = [{"ok": True}]

        assert Vision._extract_ocr_payload(DummyPrediction()) == {"ok": True}

    @patch("autocv.core.vision.Vision._ensure_ocr")
    def test_get_text_accepts_polygons(self, mock_ensure_ocr, autocv):
        mock_ensure_ocr.return_value.predict.return_value = {
            "rec_texts": ["foo"],
            "rec_scores": [0.9],
            "rec_polys": [[[0, 0], [2, 0], [2, 2], [0, 2]]],
        }
        assert autocv.get_text(confidence=0.8) == [{"text": "foo", "rect": [0, 0, 1, 1], "confidence": 0.9}]

    @patch("autocv.core.vision.Vision._ensure_ocr")
    def test_get_text_confidence_none_disables_filtering(self, mock_ensure_ocr, autocv):
        mock_ensure_ocr.return_value.predict.return_value = {
            "rec_texts": ["foo"],
            "rec_scores": [0.1],
            "rec_boxes": [[0, 0, 2, 2]],
        }
        assert autocv.get_text(confidence=None) == [{"text": "foo", "rect": [0, 0, 1, 1], "confidence": 0.1}]

    def test_prepare_ocr_input_converts_color_to_bgr(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        prepared = Vision._prepare_ocr_input(image)
        assert prepared.shape[-1] == 3

    def test_box_to_bbox_rejects_invalid_length(self):
        with pytest.raises(ValueError):
            Vision._box_to_bbox([0, 0, 1])

    def test_ensure_ocr_raises_runtime_error_when_paddle_missing(self, autocv):
        def raise_missing_paddle(*args, **kwargs):
            err = ModuleNotFoundError("No module named 'paddle'")
            err.name = "paddle"
            raise err

        with patch("autocv.core.vision.PaddleOCR", side_effect=raise_missing_paddle):
            with pytest.raises(RuntimeError):
                autocv._ensure_ocr()

    def test_ensure_ocr_reraises_other_module_not_found(self, autocv):
        def raise_missing_other(*args, **kwargs):
            err = ModuleNotFoundError("No module named 'something_else'")
            err.name = "something_else"
            raise err

        with patch("autocv.core.vision.PaddleOCR", side_effect=raise_missing_other):
            with pytest.raises(ModuleNotFoundError):
                autocv._ensure_ocr()

    def test_ensure_ocr_returns_cached_instance(self, autocv):
        cached = MagicMock()
        autocv.api = cached
        assert autocv._ensure_ocr() is cached

    def test_set_backbuffer_accepts_pil_image(self, autocv):
        pil = Image.new("RGB", (2, 2), (1, 2, 3))
        autocv.set_backbuffer(pil)
        assert autocv.opencv_image.shape == (2, 2, 3)
        assert autocv.opencv_image[0, 0].tolist() == [3, 2, 1]

    def test_get_color_rejects_non_color_image(self, autocv):
        autocv.opencv_image = np.ones((10, 10), dtype=np.uint8)
        with pytest.raises(ValueError):
            autocv.get_color((0, 0))

    def test_get_color_rejects_out_of_bounds(self, autocv):
        with pytest.raises(IndexError):
            autocv.get_color((99, 99))

    @patch("autocv.core.vision.filter_colors", return_value=np.array([[255, 0], [0, 0]], dtype=np.uint8))
    def test_find_color_offsets_points_for_rect(self, mock_filter, autocv):
        points = autocv.find_color((255, 255, 255), rect=(5, 5, 2, 2))
        assert points == [(5, 5)]

    def test_get_most_common_color_validates_ignore_colors_shape(self, autocv):
        with pytest.raises(ValueError):
            autocv.get_most_common_color(ignore_colors=[(1, 2)])

    def test_get_most_common_color_raises_when_all_pixels_ignored(self, autocv):
        with pytest.raises(ValueError):
            autocv.get_most_common_color(ignore_colors=(255, 255, 255))
