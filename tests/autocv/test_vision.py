import numpy as np
import pytest
from mock import MagicMock, patch
from autocv import AutoCV
from autocv.core.vision import _bgr_to_rgb, _to_bgr


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
        assert autocv.opencv_image.flags["C_CONTIGUOUS"]

    def test_set_backbuffer_from_pil(self, autocv):
        from PIL import Image

        pil_image = Image.fromarray(np.zeros((5, 5, 3), dtype=np.uint8))
        autocv.set_backbuffer(pil_image)
        assert autocv.opencv_image.shape == (5, 5, 3)
        assert autocv.opencv_image.flags["C_CONTIGUOUS"]

    @patch("autocv.core.vision.filter_colors", return_value=np.ones((10, 10), dtype=np.uint8))
    def test_find_color(self, mock_filter, autocv):
        result = autocv.find_color((255, 255, 255))
        assert all(isinstance(x, tuple) and len(x) == 2 for x in result)

    def test_draw_points_handles_empty(self, autocv):
        before = autocv.opencv_image.copy()
        autocv.draw_points([])
        assert np.array_equal(autocv.opencv_image, before)

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
        assert isinstance(results[0][0], tuple)
        assert isinstance(results[0][1], int)

    @patch("autocv.core.Vision._get_grouped_text")
    def test_get_text_filters_confidence(self, mock_grouped, autocv):
        mock_grouped.return_value = MagicMock()
        mock_grouped.return_value.filter.return_value.select.return_value.with_columns.return_value.select.return_value.collect.return_value.to_dicts.return_value = [
            {"text": "foo", "rect": [1, 1, 1, 1], "confidence": 0.9}
        ]
        text = autocv.get_text(confidence=0.8)
        assert isinstance(text, list)

    @staticmethod
    def test_to_bgr_helper():
        assert _to_bgr((10, 20, 30)) == (30, 20, 10)

    @staticmethod
    def test_bgr_to_rgb_helper():
        assert _bgr_to_rgb((30, 20, 10)) == (10, 20, 30)
        assert _bgr_to_rgb(np.array([30, 20, 10], dtype=np.uint8)) == (10, 20, 30)
