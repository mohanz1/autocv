import os
import numpy as np
import pytest
import types
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
    def test_init_sets_model_source_check_env_when_requested(self, monkeypatch):
        monkeypatch.delenv("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", raising=False)

        vision = Vision(disable_model_source_check=True)

        assert vision._ocr_lang == "en"
        assert vision._ocr_det_side_len > 0
        assert vision._ocr_det_box_thresh > 0
        assert vision._ocr_det_unclip_ratio > 0
        assert vision._ocr_conf_threshold == pytest.approx(0.60)
        assert vision._ocr_device is None
        assert os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] == "True"

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

    def test_refresh_updates_backbuffer_when_persisting(self, autocv):
        frame = np.zeros((2, 2, 3), dtype=np.uint8)

        with patch.object(Vision, "capture_frame", return_value=frame):
            result = autocv.refresh(set_backbuffer=True)

        assert result is None
        assert autocv.opencv_image is frame

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

    @patch("autocv.core.vision.filter_colors", return_value=np.ones((10, 10), dtype=np.uint8))
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

    def test_get_pixel_change_accepts_grayscale_images(self, autocv):
        autocv.opencv_image = np.zeros((2, 2), dtype=np.uint8)
        with patch.object(Vision, "capture_frame", return_value=np.ones((2, 2), dtype=np.uint8)):
            assert autocv.get_pixel_change() == 4

    def test_extract_ocr_payload_variants(self):
        assert Vision._extract_ocr_payload([]) is None
        assert Vision._extract_ocr_payload([{"ok": True}]) == {"ok": True}
        assert Vision._extract_ocr_payload({"ok": True}) == {"ok": True}
        assert Vision._extract_ocr_payload(["bad"]) is None

        class DummyPrediction:
            json = [{"ok": True}]

        assert Vision._extract_ocr_payload(DummyPrediction()) == {"ok": True}

        class DummyPredictionWithMethod:
            @staticmethod
            def json() -> list[dict[str, bool]]:
                return [{"ok": True}]

        assert Vision._extract_ocr_payload(DummyPredictionWithMethod()) == {"ok": True}

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

    @patch("autocv.core.vision.Vision._ensure_ocr")
    def test_get_text_falls_back_to_polygons_when_boxes_are_missing(self, mock_ensure_ocr, autocv):
        mock_ensure_ocr.return_value.predict.return_value = {
            "rec_texts": ["foo", "bar"],
            "rec_scores": [0.9, 0.95],
            "rec_boxes": [[0, 0, 2, 2]],
            "rec_polys": [
                [[0, 0], [2, 0], [2, 2], [0, 2]],
                [[4, 4], [8, 4], [8, 8], [4, 8]],
            ],
        }
        assert autocv.get_text(confidence=0.8) == [
            {"text": "foo", "rect": [0, 0, 1, 1], "confidence": 0.9},
            {"text": "bar", "rect": [2, 2, 2, 2], "confidence": 0.95},
        ]

    def test_prepare_ocr_input_converts_color_to_bgr(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        prepared = Vision._prepare_ocr_input(image)
        assert prepared.shape[-1] == 3

    def test_prepare_ocr_input_rejects_invalid_shape(self):
        with pytest.raises(ValueError, match="requires a grayscale image or a 3-channel BGR image"):
            Vision._prepare_ocr_input(np.zeros((10, 10, 4), dtype=np.uint8))

        with pytest.raises(ValueError, match="requires a grayscale image or a 3-channel BGR image"):
            Vision._prepare_ocr_input(np.zeros((10,), dtype=np.uint8))

    def test_box_to_bbox_rejects_invalid_length(self):
        with pytest.raises(ValueError):
            Vision._box_to_bbox([0, 0, 1])

    def test_ensure_ocr_raises_runtime_error_when_paddle_missing(self, autocv):
        def raise_missing_paddle(*args, **kwargs):
            err = ModuleNotFoundError("No module named 'paddle'")
            err.name = "paddle"
            raise err

        fake_module = types.SimpleNamespace(PaddleOCR=MagicMock(side_effect=raise_missing_paddle))
        with patch.dict("sys.modules", {"paddleocr": fake_module}):
            with pytest.raises(RuntimeError):
                autocv._ensure_ocr()

    def test_ensure_ocr_reraises_other_module_not_found(self, autocv):
        def raise_missing_other(*args, **kwargs):
            err = ModuleNotFoundError("No module named 'something_else'")
            err.name = "something_else"
            raise err

        fake_module = types.SimpleNamespace(PaddleOCR=MagicMock(side_effect=raise_missing_other))
        with patch.dict("sys.modules", {"paddleocr": fake_module}):
            with pytest.raises(ModuleNotFoundError):
                autocv._ensure_ocr()

    def test_ensure_ocr_returns_cached_instance(self, autocv):
        cached = MagicMock()
        autocv.api = cached
        assert autocv._ensure_ocr() is cached

    @patch("autocv.core.vision.filter_colors", return_value=np.zeros((10, 10, 3), dtype=np.uint8))
    @patch("autocv.core.vision.Vision._ensure_ocr")
    def test_get_text_skips_malformed_entries_and_applies_rect_offset(self, mock_ensure_ocr, mock_filter, autocv):
        autocv.opencv_image = np.zeros((40, 40, 3), dtype=np.uint8)
        mock_ensure_ocr.return_value.predict.return_value = {
            "rec_texts": [" ", "low", "badbox", "good", "polygood"],
            "rec_scores": [0.95, 0.10, 0.95, 0.95, 0.95],
            "rec_boxes": [
                [0, 0, 2, 2],
                [0, 0, 2, 2],
                [0, 0, 1],
                [4, 4, 8, 8],
            ],
            "rec_polys": [
                [[0, 0], [2, 0], [2, 2], [0, 2]],
                [[0, 0], [2, 0], [2, 2], [0, 2]],
                [[0]],
                [[4, 4], [8, 4], [8, 8], [4, 8]],
                [[8, 8], [12, 8], [12, 12], [8, 12]],
            ],
        }

        result = autocv.get_text(rect=(10, 20, 10, 10), colors=(255, 255, 255), tolerance=3, confidence=0.8)

        assert result == [
            {"text": "good", "rect": [12, 22, 2, 2], "confidence": 0.95},
            {"text": "polygood", "rect": [14, 24, 2, 2], "confidence": 0.95},
        ]
        mock_filter.assert_called_once()

    def test_get_ocr_bbox_returns_none_for_invalid_box_and_polygon(self):
        assert Vision._get_ocr_bbox(index=0, rec_boxes=[[0, 1, 2]], rec_polys=[[[0]]]) is None

    def test_set_backbuffer_accepts_pil_image(self, autocv):
        pil = Image.new("RGB", (2, 2), (1, 2, 3))
        autocv.set_backbuffer(pil)
        assert autocv.opencv_image.shape == (2, 2, 3)
        assert autocv.opencv_image[0, 0].tolist() == [3, 2, 1]

    @patch("autocv.core.vision.cv.imwrite", return_value=False)
    def test_save_backbuffer_to_file_raises_when_write_fails(self, mock_imwrite, autocv):
        with pytest.raises(OSError, match="Failed to write backbuffer"):
            autocv.save_backbuffer_to_file("missing.png")

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

    def test_find_color_rejects_non_color_image(self, autocv):
        autocv.opencv_image = np.ones((10, 10), dtype=np.uint8)
        with pytest.raises(ValueError, match="3-channel BGR image"):
            autocv.find_color((255, 255, 255))

    def test_get_most_common_color_validates_ignore_colors_shape(self, autocv):
        with pytest.raises(ValueError):
            autocv.get_most_common_color(ignore_colors=[(1, 2)])

    def test_get_most_common_color_raises_when_all_pixels_ignored(self, autocv):
        with pytest.raises(ValueError):
            autocv.get_most_common_color(ignore_colors=(255, 255, 255))

    def test_get_most_common_color_validates_index(self, autocv):
        with pytest.raises(ValueError, match="index must be >= 1"):
            autocv.get_most_common_color(index=0)

    def test_get_median_color_rejects_non_color_image(self, autocv):
        autocv.opencv_image = np.ones((9, 9), dtype=np.uint8)
        with pytest.raises(ValueError, match="3-channel BGR image"):
            autocv.get_median_color()
