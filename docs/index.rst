=================
Welcome to AutoCV
=================

AutoCV is a Windows-first computer vision automation toolkit for capturing game or desktop windows, analyzing frames with
OpenCV, and steering Win32 input back into the client.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Contents:

   autocv.autocv
   autocv.core.window_capture
   autocv.core.vision
   autocv.core.input
   autocv.tools
   autocv.models
   autocv.utils

Features
--------

- Win32 window discovery and capture via :class:`~autocv.core.window_capture.WindowCapture` /
  :class:`~autocv.core.vision.Vision`
- Template matching, contour detection, and color filtering helpers
- Human-like mouse motion and keyboard/mouse message sending via :class:`~autocv.core.input.Input`
- Optional OCR via PaddleOCR (:meth:`~autocv.core.vision.Vision.get_text`)
- Interactive tuning tools (:class:`~autocv.auto_color_aid.AutoColorAid`, :class:`~autocv.color_picker.ColorPicker`,
  :class:`~autocv.image_picker.ImagePicker`, :class:`~autocv.image_filter.ImageFilter`)

Installation
------------

AutoCV is not published on PyPI. Install from source:

.. code-block:: powershell

   git clone https://github.com/mohanz1/autocv.git
   cd autocv
   uv sync --locked --no-sources --group dev

OCR support is optional and requires extra dependencies:

.. code-block:: powershell

   python scripts/install_ocr.py --backend cpu
   # or on Windows x64 with GPU support:
   python scripts/install_ocr.py --backend gpu

Requirements
------------

- Windows (AutoCV uses ``pywin32`` for capture/input)
- Python 3.11+

OCR Notes
---------

The first call to :meth:`~autocv.core.vision.Vision.get_text` may download OCR models and can require network access.
If your environment blocks PaddleOCR model-host checks, set
``PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True`` before importing/using AutoCV, or pass
``disable_model_source_check=True`` when constructing :class:`~autocv.core.vision.Vision`.

Interactive Tool Notes
----------------------

``AutoCV.image_picker()`` preserves the historical ``(image, rect)`` return shape, where ``rect`` is the full target
window bounds. New code that needs the actual selected ROI should use
:meth:`~autocv.autocv.AutoCV.image_picker_capture` and read ``selection_rect``.

For Windows-side regression checks after capture/UI changes, run:

.. code-block:: powershell

   python scripts/windows_smoke_test.py --exercise-cursor

Reading the Documentation
-------------------------

The class structure and inheritance within the documentation is organized as follows:

- ``AutoCV`` inherits from ``Input`` → ``Vision`` → ``WindowCapture``.

This means :class:`~autocv.autocv.AutoCV` has access to all methods and properties of its parent classes. The reference
pages are structured to reflect this hierarchy.

Example
-------

.. code-block:: python

   from autocv import AutoCV

   autocv = AutoCV()
   autocv.set_hwnd_by_title("RuneLite")

   # Optionally walk down to an inner canvas handle.
   while autocv.set_inner_hwnd_by_title("SunAwtCanvas"):
       pass

   # Patch GetCursorPos checks (requires the bundled `antigcp` extension).
   autocv.antigcp()

   autocv.refresh()
   points = autocv.find_color((0, 255, 0), tolerance=50)
   if points:
       autocv.move_mouse(*points[0])
       autocv.click_mouse()
