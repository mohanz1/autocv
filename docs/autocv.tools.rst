Interactive Tools
=================

AutoCV ships with a handful of interactive helpers for live tuning and inspection. These tools open GUI windows and are
intended for local, manual workflows.

AutoColorAid
------------

.. autoclass:: autocv.auto_color_aid.AutoColorAid
   :members:
   :show-inheritance:

ColorPicker
-----------

.. autoclass:: autocv.color_picker.ColorPicker
   :members:

ImagePicker
-----------

.. autoclass:: autocv.image_picker.ImagePicker
   :members:

``ImagePicker.rect`` keeps the legacy full-window bounds, while ``ImagePicker.selection_rect`` stores the actual
selected ROI. For new top-level API usage, prefer :meth:`autocv.autocv.AutoCV.image_picker_capture`.

ImageFilter
-----------

.. autoclass:: autocv.image_filter.ImageFilter
   :members:
