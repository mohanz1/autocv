Models
======
These are models that are used as return-types for functions.

Circle
------

.. autoclass:: autocv.models.circle.Circle
   :members: center, random_point

Color
-----

.. autoclass:: autocv.models.color.Color
   :members: is_color_within_color_and_tolerance, invert, to_decimal, to_hex

ColorWithPoint
--------------

.. autoclass:: autocv.models.color_with_point.ColorWithPoint
   :members: point, color

Contour
-------

.. autoclass:: autocv.models.contour.Contour
   :members: area, perimeter, centroid, center, is_point_inside_contour, random_point, to_points, get_bounding_rect, get_bounding_circle

FilterSettings
--------------

.. autoclass:: autocv.models.filter_settings.FilterSettings
   :members:

Point
-----

.. autoclass:: autocv.models.point.Point
   :members: center, random_point, distance_to

Rectangle
---------

.. autoclass:: autocv.models.rectangle.Rectangle
   :members: right, bottom, area, get_overlap, center, random_point

ShapeList
---------

.. autoclass:: autocv.models.shape_list.ShapeList
   :members:

TextInfo
--------

.. autoclass:: autocv.models.text_info.TextInfo
   :members:
