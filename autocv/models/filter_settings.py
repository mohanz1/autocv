from dataclasses import dataclass

__all__ = ("FilterSettings",)


SLOTS_DATACLASS = dict(slots=True) if "slots" in dataclass.__kwdefaults__ else {}


@dataclass(**SLOTS_DATACLASS)
class FilterSettings:
    """Class to store settings for image processing filters.

    Attributes
    ----------
        h_min (int): The minimum hue value.
        h_max (int): The maximum hue value.
        s_min (int): The minimum saturation value.
        s_max (int): The maximum saturation value.
        v_min (int): The minimum value (brightness) value.
        v_max (int): The maximum value (brightness) value.
        s_add (int): The amount to add to the saturation value.
        s_subtract (int): The amount to subtract from the saturation value.
        v_add (int): The amount to add to the value (brightness) value.
        v_subtract (int): The amount to subtract from the value (brightness) value.
        canny_threshold1 (int): The first threshold value for the Canny edge detection algorithm.
        canny_threshold2 (int): The second threshold value for the Canny edge detection algorithm.
        erode_kernel_size (int): The kernel size for the erosion filter.
        dilate_kernel_size (int): The kernel size for the dilation filter.
    """

    h_min: int = 0
    h_max: int = 179
    s_min: int = 0
    s_max: int = 255
    v_min: int = 0
    v_max: int = 255
    s_add: int = 0
    s_subtract: int = 0
    v_add: int = 0
    v_subtract: int = 0
    canny_threshold1: int = 0
    canny_threshold2: int = 0
    erode_kernel_size: int = 0
    dilate_kernel_size: int = 0
