from autocv import AutoCV
from autocv.models import OrderBy

if __name__ == "__main__":
    autocv = AutoCV()
    autocv.set_hwnd_by_title("RuneLite")
    while autocv.set_inner_hwnd_by_title("SunAwtCanvas"):
        pass

    autocv.refresh()

    # if contours := autocv.find_contours((255, 255, 0), min_area=20, tolerance=50):
    #     print(contours.order_by(OrderBy.RIGHT_TO_LEFT))
    #     for contour in contours:
    #         autocv.draw_contours(contour)
    #         autocv.show_backbuffer()

    if contours := autocv.find_color((255, 255, 0), tolerance=50):
        print(contours.order_by(OrderBy.RIGHT_TO_LEFT))
        autocv.draw_points(contours)
        autocv.show_backbuffer()
