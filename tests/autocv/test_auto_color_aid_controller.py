from unittest.mock import MagicMock, patch

from autocv.auto_color_aid import AutoColorAidController, ColorSelectionState


def test_color_selection_state_add_and_remove_recomputes():
    state = ColorSelectionState()
    assert state.best_color == (0, 0, 0)
    assert state.best_tolerance == -1

    state.add((0, 0, 10, 20, 30))
    assert state.best_color == (10, 20, 30)
    assert state.best_tolerance == 1

    state.add((1, 1, 30, 40, 50))
    assert state.best_color == (20, 30, 40)
    assert state.best_tolerance == 21

    state.remove_indices({0})
    assert state.best_color == (30, 40, 50)
    assert state.best_tolerance == 1

    state.remove_indices(set())
    assert state.best_color == (30, 40, 50)

    state.remove_indices({0})
    assert state.best_color == (0, 0, 0)
    assert state.best_tolerance == -1


@patch("autocv.auto_color_aid.AutoCV")
def test_controller_titles_attach_and_add_sample(mock_autocv_cls):
    mock_autocv = mock_autocv_cls.return_value
    mock_autocv.get_windows_with_hwnds.return_value = [(1, "Main")]
    mock_autocv.get_child_windows.return_value = [(2, "Child")]
    mock_autocv.set_hwnd_by_title.return_value = True
    mock_autocv.set_inner_hwnd_by_title.return_value = False

    controller = AutoColorAidController()
    assert controller.window_titles() == ["Main"]
    assert controller.child_titles() == ["Child"]
    assert controller.attach_main("Main") is True
    assert controller.attach_child("Child") is False

    controller.add_sample(10, 20, (1, 2, 3))
    assert controller.state.pixels[-1] == (10, 20, 3, 2, 1)


@patch("autocv.auto_color_aid.AutoCV")
def test_controller_draw_markers_best_only(mock_autocv_cls):
    mock_autocv = mock_autocv_cls.return_value
    mock_autocv.find_color.return_value = [(1, 2)]

    controller = AutoColorAidController()
    controller.state.pixels = [(0, 0, 10, 20, 30)]
    controller.state.best_color = (10, 20, 30)
    controller.state.best_tolerance = 5

    controller.draw_markers(mark_best_only=True)

    mock_autocv.find_color.assert_called_once_with((10, 20, 30), tolerance=5)
    mock_autocv.draw_points.assert_called_once_with([(1, 2)])


@patch("autocv.auto_color_aid.AutoCV")
def test_controller_draw_markers_all_samples(mock_autocv_cls):
    mock_autocv = mock_autocv_cls.return_value
    mock_autocv.find_color.side_effect = [[(1, 2)], []]

    controller = AutoColorAidController()
    controller.state.pixels = [(0, 0, 10, 20, 30), (1, 1, 11, 22, 33)]

    controller.draw_markers(mark_best_only=False)

    assert mock_autocv.find_color.call_count == 2
    mock_autocv.draw_points.assert_called_once_with([(1, 2)])


@patch("autocv.auto_color_aid.AutoCV")
def test_controller_draw_markers_no_pixels_no_calls(mock_autocv_cls):
    controller = AutoColorAidController()

    controller.draw_markers(mark_best_only=True)

    mock_autocv_cls.return_value.find_color.assert_not_called()
