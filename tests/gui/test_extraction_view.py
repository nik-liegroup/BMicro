import pathlib
from collections import namedtuple

import pytest
import numpy as np

from bmlab.session import Session

from bmicro.gui.main import BMicro


def data_file_path(file_name):
    return pathlib.Path(__file__).parent.parent / 'data' / file_name


Event = namedtuple('Event', 'xdata ydata')


@pytest.fixture
def window(mocker):
    window = BMicro()
    Session.get_instance().clear()
    file_name = data_file_path('Water.h5')

    def mock_getOpenFileName(self, *args, **kwargs):
        return file_name, None

    mocker.patch('PyQt6.QtWidgets.QFileDialog.getOpenFileName',
                 mock_getOpenFileName)
    window.open_file()
    yield window
    window.close()


def test_clicking_in_select_mode_adds_points(qtbot, window):
    ev = window.widget_extraction_view
    # Change to select mode:
    ev.toggle_mode()

    event = Event(50, 50)
    ev.on_click_image(event)
    event = Event(150, 150)
    ev.on_click_image(event)

    session = Session.get_instance()
    assert session.extraction_model().get_points('1') == [(50, 50), (150, 150)]
    assert session.extraction_model().get_points('2') == []

    # Change back to default mode
    ev.toggle_mode()

    # Clicking should not add point
    event = Event(111, 111)
    ev.on_click_image(event)
    assert session.extraction_model().get_points('1') == [(50, 50), (150, 150)]


def test_selecting_three_points_creates_circle_fit(qtbot, window):
    ev = window.widget_extraction_view
    # Change to select mode:
    ev.toggle_mode()
    event = Event(0, 100)
    ev.on_click_image(event)
    event = Event(100, 0)
    ev.on_click_image(event)
    session = Session.get_instance()
    np.testing.assert_array_equal(
        session.extraction_model().get_arc_by_calib_key('1'),
        np.empty(0)
    )
    event = Event(100/(2**0.5), 100/(2**0.5))
    ev.on_click_image(event)
    fit = session.extraction_model().get_arc_by_calib_key('1')
    assert fit is not None
    assert len(fit) == 500
    np.testing.assert_array_almost_equal(fit[0, 2, :], [0, 100])
    np.testing.assert_array_almost_equal(fit[0, 2, :], [0, 100])
    np.testing.assert_array_almost_equal(fit[-1, 2, :], [100, 0])


def test_clicking_clear_deletes_points(qtbot, window):
    ev = window.widget_extraction_view
    # Change to select mode:
    ev.toggle_mode()

    event = Event(50, 50)
    ev.on_click_image(event)

    session = Session.get_instance()
    assert session.extraction_model().get_points('1') == [(50, 50)]
    ev.clear_points()
    assert session.extraction_model().get_points('1') == []


def test_clicking_find_points_finds_points(qtbot, window):
    ev = window.widget_extraction_view
    ev.clear_points()

    session = Session.get_instance()
    assert session.extraction_model().get_points('1') == []

    ev.find_points()
    assert len(session.extraction_model().get_points('1')) > 0


def test_checking_use_zoom_as_crop_restricts_analysis(qtbot, window):
    ev = window.widget_extraction_view
    session = Session.get_instance()

    full_shape = session.get_payload_image('1', 0).shape
    assert session.get_crop_bounds() is None

    # Simulate the user zooming into a sub-region with the plot's Zoom
    # tool, then checking the box to lock that in as the crop.
    ev.image_plot.set_xlim(50, 250)
    ev.image_plot.set_ylim(100, 300)
    ev.checkbox_use_zoom_as_crop.setChecked(True)

    assert session.get_crop_bounds() == (50, 250, 100, 300)
    cropped_shape = session.get_payload_image('1', 0).shape
    assert cropped_shape == (200, 200)
    assert cropped_shape != full_shape

    # Unchecking restores the full image.
    ev.checkbox_use_zoom_as_crop.setChecked(False)
    assert session.get_crop_bounds() is None
    assert session.get_payload_image('1', 0).shape == full_shape


def test_zoom_label_updates_live_with_plot_zoom(qtbot, window):
    ev = window.widget_extraction_view

    ev.image_plot.set_xlim(50, 250)
    ev.image_plot.set_ylim(100, 300)
    assert ev.label_zoom_bounds.text() == 'Zoom: x [50, 250]  y [100, 300]'

    # Updates again on a further zoom, without needing setChecked/refresh.
    ev.image_plot.set_xlim(0, 400)
    ev.image_plot.set_ylim(0, 400)
    assert ev.label_zoom_bounds.text() == 'Zoom: x [0, 400]  y [0, 400]'
