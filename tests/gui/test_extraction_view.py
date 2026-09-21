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


def test_checking_use_zoom_as_crop_then_zooming_tracks_new_view(
        qtbot, window):
    """
    Regression test: checking checkbox_use_zoom_as_crop and only then
    zooming further must still update the crop to match the new view -
    previously the crop was captured once, at check time, and stayed
    stuck at whatever (typically the full, un-zoomed) view was showing
    then, only refreshed by re-toggling the checkbox afterwards. See
    ExtractionView.on_zoom_changed().
    """
    ev = window.widget_extraction_view
    session = Session.get_instance()

    # Check the box first, while still at the full (un-zoomed) view.
    ev.checkbox_use_zoom_as_crop.setChecked(True)
    full_bounds = session.get_crop_bounds()
    assert full_bounds is not None

    # Now zoom - the crop must follow, not stay stuck at full_bounds.
    # set_xlim()/set_ylim() fire as two separate events; the actual
    # crop update is deferred to the next event-loop spin (see
    # ExtractionView._apply_zoom_as_crop()) so both are settled before
    # it runs - qtbot.wait() pumps the loop to let that happen.
    ev.image_plot.set_xlim(50, 250)
    ev.image_plot.set_ylim(100, 300)
    qtbot.wait(10)
    assert session.get_crop_bounds() == (50, 250, 100, 300)
    assert session.get_crop_bounds() != full_bounds
    assert session.get_payload_image('1', 0).shape == (200, 200)

    # Zooming again, now within the already-cropped view, must compose
    # against the existing crop's own offset (50, 100) rather than
    # overwrite it with coordinates local to the cropped image.
    ev.image_plot.set_xlim(10, 60)
    ev.image_plot.set_ylim(20, 70)
    qtbot.wait(10)
    assert session.get_crop_bounds() == (60, 110, 120, 170)


def test_refresh_does_not_shrink_crop_for_smaller_image(qtbot, window):
    """
    Regression test: Crop.apply() clamps a crop's upper bounds down to
    whatever a smaller image actually has (e.g. a repetition captured
    at a smaller camera ROI than the one the crop was originally drawn
    on) - that clamped, on-screen extent must not be mistaken for a
    real user zoom and fed back into shrinking the stored crop. This
    reproduced just by opening a file with checkbox_use_zoom_as_crop
    already checked from a saved session (no interactive zooming
    needed): refresh_image_plot()'s own imshow() autoscale looked
    exactly like the user having zoomed to the clamped extent.
    """
    ev = window.widget_extraction_view
    session = Session.get_instance()

    # Water.h5's calibration image is 400x400 - a crop reaching past
    # that (e.g. as saved for a repetition captured at a larger camera
    # ROI than this one) gets clamped down to (100, 400, 50, 400) only
    # for display, via Crop.apply().
    session.set_crop_bounds((100, 500, 50, 450))
    assert session.get_crop_bounds() == (100, 500, 50, 450)

    # Check the box without going through its own toggled handler,
    # which would just capture the current (unrelated) view as a fresh
    # crop - mirrors how update_ui() reflects a crop already restored
    # from a saved session.
    ev.checkbox_use_zoom_as_crop.blockSignals(True)
    ev.checkbox_use_zoom_as_crop.setChecked(True)
    ev.checkbox_use_zoom_as_crop.blockSignals(False)

    ev.refresh_image_plot()
    qtbot.wait(10)

    assert session.get_crop_bounds() == (100, 500, 50, 450)


def test_zooming_x_and_y_separately_does_not_drift_crop(qtbot, window):
    """
    Regression test: xlim_changed and ylim_changed fire as two separate
    events for a single zoom gesture that changes both axes (matplotlib
    calls set_xlim() then set_ylim() as two independent calls, not
    atomically) - applying the crop on the first of the two would
    compose it against the other axis' still-stale bounds, drifting the
    persisted crop away from what was actually shown.
    """
    ev = window.widget_extraction_view
    session = Session.get_instance()

    ev.checkbox_use_zoom_as_crop.setChecked(True)

    # Simulate the two events landing one at a time, as matplotlib's
    # box/scroll zoom actually does, instead of both being in place
    # before either callback fires.
    ev.image_plot.set_xlim(50, 250)
    ev.image_plot.set_ylim(100, 300)
    qtbot.wait(10)

    assert session.get_crop_bounds() == (50, 250, 100, 300)


def test_zoom_label_updates_live_with_plot_zoom(qtbot, window):
    ev = window.widget_extraction_view

    ev.image_plot.set_xlim(50, 250)
    ev.image_plot.set_ylim(100, 300)
    assert ev.label_zoom_bounds.text() == 'Zoom: x [50, 250]  y [100, 300]'

    # Updates again on a further zoom, without needing setChecked/refresh.
    ev.image_plot.set_xlim(0, 400)
    ev.image_plot.set_ylim(0, 400)
    assert ev.label_zoom_bounds.text() == 'Zoom: x [0, 400]  y [0, 400]'
