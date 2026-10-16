import types

from bmicro.BGThread import BGThread
from bmicro.gui.calibration.calibration_view import CalibrationView


def test_calibrate_returns_when_the_calibration_ends_early(qtbot, mocker):
    """
    The calibration can end without its counter ever reaching the maximum
    (nothing to calibrate, a frame that cannot be fitted, an error): the
    window must then not wait for the counter - it used to wait forever,
    with the whole batch.
    """
    waits = []

    def sleep(seconds):
        waits.append(seconds)
        if len(waits) > 100:  # fail instead of hanging if that returns
            raise RuntimeError('still waiting for the counter')

    mocker.patch('bmicro.gui.calibration.calibration_view.time.sleep', sleep)
    controller = types.SimpleNamespace(
        calibrate=lambda calib_key, count, max_count: None)  # sets nothing
    view = types.SimpleNamespace(
        combobox_calibration=types.SimpleNamespace(currentText=lambda: '1'),
        thread=BGThread(), calibration_controller=controller,
        calibration_progress=mocker.MagicMock(),
        refresh_plot=mocker.MagicMock())

    CalibrationView.calibrate(view)

    view.refresh_plot.assert_called_once()
    assert len(waits) <= 100
