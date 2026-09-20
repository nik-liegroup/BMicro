import pathlib
import shutil

import h5py
import numpy as np

from bmlab.controllers import BackgroundController, CalibrationController, \
    ExtractionController, PeakSelectionController
from bmlab.models import Orientation
from bmlab.session import Session

from bmicro.gui.main import BMicro


def data_file_path(file_name):
    return pathlib.Path(__file__).parent.parent / 'data' / file_name


def _make_background_fixture(tmp_path):
    """
    Water.h5 already has an (empty) 'background' group - it predates
    the BrillouinAcquisition version that first wrote real background
    reference points into it (see bmlab.file.Background). Copies
    payload point '0's own spectrum dataset into background/data/'0'
    so there is a real (if synthetic) point for the Background tab to
    fit - see bmlab/tests/test_background_controller.py, which this
    mirrors.
    """
    src = data_file_path('Water.h5')
    dst = tmp_path / 'water_with_background.h5'
    shutil.copy(src, dst)

    with h5py.File(dst, 'r+') as h5:
        payload_point = h5['Brillouin/0/payload/data/0']
        background_data = h5['Brillouin/0/background/data']
        h5.copy(payload_point, background_data, name='0')
        background_data['0'].attrs['channel'] = \
            np.array([b'Background'], dtype='S')
        background_data['0'].attrs['position_x_um'] = [1.0]
        background_data['0'].attrs['position_y_um'] = [2.0]
        background_data['0'].attrs['position_z_um'] = [3.0]

    return dst


def test_background_view_shows_no_points_by_default(qtbot):
    window = BMicro()
    qtbot.add_widget(window)

    Session.get_instance().clear()
    window.open_file(str(data_file_path('Water.h5')))
    window.widget_background_view.update_ui()

    assert window.widget_background_view.table.rowCount() == 0
    assert not window.widget_background_view.button_evaluate.isEnabled()
    assert 'no background reference points' in \
        window.widget_background_view.label_status.text().lower()

    window.close()


def test_background_view_populates_table_after_evaluate(qtbot, tmp_path):
    window = BMicro()
    qtbot.add_widget(window)

    fixture = _make_background_fixture(tmp_path)
    Session.get_instance().clear()
    window.open_file(str(fixture))

    session = Session.get_instance()
    session.orientation = Orientation(rotation=1, reflection={
        'vertically': False, 'horizontally': False
    })
    ExtractionController().find_points_all()
    cc = CalibrationController()
    for calib_key in session.get_calib_keys():
        cc.find_peaks(calib_key)
        cc.calibrate(calib_key)
    psc = PeakSelectionController()
    psc.add_brillouin_region_frequency((4.0e9, 6.0e9))
    psc.add_rayleigh_region_frequency((-2.0e9, 2.0e9))

    window.widget_background_view.update_ui()
    assert window.widget_background_view.button_evaluate.isEnabled()

    # Run the fit synchronously (bypassing the tab's own background
    # thread - see test_quality.py's evaluate_water() for the same
    # convention) and refresh the view the same way its worker thread's
    # `finished` signal would.
    BackgroundController().evaluate()
    window.widget_background_view.refresh_table()

    table = window.widget_background_view.table
    assert table.rowCount() == 1
    assert table.item(0, 0).text() == '0'
    assert float(table.item(0, 1).text()) == 1.0
    assert float(table.item(0, 2).text()) == 2.0
    assert float(table.item(0, 3).text()) == 3.0
    shift = float(table.item(0, 4).text())
    assert 4.5 < shift < 5.5

    window.close()
