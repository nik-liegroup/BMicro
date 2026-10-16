"""
The Data tab's "Acquisition" box: how a repetition was measured, from what
BrillouinAcquisition wrote into the file (the grid, the overviews, the drift
correction, whether it ended normally).
"""
import pathlib
import shutil

import h5py
import numpy as np
import pytest

from bmlab.file import BrillouinFile

from bmicro.gui.data.acquisition_info import acquisition_summary
from bmicro.gui.main import BMicro


def data_file_path(file_name):
    return pathlib.Path(__file__).parent.parent / 'data' / file_name


def put(group, name, values):
    group.create_dataset(name, data=np.atleast_1d(np.asarray(values, float)))


def add_acquisition(path, decisions, state=b'finished', offset=0.0):
    """What BrillouinAcquisition writes, added to Water.h5 (10 x 1 x 1)."""
    codes = {'reference': 0, 'applied': 1, 'no match': 2}
    with h5py.File(path, 'r+') as f:
        f['Brillouin/0'].attrs['acquisition_state'] = np.array([state])
        payload = f['Brillouin/0/payload']
        put(payload, 'positions-grid-x-range-um-used', [0, 90])
        put(payload, 'positions-grid-y-range-um-used', [0, 0])
        put(payload, 'positions-grid-z-range-um-used', [0, 0])
        put(payload, 'positions-grid-coordinates-absolute', 1)
        put(payload, 'positions-grid-scan-order-used', [0, 1, 2])
        put(payload, 'positions-drift-correction-used', 1)
        put(payload, 'positions-drift-objective-slot-used', 6)
        put(payload, 'positions-drift-min-ncc-used', 0.3)
        put(payload, 'positions-overview-objective-6-magnification', 10)
        put(payload, 'positions-overview-objective-6-fov-offset-um',
            [1, 119.2, -166.3, 0.4])
        for number, decision in enumerate(decisions):
            prefix = 'positions-overview-event-%04d-' % number
            put(payload, prefix + 'kind', 0 if number == 0 else 1)
            put(payload, prefix + 'drift-evaluated', 1)
            put(payload, prefix + 'drift-decision', codes[decision])
            put(payload, prefix + 'drift-step-um', [0, 0])
            put(payload, prefix + 'drift-total-um', [offset, 0])
            put(payload, prefix + 'grid-offset-after-um', [offset, 0])
            put(payload, prefix + 'drift-mean-ncc', 0.9)
            put(payload, prefix + 'drift-tiles-compared-used', [1, 1])
            put(payload, prefix + 'drift-seconds-since-reference',
                9000 if decision == 'no match' else 0)
        # the offset every point was measured with (needs the scan plan)
        for axis in 'xyz':
            put(payload, 'positions-resume-plan-index-' + axis,
                np.concatenate([[10], np.arange(10) if axis == 'x'
                                else np.zeros(10)]))
        put(payload, 'positions-grid-offset-per-point-x-um',
            np.full(10, offset))
        put(payload, 'positions-grid-offset-per-point-y-um', np.zeros(10))


@pytest.fixture
def water(tmp_path):
    path = tmp_path / 'Water.h5'
    shutil.copy(data_file_path('Water.h5'), path)
    return path


def test_summary_of_a_drift_correction_that_worked(water):
    add_acquisition(water, ['reference', 'applied', 'applied'], offset=4.0)
    file = BrillouinFile(water)
    text = acquisition_summary(file.get_repetition('0'))
    file.close()

    assert 'Finished normally.' in text
    assert 'Grid:' in text and 'x: 10 points, 0.0 to 90.0 um' in text
    assert 'every 10.00 um' in text
    assert 'scan order (innermost loop first): x, y, z' in text
    assert 'absolute (stage)' in text
    assert '3 events (1 before measurement, 2 after z plane)' in text
    assert 'objective in slot 6, 10x' in text
    assert 'Drift correction while measuring: on' in text
    assert '1 reference, 2 applied' in text
    assert 'followed the sample, by up to 4.0 um' in text
    assert 'NOTHING WAS CORRECTED' not in text


def test_summary_warns_when_the_drift_correction_never_matched(water):
    add_acquisition(water, ['no match', 'no match'])
    file = BrillouinFile(water)
    text = acquisition_summary(file.get_repetition('0'))
    file.close()

    assert '2 no match' in text
    assert 'NOTHING WAS CORRECTED' in text
    assert 'still needed' in text
    # 9000 s at the first overview: a reference of an earlier run
    assert '150 min old' in text


def test_summary_of_a_stopped_repetition(water):
    add_acquisition(water, ['reference'], state=b'stopped')
    file = BrillouinFile(water)
    text = acquisition_summary(file.get_repetition('0'))
    file.close()
    assert text.startswith('NOT FINISHED: stopped (cancelled).')


def test_summary_of_an_old_file_is_short_and_does_not_fail(water):
    file = BrillouinFile(water)
    text = acquisition_summary(file.get_repetition('0'))
    file.close()
    assert 'Drift correction' not in text
    assert 'Overviews' not in text
    assert 'Finished' not in text and 'NOT FINISHED' not in text
    assert 'Grid:' in text          # the number of points is always there


def test_data_tab_shows_the_acquisition(qtbot, mocker, water):
    add_acquisition(water, ['no match', 'no match'])
    window = BMicro()
    mocker.patch(
        'PyQt6.QtWidgets.QFileDialog.getOpenFileName',
        lambda *args, **kwargs: (water, None))
    window.open_file()
    qtbot.addWidget(window)

    box = window.widget_data_view.textedit_acquisition
    assert 'NOTHING WAS CORRECTED' in box.toPlainText()
    window.close()
