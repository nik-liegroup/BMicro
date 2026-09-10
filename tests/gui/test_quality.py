import pathlib

import numpy as np

from bmlab.controllers import Controller, EvaluationController
from bmlab.models import Orientation
from bmlab.models.setup import AVAILABLE_SETUPS
from bmlab.session import Session

from bmicro.gui.main import BMicro


def data_file_path(file_name):
    return pathlib.Path(__file__).parent.parent / 'data' / file_name


def evaluate_water(file_name='Water.h5'):
    """
    Runs the real evaluation pipeline (not a hand-built results array,
    see tests/test_run_bmlab_pipeline.py in bmlab) against the given
    file, on the same Session singleton BMicro's own views read from -
    so a GUI view can be refreshed against genuinely fitted data
    without going through the GUI's own background-thread Evaluate
    button.
    """
    setup = AVAILABLE_SETUPS[0]
    orientation = Orientation(
        rotation=1, reflection={'vertically': False, 'horizontally': False})
    brillouin_regions = [(4.0e9, 6.0e9), (9.0e9, 11.0e9)]
    rayleigh_regions = [(-2.0e9, 2.0e9), (13.0e9, 17.0e9)]
    Controller().evaluate(
        data_file_path(file_name), setup, orientation,
        brillouin_regions, rayleigh_regions)


def test_quality_view_rows_cover_every_metric(qtbot):
    window = BMicro()
    qtbot.add_widget(window)

    assert set(window.widget_quality_view._rows) == \
        set(EvaluationController.QUALITY_METRIC_KEYS)

    window.close()


def test_quality_view_threshold_updates_pass_count(qtbot):
    window = BMicro()
    qtbot.add_widget(window)

    evaluate_water()
    window.widget_quality_view.update_ui()

    label_no_threshold = window.widget_quality_view\
        .label_points_remaining.text()
    assert label_no_threshold.startswith('Points passing:')
    assert ' - / -' not in label_no_threshold

    # Enable a deliberately impossible SNR threshold - nothing should
    # pass, and the label must reflect that live, without needing to
    # press "Apply to results" first.
    row = window.widget_quality_view._rows['brillouin_peak_snr']
    row['min_check'].setChecked(True)
    row['min_box'].setValue(1e8)

    label_filtered = window.widget_quality_view\
        .label_points_remaining.text()
    assert label_filtered.startswith('Points passing: 0 /')

    # Applying commits that preview into evm.results['quality_pass'],
    # read back the same way any consumer (a plot, the export CSV)
    # would - through get_data(), not the raw 6D results array.
    window.widget_quality_view.on_apply()
    evc = window.widget_quality_view.evaluation_controller
    quality_pass, _, _, _ = evc.get_data('quality_pass', 0)
    time_data, _, _, _ = evc.get_data('time', 0)
    measured = ~np.isnan(time_data)
    assert measured.sum() > 0
    assert (quality_pass[measured] == 0).all()
    assert np.isnan(quality_pass[~measured]).all()

    Session.get_instance().clear()
    window.close()
