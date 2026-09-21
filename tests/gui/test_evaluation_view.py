import pathlib

import numpy as np
import pytest

from bmlab.session import Session

from bmicro.gui.main import BMicro
from bmicro.gui.evaluation.evaluation_view import EvaluationView


def data_file_path(file_name):
    return pathlib.Path(__file__).parent.parent / 'data' / file_name


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


def test_first_populated_z_slice_skips_empty_preferred_index():
    """
    Regression test: a ROI-restricted grid can have entire slices along
    whichever axis the z-slider ends up using (see refresh_plot()'s own
    "shortest dimension" comment) with zero measured points at all -
    e.g. index 0 sitting entirely outside the drawn ROI. The default
    slice must not silently land on one of those and show an empty
    plot when other slices actually have data.
    """
    data = np.full((4, 3, 2), np.nan)
    data[2, 1, 0] = 5.0  # only slice index 2 (of axis 0) has data

    # Preferred (previous/default) index 0 is empty - must skip to the
    # first index that actually has something, here 2.
    assert EvaluationView._first_populated_z_slice(data, 0, 0) == 2

    # Preferred index already has data - keep it, don't hunt further.
    assert EvaluationView._first_populated_z_slice(data, 0, 2) == 2

    # No slice anywhere on this axis has any data - falls back to
    # whatever was preferred, unchanged.
    empty = np.full((4, 3, 2), np.nan)
    assert EvaluationView._first_populated_z_slice(empty, 0, 1) == 1
