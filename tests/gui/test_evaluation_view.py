import pathlib

import pytest

from bmlab.session import Session

from bmicro.gui.main import BMicro


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


def test_select_brillouin_shift_method(qtbot, window):
    view = window.widget_evaluation_view
    session = Session.get_instance()
    evm = session.evaluation_model()

    # Default is the Rayleigh peak based method
    assert view.combobox_shift_method.currentIndex() == 0
    assert evm.get_brillouin_shift_method() == 'rayleigh'

    view.combobox_shift_method.setCurrentIndex(1)
    assert evm.get_brillouin_shift_method() == 'fsr'

    view.combobox_shift_method.setCurrentIndex(0)
    assert evm.get_brillouin_shift_method() == 'rayleigh'


def test_shift_method_restored_on_update_ui(qtbot, window):
    view = window.widget_evaluation_view
    session = Session.get_instance()
    evm = session.evaluation_model()

    evm.set_brillouin_shift_method('fsr')
    view.update_ui()

    assert view.combobox_shift_method.currentIndex() == 1
