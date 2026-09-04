import pathlib
import shutil

from bmlab.session import Session

from bmicro.gui.main import BMicro


def data_file_path(file_name):
    return pathlib.Path(__file__).parent.parent / 'data' / file_name


def test_export_dialog_has_additional_data_checkboxes(qtbot, mocker):
    window = BMicro()
    qtbot.add_widget(window)
    file_name = 'Water.h5'
    file_path = data_file_path(file_name)

    def mock_getOpenFileName(self, *args, **kwargs):
        return file_path, None

    mocker.patch('PyQt6.QtWidgets.QFileDialog.getOpenFileName',
                 mock_getOpenFileName)
    window.open_file()

    window.on_action_export_file()

    checkbox_overview = \
        window.export_dialog.checkbox_export_overview_brightfield
    checkbox_surface = window.export_dialog.checkbox_export_surface

    # Both are enabled by default, matching ExportController's default
    # configuration.
    assert checkbox_overview.isChecked()
    assert checkbox_surface.isChecked()
    assert window.export_config['overviewBrightfield']['export'] is True
    assert window.export_config['surface']['export'] is True

    checkbox_overview.setChecked(False)
    assert window.export_config['overviewBrightfield']['export'] is False

    checkbox_surface.setChecked(False)
    assert window.export_config['surface']['export'] is False

    window.close_export_dialog()
    window.close()


def test_export_surface_and_overview_brightfield(qtbot, mocker, tmp_path):
    file_name = 'SurfaceScan.h5'
    file_path = tmp_path / file_name
    shutil.copy(data_file_path(file_name), file_path)

    window = BMicro()
    qtbot.add_widget(window)

    def mock_getOpenFileName(self, *args, **kwargs):
        return file_path, None

    mocker.patch('PyQt6.QtWidgets.QFileDialog.getOpenFileName',
                 mock_getOpenFileName)
    window.open_file()

    # Only export the new data, skip the rest to keep the test fast
    window.export_config['fluorescence']['export'] = False
    window.export_config['fluorescenceCombined']['export'] = False
    window.export_config['brillouin']['export'] = False

    window.export_file()

    # SurfaceScan.h5 is not inside a 'RawData' folder here, so bmlab's
    # exporters write directly next to the source file (see e.g.
    # FluorescenceExport.export / SurfaceExport._plot_path).
    assert (
        tmp_path / 'SurfaceScan_BMrep0_surface_found_mask.png').exists()
    assert (
        tmp_path
        / 'SurfaceScan_FLrep0_overviewBrightfield_000.png').exists()
    assert (
        tmp_path / 'SurfaceScan_BMrep0_surface_metrics.json').exists()

    Session.get_instance().clear()
    window.close()
