import pathlib
import os

from PyQt6.QtWidgets import QWidget
from PyQt6 import QtCore

from bmlab.session import Session

from bmicro.gui.main import BMicro, check_event_mime_data


def data_file_path(file_name):
    return pathlib.Path(__file__).parent.parent / 'data' / file_name


def test_main_window_can_activate_all_tabs(qtbot):

    window = BMicro()
    qtbot.add_widget(window)
    tab_texts = ['Data', 'Extraction', 'Calibration',
                 'Peak Selection', 'Evaluation']
    tab_names = ['tab_data', 'tab_extraction', 'tab_calibration',
                 'tab_peak_selection', 'tab_evaluation']

    for idx, (tab_text, tab_name) in enumerate(zip(tab_texts, tab_names)):
        current_tab = window.tabWidget.findChild(QWidget, tab_name)
        assert current_tab
        qtbot.mouseClick(current_tab, QtCore.Qt.MouseButton.LeftButton)
        assert window.tabWidget.tabText(idx) == tab_text

    window.close()


def test_open_file_shows_metadata(qtbot, mocker):
    window = BMicro()
    file_name = 'Water.h5'
    file_path = data_file_path(file_name)

    def mock_getOpenFileName(self, *args, **kwargs):
        return file_path, None

    mocker.patch('PyQt6.QtWidgets.QFileDialog.getOpenFileName',
                 mock_getOpenFileName)

    window.open_file()

    w = window.widget_data_view
    assert w.label_selected_file.text() == str(file_name)
    assert w.label_selected_file.toolTip() == str(file_path)
    assert w.label_date.text() == '2020-11-03 15:20'
    assert w.label_resolution_x.text() == '10'
    assert w.label_resolution_y.text() == '1'
    assert w.label_resolution_z.text() == '1'
    assert w.label_calibration.text() == 'True'
    assert w.textedit_comment.toPlainText() == 'Brillouin data'

    window.close()


def test_start_batch_evaluation_refuses_while_export_running(qtbot, mocker):
    """
    start_batch_evaluation() must not start a batch run while an
    export is already in progress - every file's evaluate() call would
    silently decline to run (EvaluationView.evaluate()'s own
    export_running guard), leaving the batch limping through files it
    never actually evaluated.
    """
    window = BMicro()
    qtbot.add_widget(window)

    warn = mocker.patch.object(window, '_warn_export_in_progress')
    run_batch = mocker.patch.object(window, 'run_batch_evaluation')

    window.export_running = True
    window.start_batch_evaluation()

    warn.assert_called_once()
    run_batch.assert_not_called()
    assert window.batch_evaluation_running is False

    window.export_running = False
    window.close()


def test_check_event_mime_data():
    class DummyEvent:
        def __init__(self, *args, **kwargs):
            self.__mimeData = QtCore.QMimeData()

        def mimeData(self):
            return self.__mimeData

    event = DummyEvent()

    """ Test for empty url """
    path = check_event_mime_data(event)
    assert path is False

    """ Test for wrong file type """
    event.mimeData().setUrls([QtCore.QUrl("file:/directory/file.txt")])
    path = check_event_mime_data(event)
    assert path is False

    """ Test for correct file type """
    event.mimeData().setUrls([QtCore.QUrl("file:/directory/file.h5")])
    path = check_event_mime_data(event)
    assert path == '/directory/file.h5'


def test_save_and_load_session():
    if os.path.exists(data_file_path('Water.session.h5')):
        os.remove(data_file_path('Water.session.h5'))
    session = Session.get_instance()
    session.set_file(data_file_path('Water.h5'))
    session.set_reflection(vertically=True)
    session.extraction_models['0'].add_point('0', 10, 30, 30)
    session.save()

    assert os.path.exists(data_file_path('Water.session.h5'))

    session.clear()

    assert session.file is None
    assert session.extraction_models == {}

    session.set_file(data_file_path('Water.h5'))

    assert session.file is not None
    assert len(session.extraction_models) == 1

    os.remove(data_file_path('Water.session.h5'))


def _copy_source_file(tmp_path, folder, name='Water.h5'):
    import shutil
    target = tmp_path / folder
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy(data_file_path(name), target / name)
    return target / name


def test_batch_scan_worker_finds_new_source_files_only(qtbot, tmp_path):
    import h5py
    from bmicro.gui.main import BatchScanWorker, file_hash
    first = _copy_source_file(tmp_path, 'a')
    second = _copy_source_file(tmp_path, 'b')
    with h5py.File(tmp_path / 'not_a_source.h5', 'w') as f:  # skipped
        f.create_group('whatever')

    scanned, found = [], []
    worker = BatchScanWorker(
        tmp_path, [file_hash(first)], lambda path: {'0': path.name})
    worker.found.connect(found.append)
    worker.scanned.connect(lambda path, reps: scanned.append((path, reps)))
    with qtbot.waitSignal(worker.finished, timeout=20000):
        worker.run()
    assert found == [1]  # `first` is known already
    assert scanned == [(second, {'0': 'Water.h5'})]


def test_batch_scan_worker_can_be_cancelled(qtbot, tmp_path):
    from bmicro.gui.main import BatchScanWorker
    for folder in ('a', 'b', 'c'):
        _copy_source_file(tmp_path, folder)
    worker = BatchScanWorker(tmp_path, [], lambda path: {})
    scanned = []

    def on_scanned(path, reps):
        scanned.append(path)
        worker.aborted = True  # as the Cancel button does

    worker.scanned.connect(on_scanned)
    with qtbot.waitSignal(worker.finished, timeout=20000):
        worker.run()
    assert len(scanned) == 1


def test_batch_add_folder_scans_in_background_and_fills_table(
        qtbot, mocker, tmp_path):
    from PyQt6 import QtWidgets
    _copy_source_file(tmp_path, 'a')
    _copy_source_file(tmp_path, 'b')
    mocker.patch.object(QtWidgets.QDialog, 'exec', return_value=0)
    mocker.patch.object(
        QtWidgets.QFileDialog, 'getExistingDirectory',
        return_value=str(tmp_path))
    window = BMicro()
    qtbot.add_widget(window)
    window.on_action_batch_evaluation()  # builds the dialog, exec() mocked

    window.batch_add_files()
    # the scan runs in its own thread: the call returns at once
    assert window.batch_scan_thread is not None
    qtbot.waitUntil(lambda: window.batch_scan_thread is None, timeout=60000)
    assert len(window.batch_files) == 2
    assert window.batch_dialog.table_files.rowCount() >= 2
    window.close()


import copy
import types

import pytest


@pytest.fixture(scope='module')
def shared_window(qapp):
    """
    One main window with the batch dialog built for all the tests below
    (creating a window takes about a second, a test needs seconds less).
    """
    from unittest import mock
    from PyQt6 import QtWidgets
    with mock.patch.object(QtWidgets.QDialog, 'exec', return_value=0):
        window = BMicro()
        window.on_action_batch_evaluation()  # builds the dialog
    yield window, copy.deepcopy(window.batch_config)
    window.close()


@pytest.fixture
def window(shared_window):
    """ The shared window, with the batch state of a fresh one. """
    from PyQt6.QtWidgets import QCheckBox
    window, defaults = shared_window
    window.batch_config = copy.deepcopy(defaults)
    window.batch_files = {}
    window.batch_reanalyse = False
    window.batch_evaluation_messages = []
    window.batch_forced = {}
    for box in window.batch_dialog.findChildren(QCheckBox):
        box.blockSignals(True)
        box.setChecked(False)
        box.blockSignals(False)
    Session.get_instance().clear()
    return window


def _open_water_in_session():
    session = Session.get_instance()
    session.clear()
    session.set_file(data_file_path('Water.h5'))
    session.set_current_repetition(session.file.repetition_keys()[0])
    # the session is a singleton: start every test from the default
    # orientation, whatever an earlier test set
    session.set_rotation(0)
    session.set_reflection(vertically=False, horizontally=False)
    return session


def _file_entry(session, progress):
    key = session.file.repetition_keys()[0]
    rep = {'valid': True, 'checked': True, 'progress': progress,
           'calib_keys': [], 'nr_spectra': 1, 'label': ''}
    return key, {'path': data_file_path('Water.h5'), 'status': 'pending',
                 'repetitions': {key: rep}}


def _batch_state(**config):
    """
    What _apply_batch_base_settings() needs of the main window, without
    building one: it only reads and fills these attributes.
    """
    from bmlab.models.setup import AVAILABLE_SETUPS
    state = types.SimpleNamespace(
        batch_reanalyse=False, batch_evaluation_messages=[],
        batch_config={
            'setup': {'set': False, 'setup': AVAILABLE_SETUPS[0]},
            'orientation': {
                'set': False, 'rotation': 0,
                'reflection': {'vertically': False,
                               'horizontally': False}}})
    for name, values in config.items():
        state.batch_config[name].update(values)
    return state


def _apply(state, file):
    BMicro._apply_batch_base_settings(state, file)


def test_batch_sets_setup_and_orientation_on_files_not_analysed_yet():
    session = _open_water_in_session()
    key, file = _file_entry(session, 'none')
    old_models = session.extraction_models[key]
    state = _batch_state(orientation={'set': True, 'rotation': 1})

    _apply(state, file)

    assert session.orientation.rotation == 1
    assert session.extraction_models[key] is old_models  # nothing to reset
    assert not state.batch_evaluation_messages


def test_batch_leaves_setup_and_orientation_of_analysed_files():
    session = _open_water_in_session()
    key, file = _file_entry(session, 'evaluation')
    session.set_rotation(0)
    state = _batch_state(orientation={'set': True, 'rotation': 1})

    _apply(state, file)
    _apply(state, file)  # called once per repetition

    assert session.orientation.rotation == 0
    assert len(state.batch_evaluation_messages) == 1
    assert 'left as saved' in state.batch_evaluation_messages[0]


def test_changed_orientation_resets_everything_built_on_it():
    session = _open_water_in_session()
    key, file = _file_entry(session, 'evaluation')
    session.set_rotation(0)
    models = [session.extraction_models[key], session.calibration_models[key],
              session.peak_selection_models[key],
              session.evaluation_models[key]]
    state = _batch_state(orientation={'set': True, 'rotation': 1})
    state.batch_reanalyse = True

    _apply(state, file)

    assert session.orientation.rotation == 1
    after = [session.extraction_models[key], session.calibration_models[key],
             session.peak_selection_models[key],
             session.evaluation_models[key]]
    assert all(new is not old for new, old in zip(after, models))
    assert 'reset from extraction' in state.batch_evaluation_messages[0]

    # applied again (next step, next repetition): nothing changes any more
    state.batch_evaluation_messages.clear()
    _apply(state, file)
    assert not state.batch_evaluation_messages
    assert session.extraction_models[key] is after[0]


def test_changed_setup_keeps_extraction_but_resets_the_rest():
    from bmlab.models.setup import AVAILABLE_SETUPS
    session = _open_water_in_session()
    key, file = _file_entry(session, 'evaluation')
    other = [s for s in AVAILABLE_SETUPS if s.key != session.setup.key]
    if not other:
        return
    extraction = session.extraction_models[key]
    calibration = session.calibration_models[key]
    state = _batch_state(setup={'set': True, 'setup': other[0]})
    state.batch_reanalyse = True

    _apply(state, file)

    assert session.setup.key == other[0].key
    assert session.extraction_models[key] is extraction
    assert session.calibration_models[key] is not calibration
    assert 'reset from calibration' in state.batch_evaluation_messages[0]


def test_ticking_setup_or_orientation_ticks_the_steps_built_on_them(window):
    dialog = window.batch_dialog
    assert not dialog.checkBox_extraction.isChecked()

    dialog.checkBox_orientation.click()

    for box in (dialog.checkBox_extraction, dialog.checkBox_calibrate,
                dialog.checkBox_peak_selection, dialog.checkBox_evaluation):
        assert box.isChecked()


def test_evaluate_alone_does_not_ask_to_replace_finished_evaluations(
        window, mocker):
    from PyQt6.QtWidgets import QMessageBox
    session = _open_water_in_session()
    key, file = _file_entry(session, 'evaluation')
    window.batch_files = {'x': file}
    window.batch_config['evaluation']['evaluate'] = True
    box = mocker.patch.object(
        QMessageBox, 'exec', return_value=QMessageBox.StandardButton.Cancel)

    window.batch_reanalyse = False
    assert window._confirm_batch_overwrite() is True
    box.assert_not_called()

    window.batch_reanalyse = True  # now they would be redone: ask
    assert window._confirm_batch_overwrite() is False
    box.assert_called_once()


def test_batch_evaluation_skips_repetitions_that_are_evaluated(
        window, mocker):
    session = _open_water_in_session()
    for name in ('open_file', 'save_session', '_refresh_file_progress',
                 '_save_calibration_plots', 'update_batch_file_table'):
        mocker.patch.object(window, name)
    mocker.patch.object(window, 'aborted', return_value=False)
    evaluate = mocker.patch.object(window.widget_evaluation_view, 'evaluate')
    window.batch_config['evaluation']['evaluate'] = True

    key, file = _file_entry(session, 'evaluation')
    window.evaluate_batch_file(file)
    assert not evaluate.called
    assert not window.batch_evaluation_messages  # skipped before any check

    # not evaluated yet: it gets as far as the evaluation checks (Water.h5
    # has no peak selection, so it is reported as skipped there)
    session = _open_water_in_session()  # the batch run closes the file
    key, file = _file_entry(session, 'peak selection')
    window.evaluate_batch_file(file)
    assert any('skipped evaluation' in m
               for m in window.batch_evaluation_messages)


def test_warning_before_setup_or_orientation_resets_analysed_files(
        window, mocker):
    from PyQt6.QtWidgets import QMessageBox
    session = _open_water_in_session()
    key, file = _file_entry(session, 'calibration')
    window.batch_files = {'x': file}
    box = mocker.patch.object(
        QMessageBox, 'exec', return_value=QMessageBox.StandardButton.Cancel)
    set_text = mocker.patch.object(QMessageBox, 'setInformativeText')
    window.batch_config['orientation']['set'] = True

    window.batch_reanalyse = False  # nothing is reset then: no question
    assert window._confirm_batch_overwrite() is True
    box.assert_not_called()

    window.batch_reanalyse = True
    assert window._confirm_batch_overwrite() is False  # Cancel stops the run
    # the saved setup/orientation is not known for this entry
    assert 'could not be read' in set_text.call_args[0][0]


def _saved_base(session):
    from bmicro.gui.base_settings import setup_signature, \
        orientation_signature
    return (setup_signature(session.setup),
            orientation_signature(session.orientation.rotation,
                                  session.orientation.reflection))


def test_warning_names_exactly_the_files_whose_base_would_change(
        window, mocker):
    from PyQt6.QtWidgets import QMessageBox
    session = _open_water_in_session()
    session.set_rotation(0)
    key, same = _file_entry(session, 'calibration')
    same['repetitions'][key]['saved_base'] = _saved_base(session)
    key, other = _file_entry(session, 'calibration')
    other['path'] = data_file_path('Other.h5')
    saved = _saved_base(session)
    other['repetitions'][key]['saved_base'] = (saved[0], (1, False, False))
    window.batch_files = {'a': same, 'b': other}
    box = mocker.patch.object(
        QMessageBox, 'exec', return_value=QMessageBox.StandardButton.Cancel)
    set_text = mocker.patch.object(QMessageBox, 'setInformativeText')
    window.batch_reanalyse = True
    window.batch_config['orientation'].update(
        set=True, rotation=0, reflection=dict(
            vertically=False, horizontally=False))

    assert window._confirm_batch_overwrite() is False
    text = set_text.call_args[0][0]
    assert '1 analysed file(s) get a different' in text
    assert 'Other.h5' in text and 'Water.h5' not in text

    # nothing differs: the setup/orientation line is gone (the dialog
    # itself stays, re-analysing redoes the ticked steps anyway)
    window.batch_files = {'a': same}
    window._confirm_batch_overwrite()
    assert 'get a different' not in set_text.call_args[0][0]
    assert box.call_count == 2


def test_batch_scan_reads_the_saved_setup_and_orientation(tmp_path):
    import shutil
    path = tmp_path / 'Water.h5'
    shutil.copy(data_file_path('Water.h5'), path)
    session = Session.get_instance()
    session.clear()
    session.set_file(path)
    session.set_rotation(1)
    session.set_reflection(vertically=True, horizontally=False)
    session.save()  # next to the file, as the app does
    session.clear()

    reps = BMicro._scan_batch_repetitions(path)

    assert reps
    for rep in reps.values():
        setup_saved, orientation_saved = rep['saved_base']
        assert orientation_saved == (1, True, False)
        assert setup_saved is not None


def test_reset_analysis_gives_new_extraction_models_their_shape():
    from bmicro.gui.base_settings import reset_analysis, has_analysis
    session = _open_water_in_session()
    key = session.file.repetition_keys()[0]
    old = session.extraction_models[key]
    old.add_point('0', 0.0, 1.0, 2.0)
    assert has_analysis(session, key)
    shape = old.image_shape

    reset_analysis(session, [key], True)

    new = session.extraction_models[key]
    assert new is not old and not has_analysis(session, key)
    assert new.image_shape == shape  # as after loading the file
    assert new.arc_width >= 1


def test_data_tab_asks_before_a_new_orientation_resets_the_analysis(
        window, mocker):
    from PyQt6.QtWidgets import QMessageBox
    session = _open_water_in_session()
    session.set_rotation(0)
    view = window.widget_data_view
    view.update_ui()
    key = session.file.repetition_keys()[0]
    session.extraction_models[key].add_point('0', 0.0, 1.0, 2.0)
    model = session.extraction_models[key]
    ask = mocker.patch.object(
        QMessageBox, 'exec', return_value=QMessageBox.StandardButton.Cancel)

    view.radio_rotation_90_cw.click()  # asked, cancelled

    ask.assert_called_once()
    assert session.orientation.rotation == 0
    assert session.extraction_models[key] is model  # analysis kept
    assert view.radio_rotation_none.isChecked()  # control is back

    view.checkbox_reflect_vertically.click()  # cancelled as well
    assert not session.orientation.reflection['vertically']
    assert not view.checkbox_reflect_vertically.isChecked()

    ask.return_value = QMessageBox.StandardButton.Ok
    view.radio_rotation_90_cw.click()  # confirmed
    assert session.orientation.rotation == 1
    assert session.extraction_models[key] is not model  # reset
    assert not session.extraction_models[key].points


def test_data_tab_does_not_ask_without_analysis_or_when_refreshing(
        window, mocker):
    from PyQt6.QtWidgets import QMessageBox
    session = _open_water_in_session()
    session.set_rotation(0)
    view = window.widget_data_view
    view.update_ui()
    ask = mocker.patch.object(
        QMessageBox, 'exec', return_value=QMessageBox.StandardButton.Cancel)

    view.radio_rotation_90_cw.click()  # nothing analysed
    assert session.orientation.rotation == 1
    ask.assert_not_called()

    key = session.file.repetition_keys()[0]
    session.extraction_models[key].add_point('0', 0.0, 1.0, 2.0)
    session.set_reflection(vertically=True)
    view.update_ui()  # copies the state into the controls
    ask.assert_not_called()
    assert view.checkbox_reflect_vertically.isChecked()
