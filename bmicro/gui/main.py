import pathlib
from importlib import resources
import copy
import hashlib
import logging
import signal
import sys
import traceback

import numpy as np

from PyQt6 import QtWidgets, uic, QtCore, QtGui
from PyQt6.QtWidgets import QFileDialog, QMessageBox, \
    QVBoxLayout, QHBoxLayout, QCheckBox, QLabel, QDoubleSpinBox, QWidget
from PyQt6.QtCore import QSize

from bmlab.session import Session, get_session_file_path
from bmlab.file import is_source_file
from bmlab.models.setup import AVAILABLE_SETUPS
from bmlab.models import EvaluationModel
from bmlab.controllers import PeakSelectionController, ExportController, \
    EvaluationController

from . import data
from . import extraction
from . import calibration
from . import peak_selection
from . import evaluation
from . import quality

from bmicro import __version__ as bmicroversion
from bmlab import __version__ as bmlabversion

logger = logging.getLogger(__name__)


def check_event_mime_data(event):
    """ Returns the path to local file if h5 file """
    if event.mimeData().hasUrls():
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path.endswith(".h5"):
                return path
    return False


class ExportWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal()

    def __init__(self, configuration):
        super().__init__()
        self.configuration = configuration

    def run(self):
        try:
            ExportController().export(self.configuration)
        except Exception:
            logger.warning('Export failed', exc_info=True)
        finally:
            self.finished.emit()


class BatchExportWorker(QtCore.QObject):
    progress = QtCore.pyqtSignal(int, str)
    finished = QtCore.pyqtSignal(int, list)

    def __init__(self, files, configuration):
        super().__init__()
        self.files = files
        self.configuration = configuration
        self.aborted = False

    def run(self):
        session = Session.get_instance()
        succeeded = 0
        failed = []
        for i, file_path in enumerate(self.files):
            if self.aborted:
                break
            self.progress.emit(i + 1, str(file_path))
            try:
                # Sessions carry state (crop, orientation, etc.) that
                # must be cleared before pointing at a new file - see
                # BMicro.open_file()/close_file().
                session.clear()
                session.set_file(file_path)
                ExportController().export(self.configuration)
            except Exception:
                logger.warning(
                    'Batch export failed for %s', file_path,
                    exc_info=True)
                failed.append(file_path)
            else:
                succeeded += 1
        session.clear()
        self.finished.emit(succeeded, failed)


class BMicro(QtWidgets.QMainWindow):
    """
    Class for the main window of BMicro.
    The application can be started from console by running

        python -m bmicro

    """

    def __init__(self, *args, **kwargs):
        # Settings are stored in the .ini file format. Even though
        # `self.settings` may return integer/bool in the same session,
        # in the next session, it will reliably return strings. Lists
        # of strings (comma-separated) work nicely though.
        QtCore.QCoreApplication.setOrganizationName("BMicro")
        QtCore.QCoreApplication.setOrganizationDomain(
            "bmicro.readthedocs.io")
        QtCore.QCoreApplication.setApplicationName("BMicro")
        QtCore.QSettings.setDefaultFormat(QtCore.QSettings.Format.IniFormat)
        # Some promoted widgets may need the above constants set in order
        # to access the settings upon initialization.

        """ Initializes BMicro."""
        super(BMicro, self).__init__(*args, **kwargs)
        ref = resources.files('bmicro.gui') / 'main.ui'
        with resources.as_file(ref) as ui_file:
            uic.loadUi(ui_file, self)
        QtCore.QCoreApplication.setApplicationName('BMicro')

        ref = resources.files('bmicro') / 'img'
        with resources.as_file(ref) as imdir:
            self.imdir = imdir

        self.tabWidget.currentChanged.connect(self.update_ui)

        self.export_dialog = None
        # Initialize the export configuration
        self.export_config = ExportController.get_configuration()
        self.export_thread = None
        self.export_worker = None
        self.export_running = False
        self.batch_export_dialog = None
        self.batch_export_progress_dialog = None
        self.batch_export_thread = None
        self.batch_export_worker = None

        self.batch_dialog = None
        self.batch_files = {}
        self.batch_config = {
            'setup': {
                'set': False,
                'setup': AVAILABLE_SETUPS[0],
            },
            'orientation': {
                'set': False,
                'rotation': 0,
                'reflection': {'vertically': False, 'horizontally': True},
            },
            'extraction': {
                'extract': False,
            },
            'calibration': {
                'find-peaks': False,
                'calibrate': False,
            },
            'peak-selection': {
                'select': False,
                'brillouin_regions': [(4.0e9, 6.0e9), (9.0e9, 11.0e9)],
                'rayleigh_regions': [(-2.0e9, 2.0e9), (13.0e9, 17.0e9)],
            },
            'evaluation': {
                'evaluate': False,
                'nr_brillouin_peaks': 1,
                'bounds_w0': None,
                'bounds_fwhm': None,
            },
            # Applied to every file right after it's evaluated (see
            # evaluate_batch_file()) - same thresholds/metrics as the
            # Quality tab (EvaluationController.QUALITY_METRIC_KEYS),
            # prefilled with EvaluationModel.get_default_quality_
            # thresholds() and editable in the "Quality" section of
            # the batch dialog (see _build_batch_quality_rows()).
            # Unlike the other sections here, this isn't optional/
            # gated by its own checkbox: whatever is set here always
            # replaces each file's own saved thresholds and is applied
            # (evc.apply_quality_thresholds(), writing quality_pass)
            # whenever "Evaluate" is ticked.
            'quality': copy.deepcopy(
                EvaluationModel.get_default_quality_thresholds()),
            'export': {
                'export': False,
            },
        }
        self.batch_evaluation_running = False
        self.batch_evaluation_messages = []

        # Build tabs
        self.widget_data_view = data.DataView(self)
        self.layout_data = QtWidgets.QVBoxLayout()
        self.tab_data.setLayout(self.layout_data)
        self.layout_data.addWidget(self.widget_data_view)
        self.widget_extraction_view = extraction.ExtractionView(self)
        self.layout_extraction = QtWidgets.QVBoxLayout()
        self.tab_extraction.setLayout(self.layout_extraction)
        self.layout_extraction.addWidget(self.widget_extraction_view)
        self.widget_calibration_view = calibration.CalibrationView(self)
        self.layout_calibration = QtWidgets.QVBoxLayout()
        self.tab_calibration.setLayout(self.layout_calibration)
        self.layout_calibration.addWidget(self.widget_calibration_view)
        self.widget_peak_selection_view = peak_selection.PeakSelectionView(
            self)
        self.layout_peak_selection = QtWidgets.QVBoxLayout()
        self.tab_peak_selection.setLayout(self.layout_peak_selection)
        self.layout_peak_selection.addWidget(self.widget_peak_selection_view)
        self.widget_evaluation_view = evaluation.EvaluationView(self)
        self.layout_evaluation = QtWidgets.QVBoxLayout()
        self.tab_evaluation.setLayout(self.layout_evaluation)
        self.layout_evaluation.addWidget(self.widget_evaluation_view)
        self.widget_quality_view = quality.QualityView(self)
        self.layout_quality = QtWidgets.QVBoxLayout()
        self.tab_quality.setLayout(self.layout_quality)
        self.layout_quality.addWidget(self.widget_quality_view)
        # Cross-link so a z-slice/autoscale/ignore-outliers/aspect-ratio
        # change in either tab's 2D view is reflected in the other -
        # both show the same measurement grid.
        self.widget_evaluation_view.set_quality_view(self.widget_quality_view)
        self.widget_quality_view.set_evaluation_view(
            self.widget_evaluation_view)

        self.connect_menu()

        # Connect drag and drop
        self.dragEnterEvent = self.drag_enter_event
        self.dropEvent = self.drop_event

        self.setAcceptDrops(True)

        self.reset_ui()

        self.settings = QtCore.QSettings()

    def connect_menu(self):
        """ Registers the menu actions """
        self.action_open.triggered.connect(self.open_file)
        self.action_close.triggered.connect(self.close_file)
        self.action_save.triggered.connect(self.save_session)
        self.action_export.triggered.connect(self.on_action_export_file)
        self.action_exit.triggered.connect(self.exit_app)

        self.action_about.triggered.connect(self.on_action_about)
        self.action_batch_evaluation.triggered.connect(
            self.on_action_batch_evaluation)
        self.action_batch_export.triggered.connect(
            self.on_action_batch_export)

    def open_file(self, file_name=None):
        """ Show open file dialog and load file. """
        if not file_name:
            file_name, _ = QFileDialog.getOpenFileName(
                self, 'Open File...',
                directory=self.settings.value("path/last-used"),
                filter='*.h5')

        """ file_name is empty if user selects 'Cancel' in FileDialog """
        if not file_name:
            return
        else:
            self.settings.setValue("path/last-used",
                                   str(pathlib.Path(file_name).parent))

        session = Session.get_instance()
        try:
            self.close_file()
            session.set_file(file_name)
            # Kept inside the try/except too: a valid file can still
            # contain a repetition the UI can't render (e.g. an
            # aborted/restarted acquisition with an empty first
            # repetition), and that shouldn't be able to crash the
            # whole application on open.
            self.update_ui()
        except FileNotFoundError as e:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setText('Unable to load file:')
            msg.setInformativeText(e.strerror)
            msg.setWindowTitle(type(e).__name__)
            msg.exec()
        except Exception as e:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setText('An unknown error occured')
            msg.setInformativeText(str(e))
            msg.setWindowTitle('Unknown Error')
            msg.exec()

    def close_file(self):
        Session.get_instance().clear()
        self.reset_ui()

    def on_action_export_file(self):
        session = Session.get_instance()
        repetition_keys = \
            session.file.repetition_keys() if session.file else []
        self.export_dialog = self._build_export_config_dialog(
            'Export configuration', self.export_file, repetition_keys)
        self.export_dialog.open()

    def _build_export_config_dialog(
            self, title, on_export, repetition_keys=None):
        """
        Builds the "what to export" dialog (checkboxes for overview
        brightfield / surface / which repetitions), shared between the
        single-file export and batch export entry points. `on_export`
        is called (with no arguments) when the dialog's Export button
        is clicked.

        There is no per-parameter choice here (and none in the "Batch
        evaluation" wizard's own export step either - see
        on_action_batch_evaluation()): BrillouinExport always writes
        every evaluated quantity into one combined CSV per repetition
        (no more per-parameter Plots/Bare and Plots/WithAxis image
        files) - there's nothing left to narrow down.

        `repetition_keys`, for the single-file export entry point, is
        the currently open file's own Brillouin repetition keys (e.g.
        ['0', '1']) - one checkbox is shown per key, all checked by
        default, so a user can restrict the export to a subset of
        repetitions. Left as None for batch export: the dialog there
        is built once for potentially many files with different (or
        as-yet-unknown) repetitions, so no per-repetition choice is
        offered and every repetition is exported for every file, as
        before.
        """
        dialog = QtWidgets.QDialog(
            self,
            QtCore.Qt.WindowType.WindowTitleHint |
            QtCore.Qt.WindowType.WindowCloseButtonHint
        )
        ref = resources.files('bmicro.gui') / 'export_configuration.ui'
        with resources.as_file(ref) as ui_file:
            uic.loadUi(ui_file, dialog)
        dialog.setWindowTitle(title)
        dialog.setWindowModality(
            QtCore.Qt.WindowModality.ApplicationModal)
        dialog.button_export.clicked.connect(lambda: on_export())
        dialog.button_cancel.clicked.connect(dialog.close)
        self.init_export_repetitions(dialog.groupBox_repetitions,
                                     repetition_keys)

        checkbox_overview = dialog.checkbox_export_overview_brightfield
        checkbox_overview.setChecked(
            self.export_config['overviewBrightfield']['export'])
        checkbox_overview.toggled.connect(
            self.on_export_overview_brightfield_checkbox)

        checkbox_surface = dialog.checkbox_export_surface
        checkbox_surface.setChecked(
            self.export_config['surface']['export'])
        checkbox_surface.toggled.connect(
            self.on_export_surface_checkbox)

        dialog.resize(QSize(500, 260))
        return dialog

    def init_export_repetitions(self, group_box, repetition_keys):
        """
        Populates `group_box` with one "all checked" checkbox per key
        in `repetition_keys`, wired to `on_export_repetition_checkbox`,
        or hides it entirely when `repetition_keys` is None (batch
        export - see `_build_export_config_dialog`). Rebuilt from
        scratch on every dialog open, so `export_config['brillouin']
        ['repetitions']` always starts as "every repetition selected"
        for whichever file/keys are current, rather than carrying over
        a stale selection from a previously opened file.
        """
        layout = group_box.layout()
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not repetition_keys:
            self.export_config['brillouin']['repetitions'] = \
                None if repetition_keys is None else []
            group_box.setVisible(False)
            return

        group_box.setVisible(True)
        self.export_config['brillouin']['repetitions'] = list(
            repetition_keys)
        for rep_key in repetition_keys:
            checkbox = QCheckBox()
            checkbox.setText(f'Repetition {rep_key}')
            checkbox.setChecked(True)
            checkbox.toggled.connect(
                lambda checked, rep=rep_key:
                self.on_export_repetition_checkbox(rep, checked)
            )
            layout.addWidget(checkbox)

    def on_export_repetition_checkbox(self, repetition, checked):
        repetitions = self.export_config['brillouin']['repetitions']
        if repetitions is None:
            return
        if checked:
            if repetition not in repetitions:
                repetitions.append(repetition)
        elif repetition in repetitions:
            repetitions.remove(repetition)

    def on_export_overview_brightfield_checkbox(self, checked):
        self.export_config['overviewBrightfield']['export'] = checked

    def on_export_surface_checkbox(self, checked):
        self.export_config['surface']['export'] = checked

    def close_export_dialog(self):
        self.export_dialog.close()

    def export_file(self, blocking=False):
        if self.export_running:
            return
        self.export_running = True
        self.statusbar.showMessage('Exporting...')
        # Close the configuration dialog now - the export itself runs
        # in the background, so it shouldn't keep blocking the app.
        if self.export_dialog is not None:
            self.close_export_dialog()

        self.export_thread = QtCore.QThread()
        self.export_worker = ExportWorker(self.export_config)
        self.export_worker.moveToThread(self.export_thread)
        self.export_thread.started.connect(self.export_worker.run)
        self.export_worker.finished.connect(self.export_thread.quit)
        self.export_worker.finished.connect(self.export_worker.deleteLater)
        self.export_worker.finished.connect(self.on_export_finished)
        self.export_thread.finished.connect(self.export_thread.deleteLater)
        self.export_thread.start()

        if blocking:
            # A real Qt event loop, not a manual processEvents()+sleep()
            # poll - see the matching comment in EvaluationView.evaluate().
            loop = QtCore.QEventLoop()
            self.export_worker.finished.connect(loop.quit)
            loop.exec()

    def on_export_finished(self):
        self.export_running = False
        self.statusbar.showMessage('Export complete', 5000)

    def on_action_batch_export(self):
        folder = QFileDialog.getExistingDirectory(
            self, 'Select a folder to search for evaluated files')
        if not folder:
            return

        files = self._find_evaluated_files(pathlib.Path(folder))
        if not files:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText('No evaluated files found.')
            msg.setInformativeText(
                'Looked for saved evaluations (*.session.h5 files) '
                'under:\n' + folder)
            msg.setWindowTitle('Batch export')
            msg.exec()
            return

        self.batch_export_dialog = self._build_export_config_dialog(
            f'Batch export configuration ({len(files)} files found)',
            lambda: self._run_batch_export(files))
        self.batch_export_dialog.open()

    @staticmethod
    def _find_evaluated_files(root):
        """
        Recursively finds source data files under `root` that have a
        saved evaluation next to them (see bmlab.session.save() /
        get_session_file_path() - depending on the source file's
        folder layout, this is either a sibling "*.session.h5" file
        or a same-named file in a sibling "EvalData" folder).
        """
        files = []
        for candidate in sorted(root.rglob('*.h5')):
            try:
                if not is_source_file(candidate):
                    continue
            except Exception:
                continue
            if get_session_file_path(candidate).exists():
                files.append(candidate)
        return files

    def _run_batch_export(self, files):
        if self.batch_export_dialog is not None:
            self.batch_export_dialog.close()

        self.batch_export_progress_dialog = QtWidgets.QDialog(
            self,
            QtCore.Qt.WindowType.WindowTitleHint |
            QtCore.Qt.WindowType.WindowCloseButtonHint
        )
        self.batch_export_progress_dialog.setWindowTitle('Batch export')
        self.batch_export_progress_dialog.setWindowModality(
            QtCore.Qt.WindowModality.ApplicationModal)
        label = QLabel(f'File 0 / {len(files)}')
        label.setWordWrap(True)
        progress_bar = QtWidgets.QProgressBar()
        progress_bar.setMaximum(len(files))
        progress_bar.setValue(0)
        button_cancel = QtWidgets.QPushButton('Cancel')
        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(progress_bar)
        layout.addWidget(button_cancel)
        self.batch_export_progress_dialog.setLayout(layout)
        self.batch_export_progress_dialog.resize(QSize(500, 120))

        self.batch_export_thread = QtCore.QThread()
        self.batch_export_worker = BatchExportWorker(
            files, self.export_config)
        self.batch_export_worker.moveToThread(self.batch_export_thread)
        self.batch_export_thread.started.connect(
            self.batch_export_worker.run)
        self.batch_export_thread.finished.connect(
            self.batch_export_thread.deleteLater)

        def on_progress(i, path):
            label.setText(f'File {i} / {len(files)}\n{path}')
            progress_bar.setValue(i)

        def on_finished(succeeded, failed):
            aborted = self.batch_export_worker.aborted
            self.batch_export_thread.quit()
            self.batch_export_worker.deleteLater()
            self.batch_export_progress_dialog.close()
            Session.get_instance().clear()
            self.reset_ui()

            summary = QMessageBox()
            summary.setIcon(
                QMessageBox.Icon.Warning if failed
                else QMessageBox.Icon.Information)
            summary.setWindowTitle('Batch export')
            if aborted:
                summary.setText(
                    f'Batch export cancelled after '
                    f'{succeeded}/{len(files)} files.')
            else:
                summary.setText(
                    f'Batch export finished: '
                    f'{succeeded}/{len(files)} succeeded.')
            if failed:
                summary.setInformativeText(
                    'Failed:\n' + '\n'.join(str(f) for f in failed))
            summary.exec()

        self.batch_export_worker.progress.connect(on_progress)
        self.batch_export_worker.finished.connect(on_finished)
        button_cancel.clicked.connect(
            lambda: setattr(self.batch_export_worker, 'aborted', True))

        self.batch_export_progress_dialog.show()
        self.batch_export_thread.start()

    def reset_ui(self):
        """
        Resets the UI if a file is closed.
        """
        self.widget_data_view.reset_ui()
        self.widget_extraction_view.reset_ui()
        self.widget_calibration_view.reset_ui()
        self.widget_peak_selection_view.reset_ui()
        self.widget_evaluation_view.reset_ui()
        self.widget_quality_view.reset_ui()

    def update_ui(self, new_tab_index=-1):
        # If no tab index is specified, we update all tabs
        # (e.g. when a new file is opened).
        # Otherwise, we only update the newly selected tab.
        if new_tab_index == -1:
            self.widget_data_view.update_ui()
            self.widget_extraction_view.update_ui()
            self.widget_calibration_view.update_ui()
            self.widget_peak_selection_view.update_ui()
            self.widget_evaluation_view.update_ui()
            self.widget_quality_view.update_ui()
        elif new_tab_index == 0:
            self.widget_data_view.update_ui()
        elif new_tab_index == 1:
            self.widget_extraction_view.update_ui()
        elif new_tab_index == 2:
            self.widget_calibration_view.update_ui()
        elif new_tab_index == 3:
            self.widget_peak_selection_view.update_ui()
        elif new_tab_index == 4:
            self.widget_evaluation_view.on_tab_activated()
        elif new_tab_index == 5:
            self.widget_quality_view.on_tab_activated()

    @staticmethod
    def drag_enter_event(event):
        """ Handles dragging a file over the GUI """
        if check_event_mime_data(event):
            event.accept()
        else:
            event.ignore()

    def drop_event(self, event):
        """ Handles dropping a file, opens if h5"""
        path = check_event_mime_data(event)
        if path:
            self.open_file(path)

    @staticmethod
    def save_session():
        """ Save the session data to file with ending '.bms' """
        Session.get_instance().save()

    @staticmethod
    def exit_app():
        QtCore.QCoreApplication.quit()

    def on_action_about(self):
        gh = "BrillouinMicroscopy/BMicro"
        rtd = "bmicro.readthedocs.io"
        about_text = f"BMicro {bmicroversion}<br><br>" \
            + "BMicro is a graphical user interface " \
            + "for the analysis of Brillouin microscopy data.<br><br>" \
            + f"Using bmlab {bmlabversion}<br><br>" \
            + "Author: Raimund Schlüßler and others<br>" \
            + "GitHub: " \
            + "<a href='https://github.com/{gh}'>{gh}</a><br>".format(gh=gh) \
            + "Documentation: " \
            + "<a href='https://{rtd}'>{rtd}</a><br>".format(rtd=rtd)
        QtWidgets.QMessageBox.about(self,
                                    f"BMicro {bmicroversion}", about_text)

    def on_action_batch_evaluation(self):
        self.batch_dialog = QtWidgets.QDialog(
            self,
            QtCore.Qt.WindowType.WindowTitleHint |
            QtCore.Qt.WindowType.WindowCloseButtonHint |
            QtCore.Qt.WindowType.WindowMaximizeButtonHint |
            QtCore.Qt.WindowType.WindowMinimizeButtonHint
        )
        ref = resources.files('bmicro.gui') / 'batch_evaluation.ui'
        with resources.as_file(ref) as ui_file:
            uic.loadUi(ui_file, self.batch_dialog)
        self.batch_dialog.setWindowTitle('Batch evaluation')
        self.batch_dialog.setWindowModality(
            QtCore.Qt.WindowModality.ApplicationModal)
        self.batch_dialog.button_start_cancel.clicked.connect(
            self.start_batch_evaluation
        )
        self.batch_dialog.button_add_folder.clicked.connect(
            self.batch_add_files
        )
        self.batch_dialog.button_remove_folder.clicked.connect(
            self.batch_remove_files
        )
        self.update_batch_file_table()
        # Batch evaluation's export step reuses the same export_config
        # as the regular Export/Batch export dialogs (see
        # _build_export_config_dialog()) and behaves the same way now:
        # every evaluated quantity, every repetition, no per-file
        # choice.
        self.export_config['brillouin']['repetitions'] = None

        self._build_batch_quality_rows()
        self.batch_dialog.adjustSize()

        self.update_batch_file_settings()

        self.batch_dialog.exec()

    def close_batch_dialog(self):
        self.batch_dialog.close()

    def _build_batch_quality_rows(self):
        """
        Populates the batch dialog's "Quality" section with one row
        per EvaluationController.QUALITY_METRIC_KEYS - a nicely
        labelled checkbox+spinbox pair for min and max, same as
        QualityView's own metric rows, prefilled from
        self.batch_config['quality'] (which starts at
        EvaluationModel.get_default_quality_thresholds() - see
        __init__). Rebuilt from scratch every time the dialog opens,
        so it always reflects the current batch_config state (which
        persists across dialog opens within this app run).
        """
        parameters = EvaluationModel.get_default_parameters()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._batch_quality_rows = {}
        for key in EvaluationController.QUALITY_METRIC_KEYS:
            threshold = self.batch_config['quality'].get(
                key, {'enabled': True, 'min': None, 'max': None})
            param = parameters.get(key, {})
            unit = f" [{param['unit']}]" \
                if param.get('unit') else ''
            label_text = f"{param.get('label', key)}{unit}"

            row_layout = QHBoxLayout()
            label = QLabel(label_text)
            label.setMinimumWidth(260)
            row_layout.addWidget(label)

            min_check = QCheckBox('min')
            min_box = QDoubleSpinBox()
            min_box.setRange(-1e9, 1e9)
            min_box.setDecimals(4)
            min_box.setMaximumWidth(100)
            has_min = threshold.get('min') is not None
            min_check.setChecked(has_min)
            min_box.setEnabled(has_min)
            if has_min:
                min_box.setValue(threshold['min'])
            row_layout.addWidget(min_check)
            row_layout.addWidget(min_box)

            max_check = QCheckBox('max')
            max_box = QDoubleSpinBox()
            max_box.setRange(-1e9, 1e9)
            max_box.setDecimals(4)
            max_box.setMaximumWidth(100)
            has_max = threshold.get('max') is not None
            max_check.setChecked(has_max)
            max_box.setEnabled(has_max)
            if has_max:
                max_box.setValue(threshold['max'])
            row_layout.addWidget(max_check)
            row_layout.addWidget(max_box)
            row_layout.addStretch()

            row_widget = QWidget()
            row_widget.setLayout(row_layout)
            layout.addWidget(row_widget)

            min_check.toggled.connect(min_box.setEnabled)
            max_check.toggled.connect(max_box.setEnabled)
            min_check.toggled.connect(
                lambda checked, k=key: self.on_batch_quality_changed(k))
            max_check.toggled.connect(
                lambda checked, k=key: self.on_batch_quality_changed(k))
            min_box.valueChanged.connect(
                lambda value, k=key: self.on_batch_quality_changed(k))
            max_box.valueChanged.connect(
                lambda value, k=key: self.on_batch_quality_changed(k))

            self._batch_quality_rows[key] = {
                'min_check': min_check, 'min_box': min_box,
                'max_check': max_check, 'max_box': max_box,
            }

        # setLayout() only works once per widget - the dialog is
        # rebuilt from the .ui file (a fresh widget_quality_rows) each
        # time on_action_batch_evaluation() runs, so this is safe.
        self.batch_dialog.widget_quality_rows.setLayout(layout)

    def on_batch_quality_changed(self, key):
        row = self._batch_quality_rows[key]
        min_value = row['min_box'].value() \
            if row['min_check'].isChecked() else None
        max_value = row['max_box'].value() \
            if row['max_check'].isChecked() else None
        self.batch_config['quality'][key] = {
            'enabled': True, 'min': min_value, 'max': max_value,
        }

    def update_batch_file_settings(self):
        # Setup
        cfg_setup = self.batch_config['setup']
        self.batch_dialog.checkBox_setup.setChecked(cfg_setup['set'])
        self.batch_dialog.checkBox_setup.clicked.connect(self.on_setup_set)
        self.batch_dialog.combobox_setup.addItems(
            [s.name for s in AVAILABLE_SETUPS])
        idx = AVAILABLE_SETUPS.index(cfg_setup['setup'])
        if idx:
            self.batch_dialog.combobox_setup.setCurrentIndex(idx)
        self.batch_dialog.combobox_setup.setEnabled(cfg_setup['set'])
        self.batch_dialog.combobox_setup.currentIndexChanged.connect(
            self.on_setup_select)

        # Orientation
        cfg_orientation = self.batch_config['orientation']
        self.batch_dialog.checkBox_orientation\
            .setChecked(cfg_orientation['set'])
        self.batch_dialog.checkBox_orientation\
            .clicked.connect(self.on_orientation_set)
        self.batch_dialog.groupBox_rotation\
            .setEnabled(cfg_orientation['set'])
        self.batch_dialog.groupBox_reflection\
            .setEnabled(cfg_orientation['set'])

        self.batch_dialog.checkbox_reflect_vertically.setChecked(
            cfg_orientation['reflection']['vertically'])
        self.batch_dialog.checkbox_reflect_horizontally.setChecked(
            cfg_orientation['reflection']['horizontally'])

        self.batch_dialog.radio_rotation_none.setChecked(
            cfg_orientation['rotation'] % 4 == 0)
        self.batch_dialog.radio_rotation_90_cw.setChecked(
            cfg_orientation['rotation'] % 4 == 1)
        self.batch_dialog.radio_rotation_90_ccw.setChecked(
            cfg_orientation['rotation'] % 4 == 3)

        self.batch_dialog.radio_rotation_none\
            .clicked.connect(self.on_rotation_clicked)
        self.batch_dialog.radio_rotation_90_cw\
            .clicked.connect(self.on_rotation_clicked)
        self.batch_dialog.radio_rotation_90_ccw\
            .clicked.connect(self.on_rotation_clicked)

        self.batch_dialog.checkbox_reflect_vertically.toggled.connect(
            self.on_reflection_clicked)
        self.batch_dialog.checkbox_reflect_horizontally.toggled.connect(
            self.on_reflection_clicked)

        # Extraction
        cfg_extraction = self.batch_config['extraction']
        self.batch_dialog.checkBox_extraction\
            .setChecked(cfg_extraction['extract'])
        self.batch_dialog.checkBox_extraction\
            .clicked.connect(self.on_extraction_set)

        # Calibration
        cfg_calibration = self.batch_config['calibration']
        self.batch_dialog.checkBox_find_peaks\
            .setChecked(cfg_calibration['find-peaks'])
        self.batch_dialog.checkBox_find_peaks\
            .clicked.connect(self.on_calibration_find_peaks)
        self.batch_dialog.checkBox_calibrate\
            .setChecked(cfg_calibration['calibrate'])
        self.batch_dialog.checkBox_calibrate\
            .clicked.connect(self.on_calibration_calibrate)

        self.batch_dialog.temperature.valueChanged.connect(
            self.temperature_changed
        )
        self.batch_dialog.shift_0.valueChanged.connect(
            self.shift_methanol_changed
        )
        self.batch_dialog.shift_1.valueChanged.connect(
            self.shift_water_changed
        )

        self.update_batch_calibration_frequencies()

        # Peak selection
        cfg_peak_selection = self.batch_config['peak-selection']
        self.batch_dialog.checkBox_peak_selection\
            .setChecked(cfg_peak_selection['select'])
        self.batch_dialog.checkBox_peak_selection\
            .clicked.connect(self.on_peak_selection_select)
        self.batch_dialog.tableWidget_Brillouin\
            .setEnabled(cfg_peak_selection['select'])
        self.batch_dialog.tableWidget_Rayleigh\
            .setEnabled(cfg_peak_selection['select'])
        self.batch_dialog.button_region_brillouin_remove.clicked.connect(
            lambda: self.remove_evaluation_regions(
                self.batch_dialog.tableWidget_Brillouin,
                self.batch_config[
                    'peak-selection']['brillouin_regions']
            )
        )
        self.batch_dialog.button_region_brillouin_add.clicked.connect(
            lambda: self.add_evaluation_regions(
                self.batch_dialog.tableWidget_Brillouin,
                self.batch_config[
                    'peak-selection']['brillouin_regions']
            )
        )
        self.batch_dialog.button_region_rayleigh_remove.clicked.connect(
            lambda: self.remove_evaluation_regions(
                self.batch_dialog.tableWidget_Rayleigh,
                self.batch_config[
                    'peak-selection']['rayleigh_regions']
            )
        )
        self.batch_dialog.button_region_rayleigh_add.clicked.connect(
            lambda: self.add_evaluation_regions(
                self.batch_dialog.tableWidget_Rayleigh,
                self.batch_config[
                    'peak-selection']['rayleigh_regions']
            )
        )

        self.update_evaluation_regions_tables()

        self.batch_dialog.tableWidget_Brillouin.itemChanged.connect(
            lambda item: self.on_region_changed(
                self.batch_config['peak-selection']['brillouin_regions'],
                item)
        )

        self.batch_dialog.tableWidget_Rayleigh.itemChanged.connect(
            lambda item: self.on_region_changed(
                self.batch_config['peak-selection']['rayleigh_regions'],
                item)
        )

        header = self.batch_dialog.tableWidget_Brillouin.horizontalHeader()
        header.setSectionResizeMode(0,
                                    QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1,
                                    QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.batch_dialog.tableWidget_Brillouin\
            .verticalHeader().setVisible(False)
        self.batch_dialog.tableWidget_Brillouin \
            .setHorizontalHeaderLabels(["start", "end"])

        header = self.batch_dialog.tableWidget_Rayleigh.horizontalHeader()
        header.setSectionResizeMode(0,
                                    QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1,
                                    QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.batch_dialog.tableWidget_Rayleigh\
            .verticalHeader().setVisible(False)
        self.batch_dialog.tableWidget_Rayleigh\
            .setHorizontalHeaderLabels(["start", "end"])

        # Evaluation
        cfg_evaluation = self.batch_config['evaluation']
        self.batch_dialog.checkBox_evaluation\
            .setChecked(cfg_evaluation['evaluate'])
        self.batch_dialog.checkBox_evaluation\
            .clicked.connect(self.on_evaluation_evaluate)
        self.batch_dialog.nrBrillouinPeaksGroup\
            .setEnabled(cfg_evaluation['evaluate'])
        self.batch_dialog.group_5b_quality\
            .setEnabled(cfg_evaluation['evaluate'])
        if cfg_evaluation['nr_brillouin_peaks'] == 1:
            self.batch_dialog.nrBrillouinPeaks_1.setChecked(True)
        elif cfg_evaluation['nr_brillouin_peaks'] == 2:
            self.batch_dialog.nrBrillouinPeaks_2.setChecked(True)
        elif cfg_evaluation['nr_brillouin_peaks'] == 4:
            self.batch_dialog.nrBrillouinPeaks_4.setChecked(True)
        self.batch_dialog.bounds_table\
            .setEnabled(cfg_evaluation['evaluate'] and
                        cfg_evaluation['nr_brillouin_peaks'] > 1)

        self.batch_dialog.bounds_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.batch_dialog.bounds_table.verticalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch)

        self.update_batch_bounds_table()
        self.batch_dialog.bounds_table.cellChanged.connect(
            self.evaluation_bounds_changed)

        self.batch_dialog.nrBrillouinPeaks_1.toggled.connect(
            lambda: self.set_nr_brillouin_peaks(1))
        self.batch_dialog.nrBrillouinPeaks_2.toggled.connect(
            lambda: self.set_nr_brillouin_peaks(2))
        self.batch_dialog.nrBrillouinPeaks_4.toggled.connect(
            lambda: self.set_nr_brillouin_peaks(4))

        # Export
        cfg_export = self.batch_config['export']
        self.batch_dialog.checkBox_export\
            .setChecked(cfg_export['export'])
        self.batch_dialog.checkBox_export\
            .clicked.connect(self.on_export_export)

    def temperature_changed(self):
        temperature = self.sender().value()

        self.batch_config['setup']['setup'].set_temperature(temperature)
        self.update_batch_calibration_frequencies()

    def shift_methanol_changed(self):
        shift = self.sender().value()

        self.batch_config['setup']['setup']\
            .calibration.set_shift_methanol(1e9 * shift)

    def shift_water_changed(self):
        shift = self.sender().value()

        self.batch_config['setup']['setup']\
            .calibration.set_shift_water(1e9 * shift)

    def update_batch_calibration_frequencies(self):
        cfg_setup = self.batch_config['setup']
        if not cfg_setup['set']:
            self.batch_dialog.shift_0.setEnabled(False)
            self.batch_dialog.label_shift_0.setEnabled(False)
            self.batch_dialog.shift_1.setEnabled(False)
            self.batch_dialog.label_shift_1.setEnabled(False)
            self.batch_dialog.temperature.setEnabled(False)
            self.batch_dialog.label_temperature.setEnabled(False)
        else:
            # Set current calibration values
            if cfg_setup['setup'].calibration.shift_methanol is not None:
                self.batch_dialog.label_shift_0.setEnabled(True)
                self.batch_dialog.shift_0.setEnabled(True)
                self.batch_dialog.shift_0.setValue(
                    1e-9 * cfg_setup['setup'].calibration.shift_methanol)
            else:
                self.batch_dialog.shift_0.setEnabled(False)
                self.batch_dialog.label_shift_0.setEnabled(False)

            if cfg_setup['setup'].calibration.shift_water is not None:
                self.batch_dialog.label_shift_1.setEnabled(True)
                self.batch_dialog.shift_1.setEnabled(True)
                self.batch_dialog.shift_1.setValue(
                    1e-9 * cfg_setup['setup'].calibration.shift_water)
            else:
                self.batch_dialog.shift_1.setEnabled(False)
                self.batch_dialog.label_shift_1.setEnabled(False)

            self.batch_dialog.temperature.setEnabled(True)
            self.batch_dialog.label_temperature.setEnabled(True)
            self.batch_dialog.temperature.setValue(
                cfg_setup['setup'].temperature - 273.15
            )

    def update_batch_bounds_table(self):
        bounds_w0 = self.batch_config['evaluation']['bounds_w0']
        bounds_fwhm = self.batch_config['evaluation']['bounds_fwhm']
        if bounds_w0 is None or bounds_fwhm is None:
            self.batch_dialog.bounds_table.setRowCount(0)
        else:
            self.batch_dialog.bounds_table.setColumnCount(4)
            self.batch_dialog.bounds_table.setRowCount(len(bounds_w0))
            for i, bound in enumerate(bounds_w0):
                item = QtWidgets.QTableWidgetItem(str(bound[0]))
                self.batch_dialog.bounds_table.setItem(i, 0, item)
                item = QtWidgets.QTableWidgetItem(str(bound[1]))
                self.batch_dialog.bounds_table.setItem(i, 1, item)
            for i, bound in enumerate(bounds_fwhm):
                item = QtWidgets.QTableWidgetItem(str(bound[0]))
                self.batch_dialog.bounds_table.setItem(i, 2, item)
                item = QtWidgets.QTableWidgetItem(str(bound[1]))
                self.batch_dialog.bounds_table.setItem(i, 3, item)

        self.batch_dialog.bounds_table.setEnabled(
            self.batch_config['evaluation']['evaluate'] and
            self.batch_config['evaluation']['nr_brillouin_peaks'] > 1)

    def set_nr_brillouin_peaks(self, nr_peaks):
        if not self.sender().isChecked():
            return
        self.batch_config['evaluation']['nr_brillouin_peaks'] = nr_peaks

        if nr_peaks == 1:
            bounds_w0 = None
            bounds_fwhm = None

        # Initialize the bounds if necessary
        if nr_peaks > 1 and\
                (self.batch_config['evaluation']['bounds_w0'] is None or
                 len(self.batch_config['evaluation']['bounds_w0'])
                 is not nr_peaks):
            bounds_w0 = [['min', 'max'] for _ in range(nr_peaks)]
        if nr_peaks > 1 and\
                (self.batch_config['evaluation']['bounds_fwhm'] is None or
                 len(self.batch_config['evaluation']['bounds_fwhm'])
                 is not nr_peaks):
            bounds_fwhm = [['0', 'inf'] for _ in range(nr_peaks)]

        self.batch_config['evaluation']['bounds_w0'] = bounds_w0
        self.batch_config['evaluation']['bounds_fwhm'] = bounds_fwhm

        self.update_batch_bounds_table()

    def evaluation_bounds_changed(self, row, column):
        if column < 2:
            self.batch_config['evaluation']['bounds_w0'][row][column] =\
                    self.batch_dialog.bounds_table.item(row, column).text()
        elif column < 4:
            self.batch_config['evaluation']['bounds_fwhm'][row][column - 2] =\
                    self.batch_dialog.bounds_table.item(row, column).text()

    def add_evaluation_regions(self, region_table, region_list):
        region_list.append((0., 0.))

        self.update_evaluation_regions_tables()

        region_table.setFocus()
        region_table.setCurrentCell(len(region_list) - 1, 0)

    def remove_evaluation_regions(self, region_table, region_list):
        selected_ranges = region_table.selectedRanges()
        sorted_ranges = sorted(
            selected_ranges,
            key=lambda item: item.topRow(),
            reverse=True
        )
        for sorted_range in sorted_ranges:
            start = sorted_range.topRow()
            end = sorted_range.bottomRow()
            for idx in reversed(range(start, end + 1)):
                del region_list[idx]
        region_table.clearSelection()

        self.update_evaluation_regions_tables()

    def update_evaluation_regions_tables(self):
        cfg_peak_selection = self.batch_config['peak-selection']

        brillouin_regions = cfg_peak_selection['brillouin_regions']
        brillouin_table = self.batch_dialog.tableWidget_Brillouin
        brillouin_table.setRowCount(len(brillouin_regions))
        for rowIdx, region in enumerate(brillouin_regions):
            # Add regions to table
            # Block signals, so the itemChanged signal is not
            # emitted during table creation
            brillouin_table.blockSignals(True)
            for columnIdx, value in enumerate(region):
                region_item = QtWidgets.QTableWidgetItem(str(1e-9 * value))
                brillouin_table.setItem(rowIdx, columnIdx, region_item)
            brillouin_table.blockSignals(False)

        rayleigh_regions = cfg_peak_selection['rayleigh_regions']
        rayleigh_table = self.batch_dialog.tableWidget_Rayleigh
        rayleigh_table.setRowCount(len(rayleigh_regions))
        for rowIdx, region in enumerate(rayleigh_regions):
            # Add regions to table
            # Block signals, so the itemChanged signal is not
            # emitted during table creation
            rayleigh_table.blockSignals(True)
            for columnIdx, value in enumerate(region):
                region_item = QtWidgets.QTableWidgetItem(str(1e-9 * value))
                rayleigh_table.setItem(rowIdx, columnIdx, region_item)
            rayleigh_table.blockSignals(False)

    def on_setup_set(self):
        self.batch_config['setup']['set'] = self.sender().isChecked()
        self.batch_dialog.combobox_setup\
            .setEnabled(self.batch_config['setup']['set'])

        self.update_batch_calibration_frequencies()

    def on_orientation_set(self):
        self.batch_config['orientation']['set'] = self.sender().isChecked()
        self.batch_dialog.groupBox_rotation\
            .setEnabled(self.batch_config['orientation']['set'])
        self.batch_dialog.groupBox_reflection\
            .setEnabled(self.batch_config['orientation']['set'])

    def on_rotation_clicked(self):
        """
        Action triggered when user clicks one of the rotation radio buttons.
        """

        radio_button = self.sender()
        if not radio_button.isChecked():
            return

        if radio_button == self.batch_dialog.radio_rotation_none:
            self.batch_config['orientation']['rotation'] = 0
        elif radio_button == self.batch_dialog.radio_rotation_90_cw:
            self.batch_config['orientation']['rotation'] = 1
        elif radio_button == self.batch_dialog.radio_rotation_90_ccw:
            self.batch_config['orientation']['rotation'] = 3

    def on_reflection_clicked(self):
        """ Triggered when a reflection checkbox is clicked """

        checkbox = self.sender()

        if checkbox == self.batch_dialog.checkbox_reflect_vertically:
            self.batch_config['orientation'][
                'reflection']['vertically'] = checkbox.isChecked()
        elif checkbox == self.batch_dialog.checkbox_reflect_horizontally:
            self.batch_config['orientation'][
                'reflection']['horizontally'] = checkbox.isChecked()

    def on_extraction_set(self):
        self.batch_config['extraction']['extract'] = self.sender().isChecked()

    def on_calibration_find_peaks(self):
        self.batch_config['calibration']['find-peaks']\
            = self.sender().isChecked()

    def on_calibration_calibrate(self):
        self.batch_config['calibration']['calibrate']\
            = self.sender().isChecked()

    def on_peak_selection_select(self):
        self.batch_config['peak-selection']['select']\
            = self.sender().isChecked()
        self.batch_dialog.tableWidget_Brillouin\
            .setEnabled(self.batch_config['peak-selection']['select'])
        self.batch_dialog.tableWidget_Rayleigh\
            .setEnabled(self.batch_config['peak-selection']['select'])

    @staticmethod
    def on_region_changed(region_table, item):
        row = item.row()
        column = item.column()
        value = float(item.text())

        current_region = np.asarray(region_table[row])
        current_region[column] = 1e9 * value
        current_region = tuple(current_region)
        region_table[row] = current_region

    def on_evaluation_evaluate(self):
        self.batch_config['evaluation']['evaluate']\
            = self.sender().isChecked()
        self.batch_dialog.nrBrillouinPeaksGroup\
            .setEnabled(self.batch_config['evaluation']['evaluate'])
        self.batch_dialog.group_5b_quality\
            .setEnabled(self.batch_config['evaluation']['evaluate'])
        self.batch_dialog.bounds_table\
            .setEnabled(
                self.batch_config['evaluation']['evaluate'] and
                self.batch_config['evaluation']['nr_brillouin_peaks'] > 1)

    def on_export_export(self):
        self.batch_config['export']['export'] = self.sender().isChecked()

    def on_setup_select(self):
        """
        Action triggered when the user selects a different setup.
        """
        name = self.batch_dialog.combobox_setup.currentText()
        setup = None
        for s in AVAILABLE_SETUPS:
            if s.name == name:
                setup = s
                break
        self.batch_config['setup']['setup'] = setup

        self.update_batch_calibration_frequencies()

    def start_batch_evaluation(self):
        self.batch_evaluation_running = not self.batch_evaluation_running
        if self.batch_evaluation_running:
            self.run_batch_evaluation()
        else:
            self.widget_evaluation_view.evaluation_abort.value = True

    def run_batch_evaluation(self):
        self.batch_dialog.button_start_cancel.setText('Cancel')
        self.batch_dialog.progressBar.setMaximum(len(self.batch_files))
        self.batch_dialog.progressBar.setValue(0)
        # Collects every problem hit along the way - both a whole file
        # failing (an exception escaping evaluate_batch_file()) and a
        # single repetition being skipped within an otherwise-fine
        # file (see the calibration/peak-selection checks in
        # evaluate_batch_file()) - so they can be shown to the user at
        # the end instead of only ever showing up as an unexplained
        # "failed" icon in the file table.
        self.batch_evaluation_messages = []
        for i, (file_hash, file) in enumerate(self.batch_files.items()):
            try:
                self.evaluate_batch_file(file)
            except BaseException as e:
                # Set the status as failed
                file['status'] = 'failed'
                self.update_batch_file_table()
                self.batch_evaluation_messages.append(
                    f"{file['path']}: {e}")
                logger.warning(
                    'Batch evaluation failed for %s', file['path'],
                    exc_info=True)
            if not self.batch_evaluation_running:
                break
            self.batch_dialog.progressBar.setValue(i + 1)
        self.batch_dialog.button_start_cancel.setText('Start')
        self.batch_evaluation_running = False
        self.update_batch_file_table()

        if self.batch_evaluation_messages:
            summary = QMessageBox()
            summary.setIcon(QMessageBox.Icon.Warning)
            summary.setWindowTitle('Batch evaluation')
            summary.setText(
                f'{len(self.batch_evaluation_messages)} issue(s) during '
                f'batch evaluation:')
            summary.setInformativeText(
                '\n\n'.join(self.batch_evaluation_messages))
            summary.exec()

    def evaluate_batch_file(self, file):
        # Set the status as in process
        file['status'] = 'in-process'
        self.update_batch_file_table()

        # Show the data tab and open the file
        self.tabWidget.setCurrentIndex(0)
        QtCore.QCoreApplication.instance().processEvents()
        self.open_file(file['path'])
        QtCore.QCoreApplication.instance().processEvents()

        """
        Evaluate the file
        """
        session = Session.get_instance()

        rep_keys = session.file.repetition_keys()

        for rep_key in rep_keys:
            # Load repetition
            session.set_current_repetition(rep_key)
            QtCore.QCoreApplication.instance().processEvents()

            # Setup
            cfg_setup = self.batch_config['setup']
            if cfg_setup['set']:
                session.set_setup(cfg_setup['setup'])

            # Orientation
            cfg_orientation = self.batch_config['orientation']
            if cfg_orientation['set']:
                session.set_rotation(cfg_orientation['rotation'])
                session.set_reflection(
                    vertically=cfg_orientation['reflection']['vertically'],
                    horizontally=cfg_orientation['reflection']['horizontally']
                )

            # Extraction
            cfg_extraction = self.batch_config['extraction']
            if cfg_extraction['extract']:
                self.tabWidget.setCurrentIndex(1)
                if self.aborted(file):
                    return
                QtCore.QCoreApplication.instance().processEvents()
                self.widget_extraction_view.find_points_all()

            # Calibration
            cfg_calibration = self.batch_config['calibration']
            if cfg_calibration['find-peaks'] or\
                    cfg_calibration['calibrate']:
                self.tabWidget.setCurrentIndex(2)
                if self.aborted(file):
                    return
                QtCore.QCoreApplication.instance().processEvents()
                if not cfg_calibration['find-peaks']:
                    self.widget_calibration_view.\
                        calibrate_all(do_not='find_peaks')
                elif not cfg_calibration['calibrate']:
                    self.widget_calibration_view.\
                        calibrate_all(do_not='calibrate')
                else:
                    self.widget_calibration_view.calibrate_all()

            # PeakSelection
            cfg_peak_selection = self.batch_config['peak-selection']
            if cfg_peak_selection['select']:
                self.tabWidget.setCurrentIndex(3)
                if self.aborted(file):
                    return
                QtCore.QCoreApplication.instance().processEvents()
                psc = PeakSelectionController()
                for brillouin_region \
                        in cfg_peak_selection['brillouin_regions']:
                    psc.add_brillouin_region_frequency(brillouin_region)
                for rayleigh_region \
                        in cfg_peak_selection['rayleigh_regions']:
                    psc.add_rayleigh_region_frequency(rayleigh_region)
                self.widget_peak_selection_view.update_ui()
                QtCore.QCoreApplication.instance().processEvents()

            # Evaluation
            cfg_evaluation = self.batch_config['evaluation']
            if cfg_evaluation['evaluate']:
                self.tabWidget.setCurrentIndex(4)
                if self.aborted(file):
                    return
                QtCore.QCoreApplication.instance().processEvents()

                # A repetition with no valid measurement grid at all
                # (an aborted/restarted acquisition that never wrote
                # any positions) has nothing to evaluate - skip it
                # quietly, same as the rest of the pipeline already
                # does for it elsewhere (e.g. BrillouinExport.export()).
                if session.get_payload_resolution() is None:
                    continue

                # If "Select peaks"/"Calibrate" aren't ticked for this
                # batch run, evaluation relies entirely on whatever was
                # already saved for this file (see set_file()/load() -
                # loaded before this loop, in open_file() above). A
                # repetition with neither would otherwise fit silently
                # against undefined regions/calibration - skip just
                # this repetition instead (a multi-repetition file can
                # easily have one aborted/never-finished repetition
                # alongside perfectly good ones, so one bad repetition
                # must not stop the rest of the file from being
                # evaluated/exported), and record why so it shows up
                # in the summary run_batch_evaluation() shows at the
                # end rather than silently producing an empty result.
                pm = session.peak_selection_model()
                cm = session.calibration_model()
                missing = []
                if not pm.get_brillouin_regions() \
                        or not pm.get_rayleigh_regions():
                    missing.append(
                        "no Brillouin/Rayleigh peak regions selected "
                        "('Select peaks' isn't ticked to define them "
                        "for this batch run)")
                if cm.frequencies_by_time_interpolator is None:
                    missing.append(
                        "no valid calibration ('Calibrate' isn't "
                        "ticked to create one for this batch run)")
                if missing:
                    self.batch_evaluation_messages.append(
                        f"{file['path']} repetition {rep_key}: "
                        f"skipped evaluation - {'; '.join(missing)}.")
                    continue

                evc = EvaluationController()
                evc.set_nr_brillouin_peaks(
                    cfg_evaluation['nr_brillouin_peaks'])
                evc.set_bounds(cfg_evaluation['bounds_w0'])
                evc.set_bounds_fwhm(cfg_evaluation['bounds_fwhm'])
                self.widget_evaluation_view.evaluate(blocking=True)
                # The "Quality" section's thresholds (see
                # _build_batch_quality_rows()) apply to every file in
                # the batch, replacing whatever that file had saved -
                # then write evm.results['quality_pass'] from them,
                # same as clicking "Apply to results" on the Quality
                # tab, so it's exported/saved without a manual step
                # per file.
                session.evaluation_model().quality_thresholds = \
                    copy.deepcopy(self.batch_config['quality'])
                evc.apply_quality_thresholds()

        # Export - once per file, after every repetition has been
        # evaluated, not once per repetition inside the loop above:
        # BrillouinExport.export() already exports every repetition of
        # the file in one pass (export_config['brillouin']
        # ['repetitions'] is None - see on_action_batch_evaluation()),
        # so exporting per-repetition was both redundant and, worse,
        # skipped entirely whenever a repetition hit one of the
        # `continue`s above (a repetition skipped for missing
        # calibration/peak-selection would take that iteration's
        # export with it, even for a file where a LATER repetition
        # still needed exporting). This also calls ExportController
        # directly and synchronously - the same way the standalone
        # Batch Export feature's BatchExportWorker.run() does it -
        # instead of going through the interactive export_file()'s own
        # QThread, which exists for keeping the single-file dialog
        # responsive and isn't needed here.
        cfg_export = self.batch_config['export']
        if cfg_export['export']:
            if self.aborted(file):
                return
            QtCore.QCoreApplication.instance().processEvents()
            try:
                ExportController().export(self.export_config)
            except Exception as e:
                # Don't let an export problem lose the evaluation work
                # already done above - still fall through to save it.
                self.batch_evaluation_messages.append(
                    f"{file['path']}: export failed - {e}")
                logger.warning(
                    'Batch export failed for %s', file['path'],
                    exc_info=True)

        # Save the evaluated data
        self.save_session()
        if self.aborted(file):
            return
        QtCore.QCoreApplication.instance().processEvents()

        # Close the file
        self.close_file()
        QtCore.QCoreApplication.instance().processEvents()

        # Set the status as done
        file['status'] = 'success'
        self.update_batch_file_table()

    def aborted(self, file):
        if not self.batch_evaluation_running:
            file['status'] = 'aborted'
            self.close_file()
            self.batch_evaluation_running = False
            self.update_batch_file_table()
            self.batch_dialog.progressBar.setValue(0)
            QtCore.QCoreApplication.instance().processEvents()
            return True

    def batch_add_files(self):
        folder_name = QFileDialog.getExistingDirectory(
            self, 'Select folder...',
            directory=self.settings.value("path/last-used")
        )

        if not folder_name:
            return

        # Find all h5 files in the selected folder
        h5_files = pathlib.Path(folder_name).glob('**/*.h5')

        # Add source files to batch if not present yet
        for h5_file in h5_files:
            h5_file_md5 = hashlib.md5(str(h5_file).encode('utf-8')).hexdigest()
            if is_source_file(h5_file)\
                    and h5_file_md5 not in self.batch_files:
                self.batch_files[h5_file_md5] = {
                    'path': h5_file,
                    'status': 'pending',
                }

        self.update_batch_file_table()

    def batch_remove_files(self):
        table = self.batch_dialog.table_files
        selected_ranges = table.selectedRanges()
        hashes = list(self.batch_files.keys())
        hashes_remove = []
        for selected_range in selected_ranges:
            start = selected_range.topRow()
            end = selected_range.bottomRow()
            for i in range(start, end + 1):
                hashes_remove.append(hashes[i])
        for hash_remove in hashes_remove:
            self.batch_files.pop(hash_remove)
        table.clearSelection()
        self.update_batch_file_table()

    def update_batch_file_table(self):
        table = self.batch_dialog.table_files
        table.setTextElideMode(QtCore.Qt.TextElideMode.ElideRight)
        table.setWordWrap(False)
        table.setColumnCount(2)
        table.setRowCount(len(self.batch_files))
        table.blockSignals(True)

        pending_icon_path = str(pathlib.Path(self.imdir) / "pending.svg")
        inprocess_icon_path = str(pathlib.Path(self.imdir) / "in-process.svg")
        success_icon_path =\
            str(pathlib.Path(self.imdir) / "check_circle_outline.svg")
        failed_icon_path = str(pathlib.Path(self.imdir) / "error.svg")
        aborted_icon_path = str(pathlib.Path(self.imdir) / "cancel.svg")

        for rowIdx, (file_hash, file) in enumerate(self.batch_files.items()):
            table.setIconSize(QtCore.QSize(20, 20))
            status = file['status']
            if status == 'in-process':
                icon_path = inprocess_icon_path
            elif status == 'success':
                icon_path = success_icon_path
            elif status == 'aborted':
                icon_path = aborted_icon_path
            elif status == 'failed':
                icon_path = failed_icon_path
            else:
                icon_path = pending_icon_path
            icon = QtGui.QIcon(icon_path)
            entry = QtWidgets.QTableWidgetItem()
            entry.setSizeHint(QtCore.QSize(20, 20))
            entry.setIcon(icon)
            table.setItem(rowIdx, 0, entry)

            path = QtWidgets.QTableWidgetItem(str(file['path']))
            table.setItem(rowIdx, 1, path)
        table.setColumnWidth(0, 18)
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setVisible(False)
        table.blockSignals(False)


def excepthook(etype, value, trace):
    """
    Handler for all unhandled exceptions.
    Parameters
    ----------
    etype: Exception
        the exception type (`SyntaxError`,
        `ZeroDivisionError`, etc...)
    value: str
        the exception error message
    trace: str
        the traceback header, if any (otherwise, it
        prints the standard Python header: ``Traceback (most recent
        call last)``.
    """
    vinfo = "Unhandled exception in BMicro version {}:\n".format(
        bmicroversion)
    tmp = traceback.format_exception(etype, value, trace)
    exception = "".join([vinfo]+tmp)

    errorbox = QtWidgets.QMessageBox()
    errorbox.addButton(QtWidgets.QPushButton('Close'),
                       QtWidgets.QMessageBox.ButtonRole.YesRole)
    errorbox.addButton(QtWidgets.QPushButton(
        'Copy text && Close'), QtWidgets.QMessageBox.ButtonRole.NoRole)
    errorbox.setText(exception)
    ret = errorbox.exec()
    if ret == 1:
        cb = QtWidgets.QApplication.clipboard()
        cb.clear(mode=cb.Mode.Clipboard)
        cb.setText(exception)


# Make Ctr+C close the app
signal.signal(signal.SIGINT, signal.SIG_DFL)
# Display exception hook in separate dialog instead of crashing
sys.excepthook = excepthook
