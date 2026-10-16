import pathlib
from importlib import resources
import copy
import hashlib
import logging
import os
import signal
import sys
import traceback

import h5py
import numpy as np

from PyQt6 import QtWidgets, uic, QtCore, QtGui, sip
from PyQt6.QtWidgets import QFileDialog, QMessageBox, \
    QVBoxLayout, QHBoxLayout, QCheckBox, QLabel, QDoubleSpinBox, QWidget
from PyQt6.QtCore import QSize

from bmlab.session import Session, get_session_file_path
from bmlab.serializer import Serializer
from bmlab.file import is_source_file, is_session_file, BrillouinFile
from bmlab.models.setup import AVAILABLE_SETUPS
from bmlab.models import EvaluationModel
from bmlab.controllers import ExportController, EvaluationController

from . import data
from . import extraction
from . import calibration
from . import peak_selection
from . import evaluation
from . import quality
from . import background
from bmicro.prefetch import FilePrefetcher
from .plot_export import export_calibration_plots, evaluated_repetitions
from .base_settings import setup_signature, orientation_signature, \
    saved_base_signature, reset_analysis
from . import batch_cache
from .batch_step_dialog import BatchStepDialog, pending_repetitions, \
    step_done, DOWNSTREAM_STEPS, calibration_gaps, calibration_gaps_text, \
    STEP_EXTRACTION, STEP_CALIBRATION, STEP_PEAK_SELECTION

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


def file_hash(path):
    """ Key of a file in the batch table. """
    return hashlib.md5(str(path).encode('utf-8')).hexdigest()


def is_analysed(repetition):
    """
    Whether a repetition of the batch table has any analysis step done
    (the batch table's 'progress', see BMicro._refresh_batch_progress()).
    """
    return repetition['valid'] \
        and repetition['progress'] not in (None, 'none')


class BatchScanWorker(QtCore.QObject):
    """
    Finds the source files below a folder and reads what the batch table
    shows for each (repetitions, analysis progress) in a background
    thread. That takes seconds per file, so doing it in the GUI thread
    froze the window for the whole folder.

    `scan` is the function that reads one file and returns its
    repetitions (BMicro._scan_batch_repetitions); it must not touch the
    GUI or the Session.
    """
    # number of new source files found
    found = QtCore.pyqtSignal(int)
    # index (0-based) of the file about to be scanned, and its path
    progress = QtCore.pyqtSignal(int, str)
    # path and repetitions of a scanned file
    scanned = QtCore.pyqtSignal(object, dict)
    finished = QtCore.pyqtSignal()

    def __init__(self, folder, known_hashes, scan):
        super().__init__()
        self.folder = folder
        self.known_hashes = set(known_hashes)
        self.scan = scan
        self.aborted = False

    def run(self):
        try:
            files = []
            for h5_file in sorted(pathlib.Path(self.folder).glob('**/*.h5')):
                if self.aborted:
                    return
                if file_hash(h5_file) not in self.known_hashes \
                        and is_source_file(h5_file):
                    files.append(h5_file)
            self.found.emit(len(files))
            for i, h5_file in enumerate(files):
                if self.aborted:
                    return
                self.progress.emit(i, str(h5_file))
                self.scanned.emit(h5_file, self.scan(h5_file))
        except Exception:
            logger.warning('Scanning %s failed', self.folder, exc_info=True)
        finally:
            self.finished.emit()


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
        self.batch_scan_thread = None
        self.batch_scan_worker = None
        self.batch_files = {}
        self.batch_config = {
            'setup': {
                'set': False,
                'setup': AVAILABLE_SETUPS[0],
            },
            'orientation': {
                'set': False,
                'rotation': 0,
                'reflection': {'vertically': False, 'horizontally': False},
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
        self.batch_reanalyse = False
        # (file path, step) -> repetitions whose upstream step was redone
        # during the current batch run, so this step must be redone too.
        self.batch_forced = {}
        # (file hash, repetition key or None) per row of the batch table
        self.batch_rows = []
        # Loads the next batch files in the background (batch runs only)
        self.prefetcher = None
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
        self.widget_background_view = background.BackgroundView(self)
        self.layout_background = QtWidgets.QVBoxLayout()
        self.tab_background.setLayout(self.layout_background)
        self.layout_background.addWidget(self.widget_background_view)

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
        self.action_save.triggered.connect(self.on_action_save)
        self.action_export.triggered.connect(self.on_action_export_file)
        self.action_exit.triggered.connect(self.exit_app)

        self.action_about.triggered.connect(self.on_action_about)
        self.action_batch_evaluation.triggered.connect(
            self.on_action_batch_evaluation)
        self.action_batch_export.triggered.connect(
            self.on_action_batch_export)

    def _warn_export_in_progress(self):
        # export_file()'s background thread reads session.file/the
        # current evaluation model directly - closing the session out
        # from under it (which close_file() always does) could hand it
        # a torn read or a session that's already gone. export_file()
        # itself already guards against a second concurrent export the
        # same way.
        QMessageBox.warning(
            self, 'Export in progress',
            'Please wait for the current export to finish before '
            'opening or closing a file.')

    def open_file(self, file_name=None, update_views=True):
        """
        Show open file dialog and load file. `update_views=False` skips
        filling the tabs, for batch steps that work in their own window.
        """
        if self.export_running:
            self._warn_export_in_progress()
            return
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
        # Ready already if a batch run loaded it in the background
        preloaded = self.prefetcher.take(file_name) \
            if self.prefetcher is not None else None
        try:
            self.close_file()
            session.set_file(file_name, preloaded=preloaded)
            # Kept inside the try/except too: a valid file can still
            # contain a repetition the UI can't render (e.g. an
            # aborted/restarted acquisition with an empty first
            # repetition), and that shouldn't be able to crash the
            # whole application on open.
            if update_views:
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
        if self.export_running:
            self._warn_export_in_progress()
            return
        Session.get_instance().clear()
        self.reset_ui()

    def on_action_save(self):
        """
        Saves the session and, if it holds evaluation results, the
        calibration plots next to it.
        """
        self.save_session()
        self._save_calibration_plots()

    @staticmethod
    def _save_calibration_plots(repetitions=None):
        """
        Writes the calibration plots of the open file, only for the
        repetitions that have been evaluated (and are in `repetitions`,
        if given) - plots are only part of saving an evaluated file.
        """
        session = Session.get_instance()
        if session.file is None:
            return
        evaluated = evaluated_repetitions(session)
        if repetitions is not None:
            evaluated = [key for key in evaluated if key in repetitions]
        if not evaluated:
            return
        # The session is already saved; a plotting problem must not
        # turn that into an error.
        try:
            export_calibration_plots(session, evaluated)
        except Exception:
            logger.warning('Could not save calibration plots',
                           exc_info=True)

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

        checkbox_fluorescence = dialog.checkbox_export_fluorescence
        checkbox_fluorescence.setChecked(
            self.export_config['fluorescence']['export'])
        checkbox_fluorescence.toggled.connect(
            self.on_export_fluorescence_checkbox)

        checkbox_fluorescence_combined = \
            dialog.checkbox_export_fluorescence_combined
        checkbox_fluorescence_combined.setChecked(
            self.export_config['fluorescenceCombined']['export'])
        checkbox_fluorescence_combined.toggled.connect(
            self.on_export_fluorescence_combined_checkbox)

        checkbox_overview = dialog.checkbox_export_overview_brightfield
        checkbox_overview.setChecked(
            self.export_config['overviewBrightfield']['export'])
        checkbox_overview.toggled.connect(
            self.on_export_overview_brightfield_checkbox)

        checkbox_tiles = dialog.checkbox_export_overview_tiles
        checkbox_tiles.setChecked(
            self.export_config['overviewBrightfield']['exportTiles'])
        checkbox_tiles.toggled.connect(
            self.on_export_overview_tiles_checkbox)

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

    def on_export_fluorescence_checkbox(self, checked):
        self.export_config['fluorescence']['export'] = checked

    def on_export_fluorescence_combined_checkbox(self, checked):
        self.export_config['fluorescenceCombined']['export'] = checked

    def on_export_overview_brightfield_checkbox(self, checked):
        self.export_config['overviewBrightfield']['export'] = checked

    def on_export_overview_tiles_checkbox(self, checked):
        self.export_config['overviewBrightfield']['exportTiles'] = checked

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
        # If the Quality tab's auto-recompute got deferred by this
        # export (see EvaluationView.evaluate()'s export_running guard
        # and QualityView.on_evaluation_deferred()), retry it now
        # instead of leaving the tab showing stale metrics until the
        # user happens to revisit it - export_running is already False
        # above, so this time evaluate() actually runs.
        self.widget_quality_view.update_ui()

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
        self.widget_background_view.reset_ui()

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
            self.widget_background_view.update_ui()
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
        elif new_tab_index == 6:
            self.widget_background_view.update_ui()

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
        self.batch_dialog.table_files.itemChanged.connect(
            self.on_batch_repetition_toggled)
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
        # Re-analyse
        self.batch_dialog.checkBox_reanalyse.setChecked(self.batch_reanalyse)
        self.batch_dialog.checkBox_reanalyse.clicked.connect(
            self.on_reanalyse_set)

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
        # The regions are chosen by hand in the peak selection window,
        # so the preset tables are not needed here.
        for name in ('label_Brillouin_regions', 'tableWidget_Brillouin',
                     'button_region_brillouin_add',
                     'button_region_brillouin_remove',
                     'label_Rayleigh_regions', 'tableWidget_Rayleigh',
                     'button_region_rayleigh_add',
                     'button_region_rayleigh_remove'):
            getattr(self.batch_dialog, name).hide()
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
        self.batch_dialog.button_export_configure\
            .clicked.connect(self.on_batch_export_configure)
        self.update_batch_export_summary()

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
        self._cascade_batch_steps()

    def on_orientation_set(self):
        self.batch_config['orientation']['set'] = self.sender().isChecked()
        self.batch_dialog.groupBox_rotation\
            .setEnabled(self.batch_config['orientation']['set'])
        self.batch_dialog.groupBox_reflection\
            .setEnabled(self.batch_config['orientation']['set'])
        self._cascade_batch_steps()

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

    def on_reanalyse_set(self):
        self.batch_reanalyse = self.sender().isChecked()

    def _cascade_batch_steps(self):
        """
        Redoing a step invalidates every step built on it, so ticking
        one ticks all later ones too: setup / orientation -> extraction
        -> calibration -> peak selection -> evaluation.
        """
        cfg_calibration = self.batch_config['calibration']
        cfg_peak_selection = self.batch_config['peak-selection']
        dialog = self.batch_dialog

        if self.batch_config['setup']['set'] \
                or self.batch_config['orientation']['set']:
            self.batch_config['extraction']['extract'] = True
        dialog.checkBox_extraction.setChecked(
            self.batch_config['extraction']['extract'])
        if self.batch_config['extraction']['extract']:
            cfg_calibration['find-peaks'] = True
            cfg_calibration['calibrate'] = True
        if cfg_calibration['find-peaks'] or cfg_calibration['calibrate']:
            cfg_peak_selection['select'] = True
        if cfg_peak_selection['select']:
            self.batch_config['evaluation']['evaluate'] = True

        dialog.checkBox_find_peaks.setChecked(cfg_calibration['find-peaks'])
        dialog.checkBox_calibrate.setChecked(cfg_calibration['calibrate'])
        dialog.checkBox_peak_selection.setChecked(cfg_peak_selection['select'])
        self._set_batch_evaluate(self.batch_config['evaluation']['evaluate'])

    def on_extraction_set(self):
        self.batch_config['extraction']['extract'] = self.sender().isChecked()
        self._cascade_batch_steps()

    def on_calibration_find_peaks(self):
        self.batch_config['calibration']['find-peaks']\
            = self.sender().isChecked()
        self._cascade_batch_steps()

    def on_calibration_calibrate(self):
        self.batch_config['calibration']['calibrate']\
            = self.sender().isChecked()
        self._cascade_batch_steps()

    def on_peak_selection_select(self):
        self.batch_config['peak-selection']['select']\
            = self.sender().isChecked()
        self._cascade_batch_steps()

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
        self._set_batch_evaluate(self.sender().isChecked())

    def _set_batch_evaluate(self, evaluate):
        self.batch_config['evaluation']['evaluate'] = evaluate
        self.batch_dialog.checkBox_evaluation.setChecked(evaluate)
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
        self.update_batch_export_summary()

    def on_batch_export_configure(self):
        """
        Opens the same "what to export" dialog the single-file Export
        button and standalone Batch Export feature use, so the subset
        exported by batch evaluation can be seen/changed from within
        the batch dialog itself - without this, self.export_config
        (see update_batch_export_summary()'s own docstring) is only
        ever visible/editable via one of those two other, easy-to-miss
        entry points, and batch evaluation would silently inherit
        whatever they last left it at, or bare defaults if neither was
        ever opened this session.
        """
        dialog = self._build_export_config_dialog(
            'Batch export configuration', lambda: None)
        # This dialog's checkboxes already write straight into
        # self.export_config as they're toggled (see
        # on_export_overview_brightfield_checkbox()/
        # on_export_surface_checkbox()) - unlike the single-file/batch
        # export entry points, there is nothing left to *do* on
        # "Export" here (the real export happens later, during the
        # batch run), so relabel it to match and just close the dialog.
        dialog.button_export.setText('Done')
        dialog.button_export.clicked.connect(
            self.update_batch_export_summary)
        dialog.finished.connect(self.update_batch_export_summary)
        dialog.open()

    def update_batch_export_summary(self):
        """
        Shows what batch evaluation's export step will actually write
        for each file, given the current self.export_config - the same
        configuration object the single-file Export button and
        standalone Batch Export feature share (see
        _build_export_config_dialog()'s own docstring), so a choice
        made in either of those carries over here silently otherwise.
        The Brillouin combined CSV has no on/off choice anywhere in
        the UI - always included - which is worth surfacing here too,
        not just the ones that do.
        """
        if not self.batch_config['export']['export']:
            self.batch_dialog.label_export_summary.setText(
                'Export step is disabled.')
            return
        included = ['Brillouin CSV']
        if self.export_config['fluorescence']['export']:
            included.append('Fluorescence images')
        if self.export_config['fluorescenceCombined']['export']:
            included.append('Combined fluorescence images')
        if self.export_config['overviewBrightfield']['export']:
            included.append('Brightfield overview images')
        if self.export_config['overviewBrightfield']['exportTiles']:
            included.append('unstitched overview tiles')
        if self.export_config['surface']['export']:
            included.append('Surface scan')
        self.batch_dialog.label_export_summary.setText(
            'Will export: ' + ', '.join(included))

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
        if not self.batch_evaluation_running and self.export_running:
            # Starting into an already-running export would have every
            # file's evaluate() call silently decline to run (see
            # EvaluationView.evaluate()'s export_running guard) - refuse
            # the whole batch up front instead of limping through it.
            # batch_dialog is application-modal, so this can only be
            # reached with an export left running from before the
            # dialog was opened, never one started during the batch
            # itself.
            self._warn_export_in_progress()
            return
        if not self.batch_evaluation_running \
                and not self._confirm_batch_overwrite():
            return
        self.batch_evaluation_running = not self.batch_evaluation_running
        if self.batch_evaluation_running:
            self.run_batch_evaluation()
        else:
            self.widget_evaluation_view.evaluation_abort.value = True

    def _confirm_batch_overwrite(self):
        """
        Asks for confirmation if this batch run would replace existing
        results: redoing a step recomputes the ones built on it, which
        overwrites the saved calibration fits (when extraction is
        redone) and evaluation results. Returns False if the user
        cancels. Existing extraction points and peak regions are kept
        as the starting point, so they don't count.
        """
        self._cascade_batch_steps()
        cfg_calibration = self.batch_config['calibration']
        ticked = {
            1: self.batch_config['extraction']['extract'],
            2: cfg_calibration['find-peaks'] or cfg_calibration['calibrate'],
            3: self.batch_config['peak-selection']['select'],
        }
        evaluate = self.batch_config['evaluation']['evaluate']

        lose_fits, lose_results, files = 0, 0, set()
        for file in self.batch_files.values():
            selected = self._selected_batch_repetitions(file)
            for key, rep in file.get('repetitions', {}).items():
                if not rep['valid'] \
                        or (selected is not None and key not in selected) \
                        or rep['progress'] not in self.PROGRESS_STEPS:
                    continue
                done = self.PROGRESS_STEPS.index(rep['progress'])
                redone = [step for step, on in ticked.items() if on and (
                    self.batch_reanalyse or done < step)]
                fits = bool(redone) and min(redone) <= 1 and done >= 2
                # Evaluation alone only runs on repetitions that are not
                # evaluated yet, unless everything is redone
                results = (bool(redone) or (
                    evaluate and self.batch_reanalyse)) and done >= 4
                lose_fits += fits
                lose_results += results
                if fits or results:
                    files.add(file['path'])
        # Setup/orientation are only changed on analysed files when
        # re-analysing, and then reset every analysed repetition of the
        # file (see _apply_batch_base_settings()) - but only those whose
        # saved setup/orientation really differs from the batch's
        reset_paths, unknown_paths = [], []
        cfg_setup = self.batch_config['setup']
        cfg_orientation = self.batch_config['orientation']
        if self.batch_reanalyse and (cfg_setup['set']
                                     or cfg_orientation['set']):
            wanted_setup = setup_signature(cfg_setup['setup'])
            wanted_orientation = orientation_signature(
                cfg_orientation['rotation'], cfg_orientation['reflection'])
            for file in self.batch_files.values():
                analysed = [rep for rep in
                            (file.get('repetitions') or {}).values()
                            if is_analysed(rep)]
                if not analysed:
                    continue
                saved = analysed[0].get('saved_base')
                if saved is None:
                    unknown_paths.append(file['path'])
                elif (cfg_setup['set'] and saved[0] != wanted_setup) or (
                        cfg_orientation['set']
                        and saved[1] != wanted_orientation):
                    reset_paths.append(file['path'])
            files.update(reset_paths, unknown_paths)
        if not files:
            return True

        lines = []
        if lose_fits:
            lines.append(f'{lose_fits} repetition(s): calibration fits')
        if lose_results:
            lines.append(f'{lose_results} repetition(s): evaluation results')
        if reset_paths:
            names = [pathlib.Path(path).name for path in reset_paths]
            lines.append(
                f'{len(reset_paths)} analysed file(s) get a different '
                f'setup/orientation than saved - every analysis step of '
                f'all their repetitions is reset: '
                + ', '.join(names[:8]) + (', ...' if len(names) > 8 else ''))
        if unknown_paths:
            lines.append(
                f'{len(unknown_paths)} analysed file(s): the saved '
                f'setup/orientation could not be read - the analysis is '
                f'reset if they differ')
        box = QMessageBox(self.batch_dialog)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle('Replace existing analysis?')
        box.setText(
            f'This run redoes analysis steps that other results depend '
            f'on. In {len(files)} file(s) the following existing results '
            f'will be replaced:')
        box.setInformativeText('\n'.join(lines))
        box.setStandardButtons(
            QMessageBox.StandardButton.Ok
            | QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(QMessageBox.StandardButton.Cancel)
        return box.exec() == QMessageBox.StandardButton.Ok

    def run_batch_evaluation(self):
        # Background loading of the next files; 'batch/prefetch-files'
        # in the settings file sets how many (0 switches it off).
        try:
            nr_prefetch = int(self.settings.value('batch/prefetch-files', 5))
        except (TypeError, ValueError):
            nr_prefetch = 5
        self.prefetcher = FilePrefetcher(max_files=nr_prefetch) \
            if nr_prefetch > 0 else None
        try:
            self._run_batch_evaluation()
        finally:
            if self.prefetcher is not None:
                self.prefetcher.shutdown()
            self.prefetcher = None

    def _prefetch_batch_files(self, files, position):
        """ Starts loading the files after `position` (up to 5). """
        if self.prefetcher is not None:
            self.prefetcher.request(
                [file['path'] for file in files[position + 1:]])

    def _run_batch_evaluation(self):
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
        self.batch_forced = {}
        self._cascade_batch_steps()

        # Extraction, calibration and peak selection need the user, so
        # they run first, one step at a time over all files.
        cfg_calibration = self.batch_config['calibration']
        interactive_steps = []
        if self.batch_config['extraction']['extract']:
            interactive_steps.append(STEP_EXTRACTION)
        if cfg_calibration['find-peaks'] or cfg_calibration['calibrate']:
            interactive_steps.append(STEP_CALIBRATION)
        if self.batch_config['peak-selection']['select']:
            interactive_steps.append(STEP_PEAK_SELECTION)
        for step in interactive_steps:
            self.run_batch_interactive_step(step)
            if not self.batch_evaluation_running:
                break

        # Files with nothing left to evaluate/export are not opened
        todo = [file for file in self.batch_files.values()
                if self._batch_file_needs_evaluation(file)]
        todo_ids = {id(file): position for position, file in enumerate(todo)}
        for i, (file_hash, file) in enumerate(self.batch_files.items()):
            if not self.batch_evaluation_running:
                break
            if id(file) not in todo_ids:
                self.batch_dialog.progressBar.setValue(i + 1)
                continue
            self._prefetch_batch_files(todo, todo_ids[id(file)] - 1)
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
        selected = self._selected_batch_repetitions(file)
        if selected is not None:
            rep_keys = [key for key in rep_keys if key in selected]

        for rep_key in rep_keys:
            # Load repetition
            session.set_current_repetition(rep_key)
            QtCore.QCoreApplication.instance().processEvents()

            # Setup and orientation (already applied in
            # run_batch_interactive_step(); a no-op then)
            self._apply_batch_base_settings(file)

            # Extraction, calibration and peak selection were already
            # done interactively before this loop (see
            # run_batch_interactive_step()).

            # Evaluation
            cfg_evaluation = self.batch_config['evaluation']
            if cfg_evaluation['evaluate']:
                # A repetition that is evaluated already keeps its
                # results unless 'Re-analyse finished files' is ticked.
                # (Upstream steps redone in this run bring the
                # repetition's progress below 'evaluation' again.)
                progress = (file.get('repetitions') or {}).get(
                    rep_key, {}).get('progress')
                if progress == 'evaluation' and not self.batch_reanalyse:
                    continue
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
                        "no Brillouin/Rayleigh peak regions selected")
                if cm.frequencies_by_time_interpolator is None:
                    missing.append("no valid calibration")
                if missing:
                    self.batch_evaluation_messages.append(
                        f"{file['path']} repetition {rep_key}: "
                        f"skipped evaluation - {'; '.join(missing)}.")
                    continue

                # Still evaluated, but flagged: the missing calibrations
                # are interpolated over from the remaining ones.
                gaps = calibration_gaps_text(*calibration_gaps(
                    session.get_calib_keys() or [],
                    session.extraction_model(), cm))
                if gaps:
                    self.batch_evaluation_messages.append(
                        f"{file['path']} repetition {rep_key}: "
                        f"evaluated with incomplete calibrations "
                        f"({gaps}).")

                evc = EvaluationController()
                evc.set_nr_brillouin_peaks(
                    cfg_evaluation['nr_brillouin_peaks'])
                evc.set_bounds(cfg_evaluation['bounds_w0'])
                evc.set_bounds_fwhm(cfg_evaluation['bounds_fwhm'])
                self.widget_evaluation_view.evaluate(
                    blocking=True, notify_blocked=False)
                # start_batch_evaluation() already refuses to start a
                # batch while an export is running, but if one somehow
                # got started anyway (or started on another instance),
                # evaluate() above silently declined to run at all -
                # applying/exporting this repetition's (stale, possibly
                # entirely absent) results would be worse than just
                # flagging it as skipped.
                if getattr(self, 'export_running', False):
                    self.batch_evaluation_messages.append(
                        f"{file['path']} repetition {rep_key}: "
                        f"skipped evaluation - an export was running.")
                    continue
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
        export_failed = False
        cfg_export = self.batch_config['export']
        if cfg_export['export']:
            if self.aborted(file):
                return
            QtCore.QCoreApplication.instance().processEvents()
            try:
                export_config = copy.deepcopy(self.export_config)
                # Only the repetitions ticked in the batch table.
                export_config['brillouin']['repetitions'] = \
                    None if selected is None else sorted(selected)
                ExportController().export(export_config)
            except Exception as e:
                # Don't let an export problem lose the evaluation work
                # already done above - still fall through to save it.
                export_failed = True
                self.batch_evaluation_messages.append(
                    f"{file['path']}: export failed - {e}")
                logger.warning(
                    'Batch export failed for %s', file['path'],
                    exc_info=True)

        # Save the evaluated data
        self.save_session()
        self._refresh_file_progress(file)
        if self.batch_config['evaluation']['evaluate']:
            self._save_calibration_plots(selected)
        if self.aborted(file):
            return
        QtCore.QCoreApplication.instance().processEvents()

        # Close the file
        self.close_file()
        QtCore.QCoreApplication.instance().processEvents()

        # Set the status as done - a file whose export step failed is
        # marked the same as any other failure (the end-of-run summary
        # message above still says specifically what failed) rather
        # than 'success', which would hide that something needs
        # attention behind a checkmark.
        file['status'] = 'failed' if export_failed else 'success'
        self.update_batch_file_table()

    # progress ('none', 'extraction', ...) a repetition needs to have
    # reached for a step to be done
    STEP_PROGRESS = {STEP_EXTRACTION: 1, STEP_CALIBRATION: 2,
                     STEP_PEAK_SELECTION: 3}

    def _batch_file_needs_step(self, file, step):
        """
        Whether a manual step has anything to do in `file`, judged from
        the batch table's progress alone, without opening the file. Only
        a file whose ticked repetitions all have the step behind them
        is skipped (it would only be opened to find that out); anything
        uncertain (progress unknown, a repetition redone upstream in
        this run, re-analysis) counts as needing it.
        """
        repetitions = file.get('repetitions')
        if self.batch_reanalyse or not repetitions \
                or self.batch_forced.get((str(file['path']), step)):
            return True
        needed = self.STEP_PROGRESS[step]
        for key in self._selected_batch_repetitions(file):
            progress = repetitions[key]['progress']
            if progress not in self.PROGRESS_STEPS \
                    or self.PROGRESS_STEPS.index(progress) < needed:
                return True
        return False

    def _batch_file_needs_evaluation(self, file):
        """
        Whether evaluating/exporting `file` has anything to do, from the
        batch table's progress alone (see _batch_file_needs_step()).
        """
        repetitions = file.get('repetitions')
        if self.batch_reanalyse or not repetitions \
                or self.batch_config['export']['export']:
            return True
        if not self.batch_config['evaluation']['evaluate']:
            return False
        return any(repetitions[key]['progress'] != 'evaluation'
                   for key in self._selected_batch_repetitions(file))

    def _apply_batch_base_settings(self, file):
        """
        Sets the batch dialog's setup and orientation on the open file.

        They are the base of every analysis step: for a file that has
        been analysed already they are left as saved, unless 'Re-analyse
        finished files' is ticked. And when they are changed there, what
        was built on them is outdated and is reset - a new orientation
        changes the geometry of the extraction points (all steps from
        extraction on), a new setup the calibration (from calibration
        on) - instead of being kept as a starting point. Both are
        properties of the file, so every analysed repetition is reset,
        ticked in the batch table or not.
        """
        cfg_setup = self.batch_config['setup']
        cfg_orientation = self.batch_config['orientation']
        if not (cfg_setup['set'] or cfg_orientation['set']):
            return
        session = Session.get_instance()
        reflection = cfg_orientation['reflection']
        setup_changes = cfg_setup['set'] and setup_signature(
            cfg_setup['setup']) != setup_signature(session.setup)
        orientation_changes = cfg_orientation['set'] \
            and orientation_signature(
                session.orientation.rotation,
                session.orientation.reflection) != orientation_signature(
                    cfg_orientation['rotation'], reflection)
        if not (setup_changes or orientation_changes):
            return

        analysed = [key for key, rep in (file.get('repetitions') or {}).items()
                    if is_analysed(rep)]
        if analysed and not self.batch_reanalyse:
            message = (f"{file['path']}: setup/orientation left as saved - "
                       f"the file is analysed already (tick 'Re-analyse "
                       f"finished files' to change them and redo the "
                       f"analysis).")
            if message not in self.batch_evaluation_messages:
                self.batch_evaluation_messages.append(message)
            return

        if setup_changes:
            session.set_setup(cfg_setup['setup'])
        if orientation_changes:
            session.set_rotation(cfg_orientation['rotation'])
            session.set_reflection(
                vertically=reflection['vertically'],
                horizontally=reflection['horizontally'])
        if analysed:
            reset_analysis(session, analysed, orientation_changes)
            first_reset = 'extraction' if orientation_changes \
                else 'calibration'
            self.batch_evaluation_messages.append(
                f"{file['path']}: setup/orientation changed - analysis "
                f"of repetition(s) {', '.join(map(str, analysed))} reset "
                f"from {first_reset} on.")

    def run_batch_interactive_step(self, step):
        """
        Runs one manual step (extraction, calibration or peak
        selection) over all files of the batch: for each file that
        still has the step open in at least one repetition, opens the
        file in a window with the regular view, waits until the user
        closes it, and saves the session. Files where the step is
        already complete are skipped.
        """
        session = Session.get_instance()
        # Files that have nothing to do for this step are not even
        # opened or loaded in the background
        files = [file for file in self.batch_files.values()
                 if self._batch_file_needs_step(file, step)]

        for position, file in enumerate(files, start=1):
            file['status'] = 'in-process'
            self.update_batch_file_table()
            QtCore.QCoreApplication.instance().processEvents()
            try:
                self._prefetch_batch_files(files, position - 2)
                self.open_file(file['path'], update_views=False)
                # While this file is worked on, the next ones load.
                self._prefetch_batch_files(files, position - 1)
                if session.file is None:
                    raise RuntimeError('the file could not be opened')

                # Setup and orientation determine the images the user
                # works on, so they must be set before the dialog opens.
                self._apply_batch_base_settings(file)

                forced = self.batch_forced.get(
                    (str(file['path']), step), set())
                pending = pending_repetitions(
                    step, include_done=self.batch_reanalyse, forced=forced,
                    allowed=self._selected_batch_repetitions(file))
                if pending:
                    # Whatever is redone here makes the steps built on
                    # it outdated for these repetitions.
                    for downstream in DOWNSTREAM_STEPS[step]:
                        self.batch_forced.setdefault(
                            (str(file['path']), downstream),
                            set()).update(pending)
                    self._prefill_batch_step(step, pending, file, forced)
                    # One window per repetition, in order
                    stop = False
                    for number, rep_key in enumerate(pending, start=1):
                        dialog = BatchStepDialog(
                            self, step, file['path'], position, len(files),
                            [rep_key], (number, len(pending)))
                        dialog.exec()
                        stop = dialog.stop_requested
                        # The dialog is parented to this window, so it
                        # (and its views and figures) would otherwise
                        # pile up with every repetition of the batch.
                        # deleteLater() is not enough: inside this loop,
                        # which never returns to the event loop it was
                        # called from, it would only delete them once
                        # the whole batch is over.
                        sip.delete(dialog)
                        if stop:
                            break
                    # Saved once per file, after its last repetition
                    self.save_session()
                    self._refresh_file_progress(file)
                    if stop:
                        self.batch_evaluation_running = False
                file['status'] = 'pending'
            except BaseException as e:
                file['status'] = 'failed'
                self.batch_evaluation_messages.append(
                    f"{file['path']}: {step} failed - {e}")
                logger.warning(
                    'Batch %s failed for %s', step, file['path'],
                    exc_info=True)
            finally:
                self.close_file()
                self.update_batch_file_table()
                QtCore.QCoreApplication.instance().processEvents()
            if not self.batch_evaluation_running:
                for other in files:
                    if other['status'] == 'pending':
                        other['status'] = 'aborted'
                self.batch_dialog.progressBar.setValue(0)
                return

    def _prefill_batch_step(self, step, repetitions, file, forced):
        """
        Runs the batch dialog's automatic calibration settings (find
        peaks / calibrate) on the repetitions still open - or outdated
        because extraction was redone - as a starting point for the
        user in the calibration window.
        """
        session = Session.get_instance()
        previous = session.current_repetition_key()
        try:
            for rep_key in repetitions:
                session.set_current_repetition(rep_key)
                if step_done(step, session) and rep_key not in forced:
                    # Re-analysis: keep the existing results as they are.
                    continue
                if step == STEP_CALIBRATION:
                    cfg = self.batch_config['calibration']
                    if not step_done(STEP_EXTRACTION, session):
                        self.batch_evaluation_messages.append(
                            f"{file['path']} repetition {rep_key}: "
                            f"extraction is incomplete, so calibration "
                            f"was not prefilled.")
                        continue
                    view = self.widget_calibration_view
                    view.update_ui()
                    if not cfg['find-peaks']:
                        view.calibrate_all(do_not='find_peaks')
                    elif not cfg['calibrate']:
                        view.calibrate_all(do_not='calibrate')
                    else:
                        view.calibrate_all()
        finally:
            session.set_current_repetition(previous)

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

        if not folder_name or self.batch_scan_thread is not None:
            return

        # Find the source files in the selected folder that are not in
        # the batch yet and read their repetitions/progress in a
        # background thread, so that the window stays responsive and
        # shows how far it is.
        progress_dialog = QtWidgets.QDialog(
            self.batch_dialog,
            QtCore.Qt.WindowType.WindowTitleHint |
            QtCore.Qt.WindowType.WindowCloseButtonHint
        )
        progress_dialog.setWindowTitle('Loading folder')
        progress_dialog.setWindowModality(
            QtCore.Qt.WindowModality.ApplicationModal)
        label = QLabel('Searching for files...')
        label.setWordWrap(True)
        progress_bar = QtWidgets.QProgressBar()
        progress_bar.setRange(0, 0)  # busy until the number of files is known
        button_cancel = QtWidgets.QPushButton('Cancel')
        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(progress_bar)
        layout.addWidget(button_cancel)
        progress_dialog.setLayout(layout)
        progress_dialog.resize(QSize(500, 120))

        self.batch_scan_thread = QtCore.QThread()
        self.batch_scan_worker = BatchScanWorker(
            folder_name, self.batch_files.keys(),
            self._scan_batch_repetitions)
        worker, thread = self.batch_scan_worker, self.batch_scan_thread
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        thread.finished.connect(thread.deleteLater)
        total = []

        def on_found(count):
            total.append(count)
            progress_bar.setRange(0, max(count, 1))
            label.setText(f'{count} file(s) found' if count
                          else 'No new files found')

        def on_progress(i, path):
            label.setText(f'File {i + 1} / {total[0]}\n{path}')
            progress_bar.setValue(i)

        def on_scanned(path, repetitions):
            self.batch_files[file_hash(path)] = {
                'path': path,
                'status': 'pending',
                'repetitions': repetitions,
            }
            # Rebuilding the table for every file would be slow
            if len(self.batch_files) % 10 == 0:
                self.update_batch_file_table()

        def on_finished():
            thread.quit()
            worker.deleteLater()
            self.batch_scan_thread = None
            self.batch_scan_worker = None
            progress_dialog.close()
            self.update_batch_file_table()

        worker.found.connect(on_found)
        worker.progress.connect(on_progress)
        worker.scanned.connect(on_scanned)
        worker.finished.connect(on_finished)
        # Cancel (or closing the window) keeps the files scanned so far
        button_cancel.clicked.connect(
            lambda: setattr(worker, 'aborted', True))
        progress_dialog.rejected.connect(
            lambda: setattr(worker, 'aborted', True))

        progress_dialog.show()
        thread.start()

    @staticmethod
    def _scan_batch_repetitions(path):
        """
        Lists the repetitions of a source file for the batch table:
        {key: {'label', 'valid', 'checked'}}. A repetition without
        grid data (an aborted/restarted acquisition that never wrote
        any positions or spectra) is not valid and can't be ticked.
        All valid ones start ticked (the others can be unticked by
        hand). The label shows the total measurement time to tell
        restarts apart.
        Returns an empty dict if the file can't be scanned, which
        means "all repetitions".

        Reading the raw file and the saved session takes seconds, so
        both are looked up in the file's batch_cache entry first, which
        is only used while they are unchanged.
        """
        cached = batch_cache.read(path, get_session_file_path(path))
        if cached is not None and cached['repetitions'] is not None:
            repetitions = {
                key: dict(info, checked=False, progress=None)
                for key, info in cached['repetitions'].items()}
        else:
            repetitions = BMicro._read_batch_repetitions(path)
        # By default every repetition that has data is analysed; one
        # that should be left out can be unticked in the table.
        for rep in repetitions.values():
            rep['checked'] = rep['valid']
        BMicro._refresh_batch_progress(path, repetitions, cached=cached)
        return repetitions

    @staticmethod
    def _read_batch_repetitions(path):
        """ The repetitions of the raw file, see _scan_batch_repetitions(). """
        repetitions = {}
        try:
            file = BrillouinFile(path)
        except Exception:
            logger.warning('Could not scan %s', path, exc_info=True)
            return repetitions
        try:
            for key in file.repetition_keys():
                try:
                    repetition = file.get_repetition(key)
                    resolution = repetition.payload.resolution
                    image_keys = repetition.payload.image_keys()
                    nr_spectra = len(image_keys)
                    # Total measurement time: first to last spectrum.
                    dates = [repetition.payload.get_date(k)
                             for k in image_keys]
                    duration = (max(dates) - min(dates)).total_seconds() \
                        if dates else 0
                    hours, rest = divmod(int(round(duration)), 3600)
                    duration_text = '%d:%02d:%02d' % (
                        hours, *divmod(rest, 60))
                except Exception:
                    resolution, nr_spectra, duration_text = None, 0, '?'
                try:
                    calib_keys = list(repetition.calibration.image_keys())
                except Exception:
                    calib_keys = []
                valid = resolution is not None and nr_spectra > 0
                label = f'Repetition {key} - measured {duration_text} h'
                if valid:
                    label += ' - %d x %d x %d grid, %d spectra' % (
                        *resolution, nr_spectra)
                else:
                    label += ' - no grid data'
                repetitions[key] = {
                    'label': label, 'valid': valid, 'checked': False,
                    'calib_keys': calib_keys, 'progress': None,
                    'nr_spectra': nr_spectra}
        finally:
            file.close()
        return repetitions

    @staticmethod
    def _refresh_batch_progress(path, repetitions, models=None,
                                cached=None):
        """
        Sets rep['progress'] for every repetition to the last analysis
        step completed so far ('none', 'extraction', 'calibration',
        'peak selection' or 'evaluation'). Each step counts only if all
        earlier ones are done too.

        Where it comes from, fastest first: the batch_cache entry
        `cached` (still valid), the `models` given (the open session,
        just saved), the saved session file (which has to be
        deserialised completely). The result is stored in the file's
        batch_cache entry for the next look.
        """
        session_path = get_session_file_path(path)
        if models is None and cached is not None \
                and cached['progress'] is not None \
                and set(cached['progress']) == set(repetitions):
            for key, rep in repetitions.items():
                rep.update(cached['progress'][key])
                rep['saved_base'] = cached['saved_base']
            return
        read_ok = True
        try:
            if models is None and os.path.exists(session_path) \
                    and is_session_file(session_path):
                with h5py.File(session_path, 'r') as f:
                    models = Serializer.deserialize(f['session'])
        except Exception:
            read_ok = False
            logger.warning(
                'Could not read progress from %s', session_path,
                exc_info=True)
        # The file's saved setup and orientation (the same for all its
        # repetitions), to tell beforehand where changing them would
        # reset the analysis
        saved_base = saved_base_signature(models) \
            if models is not None else None
        for key, rep in repetitions.items():
            progress = 'none'
            rep['warning'] = None
            rep['saved_base'] = saved_base
            try:
                if models is not None and rep['valid']:
                    em = models.extraction_models[key]
                    cm = models.calibration_models[key]
                    pm = models.peak_selection_models[key]
                    evm = models.evaluation_models[key]
                    calib_keys = rep['calib_keys']
                    total, no_arc, no_fit = calibration_gaps(
                        calib_keys, em, cm)
                    # Only a partly done step is worth a warning; one
                    # that hasn't been started shows up as progress.
                    rep['warning'] = calibration_gaps_text(
                        total,
                        no_arc if 0 < no_arc < total else 0,
                        no_fit if 0 < no_fit < total else 0)
                    steps = [
                        ('extraction', all(
                            len(em.get_points(k)) >= 3
                            for k in calib_keys)),
                        ('calibration', all(
                            k in cm.frequencies for k in calib_keys)),
                        # Sessions from before the GHz regions (see
                        # Session.run_migrations()) have neither yet.
                        ('peak selection', bool(
                            getattr(pm, 'brillouin_regions_f', None)
                            and getattr(pm, 'rayleigh_regions_f', None))),
                        ('evaluation', bool(np.any(np.isfinite(
                            evm.results['brillouin_peak_position_f'])))),
                    ]
                    for name, done in steps:
                        if not done:
                            break
                        progress = name
            except Exception:
                progress = 'unknown'
            rep['progress'] = progress
        if read_ok:
            batch_cache.write(path, session_path, repetitions, saved_base)

    PROGRESS_STEPS = ['none', 'extraction', 'calibration',
                      'peak selection', 'evaluation']

    @staticmethod
    def _progress_text(progress):
        if progress == 'none':
            return 'not analysed yet'
        if progress in (None, 'unknown'):
            return 'progress unknown'
        return 'analysed up to: ' + progress

    def _file_progress_text(self, file):
        """
        The least advanced progress of the file's usable repetitions,
        plus a warning if any of them has incomplete calibrations.
        """
        reps = [rep for rep in file.get('repetitions', {}).values()
                if rep['valid']]
        progress = [rep['progress'] for rep in reps]
        if not progress:
            return 'no repetitions to analyse'
        if any(p not in self.PROGRESS_STEPS for p in progress):
            text = self._progress_text('unknown')
        else:
            text = self._progress_text(
                min(progress, key=self.PROGRESS_STEPS.index))
        if any(rep.get('warning') for rep in reps):
            text += ' - WARNING: incomplete calibrations'
        return text

    def _refresh_file_progress(self, file):
        """
        Updates the batch table's progress of a file after it was saved.
        The open session is that file's, just saved, so it is used
        directly instead of reading the saved session again.
        """
        session = Session.get_instance()
        models = None
        if session.file is not None and pathlib.Path(
                session.file.path).resolve() == pathlib.Path(
                file['path']).resolve():
            models = session
        self._refresh_batch_progress(
            file['path'], file.get('repetitions', {}), models=models)

    @staticmethod
    def _selected_batch_repetitions(file):
        """
        The repetition keys ticked for `file` in the batch table, or
        None if the file has no repetition list (use all of them).
        """
        repetitions = file.get('repetitions')
        if not repetitions:
            return None
        return {key for key, rep in repetitions.items()
                if rep['valid'] and rep['checked']}

    def on_batch_repetition_toggled(self, item):
        row = item.row()
        if row >= len(self.batch_rows):
            return
        file_hash, rep_key = self.batch_rows[row]
        if rep_key is None:
            return
        self.batch_files[file_hash]['repetitions'][rep_key]['checked'] = \
            item.checkState() == QtCore.Qt.CheckState.Checked

    def batch_remove_files(self):
        table = self.batch_dialog.table_files
        hashes_remove = set()
        for selected_range in table.selectedRanges():
            for i in range(selected_range.topRow(),
                           selected_range.bottomRow() + 1):
                file_hash, rep_key = self.batch_rows[i]
                if rep_key is None:
                    hashes_remove.add(file_hash)
        for hash_remove in hashes_remove:
            self.batch_files.pop(hash_remove)
        table.clearSelection()
        self.update_batch_file_table()

    def update_batch_file_table(self):
        table = self.batch_dialog.table_files
        table.setTextElideMode(QtCore.Qt.TextElideMode.ElideRight)
        table.setWordWrap(False)
        table.setColumnCount(2)
        table.blockSignals(True)

        # One row per file, followed by one row per repetition of it.
        self.batch_rows = []
        for file_hash, file in self.batch_files.items():
            self.batch_rows.append((file_hash, None))
            for rep_key in file.get('repetitions', {}):
                self.batch_rows.append((file_hash, rep_key))
        table.setRowCount(len(self.batch_rows))

        pending_icon_path = str(pathlib.Path(self.imdir) / "pending.svg")
        inprocess_icon_path = str(pathlib.Path(self.imdir) / "in-process.svg")
        success_icon_path =\
            str(pathlib.Path(self.imdir) / "check_circle_outline.svg")
        failed_icon_path = str(pathlib.Path(self.imdir) / "error.svg")
        aborted_icon_path = str(pathlib.Path(self.imdir) / "cancel.svg")

        table.setIconSize(QtCore.QSize(20, 20))
        for rowIdx, (file_hash, rep_key) in enumerate(self.batch_rows):
            file = self.batch_files[file_hash]
            if rep_key is not None:
                rep = file['repetitions'][rep_key]
                check = QtWidgets.QTableWidgetItem()
                text = '    ' + rep['label']
                if rep['valid']:
                    text += ' - ' + self._progress_text(rep['progress'])
                    if rep.get('warning'):
                        text += ' - WARNING: ' + rep['warning']
                label = QtWidgets.QTableWidgetItem(text)
                if rep['valid']:
                    check.setFlags(
                        QtCore.Qt.ItemFlag.ItemIsUserCheckable |
                        QtCore.Qt.ItemFlag.ItemIsEnabled)
                    check.setCheckState(
                        QtCore.Qt.CheckState.Checked if rep['checked']
                        else QtCore.Qt.CheckState.Unchecked)
                    label.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
                else:
                    # Greyed out: nothing to analyse in this repetition.
                    check.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
                    label.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
                table.setItem(rowIdx, 0, check)
                table.setItem(rowIdx, 1, label)
                continue
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

            path = QtWidgets.QTableWidgetItem(
                f"{file['path']}   [{self._file_progress_text(file)}]")
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

    app = QtCore.QCoreApplication.instance()
    if app is not None and QtCore.QThread.currentThread() is not app.thread():
        # Raised in a background thread (e.g. a calibration running in a
        # BGThread). Qt cannot show a window from here, and blocking in
        # one would leave whoever waits for this thread (the GUI thread,
        # in thread.wait()) waiting forever - the whole app would hang
        # without any message. So log it, return, and let the GUI thread
        # show the dialog once it is back in its event loop.
        logger.error('Unhandled exception in a background thread:\n%s',
                     exception)
        error_presenter.show.emit(exception)
        return
    show_error_box(exception)


class ErrorPresenter(QtCore.QObject):
    """
    Shows error dialogs in the GUI thread for exceptions raised in other
    threads: a signal emitted from there is delivered to this object
    (which lives in the thread that created it, the main thread) when
    that thread is back in its event loop.
    """
    show = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.show.connect(self._show)

    @staticmethod
    def _show(text):
        show_error_box(text)


error_presenter = ErrorPresenter()


def show_error_box(exception):
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
