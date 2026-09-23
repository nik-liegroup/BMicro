import logging
import multiprocessing as mp

import numpy as np

from PyQt6 import QtWidgets
from PyQt6.QtCore import QObject, QTimer, QThread, pyqtSignal
from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QPushButton, \
    QProgressBar, QLabel, QTableWidget, QTableWidgetItem, QHeaderView

from bmlab.session import Session
from bmlab.controllers import BackgroundController

logger = logging.getLogger(__name__)


class Worker(QObject):
    finished = pyqtSignal()

    def __init__(self, fkw):
        super().__init__()
        self.background_controller = BackgroundController()
        self.fkw = fkw

    def run(self):
        self.background_controller.evaluate(**self.fkw)
        self.finished.emit()


class BackgroundView(QtWidgets.QWidget):
    """
    Fits the repetition's background reference points
    (BrillouinAcquisition's second, independent ROI - real spectra
    captured after the main grid, e.g. to sanity-check against a known
    sample like water - see bmlab.file.Background) against the same
    calibration/extraction/peak-selection setup already configured on
    the other tabs (bmlab.controllers.BackgroundController.evaluate()
    reuses them exactly as EvaluationController does for the main
    grid), and lists the fitted Brillouin shift and fit-quality
    diagnostics for every point in a flat table - there is no spatial
    grid to lay these out on as an image (see BackgroundController.
    get_data()'s own docstring), unlike the Evaluation tab's 2D/3D map.

    Built without a Designer .ui file (unlike this app's other tabs):
    everything here is a single button + progress bar + table, simple
    enough that a plain programmatic layout is clearer than a separate
    XML resource.
    """

    # (BackgroundModel parameter key, column header) - 'key'/'x'/'y'/'z'
    # aren't BackgroundModel.results entries, handled specially in
    # refresh_table() below.
    COLUMNS = [
        ('key', 'Point'),
        ('x', 'x [µm]'),
        ('y', 'y [µm]'),
        ('z', 'z [µm]'),
        ('brillouin_shift_f', 'Brillouin shift [GHz]'),
        ('brillouin_shift_f_stokes_anti_stokes',
         'Brillouin shift S-AS [GHz]'),
        ('rayleigh_peak_position_f', 'Rayleigh position [GHz]'),
        ('brillouin_peak_snr', 'Brillouin SNR'),
        ('rayleigh_peak_snr', 'Rayleigh SNR'),
    ]

    def __init__(self, *args, **kwargs):
        super(BackgroundView, self).__init__(*args, **kwargs)

        self.session = Session.get_instance()
        self.background_controller = BackgroundController()

        self.evaluation_running = False
        self.evaluation_abort = mp.Value('I', False, lock=True)
        self.count = mp.Value('I', 0, lock=True)
        self.max_count = mp.Value('i', -1, lock=True)
        self.thread = None
        self.worker = None

        self.evaluation_timer = QTimer()
        self.evaluation_timer.timeout.connect(self.refresh_ui)

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)

        button_row = QHBoxLayout()
        self.button_evaluate = QPushButton('Evaluate background points')
        self.button_evaluate.clicked.connect(self.evaluate)
        button_row.addWidget(self.button_evaluate)
        self.label_status = QLabel('No file open.')
        button_row.addWidget(self.label_status)
        button_row.addStretch()
        layout.addLayout(button_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.table = QTableWidget()
        self.table.setColumnCount(len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(
            [label for _, label in self.COLUMNS])
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table)

        self.setLayout(layout)

    def reset_ui(self):
        self.evaluation_timer.stop()
        self.evaluation_running = False
        self.button_evaluate.setText('Evaluate background points')
        self.button_evaluate.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.label_status.setText('No file open.')
        self.table.setRowCount(0)

    def update_ui(self):
        if self.session.file is None or self.session.background_model() \
                is None:
            self.button_evaluate.setEnabled(False)
            self.label_status.setText('No file open.')
            self.table.setRowCount(0)
            return

        has_points = bool(self.session.get_background_keys())
        self.button_evaluate.setEnabled(
            has_points and not self.evaluation_running)
        self.refresh_table()

    def evaluate(self):
        if self.session.file is None:
            return
        if self.evaluation_running:
            self.evaluation_abort.value = True
            return

        background_keys = self.session.get_background_keys()
        if not background_keys:
            self.label_status.setText(
                'This repetition has no background reference points.')
            return

        self.evaluation_abort.value = False
        self.evaluation_running = True
        self.button_evaluate.setText('Cancel')
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.label_status.setText('Evaluating background points...')

        self.count = mp.Value('I', 0, lock=True)
        self.max_count = mp.Value('i', len(background_keys), lock=True)

        dnkw = {
            'count': self.count,
            'max_count': self.max_count,
            'abort': self.evaluation_abort,
        }

        self.thread = QThread()
        self.worker = Worker(fkw=dnkw)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.finished.connect(self.refresh_ui)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

        self.evaluation_timer.start(500)

    def refresh_ui(self):
        # If evaluation is aborted by user, couldn't start, or is
        # finished, we stop the timer - same convention as
        # EvaluationView.refresh_ui().
        if self.evaluation_abort.value or \
                self.max_count.value < 0 or \
                self.count.value >= self.max_count.value:
            self.evaluation_timer.stop()
            self.evaluation_running = False
            self.button_evaluate.setText('Evaluate background points')
            self.button_evaluate.setEnabled(True)
            self.progress_bar.setVisible(False)
            self.refresh_table()

        if self.max_count.value >= 0:
            self.progress_bar.setMaximum(self.max_count.value)
        self.progress_bar.setValue(self.count.value)

    def refresh_table(self):
        bgm = self.session.background_model()
        if bgm is None or not bgm.point_keys:
            self.table.setRowCount(0)
            has_points = bool(self.session.get_background_keys())
            self.label_status.setText(
                'No background points evaluated yet.' if has_points
                else 'This repetition has no background reference points.')
            return

        point_keys = bgm.point_keys
        self.table.setRowCount(len(point_keys))

        data_by_column = {}
        for key, _ in self.COLUMNS:
            if key in ('key', 'x', 'y', 'z'):
                continue
            data, _, _ = self.background_controller.get_data(key)
            data_by_column[key] = data

        for row, point_key in enumerate(point_keys):
            for col, (key, _) in enumerate(self.COLUMNS):
                if key == 'key':
                    text = point_key
                elif key in ('x', 'y', 'z'):
                    values = bgm.positions.get(key)
                    value = values[row] if values is not None else None
                    text = '' if value is None or np.isnan(value) \
                        else f'{value:.2f}'
                else:
                    data = data_by_column.get(key)
                    value = data[row] if data is not None else None
                    text = '' if value is None or np.isnan(value) \
                        else f'{value:.4f}'
                self.table.setItem(row, col, QTableWidgetItem(text))

        self.label_status.setText(
            f'{len(point_keys)} background point'
            f'{"s" if len(point_keys) != 1 else ""} evaluated.')
