from PyQt6 import QtWidgets, QtCore

from bmlab.session import Session

from bmicro.gui import extraction, calibration, peak_selection


STEP_EXTRACTION = 'extraction'
STEP_CALIBRATION = 'calibration'
STEP_PEAK_SELECTION = 'peak-selection'

STEP_TITLES = {
    STEP_EXTRACTION: 'Extraction',
    STEP_CALIBRATION: 'Calibration',
    STEP_PEAK_SELECTION: 'Peak selection',
}

STEP_VIEWS = {
    STEP_EXTRACTION: extraction.ExtractionView,
    STEP_CALIBRATION: calibration.CalibrationView,
    STEP_PEAK_SELECTION: peak_selection.PeakSelectionView,
}


def step_done(step, session=None):
    """
    Returns whether `step` is complete for the session's *current*
    repetition, i.e. whether a batch run can skip it.

    A repetition without any calibration/payload images has nothing to
    do and counts as done.
    """
    session = session or Session.get_instance()
    if session.current_repetition() is None:
        return True

    if step == STEP_PEAK_SELECTION:
        if not session.get_image_keys():
            return True
        pm = session.peak_selection_model()
        return bool(pm.get_brillouin_regions()
                    and pm.get_rayleigh_regions())

    calib_keys = session.get_calib_keys()
    if not calib_keys:
        return True

    if step == STEP_EXTRACTION:
        em = session.extraction_model()
        return all(len(em.get_points(key)) >= 3 for key in calib_keys)

    if step == STEP_CALIBRATION:
        # Every calibration needs its own fit; a repetition with only
        # some of them fitted is still unfinished.
        cm = session.calibration_model()
        return all(key in cm.frequencies for key in calib_keys)

    raise ValueError(step)


def calibration_gaps(calib_keys, extraction_model, calibration_model):
    """
    Returns (number of calibrations, number without an arc, number
    without a fit) for one repetition. Such a repetition can still be
    evaluated - the arc and the frequency axis are then interpolated
    from the remaining calibrations - but less accurately.
    """
    no_arc = sum(1 for key in calib_keys
                 if len(extraction_model.get_points(key)) < 3)
    no_fit = sum(1 for key in calib_keys
                 if key not in calibration_model.frequencies)
    return len(calib_keys), no_arc, no_fit


def calibration_gaps_text(total, no_arc, no_fit):
    """ Warning text for calibration_gaps(), or None if complete. """
    parts = []
    if no_arc:
        parts.append(f'{no_arc}/{total} calibrations without arc')
    if no_fit:
        parts.append(f'{no_fit}/{total} calibrations not fitted')
    return ', '.join(parts) or None


# Redoing a step invalidates everything built on it.
DOWNSTREAM_STEPS = {
    STEP_EXTRACTION: (STEP_CALIBRATION, STEP_PEAK_SELECTION),
    STEP_CALIBRATION: (STEP_PEAK_SELECTION,),
    STEP_PEAK_SELECTION: (),
}


def _has_work(step, session):
    """ Whether the current repetition has images to do `step` on. """
    if session.current_repetition() is None:
        return False
    if step == STEP_PEAK_SELECTION:
        return bool(session.get_image_keys())
    return bool(session.get_calib_keys())


def pending_repetitions(step, include_done=False, forced=(), allowed=None):
    """
    Returns the keys of all repetitions of the open file for which
    `step` is not done yet - or, with `include_done`, of every
    repetition that has something to work on for the step. Repetitions
    in `forced` (those whose upstream step was just redone) count as
    not done. With `allowed`, only those repetition keys are considered.
    Leaves the current repetition unchanged.
    """
    session = Session.get_instance()
    if session.file is None:
        return []
    previous = session.current_repetition_key()
    pending = []
    try:
        for rep_key in session.file.repetition_keys():
            if allowed is not None and rep_key not in allowed:
                continue
            session.set_current_repetition(rep_key)
            if not step_done(step, session) or (
                    (include_done or rep_key in forced)
                    and _has_work(step, session)):
                pending.append(rep_key)
    finally:
        session.set_current_repetition(previous)
    return pending


class BatchStepDialog(QtWidgets.QDialog):
    """
    Hosts one of the regular analysis views (extraction, calibration or
    peak selection) for a single file of a batch run.

    Closing the dialog - or clicking "Save and next file" - keeps the
    work and moves the batch on; "Stop batch" additionally sets
    `stop_requested` so the caller ends the run.
    """

    def __init__(self, parent, step, file_path, position, total,
                 repetition_keys):
        super().__init__(
            parent,
            QtCore.Qt.WindowType.WindowTitleHint |
            QtCore.Qt.WindowType.WindowCloseButtonHint |
            QtCore.Qt.WindowType.WindowMaximizeButtonHint)
        self.stop_requested = False
        self.setWindowTitle(
            f'Batch {STEP_TITLES[step].lower()} - file {position}/{total}')
        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        self.resize(1200, 800)

        layout = QtWidgets.QVBoxLayout(self)

        header = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel(str(file_path))
        label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        header.addWidget(label, 1)
        header.addWidget(QtWidgets.QLabel('Repetition:'))
        self.combobox_repetition = QtWidgets.QComboBox()
        for rep_key in repetition_keys:
            self.combobox_repetition.addItem(str(rep_key), rep_key)
        header.addWidget(self.combobox_repetition)
        layout.addLayout(header)

        self.view = STEP_VIEWS[step](self)
        layout.addWidget(self.view, 1)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        button_stop = QtWidgets.QPushButton('Stop batch')
        button_stop.clicked.connect(self.on_stop)
        button_next = QtWidgets.QPushButton('Save and next file')
        button_next.setDefault(True)
        button_next.clicked.connect(self.accept)
        buttons.addWidget(button_stop)
        buttons.addWidget(button_next)
        layout.addLayout(buttons)

        self.combobox_repetition.currentIndexChanged.connect(
            self.on_repetition_changed)
        self.on_repetition_changed()

    def on_repetition_changed(self):
        rep_key = self.combobox_repetition.currentData()
        if rep_key is None:
            return
        Session.get_instance().set_current_repetition(rep_key)
        self.view.update_ui()

    def on_stop(self):
        self.stop_requested = True
        self.accept()
