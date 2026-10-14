from importlib import resources
import logging

from PyQt6 import uic, QtWidgets, QtCore
import matplotlib

from bmlab.models.setup import AVAILABLE_SETUPS
from bmlab.session import Session
from bmlab.controllers import PeakSelectionController, EvaluationController

from bmicro.gui.mpl import MplCanvas
from bmicro.gui.base_settings import has_analysis, reset_analysis


import os

matplotlib.use('Qt5Agg')
logger = logging.getLogger(__name__)


class DataView(QtWidgets.QWidget):
    """
    Class for the data widget
    """

    def __init__(self, *args, **kwargs):
        super(DataView, self).__init__(*args, **kwargs)

        ref = resources.files('bmicro.gui.data') / 'data_view.ui'
        with resources.as_file(ref) as ui_file:
            uic.loadUi(ui_file, self)

        self.parent = args[0]
        # True while update_ui() copies the session state into the
        # controls - their change handlers must not ask anything then
        self._syncing = False

        self.mplcanvas = MplCanvas(self.image_preview_widget)
        self.preview = self.mplcanvas.get_figure().add_subplot(111)
        self.preview.axis('off')

        self.radio_rotation_none.clicked.connect(self.on_rotation_clicked)
        self.radio_rotation_90_cw.clicked.connect(self.on_rotation_clicked)
        self.radio_rotation_90_ccw.clicked.connect(self.on_rotation_clicked)

        self.checkbox_reflect_vertically.toggled.connect(
            self.on_reflection_clicked)
        self.checkbox_reflect_horizontally.toggled.connect(
            self.on_reflection_clicked)

        self.comboBox_repetition.currentIndexChanged.connect(
            self.on_select_repetition)

        self.combobox_setup.addItems([s.name for s in AVAILABLE_SETUPS])
        self.combobox_setup.currentIndexChanged.connect(
            self.on_select_setup)

        self.auto_evaluation.clicked.connect(self.on_auto_evaluation)

        # Initialize current setup to first entry
        Session.get_instance().set_setup(AVAILABLE_SETUPS[0])

        self.update_ui()

    def update_ui(self):
        session = Session.get_instance()
        if not session.file:
            return

        filename = str(os.path.basename(session.file.path))
        self.label_selected_file.setText(filename)
        self.label_selected_file.setToolTip(str(session.file.path))
        self.label_selected_file.adjustSize()
        rep_keys = session.file.repetition_keys()
        # Update repetition keys if they have changed
        current_keys = [self.comboBox_repetition.itemText(i)
                        for i in range(self.comboBox_repetition.count())]
        if current_keys != rep_keys:
            self.comboBox_repetition.clear()
            self.comboBox_repetition.addItems(rep_keys)

        if rep_keys and self.comboBox_repetition.currentText():
            repetition = session.current_repetition()
            res = repetition.payload.resolution
            if res:
                self.label_resolution_x.setText(str(res[0]))
                self.label_resolution_y.setText(str(res[1]))
                self.label_resolution_z.setText(str(res[2]))
            date = repetition.date.strftime('%Y-%m-%d %H:%M')
            self.label_date.setText(date)
            self.textedit_comment.setText(session.file.comment)
            self.label_calibration.setText(
                str(not repetition.calibration.is_empty()))

        self._syncing = True
        try:
            if session.setup is not None:
                self.combobox_setup.setCurrentText(session.setup.name)
            self._show_orientation()
        finally:
            self._syncing = False

        self.update_preview()

    def _show_orientation(self):
        """ Shows the session's orientation in the controls. """
        session = Session.get_instance()
        self.checkbox_reflect_vertically.setChecked(
            session.orientation.reflection['vertically'])
        self.checkbox_reflect_horizontally.setChecked(
            session.orientation.reflection['horizontally'])

        self.radio_rotation_none.setChecked(
            session.orientation.rotation % 4 == 0)
        self.radio_rotation_90_cw.setChecked(
            session.orientation.rotation % 4 == 1)
        self.radio_rotation_90_ccw.setChecked(
            session.orientation.rotation % 4 == 3)

    def _repetitions_with_analysis(self):
        session = Session.get_instance()
        if session.file is None:
            return []
        return [key for key in session.file.repetition_keys()
                if has_analysis(session, key)]

    def _confirm_base_change(self, what, first_reset):
        """
        Setup and orientation are the base of the analysis: changing
        them makes what was built on them outdated, which is reset. If
        there is anything analysed, asks for confirmation. Returns the
        repetitions to reset ([] if none), or None if the user cancels.
        """
        keys = self._repetitions_with_analysis()
        if not keys:
            return []
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        box.setWindowTitle(f'Change {what}?')
        box.setText(
            f'Changing the {what} makes the existing analysis of '
            f'repetition(s) {", ".join(keys)} outdated. It is reset from '
            f'{first_reset} on: everything built on it has to be done '
            f'again.')
        box.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok
                               | QtWidgets.QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        if box.exec() != QtWidgets.QMessageBox.StandardButton.Ok:
            return None
        return keys

    def _reset_after_base_change(self, keys, from_extraction):
        """ Resets the analysis of `keys` and updates all tabs. """
        if not keys:
            return
        reset_analysis(Session.get_instance(), keys, from_extraction)
        if hasattr(self.parent, 'update_ui'):
            self.parent.update_ui()

    def reset_ui(self):
        """
        Reset the tab to the state when no file is loaded.
        """
        self.label_selected_file.setText('')
        self.label_selected_file.setToolTip('')
        self.comboBox_repetition.clear()
        self.label_date.setText('')
        self.label_resolution_x.setText('')
        self.label_resolution_y.setText('')
        self.label_resolution_z.setText('')
        self.label_calibration.setText('')
        self.textedit_comment.setText('')
        self.combobox_setup.setCurrentText(AVAILABLE_SETUPS[0].name)
        self.radio_rotation_none.setChecked(True)
        self.checkbox_reflect_vertically.setChecked(False)
        self.checkbox_reflect_horizontally.setChecked(False)
        self.update_preview()

    def on_auto_evaluation(self):
        session = Session.get_instance()
        if session.file is None:
            return

        rep_keys = session.file.repetition_keys()

        for rep_key in rep_keys:
            # Load repetition
            session.set_current_repetition(rep_key)
            QtCore.QCoreApplication.instance().processEvents()

            # Setup
            session.set_setup(AVAILABLE_SETUPS[0])

            # Orientation
            session.set_rotation(0)
            session.set_reflection(
                vertically=False,
                horizontally=True
            )
            self.update_ui()

            # Extraction
            self.parent.tabWidget.setCurrentIndex(1)
            QtCore.QCoreApplication.instance().processEvents()
            self.parent.widget_extraction_view.find_points_all()

            # Calibration
            self.parent.tabWidget.setCurrentIndex(2)
            QtCore.QCoreApplication.instance().processEvents()
            self.parent.widget_calibration_view.calibrate_all()

            # PeakSelection
            self.parent.tabWidget.setCurrentIndex(3)
            QtCore.QCoreApplication.instance().processEvents()
            psc = PeakSelectionController()
            psc.add_brillouin_region_frequency((4.0e9, 6.0e9))
            psc.add_brillouin_region_frequency((9.0e9, 11.0e9))
            psc.add_rayleigh_region_frequency((-2.0e9, 2.0e9))
            psc.add_rayleigh_region_frequency((13.0e9, 17.0e9))
            self.parent.widget_peak_selection_view.update_ui()
            QtCore.QCoreApplication.instance().processEvents()

            # Evaluation
            self.parent.tabWidget.setCurrentIndex(4)
            QtCore.QCoreApplication.instance().processEvents()
            evc = EvaluationController()
            evc.set_nr_brillouin_peaks(1)
            self.parent.widget_evaluation_view.evaluate(
                blocking=True, notify_blocked=False)

    def on_rotation_clicked(self):
        """
        Action triggered when user clicks one of the rotation radio buttons.
        """

        radio_button = self.sender()
        if not radio_button.isChecked():
            return

        session = Session.get_instance()

        if radio_button == self.radio_rotation_none:
            rotation = 0
        elif radio_button == self.radio_rotation_90_cw:
            rotation = 1
        elif radio_button == self.radio_rotation_90_ccw:
            rotation = 3
        else:
            return

        reset = []
        if rotation != session.orientation.rotation % 4:
            reset = self._confirm_base_change('orientation', 'extraction')
            if reset is None:  # cancelled: back to the current rotation
                self._show_orientation()
                return
        session.set_rotation(rotation)
        self._reset_after_base_change(reset, True)

        self.update_preview()

    def on_reflection_clicked(self):
        """ Triggered when a reflection checkbox is clicked """

        checkbox = self.sender()

        session = Session.get_instance()
        if checkbox == self.checkbox_reflect_vertically:
            axis = 'vertically'
        elif checkbox == self.checkbox_reflect_horizontally:
            axis = 'horizontally'
        else:
            return

        reset = []
        changed = checkbox.isChecked() != bool(
            session.orientation.reflection[axis])
        if changed and not self._syncing:
            reset = self._confirm_base_change('orientation', 'extraction')
            if reset is None:  # cancelled: back to the current state
                checkbox.blockSignals(True)
                checkbox.setChecked(not checkbox.isChecked())
                checkbox.blockSignals(False)
                return
        session.set_reflection(**{axis: checkbox.isChecked()})
        self._reset_after_base_change(reset, True)

        self.update_preview()

    def update_preview(self):
        """
        Updates the preview plot based on current session settings.
        """
        session = Session.get_instance()
        rep = session.current_repetition()

        if rep and rep.payload.image_keys():
            first_key = rep.payload.image_keys()[0]
            img = session.get_payload_image(first_key, 0)

            self.preview.clear()

            # imshow should always get the transposed image such that
            # the horizontal axis of the plot coincides with the
            # 0-axis of the plotted array:
            self.preview.imshow(img.T, origin='lower', vmin=100, vmax=300)
            self.preview.axis('off')
            self.mplcanvas.draw()
        else:
            self.mplcanvas.get_figure().clf()
            self.preview = self.mplcanvas.get_figure().add_subplot(111)
            self.mplcanvas.draw()

    def on_select_repetition(self):
        """
        Action triggered when the user selects a different repetition.
        """
        session = Session.get_instance()
        rep_key = self.comboBox_repetition.currentText()
        session.set_current_repetition(rep_key)
        self.update_ui()

    def on_select_setup(self):
        """
        Action triggered when the user selects a different setup.
        """
        name = self.combobox_setup.currentText()
        setup = None
        session = Session.get_instance()
        for s in AVAILABLE_SETUPS:
            if s.name == name:
                setup = s
                break
        reset = []
        if not self._syncing and session.setup is not None \
                and getattr(setup, 'key', None) != session.setup.key:
            reset = self._confirm_base_change('setup', 'calibration')
            if reset is None:  # cancelled: back to the current setup
                self.combobox_setup.blockSignals(True)
                self.combobox_setup.setCurrentText(session.setup.name)
                self.combobox_setup.blockSignals(False)
                return
        session.set_setup(setup)
        self._reset_after_base_change(reset, False)
