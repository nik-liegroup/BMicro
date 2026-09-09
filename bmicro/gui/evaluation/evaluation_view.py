from importlib import resources
import logging
import numpy as np
import matplotlib
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import warnings

import time

from PyQt6 import QtWidgets, uic, QtCore
from PyQt6.QtCore import QObject, QTimer, QThread, pyqtSignal, QCoreApplication
import multiprocessing as mp

from bmlab.session import Session
from bmlab.fits import lorentz

from bmicro.gui.mpl import MplCanvas

from bmlab.controllers import EvaluationController

logger = logging.getLogger(__name__)


class Worker(QObject):
    finished = pyqtSignal()

    def __init__(self, fkw):
        super().__init__()
        self.evaluation_controller = EvaluationController()
        self.fkw = fkw

    def run(self):
        self.evaluation_controller.evaluate(**self.fkw)
        self.finished.emit()


class EvaluationView(QtWidgets.QWidget):
    """
    Class for the evaluation widget
    """

    def __init__(self, *args, **kwargs):
        super(EvaluationView, self).__init__(*args, **kwargs)

        ref = resources.files('bmicro.gui.evaluation') / 'evaluation_view.ui'
        with resources.as_file(ref) as ui_file:
            uic.loadUi(ui_file, self)

        self.mplcanvas = MplCanvas(self.image_widget,
                                   toolbar=('Home', 'Pan', 'Zoom'))
        self.mplcanvas.get_figure().canvas.mpl_connect(
            'button_press_event', self.on_click_image)
        self.plot = self.mplcanvas.get_figure().add_subplot(111)
        self.image_map = None
        self.colorbar = None

        self.image_spectrum_dialog = None
        self.isd_image_canvas = None
        self.isd_image_plot = None
        self.isd_image_map = None
        self.isd_image_colorbar = None
        self.isd_spectrum_canvas = None
        self.isd_spectrum_plot = None

        self.button_evaluate.released.connect(self.evaluate)

        self.setup_parameter_selection_combobox()

        self.combobox_parameter.currentIndexChanged.connect(
            self.on_select_parameter)
        self.combobox_peak_number.currentIndexChanged.connect(
            self.on_select_parameter)

        self.aspect_ratio.clicked.connect(
            self.refresh_plot)

        # 3D data can be viewed either as a single 2D z-slice (paged
        # through with a slider) or as actual 3D cubes at the real
        # measurement positions - see refresh_plot(). Both widgets are
        # hidden for non-3D data.
        self.show_3d = False
        self.z_slice_index = 0
        self.button_toggle_3d.setVisible(False)
        self.button_toggle_3d.clicked.connect(self.on_toggle_3d_view)
        self.z_slider.setVisible(False)
        self.label_z_slider.setVisible(False)
        self.label_z_value.setVisible(False)
        self.z_slider.valueChanged.connect(self.on_z_slider_changed)
        self.checkbox_transparent_blocks.setVisible(False)
        self.checkbox_transparent_blocks.clicked.connect(self.refresh_plot)

        # Set by refresh_plot() when a 2D image (a genuine 2D dataset or
        # a single z-slice of a 3D one) is drawn, so on_click_image()
        # knows how to map a click back onto the data/positions arrays.
        self._click_context = None

        self.autoscale.clicked.connect(
            self.on_scale_changed)
        self.ignore_outliers.clicked.connect(
            self.on_scale_changed)
        self.value_min.valueChanged.connect(
            self.on_scale_changed)
        self.value_max.valueChanged.connect(
            self.on_scale_changed)

        self.evaluation_controller = EvaluationController()

        self.evaluation_abort = mp.Value('I', False, lock=True)
        self.evaluation_running = False

        self.session = Session.get_instance()

        self.evaluation_timer = QTimer()
        self.evaluation_timer.timeout.connect(self.refresh_ui)
        self.count = None
        self.max_count = None
        self.thread = None
        self.worker = None
        # Currently used to determine if we should update the plot
        # Might not be necessary anymore once the plot is fast enough.
        self.plot_count = 0

        self.bounds_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.bounds_table.verticalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.bounds_table.cellChanged.connect(self.boundsChanged)

        self.nrBrillouinPeaks_1.toggled.connect(
            lambda: self.setNrBrillouinPeaks(1))
        self.nrBrillouinPeaks_2.toggled.connect(
            lambda: self.setNrBrillouinPeaks(2))
        self.nrBrillouinPeaks_4.toggled.connect(
            lambda: self.setNrBrillouinPeaks(4))

    def update_ui(self):
        session = Session.get_instance()
        evm = session.evaluation_model()
        if evm is None:
            return

        if evm.nr_brillouin_peaks == 1:
            self.nrBrillouinPeaks_1.setChecked(True)
            self.bounds_table.setEnabled(False)
        elif evm.nr_brillouin_peaks == 2:
            self.nrBrillouinPeaks_2.setChecked(True)
            self.bounds_table.setEnabled(True)
        elif evm.nr_brillouin_peaks == 4:
            self.nrBrillouinPeaks_4.setChecked(True)
            self.bounds_table.setEnabled(True)

        self.updateBoundsTable()
        self.setup_parameter_selection_combobox()
        self.refresh_plot()

    def updateBoundsTable(self):
        session = Session.get_instance()
        evm = session.evaluation_model()
        if evm is None:
            self.bounds_table.setRowCount(0)
            self.bounds_table.setEnabled(False)
            return
        bounds_w0 = evm.bounds_w0
        bounds_fwhm = evm.bounds_fwhm
        if bounds_w0 is None or bounds_fwhm is None:
            self.bounds_table.setRowCount(0)
        else:
            self.bounds_table.setColumnCount(4)
            self.bounds_table.setRowCount(len(bounds_w0))
            for i, bound in enumerate(bounds_w0):
                item = QtWidgets.QTableWidgetItem(str(bound[0]))
                self.bounds_table.setItem(i, 0, item)
                item = QtWidgets.QTableWidgetItem(str(bound[1]))
                self.bounds_table.setItem(i, 1, item)
            for i, bound in enumerate(bounds_fwhm):
                item = QtWidgets.QTableWidgetItem(str(bound[0]))
                self.bounds_table.setItem(i, 2, item)
                item = QtWidgets.QTableWidgetItem(str(bound[1]))
                self.bounds_table.setItem(i, 3, item)

        self.bounds_table.setEnabled(evm.nr_brillouin_peaks > 1)

    def on_click_image(self, event):
        """
        Action triggered when user clicks Brillouin map.

        Parameters
        ----------
        event: matplotlib event object
            The mouse click event.
        """
        # If the click is outside the plot axes, skip it (this also
        # rejects clicks on the colorbar, which has its own axes)
        if event.inaxes is not self.plot:
            return
        # If we don't have a session we have not loaded data yet
        session = Session.get_instance()
        if session is None:
            return
        # Only available while a 1D/2D plot is shown - see refresh_plot().
        # For a 2D slice of a 3D dataset this also covers the out-of-plane
        # (e.g. z) axis, which is pinned to the currently displayed index.
        if self._click_context is None:
            return

        # Get the resolution of the measurement
        resolution = self.session.get_payload_resolution()
        if resolution is None:
            return

        # Get the positions. x and y are centered around zero (they are
        # relative sample coordinates), but z is left as the actual
        # acquired stage position.
        positions = list(self.session.get_payload_positions().values())
        for position in positions[:2]:
            position -= np.nanmean(position)

        idx = self._click_context['idx']
        dslice = self._click_context['dslice']

        # Start from the indices of the currently displayed slice (this
        # carries over any axis - e.g. z - that is fixed rather than
        # shown in-plane), then determine the click position along each
        # in-plane axis.
        click_pos = (event.xdata, event.ydata)
        indices = np.array(
            [0 if isinstance(v, slice) else v for v in dslice],
            dtype="int"
        )
        for ind, p_ind in enumerate(idx):
            pslice = [dslice[i] if i not in idx
                      else (slice(None) if i == p_ind else 0)
                      for i in range(len(resolution))]
            r = abs(positions[p_ind][tuple(pslice)] - click_pos[ind])
            indices[p_ind] = int(np.argmin(r))

        # Convert indices to key
        image_key = self.evaluation_controller\
            .get_key_from_indices(resolution, *indices)

        # If the modal is not open yet, open it
        self.open_image_spectrum()

        # Show position and key
        self.image_spectrum_dialog.image_spectrum_label.setText(
            "Position "
            f"x: {positions[0][tuple(indices)]:#0.1f} [µm], "
            f"y: {positions[1][tuple(indices)]:#0.1f} [µm], "
            f"z: {positions[2][tuple(indices)]:#0.1f} [µm], "
            f"image key: {image_key}"
        )

        # Get the image plot it
        image = session.get_payload_image(image_key, 0)
        if image is not None:
            if isinstance(self.isd_image_map, matplotlib.image.AxesImage):
                self.isd_image_map.set_data(image.T)
            else:
                self.isd_image_map = self.isd_image_plot.imshow(
                    image.T, origin='lower', vmin=100, vmax=300
                )
                self.isd_image_colorbar = \
                    self.isd_image_canvas\
                        .get_figure().colorbar(self.isd_image_map)
        self.isd_image_canvas.draw()

        # Get spectrum and plot it
        self.isd_spectrum_plot.cla()
        spectra = session.evaluation_model().get_spectra(image_key)
        cm = session.calibration_model()
        payload_time = session.get_payload_time(image_key)
        frequencies = cm.get_frequencies_by_time(payload_time)

        if spectra is not None and frequencies is not None:
            image_nr = 0
            spectrum = np.nanmean(spectra, image_nr)

            # Also try to get the fit
            brillouin_fits, rayleigh_fits =\
                self.evaluation_controller.get_fits(image_key)

            # Show the measured data
            self.isd_spectrum_plot.plot(1e-9 * frequencies[0],
                                        spectrum, color='tab:blue')
            self.isd_spectrum_plot.set_xlabel('$f$ [GHz]')

            pm = session.peak_selection_model()
            evm = session.evaluation_model()
            if pm is not None and evm is not None:
                # Show the Brillouin peaks
                brillouin_regions = pm.get_brillouin_regions()
                # Iterate over the regions
                for region_nr in range(brillouin_fits[0].shape[1]):
                    x = np.linspace(
                        brillouin_regions[region_nr][0],
                        brillouin_regions[region_nr][1],
                        200
                    )
                    # First entry is always a single-peak fit,
                    # the following entries belong to a multi-peak fit
                    for peak_nr in range(brillouin_fits[0].shape[2]):
                        idx = (image_nr, region_nr, peak_nr)
                        current_fit = lorentz(
                            x,
                            brillouin_fits[0][idx],
                            brillouin_fits[1][idx],
                            brillouin_fits[2][idx]
                        )
                        if peak_nr < 2:
                            y = current_fit
                        else:
                            y += current_fit
                        if peak_nr == 0:
                            color = 'tab:red'
                        else:
                            color = 'tab:orange'
                        # We plot the fit for the first and last entry
                        if peak_nr == 0 or\
                                peak_nr == (brillouin_fits[0].shape[2] - 1):
                            self.isd_spectrum_plot.plot(
                                1e-9 * x,
                                y + brillouin_fits[3][idx],
                                color=color
                            )

                # Show the Rayleigh peaks
                rayleigh_regions = pm.get_rayleigh_regions()
                # Iterate over the regions
                for region_nr in range(rayleigh_fits[0].shape[1]):
                    x = np.linspace(
                        rayleigh_regions[region_nr][0],
                        rayleigh_regions[region_nr][1],
                        200
                    )
                    idx = (image_nr, region_nr, 0)
                    y = lorentz(
                        x,
                        rayleigh_fits[0][idx],
                        rayleigh_fits[1][idx],
                        rayleigh_fits[2][idx]
                    ) + rayleigh_fits[3][idx]
                    self.isd_spectrum_plot.plot(1e-9 * x,
                                                y, color='tab:purple')

        self.isd_spectrum_canvas.draw()

    def open_image_spectrum(self):
        if self.image_spectrum_dialog is None:
            self.image_spectrum_dialog = QtWidgets.QDialog(
                self,
                QtCore.Qt.WindowType.WindowTitleHint |
                QtCore.Qt.WindowType.WindowCloseButtonHint |
                QtCore.Qt.WindowType.WindowMaximizeButtonHint |
                QtCore.Qt.WindowType.WindowMinimizeButtonHint
            )
            ref = resources.files('bmicro.gui.evaluation') / 'spectrum_view.ui'
            with resources.as_file(ref) as ui_file:
                uic.loadUi(ui_file, self.image_spectrum_dialog)
            self.image_spectrum_dialog\
                .setWindowTitle('Camera image & spectrum')
            self.image_spectrum_dialog.setWindowModality(
                QtCore.Qt.WindowModality.NonModal)

            self.image_spectrum_dialog.show()

            self.isd_image_canvas = MplCanvas(
                self.image_spectrum_dialog.image_widget,
                toolbar=('Home', 'Pan', 'Zoom'))
            self.isd_image_plot =\
                self.isd_image_canvas.get_figure().add_subplot(111)
            self.isd_image_map = None
            self.isd_image_colorbar = None

            self.isd_spectrum_canvas = MplCanvas(
                self.image_spectrum_dialog.spectrum_widget,
                toolbar=('Home', 'Pan', 'Zoom'))
            self.isd_spectrum_plot =\
                self.isd_spectrum_canvas.get_figure().add_subplot(111)

        if self.image_spectrum_dialog.isVisible() is False:
            self.image_spectrum_dialog.setVisible(True)

    def setNrBrillouinPeaks(self, nr_brillouin_peaks):
        if not self.sender().isChecked():
            return
        session = Session.get_instance()
        evm = session.evaluation_model()
        if evm is None:
            return
        evm.setNrBrillouinPeaks(nr_brillouin_peaks)
        self.combobox_peak_number.setEnabled(nr_brillouin_peaks > 1)
        self.combobox_peak_number.blockSignals(True)
        self.combobox_peak_number.clear()
        peak_number_labels = []
        if nr_brillouin_peaks == 1:
            peak_number_labels = ['Single-Peak-Fit']
        elif nr_brillouin_peaks == 2:
            peak_number_labels = [
                'Single-Peak-Fit',
                'Two-Peak-Fit - Peak 1',
                'Two-Peak-Fit - Peak 2',
                'Two-Peak-Fit - Mean',
                'Two-Peak-Fit - Weighted Mean'
            ]
        elif nr_brillouin_peaks == 4:
            peak_number_labels = [
                'Single-Peak-Fit',
                'Four-Peak-Fit - Peak 1',
                'Four-Peak-Fit - Peak 2',
                'Four-Peak-Fit - Peak 3',
                'Four-Peak-Fit - Peak 4',
                'Four-Peak-Fit - Mean',
                'Four-Peak-Fit - Weighted Mean'
            ]
        self.combobox_peak_number.addItems(peak_number_labels)
        self.combobox_peak_number.blockSignals(False)

        self.updateBoundsTable()

    def boundsChanged(self, row, column):
        session = Session.get_instance()
        evm = session.evaluation_model()
        if evm is None:
            return

        if evm.bounds_w0 is not None:
            if column < 2:
                evm.bounds_w0[row][column] =\
                    self.bounds_table.item(row, column).text()
            elif column < 4:
                evm.bounds_fwhm[row][column - 2] =\
                    self.bounds_table.item(row, column).text()

    def clear_plots(self):
        self._remove_image_map()
        self._remove_colorbar()

    def _remove_colorbar(self):
        # Guard against colorbar.ax already being gone from the figure
        # (e.g. a prior figure.clf() destroyed it without going through
        # this method) - Colorbar.remove() raises a KeyError from mpl's
        # own axes bookkeeping in that case instead of being a no-op.
        if isinstance(self.colorbar, matplotlib.colorbar.Colorbar) \
                and self.colorbar.ax in self.mplcanvas.get_figure().axes:
            try:
                self.colorbar.remove()
            except Exception:
                # Colorbar.remove() can also fail on internal state we
                # don't fully control: we deliberately reuse a colorbar
                # across refreshes via update_normal() (see
                # _update_colorbar) instead of always recreating it, to
                # keep the 3D view's axes position stable - and that
                # apparently leaves mpl's own bookkeeping inconsistent
                # in ways that show up as different exceptions (seen:
                # KeyError disconnecting an already-gone axes,
                # AttributeError disconnecting a mappable callback that
                # was never registered). Rather than chase every such
                # case, fall back to removing the axes directly - a
                # crash here would lose the user's session, whereas
                # worst case this leaves a stale colorbar visible until
                # the next full figure clear.
                logger.warning(
                    'Colorbar.remove() failed, removing its axes '
                    'directly instead', exc_info=True)
                try:
                    self.mplcanvas.get_figure().delaxes(self.colorbar.ax)
                except Exception:
                    logger.warning(
                        'Removing the colorbar axes directly also '
                        'failed', exc_info=True)
        self.colorbar = None

    def _remove_image_map(self):
        if isinstance(self.image_map, list):
            for m in self.image_map:
                m.remove()
            self.image_map = None
        if self.image_map is not None:
            self.image_map.remove()
            self.image_map = None

    def _update_colorbar(self, mappable, label):
        """
        Creates the colorbar on first use, and just updates the existing
        one afterwards. Removing and re-creating it on every refresh
        (as used to happen for the 3D block view, which has to redraw
        its Poly3DCollection from scratch on every update) shifts the
        main plot's axes position a little each time on Axes3D - mpl
        does not fully restore the pre-colorbar position on remove() -
        which is what caused the plot to keep resizing during live
        updates/fitting.
        """
        if isinstance(self.colorbar, matplotlib.colorbar.Colorbar):
            self.colorbar.update_normal(mappable)
        else:
            self.colorbar = self.mplcanvas.get_figure().colorbar(
                mappable, ax=self.plot)
        self.colorbar.ax.set_title(label)

    def reset_ui(self):
        self.evaluation_progress.setValue(0)
        self.show_3d = False
        self.button_toggle_3d.setText('Switch to 3D view')
        self.button_toggle_3d.setVisible(False)
        self.z_slider.setVisible(False)
        self.label_z_slider.setVisible(False)
        self.label_z_value.setVisible(False)
        self.checkbox_transparent_blocks.setVisible(False)
        self._click_context = None
        # A full figure.clf() + fresh subplot, rather than clear_plots()
        # + plot.cla(): the latter only clears what's *drawn* on the
        # existing Axes, but leaves it (and any layout/subplotspec state
        # matplotlib accumulated on it and the figure across repeated
        # colorbar/3D-view create-destroy cycles - see _update_colorbar
        # and the 3D<->2D toggle in refresh_plot) in place, ready to
        # carry stale internal state into whatever file gets opened
        # next. clf() and a brand new Axes sidesteps that class of
        # matplotlib-internal-state bugs entirely instead of chasing
        # each way it can surface (a KeyError, then an AttributeError,
        # then this - a NoneType with no set_subplotspec - all from the
        # same "reuse across transitions" optimization, just triggered
        # by different transitions). It's fine to fully rebuild here:
        # unlike the live-update case _update_colorbar optimizes for,
        # switching files means starting over is exactly right anyway.
        self.mplcanvas.get_figure().clf()
        self.plot = self.mplcanvas.get_figure().add_subplot(111)
        self.image_map = None
        self.colorbar = None
        self.updateBoundsTable()
        self.nrBrillouinPeaks_1.setChecked(True)
        # Blocked for the same reason as combobox_peak_number just
        # below: unblocked, clear() changes the current index and fires
        # currentIndexChanged -> on_select_parameter -> refresh_plot(),
        # re-entrantly, while the session still points at whatever file
        # was open before this reset - not the new one being switched
        # to (reset_ui() runs before that happens). That stray render
        # was creating a colorbar from stale data mid-teardown, which
        # is what later broke when the real refresh for the new file
        # tried to clean it up.
        self.combobox_parameter.blockSignals(True)
        self.combobox_parameter.clear()
        self.combobox_parameter.blockSignals(False)
        self.combobox_peak_number.setEnabled(False)
        self.combobox_peak_number.blockSignals(True)
        self.combobox_peak_number.clear()
        self.combobox_peak_number.addItems(['Single-Peak-Fit'])
        self.combobox_peak_number.blockSignals(False)

        self.mplcanvas.draw()

        if self.image_spectrum_dialog is not None\
                and self.image_spectrum_dialog.isVisible():
            self.image_spectrum_dialog.close()

    def setup_parameter_selection_combobox(self):

        session = Session.get_instance()
        evm = session.evaluation_model()
        if evm is None:
            return

        parameters = evm.get_parameter_keys()

        param_labels = []
        for key, parameter in parameters.items():
            param_labels.append(
                parameter['label'] + ' [' + parameter['unit'] + ']'
            )

        self.combobox_parameter.blockSignals(True)
        self.combobox_parameter.clear()
        self.combobox_parameter.addItems(param_labels)
        self.combobox_parameter.blockSignals(False)

    def on_select_parameter(self):
        self.refresh_plot()

    def on_scale_changed(self):
        autoscale = self.autoscale.isChecked()
        self.value_min.setDisabled(autoscale)
        self.value_max.setDisabled(autoscale)
        self.ignore_outliers.setDisabled(not autoscale)
        self.refresh_plot()

    def evaluate(self, blocking=False):
        # Check that a file is open
        if self.session.file is None:
            return
        # If the evaluation is already running, we abort it and reset
        #  the button label
        if self.evaluation_running:
            self.evaluation_abort.value = True
            self.refresh_ui()
            return

        self.evaluation_abort.value = False
        self.evaluation_running = True
        self.button_evaluate.setText('Cancel')
        # While the evaluation is running, we
        # disable switching to multi-peak fit and adjusting bounds
        self.nrBrillouinPeaksGroup.setEnabled(False)
        self.bounds_table.setEnabled(False)
        self.evaluation_timer.start(500)

        self.plot_count = 0
        self.count = mp.Value('I', 0, lock=True)

        # We have to initialize the value correctly here,
        # otherwise the evaluation might abort immediately
        # if max_count is set only after the evaluation_timer
        # triggered for the first time
        image_keys = self.session.get_image_keys()
        self.max_count = mp.Value('i', len(image_keys), lock=True)

        dnkw = {
            "count": self.count,
            "max_count": self.max_count,
            "abort": self.evaluation_abort,
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

        if blocking:
            while self.evaluation_running:
                QCoreApplication.instance().processEvents()
                time.sleep(0.1)

    def refresh_ui(self):
        # If evaluation is aborted by user,
        # couldn't start or is finished,
        # we stop the timer
        if self.evaluation_abort.value or\
           self.max_count.value < 0 or\
           self.count.value >= self.max_count.value:
            self.evaluation_timer.stop()
            self.evaluation_running = False
            self.button_evaluate.setText('Evaluate')
            self.nrBrillouinPeaksGroup.setEnabled(True)
            session = Session.get_instance()
            if session.evaluation_model().nr_brillouin_peaks > 1:
                self.bounds_table.setEnabled(True)
            self.refresh_plot()

        if self.max_count.value >= 0:
            self.evaluation_progress.setMaximum(self.max_count.value)
        self.evaluation_progress.setValue(self.count.value)

        # We refresh the image every thirty points to not slow down to much
        if (self.count.value - self.plot_count) > 30:
            self.plot_count = self.count.value
            self.refresh_plot()

    def refresh_plot(self):
        session = Session.get_instance()
        evm = session.evaluation_model()
        if evm is None:
            return

        parameters = evm.get_parameter_keys()

        parameter_index = self.combobox_parameter.currentIndex()
        brillouin_peak_index = self.combobox_peak_number.currentIndex()
        parameter_key = list(parameters.keys())[parameter_index]

        data, positions, dimensionality, labels =\
            self.evaluation_controller.\
            get_data(parameter_key, brillouin_peak_index)

        # Reset click-to-spectrum context; it is only set below when we
        # actually draw a 2D image (a genuine 2D dataset, or a single
        # z-slice of a 3D one) that clicking can be mapped onto.
        self._click_context = None

        # The current repetition has no valid measurement grid
        # (e.g. an aborted/restarted acquisition that never wrote
        # any positions) - there is nothing to plot.
        if dimensionality is None:
            # Full rebuild, not just clear_plots() + plot.cla() - see
            # the comment in reset_ui() for why: this is a similar
            # "starting over" transition (switching to a repetition
            # with no valid data), so the same stale-layout-state risk
            # applies here.
            self.mplcanvas.get_figure().clf()
            self.plot = self.mplcanvas.get_figure().add_subplot(111)
            self.image_map = None
            self.colorbar = None
            self.mplcanvas.draw()
            return

        # Center x and y around zero (relative sample coordinates);
        # z is left as the actual acquired stage position.
        for position in positions[:2]:
            position -= np.nanmean(position)

        self.button_toggle_3d.setVisible(bool(dimensionality == 3))
        self.checkbox_transparent_blocks.setVisible(
            bool(dimensionality == 3 and self.show_3d))

        # Check that we have the axes type the current mode needs.
        want_3d_axes = dimensionality == 3 and self.show_3d
        if want_3d_axes and not isinstance(self.plot, Axes3D):
            self.mplcanvas.get_figure().clf()
            self.plot = self.mplcanvas.\
                get_figure().add_subplot(111, projection='3d')
            # The old image_map/colorbar were destroyed by clf() above,
            # not just logically stale - don't try to .remove() them.
            self.image_map = None
            self.colorbar = None
        elif not want_3d_axes and isinstance(self.plot, Axes3D):
            # delaxes() only removes the 3D plot axes. The colorbar lives
            # in its own separate axes (created by fig.colorbar()) and
            # is untouched by that, so without removing it explicitly it
            # is left behind as a stale, second colorbar when we then
            # create a fresh one for the 2D view.
            self._remove_colorbar()
            self.mplcanvas.get_figure().delaxes(self.plot)
            self.plot = self.mplcanvas.\
                get_figure().add_subplot(111)
            self.image_map = None

        # Create the slices list
        dslice = [slice(None) if dim > 1 else 0 for dim in data.shape]
        idx = [idx for idx, dim in enumerate(data.shape) if dim > 1]

        # A 3D dataset is either shown as actual 3D cubes at the real
        # measurement positions (show_3d), or one 2D z-slice at a time
        # via the slider, rendered exactly like a regular 2D dataset
        # (see dimensionality == 2, further down).
        if dimensionality == 3 and not self.show_3d:
            # Slice along the last occurrence of the shortest dimension
            b = data.shape[::-1]
            z_axis = len(b) - np.argmin(b) - 1
            z_len = data.shape[z_axis]
            self.z_slice_index = min(self.z_slice_index, z_len - 1)
            self.z_slider.blockSignals(True)
            self.z_slider.setMinimum(0)
            self.z_slider.setMaximum(z_len - 1)
            self.z_slider.setValue(self.z_slice_index)
            self.z_slider.blockSignals(False)
            self.z_slider.setVisible(True)
            self.label_z_slider.setVisible(True)
            self.label_z_value.setVisible(True)

            dslice[z_axis] = self.z_slice_index
            idx = [i for i in idx if i != z_axis]
            z_value = np.nanmean(positions[z_axis][tuple(dslice)])
            self.label_z_value.setText('z = %.2f' % z_value)
        else:
            self.z_slider.setVisible(False)
            self.label_z_slider.setVisible(False)
            self.label_z_value.setVisible(False)

        # Remember how the plotted data maps onto the positions arrays,
        # so on_click_image() can look up the spectrum for a clicked
        # point/pixel. Covers a 1D line plot and a 2D image (which
        # includes a single z-slice of a 3D dataset, whose dslice/idx
        # were already pinned down to the in-plane axes above). Not set
        # for a single 0D point or the 3D cube view, where a screen
        # click doesn't map onto a single data point.
        if len(idx) in (1, 2):
            self._click_context = {'idx': tuple(idx), 'dslice': list(dslice)}

        try:
            if dimensionality == 0:
                # If this is a line plot already, just set new data
                if isinstance(self.image_map, list) and\
                        isinstance(self.image_map[0], matplotlib.lines.Line2D):
                    self.image_map[0].set_data(0, data[tuple(dslice)])
                else:
                    self.clear_plots()
                    self.image_map =\
                        self.plot.plot(0, data[tuple(dslice)], marker='x')
                self.plot.set_xlabel('')
                self.plot.set_title(parameters[parameter_key]['label'])
                ylabel = parameters[parameter_key]['symbol'] +\
                    ' [' + parameters[parameter_key]['unit'] + ']'
                self.plot.set_ylabel(ylabel)
                self.plot.axis('auto')
                self.plot.set_ylim(
                    tuple(
                        data[tuple(dslice)] * np.array([0.99, 1.01])
                    )
                )
            if dimensionality == 1:
                # If this is a line plot already, just set new data
                if isinstance(self.image_map, list) and\
                        isinstance(self.image_map[0], matplotlib.lines.Line2D):
                    self.image_map[0].set_data(
                        positions[idx[0]][tuple(dslice)],
                        data[tuple(dslice)]
                    )
                else:
                    self.clear_plots()
                    self.image_map = self.plot.plot(
                        positions[idx[0]][tuple(dslice)],
                        data[tuple(dslice)]
                    )
                self.plot.set_title(parameters[parameter_key]['label'])
                self.plot.set_xlabel(labels[idx[0]])
                ylabel = parameters[parameter_key]['symbol'] +\
                    ' [' + parameters[parameter_key]['unit'] + ']'
                self.plot.set_ylabel(ylabel)
                self.plot.axis('auto')
                minx = np.nanmin(positions[idx[0]][tuple(dslice)])
                maxx = np.nanmax(positions[idx[0]][tuple(dslice)])
                if minx < maxx:
                    self.plot.set_xlim(minx, maxx)
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        action='ignore',
                        message='All-NaN slice encountered'
                    )
                    (value_min, value_max) = self.get_plot_limits(data)
                    if value_min < value_max:
                        self.plot.set_ylim(
                            value_min,
                            value_max
                        )
            if dimensionality == 2 or (dimensionality == 3
                                       and not self.show_3d):
                # We rotate the array so the x-axis is shown as the
                # horizontal axis
                image_map = data[tuple(dslice)]
                image_map = np.rot90(image_map)
                extent = np.nanmin(positions[idx[0]][tuple(dslice)]), \
                    np.nanmax(positions[idx[0]][tuple(dslice)]), \
                    np.nanmin(positions[idx[1]][tuple(dslice)]), \
                    np.nanmax(positions[idx[1]][tuple(dslice)])
                if isinstance(self.image_map, matplotlib.image.AxesImage):
                    self.image_map.set_data(image_map)
                    self.image_map.set_extent(extent)
                else:
                    self._remove_image_map()
                    self.image_map = self.plot.imshow(
                        image_map, interpolation='nearest',
                        extent=extent
                    )
                    self._update_colorbar(self.image_map, '')

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        action='ignore',
                        message='All-NaN slice encountered'
                    )
                    (value_min, value_max) = self.get_plot_limits(data)
                    if value_min < value_max:
                        self.image_map.set_clim(value_min, value_max)
                        # For some reason we have to apply the color limits
                        # twice to make it work properly
                        self.image_map.set_clim(value_min, value_max)
                self.plot.set_title(parameters[parameter_key]['label'])
                self.plot.set_xlabel(labels[idx[0]])
                self.plot.set_ylabel(labels[idx[1]])
                cb_label = parameters[parameter_key]['symbol'] +\
                    ' [' + parameters[parameter_key]['unit'] + ']'
                self.colorbar.ax.set_title(cb_label)
                if self.aspect_ratio.isChecked():
                    self.plot.axis('scaled')
                else:
                    self.plot.axis('auto')
                self.plot.set_xlim(
                    np.nanmin(positions[idx[0]][tuple(dslice)]),
                    np.nanmax(positions[idx[0]][tuple(dslice)])
                )
                self.plot.set_ylim(
                    np.nanmin(positions[idx[1]][tuple(dslice)]),
                    np.nanmax(positions[idx[1]][tuple(dslice)])
                )
            if dimensionality == 3 and self.show_3d:
                # dslice here still has all three spatial axes as
                # slice(None) (only 3D-and-not-show_3d reduces it, above),
                # so this is the actual full 3D data/positions.
                self._draw_3d_cubes(
                    positions[0][tuple(dslice)],
                    positions[1][tuple(dslice)],
                    positions[2][tuple(dslice)],
                    data[tuple(dslice)],
                    labels, parameter_key, parameters
                )

            self.mplcanvas.draw()
        except Exception as e:
            self.reset_ui()
            raise e

    def on_z_slider_changed(self, value):
        self.z_slice_index = value
        self.refresh_plot()

    def on_toggle_3d_view(self):
        self.show_3d = not self.show_3d
        self.button_toggle_3d.setText(
            'Switch to 2D view' if self.show_3d else 'Switch to 3D view')
        self.refresh_plot()

    def _draw_3d_cubes(self, x, y, z, data, labels, parameter_key,
                        parameters):
        """
        Renders one cuboid per measurement point, centered on its
        actual (x, y, z) position (not a nominal/regular grid position -
        e.g. under surface-following, z varies per (x, y) to trace the
        found surface, and each point's real position is used as-is)
        and sized to the median physical spacing between neighboring
        points along each axis, so cube size reflects the real step size
        of the scan rather than an arbitrary constant. Cuboids are drawn
        semi-transparent unless the "Transparent Blocks" checkbox is
        unchecked, so blocks further inside the volume remain visible.
        """
        valid = ~np.isnan(data)
        if not np.any(valid):
            self.image_map = None
            return

        with warnings.catch_warnings():
            warnings.filterwarnings(
                action='ignore', message='All-NaN slice encountered')
            (value_min, value_max) = self.get_plot_limits(data)
        if not value_min < value_max:
            value_min, value_max = 0.0, 1.0
        scalar_map = matplotlib.cm.ScalarMappable(
            norm=Normalize(vmin=value_min, vmax=value_max),
            cmap=matplotlib.cm.viridis
        )

        def median_step(coord, axis):
            step = np.abs(np.diff(coord, axis=axis))
            step = step[np.isfinite(step) & (step > 0)]
            return float(np.median(step)) if step.size else 1.0

        half_extent = np.array([
            median_step(x, 0), median_step(y, 1), median_step(z, 2)
        ]) / 2

        centers = np.stack(
            [x[valid], y[valid], z[valid]], axis=1)  # (N, 3)

        # Unit-cube corner offsets (bottom face then top face).
        offsets = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        ], dtype=float)
        verts = centers[:, None, :] + offsets[None, :, :] * half_extent

        face_corner_idx = [
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7],
        ]
        faces = verts[:, face_corner_idx, :].reshape(-1, 4, 3)
        alpha = 0.5 if self.checkbox_transparent_blocks.isChecked() else 1.0
        face_colors = np.repeat(
            scalar_map.to_rgba(data[valid], alpha=alpha), 6, axis=0)

        self._remove_image_map()
        poly = Poly3DCollection(
            faces, facecolors=face_colors, edgecolors='black',
            linewidths=0.2
        )
        self.plot.add_collection3d(poly)
        self.image_map = poly

        x_min, x_max = np.nanmin(x) - half_extent[0], np.nanmax(x) + half_extent[0]
        y_min, y_max = np.nanmin(y) - half_extent[1], np.nanmax(y) + half_extent[1]
        z_min, z_max = np.nanmin(z) - half_extent[2], np.nanmax(z) + half_extent[2]
        self.plot.set_xlim(x_min, x_max)
        self.plot.set_ylim(y_min, y_max)
        self.plot.set_zlim(z_min, z_max)

        # Same "Aspect ratio" checkbox the 2D view uses (there: axis('scaled')
        # vs axis('auto')) - here, a box aspect matching the real x/y/z extents
        # vs mpl3d's own fixed 4:4:3 default, so unchecked looks like it always
        # did before this box-aspect handling existed.
        if self.aspect_ratio.isChecked():
            box_aspect = (
                max(x_max - x_min, 1e-9),
                max(y_max - y_min, 1e-9),
                max(z_max - z_min, 1e-9),
            )
        else:
            box_aspect = None
        self.plot.set_box_aspect(box_aspect)

        self.plot.set_xlabel(labels[0])
        self.plot.set_ylabel(labels[1])
        self.plot.set_zlabel(labels[2])

        cb_label = parameters[parameter_key]['symbol'] +\
            ' [' + parameters[parameter_key]['unit'] + ']'
        self._update_colorbar(scalar_map, cb_label)

    def get_plot_limits(self, data):
        if self.autoscale.isChecked():

            if self.ignore_outliers.isChecked():
                # Ignore outliers with 1.5 IQR rule
                q1 = np.nanpercentile(data, 25)
                q3 = np.nanpercentile(data, 75)
                iqr = q3 - q1
                data[data < (q1 - 1.5 * iqr)] = np.nan
                data[data > (q3 + 1.5 * iqr)] = np.nan

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    action='ignore',
                    message='All-NaN slice encountered'
                )
                value_min = np.nanmin(data)
                value_max = np.nanmax(data)

            self.value_min.blockSignals(True)
            self.value_min.setValue(value_min)
            self.value_min.blockSignals(False)
            self.value_max.blockSignals(True)
            self.value_max.setValue(value_max)
            self.value_max.blockSignals(False)
        else:
            value_min = self.value_min.value()
            value_max = self.value_max.value()
            if value_min > value_max:
                value_min = np.nanmin(data)
                value_max = np.nanmax(data)
        if value_min < value_max:
            # Adjust the double spin box step size
            single_step = (value_max - value_min) / 15
            self.value_min.setSingleStep(single_step)
            self.value_max.setSingleStep(single_step)

        return value_min, value_max
