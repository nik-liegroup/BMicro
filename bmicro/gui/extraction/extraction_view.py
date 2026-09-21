from importlib import resources
import logging

from PyQt6 import QtWidgets, QtCore, uic
from matplotlib.patches import Circle as MPLCircle

import matplotlib
import numpy as np

from bmlab.session import Session
from bmlab.controllers import ExtractionController

from bmicro.BGThread import BGThread
from bmicro.gui.mpl import MplCanvas


MODE_DEFAULT = 0
MODE_SELECT = 1

logger = logging.getLogger(__name__)


class ExtractionView(QtWidgets.QWidget):
    """
    Class for the extraction widget
    """

    def __init__(self, *args, **kwargs):
        super(ExtractionView, self).__init__(*args, **kwargs)

        ref = resources.files('bmicro.gui.extraction') / 'extraction_view.ui'
        with resources.as_file(ref) as ui_file:
            uic.loadUi(ui_file, self)

        self.mode = MODE_DEFAULT
        self.current_frame = 0
        # Reentrancy guard for on_zoom_changed(): applying a crop calls
        # refresh_image_plot(), which redraws the (now cropped) image and
        # fires xlim_changed/ylim_changed again - without this, that
        # would recurse into on_zoom_changed() a second time before the
        # first call has returned.
        self._syncing_crop = False

        self.mplcanvas = MplCanvas(
            self.image_widget, toolbar=('Home', 'Pan', 'Zoom'))
        self.image_plot = self.mplcanvas.get_figure().add_subplot(111)
        self.image_plot.axis('off')
        self.mplcanvas.get_figure().canvas.mpl_connect(
            'button_press_event', self.on_click_image)

        self.thread = BGThread()

        self.combobox_datasets.currentIndexChanged.connect(
            self.on_select_dataset)

        self.table_selected_points.itemChanged.connect(
            lambda item: self.on_points_changed(item))

        self.setupTable()

        self.button_select_done.clicked.connect(self.toggle_mode)
        self.button_clear.clicked.connect(self.clear_points)
        self.button_optimize.clicked.connect(self.optimize_points)
        self.button_find_points.clicked.connect(self.find_points)
        self.button_find_points_all.clicked.connect(self.find_points_all)

        self.button_prev_frame.clicked.connect(self.prev_frame)
        self.button_next_frame.clicked.connect(self.next_frame)

        self.checkbox_use_zoom_as_crop.toggled.connect(
            self.on_toggle_use_zoom_as_crop)

        self.update_ui()
        self.checkFrameNavigationButtons()

        self.extraction_controller = ExtractionController()

    def update_ui(self):
        session = Session.get_instance()
        self.combobox_datasets.clear()

        # Reflect whatever crop is already set on the session (e.g. loaded
        # from a session file) without re-triggering the toggle handler,
        # which would overwrite it with the current (unrelated) zoom.
        self.checkbox_use_zoom_as_crop.blockSignals(True)
        self.checkbox_use_zoom_as_crop.setChecked(
            session.get_crop_bounds() is not None)
        self.checkbox_use_zoom_as_crop.blockSignals(False)

        if not session.current_repetition():
            return

        calib_keys = session.get_calib_keys(sort_by_time=True)
        self.combobox_datasets.addItems(calib_keys)

    def reset_ui(self):
        self.combobox_datasets.clear()
        self.table_selected_points.setRowCount(0)
        self.checkbox_use_zoom_as_crop.blockSignals(True)
        self.checkbox_use_zoom_as_crop.setChecked(False)
        self.checkbox_use_zoom_as_crop.blockSignals(False)
        self.refresh_image_plot()

    def prev_frame(self):
        if self.current_frame > 0:
            self.current_frame -= 1
            self.refresh_image_plot()
            self.checkFrameNavigationButtons()

    def next_frame(self):
        session = Session.get_instance()
        calib_key = self.combobox_datasets.currentText()
        frame_count = session.get_calibration_image_count(calib_key)
        if self.current_frame < frame_count - 1:
            self.current_frame += 1
            self.refresh_image_plot()
            self.checkFrameNavigationButtons()

    def checkFrameNavigationButtons(self):
        if self.current_frame > 0:
            self.button_prev_frame.setEnabled(True)
        else:
            self.button_prev_frame.setEnabled(False)

        session = Session.get_instance()
        calib_key = self.combobox_datasets.currentText()
        if calib_key:
            frame_count = session.get_calibration_image_count(calib_key)
            if self.current_frame < frame_count - 1:
                self.button_next_frame.setEnabled(True)
            else:
                self.button_next_frame.setEnabled(False)

    def on_select_dataset(self):
        """
        Action triggered when user selects a calibration dataset.
        """
        self.checkFrameNavigationButtons()
        self.refresh_image_plot()

    def on_toggle_use_zoom_as_crop(self, checked):
        """
        Locks in the plot's current zoom (from the toolbar's Pan/Zoom
        tool) as a pixel crop applied to every calibration/payload image
        used in extraction, calibration fitting, and evaluation for the
        whole file - not just the currently displayed one. Useful when
        the camera ROI used at acquisition time was wider than a single
        VIPA order, without needing to re-acquire the data.

        While checked, the crop keeps tracking further zooming too (see
        on_zoom_changed()) - this only needs to capture the zoom as it
        is right now, at the moment of checking.

        Unchecking removes the crop and restores the full image.
        """
        session = Session.get_instance()
        if checked:
            x_min, x_max = self.image_plot.get_xlim()
            y_min, y_max = self.image_plot.get_ylim()
            session.set_crop_bounds((x_min, x_max, y_min, y_max))
        else:
            session.clear_crop()
        self.refresh_image_plot()

    def on_click_image(self, event):
        """
        Action triggered when user clicks on preview image in selection mode.

        Parameters
        ----------
        event: matplotlib event object
            The mouse click event.
        """
        if self.mode != MODE_SELECT:
            return
        ec = ExtractionController()
        calib_key = self.combobox_datasets.currentText()
        logger.debug('Adding point (%f, %f) for calibration key %s' % (
            event.xdata, event.ydata, calib_key
        ))
        ec.add_point(calib_key,
                     (event.xdata, event.ydata))
        self.refresh_image_plot()

    def refresh_image_plot(self):
        """
        Updates the plot of the selected calibration image.
        """
        self.image_plot.cla()
        # cla() drops any callbacks registered on the axes, so reconnect
        # these every refresh rather than once in __init__: fires on
        # every zoom/pan (toolbar or programmatic) so the label - and,
        # while checkbox_use_zoom_as_crop is checked, the crop itself -
        # stay live while the user is dragging the Zoom tool, not just
        # when they let go of the mouse.
        self.image_plot.callbacks.connect(
            'xlim_changed', self.on_zoom_changed)
        self.image_plot.callbacks.connect(
            'ylim_changed', self.on_zoom_changed)
        session = Session.get_instance()
        calib_key = self.combobox_datasets.currentText()
        if not calib_key:
            self.mplcanvas.draw()
            self.update_zoom_label()
            return

        em = session.extraction_model()
        img = session.get_calibration_image(calib_key, self.current_frame)

        # imshow()'s own initial autoscale - and the explicit
        # on_zoom_changed() call below - must not be mistaken for a
        # real user zoom: Crop.apply() clamps a too-large crop down to
        # whatever a smaller image actually has (e.g. a repetition
        # captured at a smaller camera ROI than the one the crop was
        # drawn on), so the freshly drawn image's own axis extent can
        # legitimately differ from session.crop's stored bounds even
        # though nothing was zoomed. Without this guard, on_zoom_
        # changed() reads that as "the view changed" and silently
        # overwrites (shrinks) the crop for every repetition, not just
        # the one being displayed - reproducible just by opening a
        # file with checkbox_use_zoom_as_crop already checked from a
        # saved session, no interactive zooming needed at all.
        self._syncing_crop = True
        try:
            # imshow should always get the transposed image such that
            # the horizontal axis of the plot coincides with the
            # 0-axis of the plotted array:
            self.image_plot.imshow(img.T, origin='lower', vmin=100, vmax=300)
            self.image_plot.set_title('Frame %d' % (self.current_frame+1))

            points = em.get_points(calib_key)

            if len(points) >= 3:
                em.set_image_shape(img.shape)

                arcs = em.get_arc_by_calib_key(calib_key)
                for arc in arcs:
                    dr = arc[-1] - arc[0]
                    line = matplotlib.patches.FancyArrow(
                        *arc[0], dr[0], dr[1], head_width=0,
                        head_length=0, color='Yellow', alpha=0.5)
                    self.image_plot.add_patch(line)

            self._plot_points(em.get_points(calib_key))

            self.mplcanvas.draw()
            self.refresh_points()
            self.on_zoom_changed()
        finally:
            self._syncing_crop = False

    def on_zoom_changed(self, *_):
        """
        Called on every xlim_changed/ylim_changed - i.e. on every zoom or
        pan, from the toolbar's Zoom/Pan tool, from a fresh
        refresh_image_plot() draw, or programmatically - and, unlike the
        checkbox's own toggled handler, regardless of when the checkbox
        was checked relative to the zoom. Without this, checking
        checkbox_use_zoom_as_crop and only then zooming left the crop
        stuck at whatever (typically the full, un-zoomed) view was
        showing at the moment of checking - a stale crop the user could
        only fix by re-toggling the checkbox after zooming instead.

        Takes an optional, unused argument so it can be used directly as
        a matplotlib axes callback (xlim_changed/ylim_changed pass the
        Axes as their sole argument).
        """
        self.update_zoom_label()

        if self._syncing_crop \
                or not self.checkbox_use_zoom_as_crop.isChecked():
            return

        # A single zoom/pan gesture that changes both axes (box zoom,
        # scroll zoom) fires xlim_changed and ylim_changed as two
        # SEPARATE events, not atomically together - matplotlib calls
        # set_xlim() and set_ylim() one after the other. Applying the
        # crop right on the first of the two would compose it against
        # the other axis' still-stale bounds (the previous view, not
        # the new one), drifting the persisted crop away from what was
        # actually shown - compounding further on every subsequent
        # zoom. Defer to the next spin of the event loop, once both
        # axes have settled, and coalesce any back-to-back events into
        # a single update.
        QtCore.QTimer.singleShot(0, self._apply_zoom_as_crop)

    def _apply_zoom_as_crop(self):
        if self._syncing_crop \
                or not self.checkbox_use_zoom_as_crop.isChecked():
            return

        session = Session.get_instance()
        x_min, x_max = self.image_plot.get_xlim()
        y_min, y_max = self.image_plot.get_ylim()

        # The displayed image is already the cropped one whenever a crop
        # is active (Crop.apply() - see its own docstring), so these
        # xlim/ylim are in that CROPPED image's own local pixel
        # coordinates, not the original image's. Crop.bounds always
        # means "relative to the original, uncropped image" - offsetting
        # by the existing crop's own x_min/y_min converts back to that,
        # rather than composing crops relative to each other, which
        # would drift off the true position after repeated zooming.
        existing = session.get_crop_bounds()
        offset_x, offset_y = (existing[0], existing[2]) if existing \
            else (0, 0)
        new_bounds = (
            x_min + offset_x, x_max + offset_x,
            y_min + offset_y, y_max + offset_y,
        )
        if existing is not None and tuple(round(v) for v in new_bounds) \
                == tuple(round(v) for v in existing):
            # Already in sync (e.g. this call is refresh_image_plot()'s
            # own redraw right after applying this same crop) - nothing
            # to do, and re-applying would otherwise recurse forever via
            # refresh_image_plot() -> cla() -> a fresh xlim_changed.
            return

        self._syncing_crop = True
        try:
            session.set_crop_bounds(new_bounds)
            self.refresh_image_plot()
        finally:
            self._syncing_crop = False

    def update_zoom_label(self, *_):
        """
        Shows the plot's current zoom as live pixel bounds, in the same
        (x, y) convention as the crop and as extraction points - compare
        directly against BrillouinAcquisition's own ROI panel
        (Left/Top/Width/Height) to line up the same physical region
        without needing to eyeball it.

        Takes an optional, unused argument so it can be used directly as
        a matplotlib axes callback (xlim_changed/ylim_changed pass the
        Axes as their sole argument).
        """
        x_min, x_max = self.image_plot.get_xlim()
        y_min, y_max = self.image_plot.get_ylim()
        self.label_zoom_bounds.setText(
            'Zoom: x [%d, %d]  y [%d, %d]' % (
                round(x_min), round(x_max), round(y_min), round(y_max)))

    def setupTable(self):
        self.table_selected_points.setColumnCount(2)
        self.table_selected_points\
            .setHorizontalHeaderLabels(["x", "y"])
        header = self.table_selected_points.horizontalHeader()
        header.setSectionResizeMode(0,
                                    QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1,
                                    QtWidgets.QHeaderView.ResizeMode.Stretch)

    def refresh_points(self):
        session = Session.get_instance()
        em = session.extraction_model()
        calib_key = self.combobox_datasets.currentText()
        if not calib_key:
            return
        if em:
            points = em.get_points(calib_key)
            self.table_selected_points.setRowCount(len(points))
            for rowIdx, point in enumerate(points):
                # Add regions to table
                # Block signals, so the itemChanged signal is not
                # emitted during table creation
                self.table_selected_points.blockSignals(True)
                for columnIdx, value in enumerate(point):
                    item = QtWidgets.QTableWidgetItem(str(value))
                    self.table_selected_points.setItem(rowIdx, columnIdx, item)
                self.table_selected_points.blockSignals(False)

    def on_points_changed(self, item):
        row = item.row()
        column = item.column()
        value = float(item.text())

        session = Session.get_instance()
        calib_key = self.combobox_datasets.currentText()

        em = session.extraction_model()
        ec = ExtractionController()
        if em:
            points = em.get_points(calib_key)
            current_point = np.asarray(points[row])
            current_point[column] = value
            current_point = tuple(current_point)
            ec.set_point(calib_key, row, current_point)
            self.refresh_image_plot()

    def _plot_points(self, points):
        for p in points:
            circle = MPLCircle(p, radius=3, color='red', alpha=0.5)
            self.image_plot.add_patch(circle)

    def toggle_mode(self):
        """
        Toggles between normal and selection mode.

        In selection mode, the user can select points in the calibration
        image.
        """
        if self.mode == MODE_DEFAULT:
            self.mode = MODE_SELECT
            self.button_select_done.setText('Done')
        else:
            self.mode = MODE_DEFAULT
            self.button_select_done.setText('Select')

    def clear_points(self):
        """
        Deletes the selected point for the current calibration image
        """
        calib_key = self.combobox_datasets.currentText()
        session = Session.get_instance()

        if not session.extraction_model():
            return
        session.extraction_model().clear_points(calib_key)
        self.refresh_image_plot()

    def optimize_points(self):
        """
        Moves the selected points in the calibration image to the nearest
        maximum values.
        """
        calib_key = self.combobox_datasets.currentText()
        ec = ExtractionController()
        ec.optimize_points(calib_key)
        self.refresh_image_plot()

    def find_points(self):
        """
        Automatically finds the Rayleigh and Brillouin peaks of interest.
        """
        calib_key = self.combobox_datasets.currentText()
        ec = ExtractionController()
        ec.find_points(calib_key)
        self.refresh_image_plot()

    def find_points_all(self):
        """
        Automatically finds the Rayleigh and Brillouin peaks of interest
        for all calibrations existing.
        """
        session = Session.get_instance()
        calib_keys = session.get_calib_keys(sort_by_time=True)

        if not calib_keys:
            return

        for i, calib_key in enumerate(calib_keys):
            self.combobox_datasets.setCurrentText(calib_key)

            dnkw = {
                "calib_key": calib_key,
            }

            self.thread.set_task(
                func=self.extraction_controller.find_points, fkw=dnkw)
            self.thread.start()
            self.thread.wait()

            self.refresh_image_plot()
            QtCore.QCoreApplication.instance().processEvents()
