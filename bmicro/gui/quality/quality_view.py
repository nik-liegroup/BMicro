from importlib import resources
import warnings

import numpy as np
import matplotlib
from matplotlib.collections import LineCollection

from PyQt6 import QtWidgets, uic
from PyQt6.QtWidgets import QHBoxLayout, QVBoxLayout, QLabel, QCheckBox, \
    QDoubleSpinBox, QRadioButton, QWidget, QButtonGroup

from bmlab.session import Session
from bmlab.controllers import EvaluationController

from bmicro.gui.quality.histogram_view import HistogramView


class QualityView(QtWidgets.QWidget):
    """
    Lets a user set/tune quality thresholds (see
    EvaluationController.QUALITY_METRIC_KEYS - fit SNR, fit-center
    uncertainty, frame-to-frame reproducibility, NRMSE, FWHM) live,
    and see - before committing anything - how many measured points
    would pass them, both as a count and as a 2D map (the same
    autoscale/ignore-outliers/aspect-ratio/z-slice view the Evaluation
    tab uses, minus its 3D-cube option) of the currently previewed
    metric, with failing points marked. The map is not just kept in
    sync with the Evaluation tab's own view via set_evaluation_view() -
    it is the *same* canvas/figure/axes, handed back and forth between
    the two tabs' containers as each becomes active (see
    on_tab_activated(), mpl.MplCanvas.attach_to()), since both show
    the same measurement grid. That also means pan/zoom carries over
    between the tabs for free, and clicking a tile here opens the
    spectrum dialog exactly as it does on the Evaluation tab (the
    click handler is connected once, on the canvas itself).
    "Apply to results" is the only action that writes anything
    (evm.results['quality_pass']); every other control here previews.
    """

    def __init__(self, *args, **kwargs):
        super(QualityView, self).__init__(*args, **kwargs)

        ref = resources.files('bmicro.gui.quality') / 'quality_view.ui'
        with resources.as_file(ref) as ui_file:
            uic.loadUi(ui_file, self)

        self.evaluation_controller = EvaluationController()
        self.session = Session.get_instance()
        self._evaluation_view = None
        self.histogram_view = None
        self._recompute_pending = False

        self.z_slice_index = 0
        self.z_slider.setVisible(False)
        self.label_z_slider.setVisible(False)
        self.label_z_value.setVisible(False)
        self.z_slider.valueChanged.connect(self.on_z_slider_changed)

        self.aspect_ratio.clicked.connect(self.on_aspect_ratio_changed)
        self.autoscale.clicked.connect(self.on_scale_changed)
        self.ignore_outliers.clicked.connect(self.on_scale_changed)
        self.value_min.valueChanged.connect(self.on_scale_changed)
        self.value_max.valueChanged.connect(self.on_scale_changed)

        self._preview_group = QButtonGroup(self)
        self._preview_group.setExclusive(True)
        self._rows = {}  # metric_key -> dict of row widgets

        self._build_metric_rows()

        self.button_apply_quality.clicked.connect(self.on_apply)
        self.button_show_histogram.clicked.connect(self.on_show_histogram)

    def set_evaluation_view(self, evaluation_view):
        """
        Cross-links this tab with the Evaluation tab's own 2D view (see
        EvaluationView.set_quality_view(), called the same way from
        MainWindow right after both tabs are built) so a z-slice or
        autoscale/ignore-outliers/aspect-ratio change in either one is
        reflected in the other.
        """
        self._evaluation_view = evaluation_view

    # The plot canvas/figure/axes/colorbar are not this tab's own -
    # they belong to (and are drawn/managed by) the Evaluation tab,
    # and are simply re-parented onto this tab's image_widget while
    # it is active (see on_tab_activated()). These properties let the
    # rest of this class keep referring to self.mplcanvas/self.plot/
    # self.image_map/self.colorbar as if they were local, unchanged
    # from before this tab had its own copies of them.
    @property
    def mplcanvas(self):
        return self._evaluation_view.mplcanvas

    @property
    def plot(self):
        return self._evaluation_view.plot

    @plot.setter
    def plot(self, value):
        self._evaluation_view.plot = value

    @property
    def image_map(self):
        return self._evaluation_view.image_map

    @image_map.setter
    def image_map(self, value):
        self._evaluation_view.image_map = value

    @property
    def colorbar(self):
        return self._evaluation_view.colorbar

    def on_tab_activated(self):
        """
        Called by MainWindow.update_ui() when this tab becomes the
        current one - (re-)attaches the shared plot canvas here (see
        EvaluationView.on_tab_activated(), mpl.MplCanvas.attach_to()),
        preserving whatever pan/zoom was set on it, then refreshes.
        """
        if self._evaluation_view is None:
            self.update_ui()
            return
        if self.mplcanvas.parentWidget() is self.image_widget:
            self.update_ui()
            return
        try:
            xlim, ylim = self.plot.get_xlim(), self.plot.get_ylim()
        except Exception:
            xlim = ylim = None
        self.mplcanvas.attach_to(self.image_widget)
        self.update_ui()
        if xlim is not None:
            try:
                self.plot.set_xlim(xlim)
                self.plot.set_ylim(ylim)
                self.mplcanvas.draw()
            except Exception:
                pass

    def sync_z_slice(self, value):
        """Called by the Evaluation tab when ITS z-slice changes."""
        if value == self.z_slice_index:
            return
        self.z_slice_index = value
        self.z_slider.blockSignals(True)
        self.z_slider.setValue(value)
        self.z_slider.blockSignals(False)
        self.refresh_plot()

    def sync_view_prefs(self, autoscale=None, ignore_outliers=None,
                        aspect_ratio=None):
        """Called by the Evaluation tab when one of these checkboxes
        changes there."""
        changed = False
        for checkbox, value in (
                (self.autoscale, autoscale),
                (self.ignore_outliers, ignore_outliers),
                (self.aspect_ratio, aspect_ratio)):
            if value is None or checkbox.isChecked() == value:
                continue
            checkbox.blockSignals(True)
            checkbox.setChecked(value)
            checkbox.blockSignals(False)
            changed = True
        if autoscale is not None:
            self.value_min.setDisabled(autoscale)
            self.value_max.setDisabled(autoscale)
            self.ignore_outliers.setDisabled(not autoscale)
        if changed:
            self.refresh_plot()

    def on_scale_changed(self):
        autoscale = self.autoscale.isChecked()
        self.value_min.setDisabled(autoscale)
        self.value_max.setDisabled(autoscale)
        self.ignore_outliers.setDisabled(not autoscale)
        self.refresh_plot()
        if self._evaluation_view is not None:
            self._evaluation_view.sync_view_prefs(
                autoscale=autoscale,
                ignore_outliers=self.ignore_outliers.isChecked())

    def on_aspect_ratio_changed(self):
        self.refresh_plot()
        if self._evaluation_view is not None:
            self._evaluation_view.sync_view_prefs(
                aspect_ratio=self.aspect_ratio.isChecked())

    def on_z_slider_changed(self, value):
        self.z_slice_index = value
        self.refresh_plot()
        if self._evaluation_view is not None:
            self._evaluation_view.sync_z_slice(value)

    def _build_metric_rows(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 6, 0, 6)
        layout.setSpacing(0)

        for i, key in enumerate(EvaluationController.QUALITY_METRIC_KEYS):
            if i > 0:
                # Between rows only, not after the last one - a
                # trailing separator with nothing below it just reads
                # as a cut-off/missing row.
                separator = QtWidgets.QFrame()
                separator.setFrameShape(QtWidgets.QFrame.Shape.HLine)
                separator.setStyleSheet('color: #ddd;')
                layout.addWidget(separator)

            # Two lines per metric (label on its own line, bounds
            # below, indented) rather than one long row - the labels
            # are too long (e.g. "Rayleigh peak center uncertainty
            # (quality) [GHz]") to share a row with both bound controls
            # without clipping one of them.
            row_layout = QVBoxLayout()
            row_layout.setSpacing(2)

            label_layout = QHBoxLayout()
            preview = QRadioButton()
            preview.setToolTip('Show this metric in the plots')
            self._preview_group.addButton(preview)
            label_layout.addWidget(preview)
            label = QLabel(key)
            label_layout.addWidget(label)
            label_layout.addStretch()
            row_layout.addLayout(label_layout)

            bounds_layout = QHBoxLayout()
            bounds_layout.addSpacing(20)

            min_check = QCheckBox('min')
            bounds_layout.addWidget(min_check)
            min_box = QDoubleSpinBox()
            min_box.setRange(-1e9, 1e9)
            min_box.setDecimals(4)
            min_box.setMaximumWidth(100)
            min_box.setEnabled(False)
            bounds_layout.addWidget(min_box)

            bounds_layout.addSpacing(12)

            max_check = QCheckBox('max')
            bounds_layout.addWidget(max_check)
            max_box = QDoubleSpinBox()
            max_box.setRange(-1e9, 1e9)
            max_box.setDecimals(4)
            max_box.setMaximumWidth(100)
            max_box.setEnabled(False)
            bounds_layout.addWidget(max_box)

            bounds_layout.addStretch()
            row_layout.addLayout(bounds_layout)

            row_widget = QWidget()
            row_widget.setLayout(row_layout)
            layout.addWidget(row_widget)

            min_check.toggled.connect(min_box.setEnabled)
            max_check.toggled.connect(max_box.setEnabled)
            min_check.toggled.connect(self.on_threshold_changed)
            max_check.toggled.connect(self.on_threshold_changed)
            min_box.valueChanged.connect(self.on_threshold_changed)
            max_box.valueChanged.connect(self.on_threshold_changed)
            preview.toggled.connect(self.on_preview_changed)

            self._rows[key] = {
                'preview': preview, 'label': label,
                'min_check': min_check, 'min_box': min_box,
                'max_check': max_check, 'max_box': max_box,
            }

        self.widget_metric_rows.setLayout(layout)
        self._rows[EvaluationController.QUALITY_METRIC_KEYS[0]][
            'preview'].setChecked(True)

    def preview_key(self):
        for key, row in self._rows.items():
            if row['preview'].isChecked():
                return key
        return EvaluationController.QUALITY_METRIC_KEYS[0]

    def set_preview_key(self, key):
        """
        Selects `key`'s radio button (used by HistogramView when
        cycling metrics there via Prev/Next, so the main map follows
        along). Returns True if this actually changed the selection
        (which itself triggers on_preview_changed() -> refresh_plot()
        and, if the histogram window is open, a matching refresh
        there too) - False if `key` was already selected or unknown,
        so the caller knows it must refresh itself instead.
        """
        row = self._rows.get(key)
        if row is None or row['preview'].isChecked():
            return False
        row['preview'].setChecked(True)
        return True

    def update_ui(self):
        """
        Called when a file is opened/an evaluation finishes (see
        MainWindow.update_ui()) - refreshes row labels (now that units
        are known) and restores any thresholds already stored on this
        evaluation model (e.g. loaded from a saved session), then
        redraws the plots.
        """
        evm = self.session.evaluation_model()
        if evm is None:
            return

        for key, row in self._rows.items():
            param = evm.parameters.get(key)
            if param is not None:
                unit = f" [{param['unit']}]" if param['unit'] else ''
                row['label'].setText(f"{param['label']}{unit}")

            threshold = evm.quality_thresholds.get(key, {})
            for widget in ('min_check', 'max_check', 'min_box', 'max_box'):
                row[widget].blockSignals(True)

            has_min = threshold.get('min') is not None
            row['min_check'].setChecked(has_min)
            row['min_box'].setEnabled(has_min)
            if has_min:
                row['min_box'].setValue(threshold['min'])

            has_max = threshold.get('max') is not None
            row['max_check'].setChecked(has_max)
            row['max_box'].setEnabled(has_max)
            if has_max:
                row['max_box'].setValue(threshold['max'])

            for widget in ('min_check', 'max_check', 'min_box', 'max_box'):
                row[widget].blockSignals(False)

        if self.evaluation_controller.quality_metrics_need_recompute():
            self._trigger_auto_recompute()
            return

        self.refresh_plot()

    def _trigger_auto_recompute(self):
        """
        This session's saved fit predates brillouin_peak_snr/_nrmse/
        _center_uncertainty (and their rayleigh_ equivalents) - only
        evaluate()'s fitting loop populates those (see
        EvaluationController.quality_metrics_need_recompute()), so
        re-run it via the Evaluation tab (reusing its background-thread
        worker; on_evaluation_finished() below refreshes this tab once
        it's done) rather than show an empty view.
        """
        if self._evaluation_view is None \
                or self._evaluation_view.evaluation_running \
                or self._recompute_pending:
            return
        self._recompute_pending = True
        self.setEnabled(False)
        self.label_points_remaining.setText(
            'Points passing: calculating quality metrics '
            '(re-running evaluation)...')
        self._evaluation_view.evaluate(blocking=False)

    def on_evaluation_finished(self):
        """
        Connected (by EvaluationView.evaluate() itself) to every
        evaluate() run's completion, whether triggered by its own
        button or by _trigger_auto_recompute() here - either way this
        tab's view is stale until it's done.
        """
        self._recompute_pending = False
        self.setEnabled(True)
        self.update_ui()

    def reset_ui(self):
        self._recompute_pending = False
        self.setEnabled(True)
        for row in self._rows.values():
            row['min_check'].setChecked(False)
            row['max_check'].setChecked(False)
        self.label_points_remaining.setText('Points passing: - / -')
        self.z_slider.setVisible(False)
        self.label_z_slider.setVisible(False)
        self.label_z_value.setVisible(False)
        # The plot canvas/figure/colorbar belong to the Evaluation tab
        # (see the mplcanvas/plot/image_map/colorbar properties above)
        # and are reset there by its own reset_ui() (which also clears
        # our fail markers, since a clf() there invalidates them) -
        # nothing to do with them here beyond dropping our own
        # (already-invalid) references.
        self._fail_markers = []
        if self.histogram_view is not None:
            self.histogram_view.close()

    def on_threshold_changed(self):
        evm = self.session.evaluation_model()
        if evm is None:
            return
        for key, row in self._rows.items():
            min_value = row['min_box'].value() \
                if row['min_check'].isChecked() else None
            max_value = row['max_box'].value() \
                if row['max_check'].isChecked() else None
            self.evaluation_controller.set_quality_threshold(
                key, enabled=True,
                min_value=min_value, max_value=max_value)
        self.refresh_plot()
        if self.histogram_view is not None:
            self.histogram_view.refresh()

    def on_preview_changed(self, checked):
        if not checked:
            return
        self.refresh_plot()
        if self.histogram_view is not None and self.histogram_view.isVisible():
            self.histogram_view.show_for_metric(self.preview_key())

    def on_apply(self):
        self.evaluation_controller.apply_quality_thresholds()

    def on_show_histogram(self):
        if self.histogram_view is None:
            self.histogram_view = HistogramView(self)
        self.histogram_view.show_for_metric(self.preview_key())
        self.histogram_view.show()
        self.histogram_view.raise_()
        self.histogram_view.activateWindow()

    def get_plot_limits(self, data):
        if self.autoscale.isChecked():
            if self.ignore_outliers.isChecked():
                # Percentile-based, not the Evaluation tab's IQR rule:
                # several quality metrics (e.g. center_uncertainty on a
                # near-singular fit) can have a large enough fraction
                # of degenerate/huge points that Q1/Q3 themselves land
                # in the outlier range, which breaks IQR-based
                # rejection. A fixed 1st/99th percentile range stays
                # bounded regardless of how large that fraction is.
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        action='ignore',
                        message='All-NaN slice encountered')
                    lo = np.nanpercentile(data, 1)
                    hi = np.nanpercentile(data, 99)
                data[data < lo] = np.nan
                data[data > hi] = np.nan
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    action='ignore', message='All-NaN slice encountered')
                value_min = np.nanmin(data)
                value_max = np.nanmax(data)
            # No finite data at all (e.g. this metric was never
            # computed for whatever fit produced the loaded session -
            # only evaluate() populates brillouin_peak_snr/_nrmse/
            # _center_uncertainty and its rayleigh_ equivalents, so a
            # file evaluated before this feature existed won't have
            # them) - fall back to a sentinel range rather than
            # display NaN clamped to the spinboxes' own bound.
            if not np.isfinite(value_min) or not np.isfinite(value_max):
                value_min, value_max = 0.0, 1.0
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
            single_step = (value_max - value_min) / 15
            self.value_min.setSingleStep(single_step)
            self.value_max.setSingleStep(single_step)
        return value_min, value_max

    def refresh_plot(self):
        evm = self.session.evaluation_model()
        if evm is None:
            return

        mask, measured = self.evaluation_controller.compute_quality_mask()
        if mask is None:
            self.label_points_remaining.setText('Points passing: - / -')
            return

        n_measured = int(measured.sum())
        n_pass = int(mask.sum())
        pct = 100 * n_pass / n_measured if n_measured else 0.0
        self.label_points_remaining.setText(
            f'Points passing: {n_pass} / {n_measured} ({pct:.1f}%)')

        # The shared canvas (see EvaluationView.on_tab_activated()/
        # mpl.MplCanvas.attach_to()) currently belongs to the
        # Evaluation tab - the pass-count label above is always kept
        # current, but there's nothing to draw here until this tab is
        # activated again (which re-attaches it and calls us again).
        if self._evaluation_view is None \
                or self.mplcanvas.parentWidget() is not self.image_widget:
            return

        key = self.preview_key()
        data, positions, dimensionality, labels = \
            self.evaluation_controller.get_data(key, 0)
        param = evm.parameters[key]

        if dimensionality is None or dimensionality < 2:
            # Nothing meaningful to draw as a 2D map (or a slice of
            # one) - clear and bail, same as the Evaluation tab does
            # for a repetition with no valid measurement grid.
            self._evaluation_view._click_context = None
            self._evaluation_view._remove_colorbar()
            self.mplcanvas.get_figure().clf()
            self.plot = self.mplcanvas.get_figure().add_subplot(111)
            self.image_map = None
            self.z_slider.setVisible(False)
            self.label_z_slider.setVisible(False)
            self.label_z_value.setVisible(False)
            self.mplcanvas.draw()
            return

        # Center x and y around zero (relative sample coordinates), as
        # the Evaluation tab does; z is left as the actual acquired
        # stage position.
        for position in positions[:2]:
            position -= np.nanmean(position)

        dslice = [slice(None) if dim > 1 else 0 for dim in data.shape]
        idx = [i for i, dim in enumerate(data.shape) if dim > 1]

        if dimensionality == 3:
            # Always a 2D slice through a slider here - no 3D-cube
            # option (unlike the Evaluation tab).
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

        if len(idx) != 2:
            # A 1-point-wide slice along one in-plane axis too - not
            # enough left to draw as an image.
            self._evaluation_view._click_context = None
            return

        # Lets a click on this map open the spectrum dialog exactly as
        # on the Evaluation tab (see EvaluationView.on_click_image()) -
        # the mapping from a click position to a measurement index only
        # depends on the shape/positions of the current slice, not on
        # which parameter/metric is being displayed.
        self._evaluation_view._click_context = {
            'idx': tuple(idx), 'dslice': list(dslice)}

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
            if self.image_map is not None:
                self.image_map.remove()
            self.image_map = self.plot.imshow(
                image_map, interpolation='nearest', extent=extent)
            self._evaluation_view._update_colorbar(self.image_map, '')

        with warnings.catch_warnings():
            warnings.filterwarnings(
                action='ignore', message='All-NaN slice encountered')
            (value_min, value_max) = self.get_plot_limits(data.copy())
            if value_min < value_max:
                self.image_map.set_clim(value_min, value_max)
                self.image_map.set_clim(value_min, value_max)

        self.plot.set_title(param['label'])
        self.plot.set_xlabel(labels[idx[0]])
        self.plot.set_ylabel(labels[idx[1]])
        cb_label = param['symbol'] + \
            (' [' + param['unit'] + ']' if param['unit'] else '')
        self.colorbar.ax.set_title(cb_label)
        self.plot.axis('scaled' if self.aspect_ratio.isChecked() else 'auto')
        self.plot.set_xlim(
            np.nanmin(positions[idx[0]][tuple(dslice)]),
            np.nanmax(positions[idx[0]][tuple(dslice)]))
        self.plot.set_ylim(
            np.nanmin(positions[idx[1]][tuple(dslice)]),
            np.nanmax(positions[idx[1]][tuple(dslice)]))

        # Mark failing points (measured, but not passing every enabled
        # threshold) directly on the map, one cross per failing tile.
        # Built from each point's integer grid position rather than
        # its raw physical position (as positions[...] would give):
        # imshow's extent divides evenly into `image_map.shape` cells,
        # which - since the given extent spans data-point min to max,
        # not cell edge to cell edge - is subtly narrower than the
        # true point spacing, so a tile's rendered center isn't quite
        # at its own physical coordinate. Deriving the cross position
        # the same way (extent + cell index) instead keeps it exactly
        # on the tile as drawn, whatever that offset is.
        self.clear_fail_markers()
        fail = (measured & ~mask)[tuple(dslice)]
        if fail.any():
            n0, n1 = fail.shape
            cell_w = (extent[1] - extent[0]) / n0
            cell_h = (extent[3] - extent[2]) / n1
            fail_i0, fail_i1 = np.nonzero(fail)
            cx = extent[0] + (fail_i0 + 0.5) * cell_w
            cy = extent[2] + (fail_i1 + 0.5) * cell_h
            # Sized as a fraction of the tile itself (in data
            # coordinates) rather than a fixed on-screen marker size,
            # so it scales with the tile when the user zooms instead
            # of staying a constant pixel size.
            hw, hh = 0.25 * cell_w, 0.25 * cell_h
            segments = [
                seg for x, y in zip(cx, cy) for seg in (
                    [(x - hw, y - hh), (x + hw, y + hh)],
                    [(x - hw, y + hh), (x + hw, y - hh)])]
            markers = LineCollection(
                segments, colors='red', linewidths=1.5,
                label='fails threshold')
            self.plot.add_collection(markers)
            self._fail_markers = [markers]
            self.plot.legend(loc='upper right', fontsize='small')

        self.mplcanvas.draw()

    def clear_fail_markers(self):
        """
        Removes this tab's failing-point cross markers (see
        refresh_plot()) from the shared plot - called before it
        redraws them itself, and by the Evaluation tab (see
        EvaluationView.refresh_plot()/reset_ui()) before it draws or
        clears its own view of the same, shared axes, since these
        markers are Quality-tab-specific and shouldn't show up there.
        """
        for artist in getattr(self, '_fail_markers', []):
            # The shared figure (see the mplcanvas/plot properties
            # above) may have been cleared from under these markers by
            # the Evaluation tab (e.g. its own clf() for a repetition
            # with no valid grid) - such a stale artist is already
            # detached and raises on remove().
            try:
                artist.remove()
            except Exception:
                pass
        self._fail_markers = []
