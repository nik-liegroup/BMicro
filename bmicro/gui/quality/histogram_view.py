import numpy as np

from PyQt6 import QtCore, QtWidgets

from bmlab.session import Session
from bmlab.controllers import EvaluationController

from bmicro.gui.mpl import MplCanvas


class HistogramView(QtWidgets.QDialog):
    """
    Independent (non-modal) window showing one quality metric's
    histogram at a time, with vertical lines at its currently set
    min/max threshold - opened from QualityView's "Show Histogram"
    button, initialized to whatever metric that tab currently has
    selected. Cycling here via Prev/Next also switches the Quality
    tab's own selected metric (and vice versa - selecting a different
    metric there updates this window too, see
    QualityView.on_preview_changed()), so the two always show the same
    metric. Call refresh() (QualityView does, on every threshold
    change) to redraw the currently shown metric against the latest
    thresholds/data.
    """

    def __init__(self, quality_view, *args, **kwargs):
        super(HistogramView, self).__init__(
            quality_view,
            QtCore.Qt.WindowType.Window |
            QtCore.Qt.WindowType.WindowTitleHint |
            QtCore.Qt.WindowType.WindowCloseButtonHint |
            QtCore.Qt.WindowType.WindowMinimizeButtonHint
        )
        self.setWindowTitle('Quality metric histogram')
        self.setWindowModality(QtCore.Qt.WindowModality.NonModal)
        self.resize(600, 500)

        self.quality_view = quality_view
        self.evaluation_controller = EvaluationController()
        self.session = Session.get_instance()
        self.metric_key = EvaluationController.QUALITY_METRIC_KEYS[0]

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        nav_layout = QtWidgets.QHBoxLayout()
        self.button_prev = QtWidgets.QPushButton('< Prev')
        self.label_metric = QtWidgets.QLabel('')
        self.label_metric.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignCenter)
        self.button_next = QtWidgets.QPushButton('Next >')
        nav_layout.addWidget(self.button_prev)
        nav_layout.addWidget(self.label_metric, stretch=1)
        nav_layout.addWidget(self.button_next)
        layout.addLayout(nav_layout)

        self.plot_widget = QtWidgets.QWidget()
        layout.addWidget(self.plot_widget, stretch=1)
        self.mplcanvas = MplCanvas(
            self.plot_widget, toolbar=('Home', 'Pan', 'Zoom'))
        self.plot = self.mplcanvas.get_figure().add_subplot(111)

        self.button_prev.clicked.connect(self.on_prev)
        self.button_next.clicked.connect(self.on_next)

    def show_for_metric(self, metric_key):
        self.metric_key = metric_key
        self.refresh()

    def on_prev(self):
        self._cycle(-1)

    def on_next(self):
        self._cycle(1)

    def _cycle(self, step):
        keys = EvaluationController.QUALITY_METRIC_KEYS
        self.metric_key = keys[
            (keys.index(self.metric_key) + step) % len(keys)]
        # Also select this metric's radio button on the Quality tab,
        # so the main map follows the histogram - that in turn fires
        # QualityView.on_preview_changed(), which calls us back via
        # show_for_metric() -> refresh(), so only refresh here
        # ourselves if that didn't already happen (e.g. the tab isn't
        # around, or this metric was already selected there).
        if not self.quality_view.set_preview_key(self.metric_key):
            self.refresh()

    def refresh(self):
        evm = self.session.evaluation_model()
        if evm is None:
            return

        param = evm.parameters[self.metric_key]
        self.label_metric.setText(param['label'])

        mask, measured = self.evaluation_controller.compute_quality_mask()
        self.plot.cla()
        if mask is None:
            self.mplcanvas.draw()
            return

        data, _, _, _ = self.evaluation_controller.get_data(
            self.metric_key, 0)
        values = data[measured]
        values = values[~np.isnan(values)]

        if values.size:
            # A robust (percentile-based) range rather than the full
            # min/max - some of these metrics (e.g. a near-singular fit
            # producing a huge center_uncertainty) have extreme
            # outliers that otherwise squash the whole informative
            # bulk of the distribution into a single bin, so the
            # histogram looks empty.
            lo, hi = np.nanpercentile(values, [1, 99])
            if lo == hi:
                lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
            if lo < hi:
                self.plot.hist(
                    values, bins=40, range=(lo, hi), color='tab:blue')
            else:
                self.plot.hist(values, bins=1, color='tab:blue')
            n_outside = int(np.sum((values < lo) | (values > hi)))
            if n_outside:
                self.plot.set_title(
                    f'{param["label"]} '
                    f'({n_outside} outlier(s) outside shown range)')
            else:
                self.plot.set_title(param['label'])
        else:
            self.plot.set_title(param['label'] + ' (no data)')

        threshold = evm.quality_thresholds.get(self.metric_key, {})
        if threshold.get('min') is not None:
            self.plot.axvline(
                threshold['min'], color='tab:red', linestyle='--')
        if threshold.get('max') is not None:
            self.plot.axvline(
                threshold['max'], color='tab:red', linestyle='--')

        xlabel = param['symbol'] + \
            (' [' + param['unit'] + ']' if param['unit'] else '')
        self.plot.set_xlabel(xlabel)
        self.plot.set_ylabel('Points')

        self.mplcanvas.get_figure().tight_layout()
        self.mplcanvas.draw()
