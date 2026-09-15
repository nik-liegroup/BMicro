from PyQt6 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent, toolbar=False, width=5, height=5, dpi=100):
        """
        A custom PyQt widget for plotting with matplotlib.

        Parameters
        ----------
        parent: QWidget
            the widget where to embed the current plot widget
        toolbar: tuple of str
            Strings defining what buttons to show on the toolbar
        width: int
            width of the figure
        height: int
            height of the figure
        dpi: int
            DPI settings
        """

        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(MplCanvas, self).__init__(self.fig)

        self.toolbar = None
        if toolbar:

            class CustomToolbar(NavigationToolbar2QT):
                toolitems = [t for t in NavigationToolbar2QT.toolitems if
                             t[0] in toolbar]

            self.toolbar = CustomToolbar(self, parent)

        self.attach_to(parent)

    def get_figure(self):
        return self.fig

    def attach_to(self, parent):
        """
        (Re-)embeds this canvas (and its toolbar, if any) in `parent`,
        removing it from wherever it was embedded before. Lets two
        tabs showing the same underlying data (e.g. Evaluation and
        Quality - see EvaluationView.on_tab_activated()) share a
        single canvas/figure/axes rather than each keeping its own -
        so pan/zoom state and the click-to-spectrum handler (connected
        once, on the canvas itself) carry over between them
        automatically, and only whichever tab currently holds the
        canvas needs to actually redraw it.
        """
        if self.parentWidget() is parent:
            return
        old_layout = self.parentWidget().layout() \
            if self.parentWidget() is not None else None
        if old_layout is not None:
            if self.toolbar is not None:
                old_layout.removeWidget(self.toolbar)
            old_layout.removeWidget(self)

        layout = parent.layout()
        if layout is None:
            layout = QtWidgets.QVBoxLayout()
            parent.setLayout(layout)
        if self.toolbar is not None:
            layout.addWidget(self.toolbar)
        layout.addWidget(self)
