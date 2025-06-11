# fmt: off

from typing import Any, Optional

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


class SimplePlottingAxes:
    def __init__(self,
                 ax: Optional[Axes] = None,
                 show: bool = False,
                 filename: str = None) -> None:
        self.ax = ax
        self.show = show
        self.filename = filename
        self.figure: Any = None  # Don't know about Figure/SubFigure etc

    def __enter__(self) -> Axes:
        if self.ax is None:
            self.figure, self.ax = plt.subplots()
        else:
            self.figure = self.ax.get_figure()

        return self.ax

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is None:
            # If there was no exception, display/write the plot as appropriate
            if self.figure is None:
                raise Exception("Something went wrong initializing matplotlib "
                                "figure")
            if self.show:
                self.figure.show()
            if self.filename is not None:
                self.figure.savefig(self.filename)

        return
