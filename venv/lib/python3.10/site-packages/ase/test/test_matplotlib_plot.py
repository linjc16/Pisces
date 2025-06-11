import os

import numpy as np
import pytest

from ase.lattice.cubic import FaceCenteredCubic
from ase.utils.plotting import SimplePlottingAxes
from ase.visualize.plot import plot_atoms


def test_matplotlib_plot_info_occupancies(plt):
    slab = FaceCenteredCubic('Au')
    slab.info['occupancy'] = {'0': {'Au': 1}}
    _fig, ax = plt.subplots()
    plot_atoms(slab, ax, show_unit_cell=0)
    assert len(ax.patches) == len(slab)


def test_matplotlib_plot(plt):
    slab = FaceCenteredCubic('Au', size=(2, 2, 2))

    _fig, ax = plt.subplots()
    plot_atoms(slab, ax, radii=0.5, rotation=('10x,10y,10z'),
               show_unit_cell=0)

    assert len(ax.patches) == len(slab)


class TestPlotManager:
    @pytest.fixture()
    def xy_data(self):
        return ([1, 2], [3, 4])

    def test_plot_manager_error(self, figure):
        with pytest.raises(AssertionError):
            with SimplePlottingAxes(ax=None, show=False, filename=None):
                raise AssertionError

    def test_plot_manager_no_file(self, xy_data, figure):
        x, y = xy_data

        with SimplePlottingAxes(ax=None, show=False, filename=None) as ax:
            ax.plot(x, y)

        assert np.allclose(ax.lines[0].get_xydata().transpose(), xy_data)

    def test_plot_manager_axis_file(self, testdir, xy_data, figure):
        filename = 'plot.png'
        x, y = xy_data
        ax = figure.add_subplot(111)
        with SimplePlottingAxes(ax=ax, show=False,
                                filename=filename) as return_ax:
            assert return_ax is ax
            ax.plot(x, y)

        assert os.path.isfile(filename)
