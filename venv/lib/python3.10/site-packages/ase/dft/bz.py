# fmt: off

from itertools import product
from math import cos, pi, sin
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d
from scipy.spatial.transform import Rotation

from ase.cell import Cell


def bz_vertices(icell, dim=3):
    """Return the vertices and the normal vector of the BZ.

    See https://xkcd.com/1421 ..."""
    from scipy.spatial import Voronoi

    icell = icell.copy()
    if dim < 3:
        icell[2, 2] = 1e-3
    if dim < 2:
        icell[1, 1] = 1e-3

    indices = (np.indices((3, 3, 3)) - 1).reshape((3, 27))
    G = np.dot(icell.T, indices).T
    vor = Voronoi(G)
    bz1 = []
    for vertices, points in zip(vor.ridge_vertices, vor.ridge_points):
        if -1 not in vertices and 13 in points:
            normal = G[points].sum(0)
            normal /= (normal**2).sum()**0.5
            bz1.append((vor.vertices[vertices], normal))
    return bz1


class FlatPlot:
    """Helper class for 1D/2D Brillouin zone plots."""

    axis_dim = 2  # Dimension of the plotting surface (2 even if it's 1D BZ).
    point_options = {'zorder': 5}

    def new_axes(self, fig):
        return fig.gca()

    def adjust_view(self, ax, minp, maxp, symmetric: bool = True):
        """Ajusting view property of the drawn BZ. (1D/2D)

        Parameters
        ----------
        ax: Axes
            matplotlib Axes object.
        minp: float
            minimum value for the plotting region, which detemines the
            bottom left corner of the figure. if symmetric is set as True,
            this value is ignored.
        maxp: float
            maximum value for the plotting region, which detemines the
            top right corner of the figure.
        symmetric: bool
            if True, set the (0,0) position (Gamma-bar position) at the center
            of the figure.

        """
        ax.autoscale_view(tight=True)
        s = maxp * 1.05
        if symmetric:
            ax.set_xlim(-s, s)
            ax.set_ylim(-s, s)
        else:
            ax.set_xlim(minp * 1.05, s)
            ax.set_ylim(minp * 1.05, s)
        ax.set_aspect('equal')

    def draw_arrow(self, ax, vector, **kwargs):
        ax.arrow(0, 0, vector[0], vector[1],
                 lw=1,
                 length_includes_head=True,
                 head_width=0.03,
                 head_length=0.05,
                 **kwargs)

    def label_options(self, point):
        ha_s = ['right', 'left', 'right']
        va_s = ['bottom', 'bottom', 'top']

        x, y = point
        ha = ha_s[int(np.sign(x))]
        va = va_s[int(np.sign(y))]
        return {'ha': ha, 'va': va, 'zorder': 4}

    def view(self):
        pass


class SpacePlot:
    """Helper class for ordinary (3D) Brillouin zone plots.

    Attributes
    ----------
    azim : float
        Azimuthal angle in radian for viewing 3D BZ.
        default value is pi/5
    elev : float
        Elevation angle in radian for viewing 3D BZ.
        default value is pi/6

    """
    axis_dim = 3
    point_options: Dict[str, Any] = {}

    def __init__(self, *, azim: Optional[float] = None,
                 elev: Optional[float] = None):
        class Arrow3D(FancyArrowPatch):
            def __init__(self, ax, xs, ys, zs, *args, **kwargs):
                FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
                self._verts3d = xs, ys, zs
                self.ax = ax

            def draw(self, renderer):
                xs3d, ys3d, zs3d = self._verts3d
                xs, ys, _zs = proj3d.proj_transform(xs3d, ys3d,
                                                   zs3d, self.ax.axes.M)
                self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
                FancyArrowPatch.draw(self, renderer)

            # FIXME: Compatibility fix for matplotlib 3.5.0: Handling of 3D
            # artists have changed and all 3D artists now need
            # "do_3d_projection". Since this class is a hack that manually
            # projects onto the 3D axes we don't need to do anything in this
            # method. Ideally we shouldn't resort to a hack like this.
            def do_3d_projection(self, *_, **__):
                return 0

        self.arrow3d = Arrow3D
        self.azim: float = pi / 5 if azim is None else azim
        self.elev: float = pi / 6 if elev is None else elev
        self.view = [
            sin(self.azim) * cos(self.elev),
            cos(self.azim) * cos(self.elev),
            sin(self.elev),
        ]

    def new_axes(self, fig):
        return fig.add_subplot(projection='3d')

    def draw_arrow(self, ax: Axes3D, vector, **kwargs):
        ax.add_artist(self.arrow3d(
            ax,
            [0, vector[0]],
            [0, vector[1]],
            [0, vector[2]],
            mutation_scale=20,
            arrowstyle='-|>',
            **kwargs))

    def adjust_view(self, ax, minp, maxp, symmetric=True):
        """Ajusting view property of the drawn BZ. (3D)

        Parameters
        ----------
        ax: Axes
            matplotlib Axes object.
        minp: float
            minimum value for the plotting region, which detemines the
            bottom left corner of the figure. if symmetric is set as True,
            this value is ignored.
        maxp: float
            maximum value for the plotting region, which detemines the
            top right corner of the figure.
        symmetric: bool
            Currently, this is not used, just for keeping consistency with 2D
            version.

        """
        import matplotlib.pyplot as plt

        # ax.set_aspect('equal') <-- won't work anymore in 3.1.0
        ax.view_init(azim=np.rad2deg(self.azim), elev=np.rad2deg(self.elev))
        # We want aspect 'equal', but apparently there was a bug in
        # matplotlib causing wrong behaviour.  Matplotlib raises
        # NotImplementedError as of v3.1.0.  This is a bit unfortunate
        # because the workarounds known to StackOverflow and elsewhere
        # all involve using set_aspect('equal') and then doing
        # something more.
        #
        # We try to get square axes here by setting a square figure,
        # but this is probably rather inexact.
        fig = ax.get_figure()
        xx = plt.figaspect(1.0)
        fig.set_figheight(xx[1])
        fig.set_figwidth(xx[0])

        ax.set_proj_type('ortho')

        minp0 = 0.9 * minp  # Here we cheat a bit to trim spacings
        maxp0 = 0.9 * maxp
        ax.set_xlim3d(minp0, maxp0)
        ax.set_ylim3d(minp0, maxp0)
        ax.set_zlim3d(minp0, maxp0)

        ax.set_box_aspect([1, 1, 1])

    def label_options(self, point):
        return dict(ha='center', va='bottom')


def normalize_name(name):
    if name == 'G':
        return '\\Gamma'

    if len(name) > 1:
        import re

        m = re.match(r'^(\D+?)(\d*)$', name)
        if m is None:
            raise ValueError(f'Bad label: {name}')
        name, num = m.group(1, 2)
        if num:
            name = f'{name}_{{{num}}}'
    return name


def bz_plot(cell: Cell, vectors: bool = False, paths=None, points=None,
            azim: Optional[float] = None, elev: Optional[float] = None,
            scale=1, interactive: bool = False,
            transforms: Optional[list] = None,
            repeat: Union[Tuple[int, int], Tuple[int, int, int]] = (1, 1, 1),
            pointstyle: Optional[dict] = None,
            ax=None, show=False, **kwargs):
    """Plot the Brillouin zone of the Cell

    Parameters
    ----------
    cell: Cell
        Cell object for BZ drawing.
    vectors : bool
        if True, show the vector.
    paths : list[tuple[str, np.ndarray]] | None
        Special point name and its coordinate position
    points : np.ndarray
        Coordinate points along the paths.
    azim : float | None
        Azimuthal angle in radian for viewing 3D BZ.
    elev : float | None
        Elevation angle in radian for viewing 3D BZ.
    scale : float
        Not used. To be removed?
    interactive : bool
        Not effectively works. To be removed?
    transforms: List
        List of linear transformation (scipy.spatial.transform.Rotation)
    repeat: Tuple[int, int] | Tuple[int, int, int]
        Set the repeating draw of BZ. default is (1, 1, 1), no repeat.
    pointstyle : Dict
        Style of the special point
    ax : Axes | Axes3D
        matplolib Axes (Axes3D in 3D) object
    show : bool
        If true, show the figure.
    **kwargs
        Additional keyword arguments to pass to ax.plot

    Returns
    -------
    ax
        A matplotlib axis object.
    """
    import matplotlib.pyplot as plt

    if pointstyle is None:
        pointstyle = {}

    if transforms is None:
        transforms = [Rotation.from_rotvec((0, 0, 0))]

    cell = cell.copy()

    dimensions = cell.rank
    if dimensions == 3:
        plotter: Union[SpacePlot, FlatPlot] = SpacePlot(azim=azim, elev=elev)
    else:
        plotter = FlatPlot()
    assert dimensions > 0, 'No BZ for 0D!'

    if ax is None:
        ax = plotter.new_axes(plt.gcf())

    assert not np.array(cell)[dimensions:, :].any()
    assert not np.array(cell)[:, dimensions:].any()

    icell = cell.reciprocal()
    kpoints = points
    bz1 = bz_vertices(icell, dim=dimensions)
    if len(repeat) == 2:
        repeat = (repeat[0], repeat[1], 1)

    maxp = 0.0
    minp = 0.0
    for bz_i in bz_index(repeat):
        for points, normal in bz1:
            shift = np.dot(np.array(icell).T, np.array(bz_i))
            for transform in transforms:
                shift = transform.apply(shift)
            ls = '-'
            xyz = np.concatenate([points, points[:1]])
            for transform in transforms:
                xyz = transform.apply(xyz)
            x, y, z = xyz.T
            x, y, z = x + shift[0], y + shift[1], z + shift[2]
            if dimensions == 3:
                if normal @ plotter.view < 0 and not interactive:
                    ls = ':'
            if plotter.axis_dim == 2:
                ax.plot(x, y, c='k', ls=ls, **kwargs)
            else:
                ax.plot(x, y, z, c='k', ls=ls, **kwargs)
            maxp = max(maxp, x.max(), y.max(), z.max())
            minp = min(minp, x.min(), y.min(), z.min())

    if vectors:
        for transform in transforms:
            icell = transform.apply(icell)
        assert isinstance(icell, np.ndarray)
        for i in range(dimensions):
            plotter.draw_arrow(ax, icell[i], color='k')

        # XXX Can this be removed?
        if dimensions == 3:
            maxp = max(maxp, 0.6 * icell.max())
        else:
            maxp = max(maxp, icell.max())

    if paths is not None:
        for names, points in paths:
            for transform in transforms:
                points = transform.apply(points)
            coords = np.array(points).T[:plotter.axis_dim, :]
            ax.plot(*coords, c='r', ls='-')

            for name, point in zip(names, points):
                name = normalize_name(name)
                for transform in transforms:
                    point = transform.apply(point)
                point = point[:plotter.axis_dim]
                ax.text(*point, rf'$\mathrm{{{name}}}$',
                        color='g', **plotter.label_options(point))

    if kpoints is not None:
        kw = {'c': 'b', **plotter.point_options, **pointstyle}
        for transform in transforms:
            kpoints = transform.apply(kpoints)
        ax.scatter(*kpoints[:, :plotter.axis_dim].T, **kw)

    ax.set_axis_off()

    if repeat == (1, 1, 1):
        plotter.adjust_view(ax, minp, maxp)
    else:
        plotter.adjust_view(ax, minp, maxp, symmetric=False)
    if show:
        plt.show()

    return ax


def bz_index(repeat):
    """BZ index from the repeat

    A helper function to iterating drawing BZ.

    Parameters
    ----------
    repeat: Tuple[int, int] | Tuple[int, int, int]
        repeating for drawing BZ

    Returns
    -------
    Iterator[Tuple[int, int, int]]

    >>> list(_bz_index((1, 2, -2)))
    [(0, 0, 0), (0, 0, -1), (0, 1, 0), (0, 1, -1)]

    """
    if len(repeat) == 2:
        repeat = (repeat[0], repeat[1], 1)
    assert len(repeat) == 3
    assert repeat[0] != 0
    assert repeat[1] != 0
    assert repeat[2] != 0
    repeat_along_a = (
        range(0, repeat[0]) if repeat[0] > 0 else range(0, repeat[0], -1)
    )
    repeat_along_b = (
        range(0, repeat[1]) if repeat[1] > 0 else range(0, repeat[1], -1)
    )
    repeat_along_c = (
        range(0, repeat[2]) if repeat[2] > 0 else range(0, repeat[2], -1)
    )
    return product(repeat_along_a, repeat_along_b, repeat_along_c)
