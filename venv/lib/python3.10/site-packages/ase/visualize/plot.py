# fmt: off

from ase.io.utils import PlottingVariables, make_patch_list


class Matplotlib(PlottingVariables):
    def __init__(self, atoms, ax,
                 rotation='', radii=None,
                 colors=None, scale=1, offset=(0, 0), **parameters):
        PlottingVariables.__init__(
            self, atoms, rotation=rotation,
            radii=radii, colors=colors, scale=scale,
            extra_offset=offset, **parameters)

        self.ax = ax
        self.figure = ax.figure
        self.ax.set_aspect('equal')

    def write(self):
        self.write_body()
        self.ax.set_xlim(0, self.w)
        self.ax.set_ylim(0, self.h)

    def write_body(self):
        patch_list = make_patch_list(self)
        for patch in patch_list:
            self.ax.add_patch(patch)


def animate(images, ax=None,
            interval=200,  # in ms; same default value as in FuncAnimation
            save_count=None,  # ignored as of 2023 with newer matplotlib
            **parameters):
    """Convert sequence of atoms objects into Matplotlib animation.

    Each image is generated using plot_atoms().  Additional parameters
    are passed to this function."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    if ax is None:
        ax = plt.gca()

    fig = ax.get_figure()

    def drawimage(atoms):
        ax.clear()
        ax.axis('off')
        plot_atoms(atoms, ax=ax, **parameters)

    animation = FuncAnimation(fig, drawimage, frames=images,
                              init_func=lambda: None,
                              interval=interval)
    return animation


def plot_atoms(atoms, ax=None, **parameters):
    """Plot an atoms object in a matplotlib subplot.

    Parameters
    ----------
    atoms : Atoms object
    ax : Matplotlib subplot object
    rotation : str, optional
        In degrees. In the form '10x,20y,30z'
    show_unit_cell : int, optional, default 2
        Draw the unit cell as dashed lines depending on value:
        0: Don't
        1: Do
        2: Do, making sure cell is visible
    radii : float, optional
        The radii of the atoms
    colors : list of strings, optional
        Color of the atoms, must be the same length as
        the number of atoms in the atoms object.
    scale : float, optional
        Scaling of the plotted atoms and lines.
    offset : tuple (float, float), optional
        Offset of the plotted atoms and lines.
    """
    if isinstance(atoms, list):
        assert len(atoms) == 1
        atoms = atoms[0]

    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt.gca()
    Matplotlib(atoms, ax, **parameters).write()
    return ax
