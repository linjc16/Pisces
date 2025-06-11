# fmt: off

from collections import namedtuple

import numpy as np

from ase.geometry import find_mic


def fit_raw(energies, forces, positions, cell=None, pbc=None):
    """Calculates parameters for fitting images to a band, as for
    a NEB plot."""
    energies = np.array(energies) - energies[0]
    n_images = len(energies)
    fit_energies = np.empty((n_images - 1) * 20 + 1)
    fit_path = np.empty((n_images - 1) * 20 + 1)

    path = [0]
    for i in range(n_images - 1):
        dR = positions[i + 1] - positions[i]
        if cell is not None and pbc is not None:
            dR, _ = find_mic(dR, cell, pbc)
        path.append(path[i] + np.sqrt((dR**2).sum()))

    lines = []  # tangent lines
    lastslope = None
    for i in range(n_images):
        if i == 0:
            direction = positions[i + 1] - positions[i]
            dpath = 0.5 * path[1]
        elif i == n_images - 1:
            direction = positions[-1] - positions[-2]
            dpath = 0.5 * (path[-1] - path[-2])
        else:
            direction = positions[i + 1] - positions[i - 1]
            dpath = 0.25 * (path[i + 1] - path[i - 1])

        direction /= np.linalg.norm(direction)
        slope = -(forces[i] * direction).sum()
        x = np.linspace(path[i] - dpath, path[i] + dpath, 3)
        y = energies[i] + slope * (x - path[i])
        lines.append((x, y))

        if i > 0:
            s0 = path[i - 1]
            s1 = path[i]
            x = np.linspace(s0, s1, 20, endpoint=False)
            c = np.linalg.solve(np.array([(1, s0, s0**2, s0**3),
                                          (1, s1, s1**2, s1**3),
                                          (0, 1, 2 * s0, 3 * s0**2),
                                          (0, 1, 2 * s1, 3 * s1**2)]),
                                np.array([energies[i - 1], energies[i],
                                          lastslope, slope]))
            y = c[0] + x * (c[1] + x * (c[2] + x * c[3]))
            fit_path[(i - 1) * 20:i * 20] = x
            fit_energies[(i - 1) * 20:i * 20] = y

        lastslope = slope

    fit_path[-1] = path[-1]
    fit_energies[-1] = energies[-1]
    return ForceFit(path, energies, fit_path, fit_energies, lines)


class ForceFit(namedtuple('ForceFit', ['path', 'energies', 'fit_path',
                                       'fit_energies', 'lines'])):
    """Data container to hold fitting parameters for force curves."""

    def plot(self, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.gca()

        ax.plot(self.path, self.energies, 'o')
        for x, y in self.lines:
            ax.plot(x, y, '-g')
        ax.plot(self.fit_path, self.fit_energies, 'k-')
        ax.set_xlabel(r'path [Å]')
        ax.set_ylabel('energy [eV]')
        Ef = max(self.energies) - self.energies[0]
        Er = max(self.energies) - self.energies[-1]
        dE = self.energies[-1] - self.energies[0]
        ax.set_title(r'$E_\mathrm{{f}} \approx$ {:.3f} eV; '
                     r'$E_\mathrm{{r}} \approx$ {:.3f} eV; '
                     r'$\Delta E$ = {:.3f} eV'.format(Ef, Er, dE))
        return ax


def fit_images(images):
    """Fits a series of images with a smoothed line for producing a standard
    NEB plot. Returns a `ForceFit` data structure; the plot can be produced
    by calling the `plot` method of `ForceFit`."""
    R = [atoms.positions for atoms in images]
    E = [atoms.get_potential_energy() for atoms in images]
    F = [atoms.get_forces() for atoms in images]  # XXX force consistent???
    A = images[0].cell
    pbc = images[0].pbc
    return fit_raw(E, F, R, A, pbc)


def force_curve(images, ax=None):
    """Plot energies and forces as a function of accumulated displacements.

    This is for testing whether a calculator's forces are consistent with
    the energies on a set of geometries where energies and forces are
    available."""

    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt.gca()

    nim = len(images)

    accumulated_distances = []
    accumulated_distance = 0.0

    # XXX force_consistent=True will work with some calculators,
    # but won't work if images were loaded from a trajectory.
    energies = [atoms.get_potential_energy()
                for atoms in images]

    for i in range(nim):
        atoms = images[i]
        f_ac = atoms.get_forces()

        if i < nim - 1:
            rightpos = images[i + 1].positions
        else:
            rightpos = atoms.positions

        if i > 0:
            leftpos = images[i - 1].positions
        else:
            leftpos = atoms.positions

        disp_ac, _ = find_mic(rightpos - leftpos, cell=atoms.cell,
                              pbc=atoms.pbc)

        def total_displacement(disp):
            disp_a = (disp**2).sum(axis=1)**.5
            return sum(disp_a)

        dE_fdotr = -0.5 * np.vdot(f_ac.ravel(), disp_ac.ravel())

        linescale = 0.45

        disp = 0.5 * total_displacement(disp_ac)

        if i == 0 or i == nim - 1:
            disp *= 2
            dE_fdotr *= 2

        x1 = accumulated_distance - disp * linescale
        x2 = accumulated_distance + disp * linescale
        y1 = energies[i] - dE_fdotr * linescale
        y2 = energies[i] + dE_fdotr * linescale

        ax.plot([x1, x2], [y1, y2], 'b-')
        ax.plot(accumulated_distance, energies[i], 'bo')
        ax.set_ylabel('Energy [eV]')
        ax.set_xlabel('Accumulative distance [Å]')
        accumulated_distances.append(accumulated_distance)
        accumulated_distance += total_displacement(rightpos - atoms.positions)

    ax.plot(accumulated_distances, energies, ':', zorder=-1, color='k')
    return ax


def plotfromfile(*fnames):
    from ase.io import read
    nplots = len(fnames)

    for i, fname in enumerate(fnames):
        images = read(fname, ':')
        import matplotlib.pyplot as plt
        plt.subplot(nplots, 1, 1 + i)
        force_curve(images)
    plt.show()


if __name__ == '__main__':
    import sys
    fnames = sys.argv[1:]
    plotfromfile(*fnames)
