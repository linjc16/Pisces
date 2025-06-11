# fmt: off

from math import pi, sqrt

import numpy as np

from ase.dft.kpoints import get_monkhorst_pack_size_and_offset
from ase.parallel import world
from ase.utils.cext import cextension


class DOS:
    def __init__(self, calc, width=0.1, window=None, npts=401, comm=world):
        """Electronic Density Of States object.

        calc: calculator object
            Any ASE compliant calculator object.
        width: float
            Width of guassian smearing.  Use width=0.0 for linear tetrahedron
            interpolation.
        window: tuple of two float
            Use ``window=(emin, emax)``.  If not specified, a window
            big enough to hold all the eigenvalues will be used.
        npts: int
            Number of points.
        comm: communicator object
            MPI communicator for lti_dos

        """

        self.comm = comm
        self.npts = npts
        self.width = width
        self.w_k = calc.get_k_point_weights()
        self.nspins = calc.get_number_of_spins()
        self.e_skn = np.array([[calc.get_eigenvalues(kpt=k, spin=s)
                                for k in range(len(self.w_k))]
                               for s in range(self.nspins)])
        try:  # two Fermi levels
            for i, eF in enumerate(calc.get_fermi_level()):
                self.e_skn[i] -= eF
        except TypeError:  # a single Fermi level
            self.e_skn -= calc.get_fermi_level()

        if window is None:
            emin = None
            emax = None
        else:
            emin, emax = window

        if emin is None:
            emin = self.e_skn.min() - 5 * self.width
        if emax is None:
            emax = self.e_skn.max() + 5 * self.width

        self.energies = np.linspace(emin, emax, npts)

        if width == 0.0:
            bzkpts = calc.get_bz_k_points()
            size, _offset = get_monkhorst_pack_size_and_offset(bzkpts)
            bz2ibz = calc.get_bz_to_ibz_map()
            shape = (self.nspins,) + tuple(size) + (-1,)
            self.e_skn = self.e_skn[:, bz2ibz].reshape(shape)
            self.cell = calc.atoms.cell

    def get_energies(self):
        """Return the array of energies used to sample the DOS.

        The energies are reported relative to the Fermi level.
        """
        return self.energies

    def delta(self, energy):
        """Return a delta-function centered at 'energy'."""
        x = -((self.energies - energy) / self.width)**2
        return np.exp(x) / (sqrt(pi) * self.width)

    def get_dos(self, spin=None):
        """Get array of DOS values.

        The *spin* argument can be 0 or 1 (spin up or down) - if not
        specified, the total DOS is returned.
        """

        if spin is None:
            if self.nspins == 2:
                # Return the total DOS
                return self.get_dos(spin=0) + self.get_dos(spin=1)
            else:
                return 2 * self.get_dos(spin=0)
        elif spin == 1 and self.nspins == 1:
            # For an unpolarized calculation, spin up and down are equivalent
            spin = 0

        if self.width == 0.0:
            dos = linear_tetrahedron_integration(self.cell, self.e_skn[spin],
                                                 self.energies, comm=self.comm)
            return dos

        dos = np.zeros(self.npts)
        for w, e_n in zip(self.w_k, self.e_skn[spin]):
            for e in e_n:
                dos += w * self.delta(e)
        return dos


def linear_tetrahedron_integration(cell, eigs, energies,
                                   weights=None, comm=world):
    """DOS from linear tetrahedron interpolation.

    cell: 3x3 ndarray-like
        Unit cell.
    eigs: (n1, n2, n3, nbands)-shaped ndarray
        Eigenvalues on a Monkhorst-Pack grid (not reduced).
    energies: 1-d array-like
        Energies where the DOS is calculated (must be a uniform grid).
    weights: ndarray of shape (n1, n2, n3, nbands) or (n1, n2, n3, nbands, nw)
        Weights.  Defaults to a (n1, n2, n3, nbands)-shaped ndarray
        filled with ones.  Can also have an extra dimednsion if there are
        nw weights.
    comm: communicator object
            MPI communicator for lti_dos

    Returns:

        DOS as an ndarray of same length as energies or as an
        ndarray of shape (nw, len(energies)).

    See:

        Extensions of the tetrahedron method for evaluating
        spectral properties of solids,
        A. H. MacDonald, S. H. Vosko and P. T. Coleridge,
        1979 J. Phys. C: Solid State Phys. 12 2991,
        :doi:`10.1088/0022-3719/12/15/008`
    """

    from scipy.spatial import Delaunay

    # Find the 6 tetrahedra:
    size = eigs.shape[:3]
    B = (np.linalg.inv(cell) / size).T
    indices = np.array([[i, j, k]
                        for i in [0, 1] for j in [0, 1] for k in [0, 1]])
    dt = Delaunay(np.dot(indices, B))

    if weights is None:
        weights = np.ones_like(eigs)

    if weights.ndim == 4:
        extra_dimension_added = True
        weights = weights[:, :, :, :, np.newaxis]
    else:
        extra_dimension_added = False

    nweights = weights.shape[4]
    dos = np.empty((nweights, len(energies)))

    lti_dos(indices[dt.simplices], eigs, weights, energies, dos, comm)

    dos /= np.prod(size)

    if extra_dimension_added:
        return dos[0]
    return dos


@cextension
def lti_dos(simplices, eigs, weights, energies, dos, world):
    shape = eigs.shape[:3]
    nweights = weights.shape[-1]
    dos[:] = 0.0
    n = -1
    for index in np.indices(shape).reshape((3, -1)).T:
        n += 1
        if n % world.size != world.rank:
            continue
        i = ((index + simplices) % shape).T
        E = eigs[i[0], i[1], i[2]].reshape((4, -1))
        W = weights[i[0], i[1], i[2]].reshape((4, -1, nweights))
        for e, w in zip(E.T, W.transpose((1, 0, 2))):
            lti_dos1(e, w, energies, dos)

    dos /= 6.0
    world.sum(dos)


def lti_dos1(e, w, energies, dos):
    i = e.argsort()
    e0, e1, e2, e3 = en = e[i]
    w = w[i]

    zero = energies[0]
    if len(energies) > 1:
        de = energies[1] - zero
        nn = (np.floor((en - zero) / de).astype(int) + 1).clip(0,
                                                               len(energies))
    else:
        nn = (en > zero).astype(int)

    n0, n1, n2, n3 = nn

    if n1 > n0:
        s = slice(n0, n1)
        x = energies[s] - e0
        f10 = x / (e1 - e0)
        f20 = x / (e2 - e0)
        f30 = x / (e3 - e0)
        f01 = 1 - f10
        f02 = 1 - f20
        f03 = 1 - f30
        g = f20 * f30 / (e1 - e0)
        dos[:, s] += w.T.dot([f01 + f02 + f03,
                              f10,
                              f20,
                              f30]) * g
    if n2 > n1:
        delta = e3 - e0
        s = slice(n1, n2)
        x = energies[s]
        f20 = (x - e0) / (e2 - e0)
        f30 = (x - e0) / (e3 - e0)
        f21 = (x - e1) / (e2 - e1)
        f31 = (x - e1) / (e3 - e1)
        f02 = 1 - f20
        f03 = 1 - f30
        f12 = 1 - f21
        f13 = 1 - f31
        g = 3 / delta * (f12 * f20 + f21 * f13)
        dos[:, s] += w.T.dot([g * f03 / 3 + f02 * f20 * f12 / delta,
                              g * f12 / 3 + f13 * f13 * f21 / delta,
                              g * f21 / 3 + f20 * f20 * f12 / delta,
                              g * f30 / 3 + f31 * f13 * f21 / delta])
    if n3 > n2:
        s = slice(n2, n3)
        x = energies[s] - e3
        f03 = x / (e0 - e3)
        f13 = x / (e1 - e3)
        f23 = x / (e2 - e3)
        f30 = 1 - f03
        f31 = 1 - f13
        f32 = 1 - f23
        g = f03 * f13 / (e3 - e2)
        dos[:, s] += w.T.dot([f03,
                              f13,
                              f23,
                              f30 + f31 + f32]) * g
