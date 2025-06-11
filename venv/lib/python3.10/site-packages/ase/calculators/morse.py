# fmt: off

import numpy as np

from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import neighbor_list as ase_neighbor_list
from ase.stress import full_3x3_to_voigt_6_stress


def fcut(r: np.ndarray, r0: float, r1: float) -> np.ndarray:
    """Piecewise quintic C^{2,1} regular polynomial for use as a smooth cutoff.

    Ported from JuLIP.jl.

    https://github.com/JuliaMolSim/JuLIP.jl/blob/master/src/cutoffs.jl

    https://en.wikipedia.org/wiki/Smoothstep

    Parameters
    ----------
    r : np.ndarray
        Distances between atoms.
    r0 : float
        Inner cutoff radius.
    r1 : float
        Outder cutoff radius.

    Returns
    -------
    np.ndarray
        Sigmoid-like function smoothly interpolating (r0, 1) and (r1, 0).

    """""
    s = 1.0 - (r - r0) / (r1 - r0)
    return (s >= 1.0) + ((s > 0.0) & (s < 1.0)) * (
        6.0 * s**5 - 15.0 * s**4 + 10.0 * s**3
    )


def fcut_d(r: np.ndarray, r0: float, r1: float) -> np.ndarray:
    """Derivative of fcut() function defined above."""
    s = 1.0 - (r - r0) / (r1 - r0)
    return -(
        ((s > 0.0) & (s < 1.0))
        * (30.0 * s**4 - 60.0 * s**3 + 30.0 * s**2)
        / (r1 - r0)
    )


class MorsePotential(Calculator):
    """Morse potential."""

    implemented_properties = [
        'energy', 'energies', 'free_energy', 'forces', 'stress',
    ]
    default_parameters = {'epsilon': 1.0,
                          'rho0': 6.0,
                          'r0': 1.0,
                          'rcut1': 1.9,
                          'rcut2': 2.7}
    nolabel = True

    def __init__(self, neighbor_list=ase_neighbor_list, **kwargs):
        r"""

        The pairwise energy between atoms *i* and *j* is given by

        .. math::

            V_{ij} = \epsilon \left(
                \mathrm{e}^{-2 \rho_0 (r_{ij} / r_0 - 1)}
                - 2 \mathrm{e}^{- \rho_0 (r_{ij} / r_0 - 1)}
            \right)

        Parameters
        ----------
        epsilon : float, default 1.0
          Absolute minimum depth.
        r0 : float, default 1.0
          Minimum distance.
        rho0 : float, default 6.0
          Exponential prefactor.
          The force constant in the potential minimum is given by

          .. math::

              k = 2 \epsilon \left(\frac{\rho_0}{r_0}\right)^2.

        rcut1 : float, default 1.9
            Distance starting a smooth cutoff normalized by ``r0``.
        rcut2 : float, default 2.7
            Distance ending a smooth cutoff normalized by ``r0``.
        neighbor_list : callable, optional
            neighbor_list function compatible with
            ase.neighborlist.neighbor_list

        Notes
        -----
        The default values are chosen to be similar as Lennard-Jones.

        """
        self.neighbor_list = neighbor_list
        Calculator.__init__(self, **kwargs)

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        epsilon = self.parameters['epsilon']
        rho0 = self.parameters['rho0']
        r0 = self.parameters['r0']
        rcut1 = self.parameters['rcut1'] * r0
        rcut2 = self.parameters['rcut2'] * r0

        number_of_atoms = len(self.atoms)

        forces = np.zeros((number_of_atoms, 3))

        i, _j, d, D = self.neighbor_list('ijdD', atoms, rcut2)
        dhat = (D / d[:, None]).T

        expf = np.exp(rho0 * (1.0 - d / r0))

        cutoff_fn = fcut(d, rcut1, rcut2)
        d_cutoff_fn = fcut_d(d, rcut1, rcut2)

        pairwise_energies = epsilon * expf * (expf - 2.0)
        self.results['energies'] = np.bincount(
            i,
            weights=0.5 * (pairwise_energies * cutoff_fn),
            minlength=number_of_atoms,
        )
        self.results['energy'] = self.results['energies'].sum()
        self.results['free_energy'] = self.results['energy']

        # derivatives of `pair_energies` with respect to `d`
        de = (-2.0 * epsilon * rho0 / r0) * expf * (expf - 1.0)

        # smoothened `de`
        de = de * cutoff_fn + pairwise_energies * d_cutoff_fn

        de_vec = (de * dhat).T
        for dim in range(3):
            forces[:, dim] = np.bincount(
                i,
                weights=de_vec[:, dim],
                minlength=number_of_atoms,
            )
        self.results['forces'] = forces

        if self.atoms.cell.rank == 3:
            stress = 0.5 * (D.T @ de_vec) / self.atoms.get_volume()
            self.results['stress'] = full_3x3_to_voigt_6_stress(stress)
