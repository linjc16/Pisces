# fmt: off

"""Effective medium theory potential."""
from collections import defaultdict
from math import log, sqrt

import numpy as np

from ase.calculators.calculator import (
    Calculator,
    PropertyNotImplementedError,
    all_changes,
)
from ase.data import atomic_numbers, chemical_symbols
from ase.neighborlist import NeighborList
from ase.units import Bohr

parameters = {
    #      E0     s0    V0     eta2    kappa   lambda  n0
    #      eV     bohr  eV     bohr^-1 bohr^-1 bohr^-1 bohr^-3
    'Al': (-3.28, 3.00, 1.493, 1.240, 2.000, 1.169, 0.00700),
    'Cu': (-3.51, 2.67, 2.476, 1.652, 2.740, 1.906, 0.00910),
    'Ag': (-2.96, 3.01, 2.132, 1.652, 2.790, 1.892, 0.00547),
    'Au': (-3.80, 3.00, 2.321, 1.674, 2.873, 2.182, 0.00703),
    'Ni': (-4.44, 2.60, 3.673, 1.669, 2.757, 1.948, 0.01030),
    'Pd': (-3.90, 2.87, 2.773, 1.818, 3.107, 2.155, 0.00688),
    'Pt': (-5.85, 2.90, 4.067, 1.812, 3.145, 2.192, 0.00802),
    # extra parameters - just for fun ...
    'H': (-3.21, 1.31, 0.132, 2.652, 2.790, 3.892, 0.00547),
    'C': (-3.50, 1.81, 0.332, 1.652, 2.790, 1.892, 0.01322),
    'N': (-5.10, 1.88, 0.132, 1.652, 2.790, 1.892, 0.01222),
    'O': (-4.60, 1.95, 0.332, 1.652, 2.790, 1.892, 0.00850)}

beta = 1.809  # (16 * pi / 3)**(1.0 / 3) / 2**0.5, preserve historical rounding


class EMT(Calculator):
    """Python implementation of the Effective Medium Potential.

    Supports the following standard EMT metals:
    Al, Cu, Ag, Au, Ni, Pd and Pt.

    In addition, the following elements are supported.
    They are NOT well described by EMT, and the parameters
    are not for any serious use:
    H, C, N, O

    Parameters
    ----------
    asap_cutoff : bool, default: False
        If True the cutoff mimics how ASAP does it; most importantly the global
        cutoff is chosen from the largest atom present in the simulation.
        If False it is chosen from the largest atom in the parameter table.
        True gives the behaviour of ASAP and older EMT implementations,
        although the results are not bitwise identical.

    Notes
    -----
    Formulation mostly follows Jacobsen *et al*. [1]_
    `Documentation in ASAP can also be referred to <https://gitlab.com/asap/
    asap/blob/master/docs/manual/potentials/emt.pdf>`_.

    .. [1] K. W. Jacobsen, P. Stoltze, and J. K. Nørskov,
       Surf. Sci. 366, 394 (1996).
    """
    implemented_properties = ['energy', 'free_energy', 'energies', 'forces',
                              'stress', 'magmom', 'magmoms']

    nolabel = True

    default_parameters = {'asap_cutoff': False}

    def __init__(self, **kwargs):
        Calculator.__init__(self, **kwargs)

    def initialize(self, atoms):
        self.rc, self.rc_list, self.acut = self._calc_cutoff(atoms)

        numbers = atoms.get_atomic_numbers()

        # ia2iz : map from idx of atoms to idx of atomic numbers in self.par
        unique_numbers, self.ia2iz = np.unique(numbers, return_inverse=True)
        self.par = defaultdict(lambda: np.empty(len(unique_numbers)))
        for i, Z in enumerate(unique_numbers):
            sym = chemical_symbols[Z]
            if sym not in parameters:
                raise NotImplementedError(f'No EMT-potential for {sym}')
            p = parameters[sym]
            s0 = p[1] * Bohr
            eta2 = p[3] / Bohr
            kappa = p[4] / Bohr
            gamma1, gamma2 = self._calc_gammas(s0, eta2, kappa)
            self.par['Z'][i] = Z
            self.par['E0'][i] = p[0]
            self.par['s0'][i] = s0
            self.par['V0'][i] = p[2]
            self.par['eta2'][i] = eta2
            self.par['kappa'][i] = kappa
            self.par['lambda'][i] = p[5] / Bohr
            self.par['n0'][i] = p[6] / Bohr**3
            self.par['inv12gamma1'][i] = 1.0 / (12.0 * gamma1)
            self.par['neghalfv0overgamma2'][i] = -0.5 * p[2] / gamma2

        self.chi = self.par['n0'][None, :] / self.par['n0'][:, None]

        self.energies = np.empty(len(atoms))
        self.forces = np.empty((len(atoms), 3))
        self.stress = np.empty((3, 3))
        self.deds = np.empty(len(atoms))

        self.nl = NeighborList([0.5 * self.rc_list] * len(atoms),
                               self_interaction=False, bothways=True)

    def _calc_cutoff(self, atoms):
        """Calculate parameters of the logistic smoothing function etc.

        The logistic smoothing function is given by

        .. math:

            w(r) = \\frac{1}{1 + \\exp a (r - r_\\mathrm{c})}

        Returns
        -------
        rc : float
            "Midpoint" of the logistic smoothing function, set to be the mean
            of the 3rd and the 4th nearest-neighbor distances in FCC.
        rc_list : float
            Cutoff radius for the neighbor search, set to be slightly larger
            than ``rc`` depending on ``asap_cutoff``.
        acut : float
            "Slope" of the smoothing function, set for the smoothing function
            value to be ``1e-4`` at the 4th nearest-neighbor distance in FCC.

        Notes
        -----
        ``maxseq`` is the present FCC Wigner-Seitz radius. ``beta * maxseq``
        (`r1nn`) is the corresponding 1st nearest-neighbor distance in FCC.
        The 2nd, 3rd, 4th nearest-neighbor distances in FCC are given using
        ``r1nn`` by ``sqrt(2) * r1nn``, ``sqrt(3) * r1nn``, ``sqrt(4) * r1nn``,
        respectively.
        """
        numbers = atoms.get_atomic_numbers()
        if self.parameters['asap_cutoff']:
            relevant_pars = {
                symb: p
                for symb, p in parameters.items()
                if atomic_numbers[symb] in numbers
            }
        else:
            relevant_pars = parameters
        maxseq = max(par[1] for par in relevant_pars.values()) * Bohr
        r1nn = beta * maxseq  # 1st NN distance in FCC
        rc = r1nn * 0.5 * (sqrt(3.0) + 2.0)  # mean of 3NN and 4NN dists.
        r4nn = r1nn * 2.0  # 4NN distance in FCC
        eps = 1e-4  # value at r4nn, should be small

        # "slope" is set so that the function value becomes eps at r4nn
        acut = log(1.0 / eps - 1.0) / (r4nn - rc)

        rc_list = rc * 1.045 if self.parameters['asap_cutoff'] else rc + 0.5

        return rc, rc_list, acut

    def _calc_gammas(self, s0, eta2, kappa):
        n = np.array([12, 6, 24])  # numbers of 1, 2, 3NN sites in fcc
        r = beta * s0 * np.sqrt([1.0, 2.0, 3.0])  # distances of 1, 2, 3NNs
        w = 1.0 / (1.0 + np.exp(self.acut * (r - self.rc)))
        x = n * w / 12.0
        gamma1 = x @ np.exp(-eta2 * (r - beta * s0))
        gamma2 = x @ np.exp(-kappa / beta * (r - beta * s0))
        return gamma1, gamma2

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        if 'numbers' in system_changes:
            self.initialize(self.atoms)

        self.nl.update(self.atoms)

        self.energies[:] = 0.0
        self.forces[:] = 0.0
        self.stress[:] = 0.0
        self.deds[:] = 0.0

        natoms = len(self.atoms)

        # store nearest neighbor info for all the atoms
        # suffixes 's' and 'o': contributions from self and the other atoms
        ps = {}
        for a1 in range(natoms):
            a2, d, r = self._get_neighbors(a1)
            if len(a2) == 0:
                continue
            w, dwdroverw = self._calc_theta(r)
            dsigma1s, dsigma1o = self._calc_dsigma1(a1, a2, r, w)
            dsigma2s, dsigma2o = self._calc_dsigma2(a1, a2, r, w)
            ps[a1] = {
                'a2': a2,
                'd': d,
                'r': r,
                'invr': 1.0 / r,
                'w': w,
                'dwdroverw': dwdroverw,
                'dsigma1s': dsigma1s,
                'dsigma1o': dsigma1o,
                'dsigma2s': dsigma2s,
                'dsigma2o': dsigma2o,
            }

        # deds is computed in _calc_e_c_a2
        # since deds for all the atoms are used later in _calc_f_c_a2,
        # _calc_e_c_a2 must be called beforehand for all the atoms
        for a1, p in ps.items():
            a2 = p['a2']
            dsigma1s = p['dsigma1s']
            self._calc_e_c_a2(a1, dsigma1s)

        for a1, p in ps.items():
            a2 = p['a2']
            d = p['d']
            invr = p['invr']
            dwdroverw = p['dwdroverw']
            dsigma1s = p['dsigma1s']
            dsigma1o = p['dsigma1o']
            dsigma2s = p['dsigma2s']
            dsigma2o = p['dsigma2o']
            self._calc_fs_c_a2(a1, a2, d, invr, dwdroverw, dsigma1s, dsigma1o)
            self._calc_efs_a1(a1, a2, d, invr, dwdroverw, dsigma2s, dsigma2o)

        # subtract E0 (ASAP convention)
        self.energies -= self.par['E0'][self.ia2iz]

        energy = np.add.reduce(self.energies, axis=0)
        self.results['energy'] = self.results['free_energy'] = energy
        self.results['energies'] = self.energies
        self.results['forces'] = self.forces

        if self.atoms.cell.rank == 3:
            self.stress = (self.stress + self.stress.T) * 0.5  # symmetrize
            self.stress /= self.atoms.get_volume()
            self.results['stress'] = self.stress.flat[[0, 4, 8, 5, 2, 1]]
        elif 'stress' in properties:
            raise PropertyNotImplementedError

    def _get_neighbors(self, a1):
        positions = self.atoms.positions
        cell = self.atoms.cell
        neighbors, offsets = self.nl.get_neighbors(a1)
        offsets = np.dot(offsets, cell)
        d = positions[neighbors] + offsets - positions[a1]
        r = np.sqrt(np.add.reduce(d**2, axis=1))
        mask = r < self.rc_list
        return neighbors[mask], d[mask], r[mask]

    def _calc_theta(self, r):
        """Calculate cutoff function and its r derivative"""
        w = 1.0 / (1.0 + np.exp(self.acut * (r - self.rc)))
        dwdroverw = self.acut * (w - 1.0)
        return w, dwdroverw

    def _calc_dsigma1(self, a1, a2, r, w):
        """Calculate contributions of neighbors to sigma1"""
        s0s = self.par['s0'][self.ia2iz[a1]]
        s0o = self.par['s0'][self.ia2iz[a2]]
        eta2s = self.par['eta2'][self.ia2iz[a1]]
        eta2o = self.par['eta2'][self.ia2iz[a2]]
        chi = self.chi[self.ia2iz[a1], self.ia2iz[a2]]

        dsigma1s = np.exp(-eta2o * (r - beta * s0o)) * chi * w
        dsigma1o = np.exp(-eta2s * (r - beta * s0s)) / chi * w

        return dsigma1s, dsigma1o

    def _calc_dsigma2(self, a1, a2, r, w):
        """Calculate contributions of neighbors to sigma2"""
        s0s = self.par['s0'][self.ia2iz[a1]]
        s0o = self.par['s0'][self.ia2iz[a2]]
        kappas = self.par['kappa'][self.ia2iz[a1]]
        kappao = self.par['kappa'][self.ia2iz[a2]]
        chi = self.chi[self.ia2iz[a1], self.ia2iz[a2]]

        dsigma2s = np.exp(-kappao * (r / beta - s0o)) * chi * w
        dsigma2o = np.exp(-kappas * (r / beta - s0s)) / chi * w

        return dsigma2s, dsigma2o

    def _calc_e_c_a2(self, a1, dsigma1s):
        """Calculate E_c and the second term of E_AS and their s derivatives"""
        e0s = self.par['E0'][self.ia2iz[a1]]
        v0s = self.par['V0'][self.ia2iz[a1]]
        eta2s = self.par['eta2'][self.ia2iz[a1]]
        lmds = self.par['lambda'][self.ia2iz[a1]]
        kappas = self.par['kappa'][self.ia2iz[a1]]
        inv12gamma1s = self.par['inv12gamma1'][self.ia2iz[a1]]

        sigma1 = np.add.reduce(dsigma1s)
        ds = -1.0 * np.log(sigma1 * inv12gamma1s) / (beta * eta2s)

        lmdsds = lmds * ds
        expneglmdds = np.exp(-1.0 * lmdsds)
        self.energies[a1] += e0s * (1.0 + lmdsds) * expneglmdds
        self.deds[a1] += -1.0 * e0s * lmds * lmdsds * expneglmdds

        sixv0expnegkppds = 6.0 * v0s * np.exp(-1.0 * kappas * ds)
        self.energies[a1] += sixv0expnegkppds
        self.deds[a1] += -1.0 * kappas * sixv0expnegkppds

        self.deds[a1] /= -1.0 * beta * eta2s * sigma1  # factor from ds/dr

    def _calc_efs_a1(self, a1, a2, d, invr, dwdroverw, dsigma2s, dsigma2o):
        """Calculate the first term of E_AS and derivatives"""
        neghalfv0overgamma2s = self.par['neghalfv0overgamma2'][self.ia2iz[a1]]
        neghalfv0overgamma2o = self.par['neghalfv0overgamma2'][self.ia2iz[a2]]
        kappas = self.par['kappa'][self.ia2iz[a1]]
        kappao = self.par['kappa'][self.ia2iz[a2]]

        es = neghalfv0overgamma2s * dsigma2s
        eo = neghalfv0overgamma2o * dsigma2o
        self.energies[a1] += 0.5 * np.add.reduce(es + eo, axis=0)

        dedrs = es * (dwdroverw - kappao / beta)
        dedro = eo * (dwdroverw - kappas / beta)
        f = ((dedrs + dedro) * invr)[:, None] * d
        self.forces[a1] += np.add.reduce(f, axis=0)
        self.stress += 0.5 * np.dot(d.T, f)  # compensate double counting

    def _calc_fs_c_a2(self, a1, a2, d, invr, dwdroverw, dsigma1s, dsigma1o):
        """Calculate forces and stress from E_c and the second term of E_AS"""
        eta2s = self.par['eta2'][self.ia2iz[a1]]
        eta2o = self.par['eta2'][self.ia2iz[a2]]

        ddsigma1sdr = dsigma1s * (dwdroverw - eta2o)
        ddsigma1odr = dsigma1o * (dwdroverw - eta2s)
        dedrs = self.deds[a1] * ddsigma1sdr
        dedro = self.deds[a2] * ddsigma1odr
        f = ((dedrs + dedro) * invr)[:, None] * d
        self.forces[a1] += np.add.reduce(f, axis=0)
        self.stress += 0.5 * np.dot(d.T, f)  # compensate double counting
