# fmt: off

from typing import Tuple

import numpy as np

from ase.data import covalent_radii
from ase.neighborlist import NeighborList
from ase.units import Bohr, Ha

from .polarizability import StaticPolarizabilityCalculator


class LippincottStuttman:
    # atomic polarizability values from:
    #   Lippincott and Stutman J. Phys. Chem. 68 (1964) 2926-2940
    #   DOI: 10.1021/j100792a033
    # see also:
    #   Marinov and Zotov Phys. Rev. B 55 (1997) 2938-2944
    #   DOI: 10.1103/PhysRevB.55.2938
    # unit: Angstrom^3
    atomic_polarizability = {
        'H': 0.592,
        'Be': 3.802,
        'B': 1.358,
        'C': 0.978,
        'N': 0.743,
        'O': 0.592,
        'Al': 3.918,
        'Si': 2.988,
        'P': 2.367,
        'S': 1.820,
    }
    # reduced electronegativity Table I
    reduced_eletronegativity = {
        'H': 1.0,
        'Be': 0.538,
        'B': 0.758,
        'C': 0.846,
        'N': 0.927,
        'O': 1.0,
        'Al': 0.533,
        'Si': 0.583,
        'P': 0.630,
        'S': 0.688,
    }

    def __call__(self, el1: str, el2: str,
                 length: float) -> Tuple[float, float]:
        """Bond polarizability

        Parameters
        ----------
        el1: element string
        el2: element string
        length: float

        Returns
        -------
        alphal: float
          Parallel component
        alphap: float
          Perpendicular component
        """
        alpha1 = self.atomic_polarizability[el1]
        alpha2 = self.atomic_polarizability[el2]
        ren1 = self.reduced_eletronegativity[el1]
        ren2 = self.reduced_eletronegativity[el2]

        sigma = 1.
        if el1 != el2:
            sigma = np.exp(- (ren1 - ren2)**2 / 4)

        # parallel component
        alphal = sigma * length**4 / (4**4 * alpha1 * alpha2)**(1. / 6)
        # XXX consider fractional covalency ?

        # prependicular component
        alphap = ((ren1**2 * alpha1 + ren2**2 * alpha2)
                  / (ren1**2 + ren2**2))
        # XXX consider fractional covalency ?

        return alphal, alphap


class Linearized:
    def __init__(self):
        self._data = {
            # L. Wirtz, M. Lazzeri, F. Mauri, A. Rubio,
            # Phys. Rev. B 2005, 71, 241402.
            #      R0     al    al'   ap    ap'
            'CC': (1.53, 1.69, 7.43, 0.71, 0.37),
            'BN': (1.56, 1.58, 4.22, 0.42, 0.90),
        }

    def __call__(self, el1: str, el2: str,
                 length: float) -> Tuple[float, float]:
        """Bond polarizability

        Parameters
        ----------
        el1: element string
        el2: element string
        length: float

        Returns
        -------
        alphal: float
          Parallel component
        alphap: float
          Perpendicular component
        """
        if el1 > el2:
            bond = el2 + el1
        else:
            bond = el1 + el2
        assert bond in self._data
        length0, al, ald, ap, apd = self._data[bond]

        return al + ald * (length - length0), ap + apd * (length - length0)


class BondPolarizability(StaticPolarizabilityCalculator):
    def __init__(self, model=LippincottStuttman()):
        self.model = model

    def __call__(self, atoms, radiicut=1.5):
        """Sum up the bond polarizability from all bonds

        Parameters
        ----------
        atoms: Atoms object
        radiicut: float
          Bonds are counted up to
          radiicut * (sum of covalent radii of the pairs)
          Default: 1.5

        Returns
        -------
        polarizability tensor with unit (e^2 Angstrom^2 / eV).
        Multiply with Bohr * Ha to get (Angstrom^3)
        """
        radii = np.array([covalent_radii[z]
                          for z in atoms.numbers])
        nl = NeighborList(radii * 1.5, skin=0,
                          self_interaction=False)
        nl.update(atoms)
        pos_ac = atoms.get_positions()

        alpha = 0
        for ia, atom in enumerate(atoms):
            indices, offsets = nl.get_neighbors(ia)
            pos_ac = atoms.get_positions() - atoms.get_positions()[ia]

            for ib, offset in zip(indices, offsets):
                weight = 1
                if offset.any():  # this comes from a periodic image
                    weight = 0.5  # count half the bond only

                dist_c = pos_ac[ib] + np.dot(offset, atoms.get_cell())
                dist = np.linalg.norm(dist_c)
                al, ap = self.model(atom.symbol, atoms[ib].symbol, dist)

                eye3 = np.eye(3) / 3
                alpha += weight * (al + 2 * ap) * eye3
                alpha += weight * (al - ap) * (
                    np.outer(dist_c, dist_c) / dist**2 - eye3)
        return alpha / Bohr / Ha
