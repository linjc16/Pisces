# fmt: off

import math

import numpy as np

from ase import Atoms
from ase.cluster.base import ClusterBase


class Cluster(Atoms, ClusterBase):
    symmetry = None
    surfaces = None
    lattice_basis = None
    resiproc_basis = None
    atomic_basis = None

    def copy(self):
        cluster = Atoms.copy(self)
        cluster.symmetry = self.symmetry
        cluster.surfaces = self.surfaces.copy()
        cluster.lattice_basis = self.lattice_basis.copy()
        cluster.atomic_basis = self.atomic_basis.copy()
        cluster.resiproc_basis = self.resiproc_basis.copy()
        return cluster

    def get_surfaces(self):
        """Returns the miller indexs of the stored surfaces of the cluster."""
        if self.surfaces is not None:
            return self.surfaces.copy()
        else:
            return None

    def get_layers(self):
        """Return number of atomic layers in stored surfaces directions."""

        layers = []

        for s in self.surfaces:
            n = self.miller_to_direction(s)
            c = self.get_positions().mean(axis=0)
            r = np.dot(self.get_positions() - c, n).max()
            d = self.get_layer_distance(s, 2)
            l_ = 2 * np.round(r / d).astype(int)

            ls = np.arange(l_ - 1, l_ + 2)
            ds = np.array([self.get_layer_distance(s, i) for i in ls])

            mask = (np.abs(ds - r) < 1e-10)

            layers.append(ls[mask][0])

        return np.array(layers, int)

    def get_diameter(self, method='volume'):
        """Returns an estimate of the cluster diameter based on two different
        methods.

        Parameters
        ----------
        method : {'volume', 'shape'}
            'volume' (default) returns the diameter of a sphere with the same
            volume as the atoms. 'shape' returns the averaged diameter
            calculated from the directions given by the defined surfaces.
        """

        if method == 'shape':
            cen = self.get_positions().mean(axis=0)
            pos = self.get_positions() - cen
            d = 0.0
            for s in self.surfaces:
                n = self.miller_to_direction(s)
                r = np.dot(pos, n)
                d += r.max() - r.min()
            return d / len(self.surfaces)
        elif method == 'volume':
            V_cell = np.abs(np.linalg.det(self.lattice_basis))
            N_cell = len(self.atomic_basis)
            N = len(self)
            return 2.0 * (3.0 * N * V_cell /
                          (4.0 * math.pi * N_cell)) ** (1.0 / 3.0)
        else:
            return 0.0
