# fmt: off

import numpy as np

from ase import units
from ase.calculators.calculator import Calculator

k_c = units.Hartree * units.Bohr


class AtomicCounterIon(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, charge, epsilon, sigma, sites_per_mol=1,
                 rc=7.0, width=1.0):
        """ Counter Ion Calculator.

        A very simple, nonbonded (Coulumb and LJ)
        interaction calculator meant for single atom ions
        to charge neutralize systems (and nothing else)...
        """
        self.rc = rc
        self.width = width
        self.sites_per_mol = sites_per_mol
        self.epsilon = epsilon
        self.sigma = sigma
        self.charge = charge
        Calculator.__init__(self)

    def add_virtual_sites(self, positions):
        return positions

    def get_virtual_charges(self, atoms):
        charges = np.tile(self.charge, len(atoms) // self.sites_per_mol)
        return charges

    def redistribute_forces(self, forces):
        return forces

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        R = atoms.get_positions()
        charges = self.get_virtual_charges(atoms)
        pbc = atoms.pbc

        energy = 0.0
        forces = np.zeros_like(atoms.get_positions())

        for m in range(len(atoms)):
            D = R[m + 1:] - R[m]
            shift = np.zeros_like(D)
            for i, periodic in enumerate(pbc):
                if periodic:
                    L = atoms.cell.diagonal()[i]
                    shift[:, i] = (D[:, i] + L / 2) % L - L / 2 - D[:, i]
            D += shift
            d2 = (D**2).sum(1)
            d = d2**0.5

            x1 = d > self.rc - self.width
            x2 = d < self.rc
            x12 = np.logical_and(x1, x2)
            y = (d[x12] - self.rc + self.width) / self.width
            t = np.zeros(len(d))  # cutoff function
            t[x2] = 1.0
            t[x12] -= y**2 * (3.0 - 2.0 * y)
            dtdd = np.zeros(len(d))
            dtdd[x12] -= 6.0 / self.width * y * (1.0 - y)

            c6 = (self.sigma**2 / d2)**3
            c12 = c6**2
            e_lj = 4 * self.epsilon * (c12 - c6)
            e_c = k_c * charges[m + 1:] * charges[m] / d

            energy += np.dot(t, e_lj)
            energy += np.dot(t, e_c)

            F = (24 * self.epsilon * (2 * c12 - c6) / d2 * t -
                 e_lj * dtdd / d)[:, None] * D

            forces[m] -= F.sum(0)
            forces[m + 1:] += F

            F = (e_c / d2 * t)[:, None] * D \
                - (e_c * dtdd / d)[:, None] * D

            forces[m] -= F.sum(0)
            forces[m + 1:] += F

        self.results['energy'] = energy
        self.results['forces'] = forces
