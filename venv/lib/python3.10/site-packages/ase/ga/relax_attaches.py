# fmt: off

""" An object which can be associated with a local relaxation in order
to make the relaxations run more smoothly."""
from math import sqrt

import numpy as np


class VariansBreak:

    """ Helper class which can be attached to a structure optimization,
        in order to terminale stalling calculations.

        Parameters:

        atoms: Atoms object being optimized
        dyn: The relaxation object being used
        min_stdev: The limiting std. deviation in forces to terminate at
        N: The number of steps used to calculate the st. dev.
    """

    def __init__(self, atoms, dyn, min_stdev=0.005, N=15):
        self.atoms = atoms
        self.dyn = dyn
        self.N = N
        self.forces = []
        self.min_stdev = min_stdev

    def write(self):
        """ The method called by the optimizer in each step. """
        if len(self.forces) >= self.N:
            self.forces.pop(0)
        fmax = (self.atoms.get_forces()**2).sum(axis=1).max()**0.5
        self.forces.append(fmax)

        m = sum(self.forces) / float(len(self.forces))

        stdev = sqrt(
            (sum((c - m)**2 for c in self.forces) / float(len(self.forces))))

        if len(self.forces) >= self.N and stdev < self.min_stdev:
            self.dyn.converged = lambda x: True


class DivergenceBreak:

    """ Helper class which can be attached to a structure optimization,
        in order to terminate diverging calculations.

        Parameters:

        atoms: Atoms object being optimized
        dyn: The relaxation object being used
        N: The maximum number of recent steps to be included in the
           evaluation of the slope
        Nmin: The minimal amount of steps required before evaluating
              the slope
    """

    def __init__(self, atoms, dyn, N=15, Nmin=5):
        self.atoms = atoms
        self.dyn = dyn
        self.N = N
        self.Nmin = 5
        self.energies = []

    def write(self):
        """ The method called by the optimizer in each step. """

        if len(self.energies) >= self.N:
            self.energies.pop(0)
        self.energies.append(self.atoms.get_potential_energy())

        if len(self.energies) > self.Nmin:
            x = np.array(range(len(self.energies)))
            y = np.array(self.energies)
            A = np.vstack([x, np.ones(len(x))]).T
            slope, _intersect = np.linalg.lstsq(A, y)[0]

            if len(self.energies) >= self.N and slope > 0:
                self.dyn.converged = lambda x: True
