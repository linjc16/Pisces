# fmt: off

"""Ideal gas calculator - the potential energy is always zero."""

import numpy as np

from ase.calculators.calculator import Calculator, all_changes


class IdealGas(Calculator):
    """The ideal gas: non-interacting atoms.

    The ideal gas is atoms that do not interact.  The potential is thus
    always zero, and so are forces and stresses (apart from the dynamic
    part of the stress, which is handled by the atoms themselves).

    This calculator is probably only useful for testing purposes.
    """

    implemented_properties = ['energy', 'energies', 'forces',
                              'stress', 'stresses']

    def calculate(self, atoms=None, properties=[],
                  system_changes=all_changes):
        """'Calculate' the zero energies and their derivatives."""
        super().calculate(atoms, properties, system_changes)
        n = len(self.atoms)
        self.results = {
            'energy': 0.0,
            'energies': np.zeros(n),
            'forces': np.zeros((n, 3)),
            'stress': np.zeros(6),
            'stresses': np.zeros((n, 6)),
        }
