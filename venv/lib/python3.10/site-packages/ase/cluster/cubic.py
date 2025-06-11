# fmt: off

"""
Function-like objects that creates cubic clusters.
"""

import numpy as np

from ase.cluster.factory import ClusterFactory
from ase.data import reference_states as _refstate


class SimpleCubicFactory(ClusterFactory):
    spacegroup = 221

    xtal_name = 'sc'

    def get_lattice_constant(self):
        "Get the lattice constant of an element with cubic crystal structure."
        symmetry = _refstate[self.atomic_numbers[0]]['symmetry']
        if symmetry != self.xtal_name:
            raise ValueError(f"Cannot guess the {self.xtal_name} " +
                             "lattice constant of an element with crystal " +
                             f"structure {symmetry}.")
        return _refstate[self.atomic_numbers[0]]['a']

    def set_basis(self):
        a = self.lattice_constant
        if not isinstance(a, (int, float)):
            raise ValueError(
                f"Improper lattice constant for {self.xtal_name} crystal.")

        self.lattice_basis = np.array([[a, 0., 0.],
                                       [0., a, 0.],
                                       [0., 0., a]])

        self.resiproc_basis = self.get_resiproc_basis(self.lattice_basis)


SimpleCubic = SimpleCubicFactory()


class BodyCenteredCubicFactory(SimpleCubicFactory):
    spacegroup = 229

    xtal_name = 'bcc'

    atomic_basis = np.array([[0., 0., 0.],
                             [.5, .5, .5]])


BodyCenteredCubic = BodyCenteredCubicFactory()


class FaceCenteredCubicFactory(SimpleCubicFactory):
    spacegroup = 225

    xtal_name = 'fcc'

    atomic_basis = np.array([[0., 0., 0.],
                             [0., .5, .5],
                             [.5, 0., .5],
                             [.5, .5, 0.]])


FaceCenteredCubic = FaceCenteredCubicFactory()
