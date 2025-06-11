from numpy.testing import assert_allclose

import ase.build
from ase.calculators.emt import EMT


def test_potential_energies():
    TOL = 1E-8

    atoms = ase.build.bulk("Ni", crystalstructure="fcc", cubic=1)
    atoms *= (2, 2, 2)
    atoms.calc = EMT()
    energies = atoms.get_potential_energies()
    energy = atoms.get_potential_energy()

    # energy sums should be identical
    assert abs(energies.sum() - energy) < TOL

    # in ideal FCC crystal per-atom energies should be identical
    assert_allclose(energies, energies[0], rtol=TOL)

    # rattle the system and check energy sums again
    atoms.rattle()
    assert abs(energies.sum() - energy) < TOL
