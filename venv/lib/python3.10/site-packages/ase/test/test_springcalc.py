import numpy as np

from ase.build import bulk
from ase.calculators.harmonic import SpringCalculator
from ase.calculators.test import gradient_test


def test_springcalc():
    # setup
    k = 3.0
    atoms_ideal = bulk('Al').repeat(3)
    calc = SpringCalculator(atoms_ideal.get_positions(), k)
    displacements = np.array([(d, 2 * d, 3 * d)
                             for d in np.linspace(0, 1, len(atoms_ideal))])

    # calc forces and energy
    atoms = atoms_ideal.copy()
    atoms.positions += displacements
    atoms.calc = calc
    forces = atoms.get_forces()
    Epot = atoms.get_potential_energy()

    # reference forces and energy
    Epot_target = np.sum(k / 2.0 * displacements**2)
    forces_target = - k * displacements

    assert np.allclose(forces, forces_target)
    assert np.isclose(Epot, Epot_target)

    # numeric forces test
    atoms_ideal.calc = calc
    f, fn = gradient_test(atoms_ideal)
    assert abs(f - fn).max() < 1e-10
