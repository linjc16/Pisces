import numpy as np

from ase.build import molecule
from ase.calculators.qmmm import ForceConstantCalculator
from ase.optimize import MDMin


def test_mdmin_maxstep():
    atoms = molecule("N2", vacuum=10)
    calc = ForceConstantCalculator(np.eye(6) * 100,
                                   atoms.copy(), np.zeros((2, 3)))
    atoms.calc = calc
    atoms.positions[1] += np.array([1.0, 1.0, 1.0])
    initial_positions = atoms.positions.copy()
    # Both atoms now have huge forces on them.

    # This is a very large dt to force the maxstep clipping to trigger.
    opt = MDMin(atoms, dt=1.0)
    opt.run(steps=1)

    assert np.max(
        np.linalg.norm(atoms.positions - initial_positions, axis=1)) <= 0.2
