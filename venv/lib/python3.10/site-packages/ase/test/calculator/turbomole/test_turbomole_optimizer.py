# type: ignore
import numpy as np
import pytest

from ase.build import molecule
from ase.calculators.turbomole import Turbomole, TurbomoleOptimizer


@pytest.fixture()
def atoms():
    return molecule('H2O')


@pytest.fixture()
def calc():
    params = {
        'title': 'water',
        'basis set name': 'sto-3g hondo',
        'total charge': 0,
        'multiplicity': 1,
        'use dft': True,
        'density functional': 'b-p',
        'use resolution of identity': True,
    }
    return Turbomole(**params)


def test_turbomole_optimizer_class(atoms, calc, turbomole_factory):
    optimizer = TurbomoleOptimizer(atoms, calc)
    optimizer.run(steps=1)
    assert isinstance(optimizer.todict(), dict)


def test_turbomole_optimizer(atoms, calc, turbomole_factory):
    optimizer = calc.get_optimizer(atoms)
    optimizer.run(fmax=0.01, steps=5)
    assert np.linalg.norm(calc.get_forces(atoms)) < 0.01
