import numpy as np
import pytest
from numpy.testing import assert_allclose

from ase.build import bulk
from ase.calculators.fd import calculate_numerical_stress
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS


@pytest.fixture(name="atoms")
def fixture_atoms():
    rng = np.random.RandomState(17)
    atoms = bulk('Pt') * (2, 2, 2)
    atoms.rattle(stdev=0.1)
    atoms.cell += 2 * rng.random((3, 3))
    return atoms


@pytest.mark.calculator_lite()
@pytest.mark.calculator('lammpslib')
def test_Pt_stress_cellopt(atoms, factory):
    """Test if the stresses and the optimized cell are as expected.

    This test is taken from the one with the same name from lammpsrun.
    """
    lmpcmds = ['pair_style eam', 'pair_coeff 1 1 Pt_u3.eam']
    with factory.calc(lmpcmds=lmpcmds) as calc:
        atoms.calc = calc
        assert_allclose(atoms.get_stress(),
                        calculate_numerical_stress(atoms),
                        atol=1e-4, rtol=1e-4)

        with BFGS(FrechetCellFilter(atoms)) as opt:
            opt.run(fmax=0.001)

        cell1_ref = np.array([
            [0.178351, 3.885347, 3.942046],
            [4.19978, 0.591071, 5.062568],
            [4.449044, 3.264038, 0.471548],
        ])

        assert_allclose(np.asarray(atoms.cell), cell1_ref,
                        atol=3e-4, rtol=3e-4)
        assert_allclose(atoms.get_stress(),
                        calculate_numerical_stress(atoms),
                        atol=1e-4, rtol=1e-4)
