import numpy as np
import pytest

from ase.build import bulk
from ase.calculators.fd import calculate_numerical_stress
from ase.calculators.lj import LennardJones
from ase.filters import UnitCellFilter
from ase.optimize import BFGS

# Theoretical infinite-cutoff LJ FCC unit cell parameters
vol0 = 4 * 0.91615977036  # theoretical minimum
a0 = vol0**(1 / 3)


@pytest.fixture()
def atoms():
    """two atoms at potential minimum"""
    atoms = bulk('X', 'fcc', a=a0)

    atoms.calc = LennardJones()

    return atoms


def test_stress_voigt_shape(atoms):
    # test voigt shape
    for ideal_gas in (False, True):
        kw = {'include_ideal_gas': ideal_gas}

        assert atoms.get_stress(voigt=True, **kw).shape == (6,)
        assert atoms.get_stress(voigt=False, **kw).shape == (3, 3)

        assert atoms.get_stresses(voigt=True, **kw).shape == (len(atoms), 6)
        assert atoms.get_stresses(voigt=False, **kw).shape == (len(atoms), 3, 3)


@pytest.mark.optimize()
@pytest.mark.slow()
def test_stress(atoms):
    cell0 = atoms.get_cell()

    atoms.set_cell(np.dot(atoms.cell,
                          [[1.02, 0, 0.03],
                           [0, 0.99, -0.02],
                           [0.1, -0.01, 1.03]]),
                   scale_atoms=True)

    atoms *= (1, 2, 3)
    cell0 *= np.array([1, 2, 3])[:, np.newaxis]

    atoms.rattle()

    # Verify analytical stress tensor against numerical value
    s_analytical = atoms.get_stress()
    s_numerical = calculate_numerical_stress(atoms, 1e-5)
    s_p_err = 100 * (s_numerical - s_analytical) / s_numerical

    print("Analytical stress:\n", s_analytical)
    print("Numerical stress:\n", s_numerical)
    print("Percent error in stress:\n", s_p_err)
    assert np.all(abs(s_p_err) < 1e-5)

    # Minimize unit cell
    opt = BFGS(UnitCellFilter(atoms))
    opt.run(fmax=1e-3)

    # Verify minimized unit cell using Niggli tensors
    g_minimized = np.dot(atoms.cell, atoms.cell.T)
    g_theory = np.dot(cell0, cell0.T)
    g_p_err = 100 * (g_minimized - g_theory) / g_theory

    print("Minimized Niggli tensor:\n", g_minimized)
    print("Theoretical Niggli tensor:\n", g_theory)
    print("Percent error in Niggli tensor:\n", g_p_err)
    assert np.all(abs(g_p_err) < 1)
