from itertools import product

import numpy as np
import pytest

import ase
from ase.build import bulk
from ase.calculators.test import gradient_test
from ase.filters import ExpCellFilter, Filter, FrechetCellFilter, UnitCellFilter
from ase.io import Trajectory
from ase.optimize import LBFGS, MDMin
from ase.units import GPa


@pytest.fixture()
def atoms(asap3) -> ase.Atoms:
    rng = np.random.RandomState(0)
    atoms = bulk('Cu', cubic=True)
    atoms.positions[:, 0] *= 0.995
    atoms.cell += rng.uniform(-1e-2, 1e-2, size=9).reshape((3, 3))
    atoms.calc = asap3.EMT()
    return atoms


@pytest.mark.optimize()
@pytest.mark.filterwarnings("ignore:Use FrechetCellFilter")
@pytest.mark.parametrize(
    'cellfilter', [UnitCellFilter, FrechetCellFilter, ExpCellFilter]
)
def test_get_and_set_positions(atoms, cellfilter):
    filter: Filter = cellfilter(atoms)
    pos = filter.get_positions()
    filter.set_positions(pos)
    pos2 = filter.get_positions()
    assert np.allclose(pos, pos2)


@pytest.mark.filterwarnings("ignore:Use FrechetCellFilter")
@pytest.mark.parametrize(
    'cellfilter', [UnitCellFilter, FrechetCellFilter, ExpCellFilter]
)
def test_pressure(atoms, cellfilter):
    xcellfilter = cellfilter(atoms, scalar_pressure=10.0 * GPa)

    # test all derivatives
    f, fn = gradient_test(xcellfilter)
    assert abs(f - fn).max() < 5e-6

    opt = LBFGS(xcellfilter)
    opt.run(1e-3)

    # check pressure is within 0.1 GPa of target
    sigma = atoms.get_stress() / GPa
    pressure = -(sigma[0] + sigma[1] + sigma[2]) / 3.0
    assert abs(pressure - 10.0) < 0.1


@pytest.mark.filterwarnings("ignore:Use FrechetCellFilter")
@pytest.mark.parametrize(
    'cellfilter', [UnitCellFilter, FrechetCellFilter, ExpCellFilter]
)
def test_cellfilter_forces(atoms, cellfilter):
    xcellfilter = cellfilter(atoms)
    f, fn = gradient_test(xcellfilter)
    assert abs(f - fn).max() < 3e-6


@pytest.mark.parametrize('cellfilter,cell_factor_name, cell_factor_value', [
    (UnitCellFilter, 'cell_factor', 1),
    (UnitCellFilter, 'cell_factor', None),
    (FrechetCellFilter, 'exp_cell_factor', 1),
    (FrechetCellFilter, 'exp_cell_factor', None),
    # Never apply this test to ExpCellFilter because it is known to fail!
])
def test_cellfilter_stress(
    atoms: ase.Atoms,
    cellfilter,
    cell_factor_name: str,
    cell_factor_value
):
    filter: Filter = cellfilter(**{
        "atoms": atoms,
        cell_factor_name: cell_factor_value
    })

    # Check gradient at other than origin
    natoms = len(atoms)
    pos0 = filter.get_positions()
    rng = np.random.RandomState(0)
    pos0[natoms:, :] += 1e-2 * rng.randn(3, 3)
    filter.set_positions(pos0)
    grads_actual = -filter.get_forces()

    eps = 1e-4
    for alpha, beta in product(range(3), repeat=2):
        pos_p = pos0.copy()
        pos_p[natoms + alpha, beta] += eps
        filter.set_positions(pos_p)
        energy_p = filter.get_potential_energy()

        pos_m = pos0.copy()
        pos_m[natoms + alpha, beta] -= eps
        filter.set_positions(pos_m)
        energy_m = filter.get_potential_energy()

        expect = (energy_p - energy_m) / (2 * eps)
        actual = grads_actual[natoms + alpha, beta]
        assert np.isclose(actual, expect, atol=1e-4)


def test_intensive_cell_gradient(atoms: ase.Atoms):
    """Gradient w.r.t. cell variables for FrechetCellFilter should provide
    intensive values with an appropriate scaling factor.
    """
    filter = FrechetCellFilter(atoms, exp_cell_factor=float(len(atoms)))
    cell_grad = filter.get_forces()[-3:]

    atoms2 = atoms.copy()
    atoms2 *= (2, 2, 2)
    atoms2.calc = atoms.calc
    filter2 = FrechetCellFilter(atoms2, exp_cell_factor=float(len(atoms2)))
    cell_grad2 = filter2.get_forces()[-3:]

    assert np.allclose(cell_grad, cell_grad2)


@pytest.mark.filterwarnings("ignore:Use FrechetCellFilter")
@pytest.mark.parametrize(
    'cellfilter', [UnitCellFilter, FrechetCellFilter, ExpCellFilter]
)
def test_constant_volume(atoms: ase.Atoms, cellfilter):
    atoms_opt = atoms.copy()
    atoms_opt.calc = atoms.calc
    filter: Filter = cellfilter(atoms_opt, constant_volume=True)
    opt = LBFGS(filter)  # type: ignore[arg-type]
    opt.run()

    # Check if volume is conserved
    assert not np.allclose(atoms.cell.array, atoms_opt.cell.array)
    assert np.isclose(atoms.get_volume(), atoms_opt.get_volume())


# XXX This test should have some assertions!  --askhl
@pytest.mark.optimize()
def test_unitcellfilter(asap3, testdir):
    cu = bulk('Cu') * (6, 6, 6)
    cu.calc = asap3.EMT()
    f = UnitCellFilter(cu, [1, 1, 1, 0, 0, 0])
    opt = LBFGS(f)

    with Trajectory('Cu-fcc.traj', 'w', cu) as t:
        opt.attach(t)
        opt.run(5.0)
    # No assertions??


@pytest.mark.optimize()
def test_unitcellfilter_hcp(asap3, testdir):
    cu = bulk('Cu', 'hcp', a=3.6 / 2.0**0.5)
    cu.cell[1, 0] -= 0.05
    cu *= (6, 6, 3)
    cu.calc = asap3.EMT()
    print(cu.get_forces())
    print(cu.get_stress())
    f = UnitCellFilter(cu)
    opt = MDMin(f, dt=0.01)
    with Trajectory('Cu-hcp.traj', 'w', cu) as t:
        opt.attach(t)
        opt.run(0.2)
    # No assertions??
