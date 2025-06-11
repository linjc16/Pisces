import numpy as np
import pytest

from ase.build import bulk
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.filters import UnitCellFilter
from ase.neighborlist import neighbor_list
from ase.optimize.precon import Precon, make_precon
from ase.utils.ff import Bond


@pytest.fixture()
def ref_atoms():
    atoms = bulk('Al', a=3.994) * 2
    atoms.calc = EMT()
    i, j, r = neighbor_list('ijd', atoms, cutoff=3.0)
    bonds = [Bond(I, J, k=1.0, b0=R) for (I, J, R) in zip(i, j, r)]
    return atoms, bonds


@pytest.fixture()
def atoms(ref_atoms):
    atoms, bonds = ref_atoms
    atoms.rattle(stdev=0.1, seed=7)
    return atoms, bonds


@pytest.fixture()
def var_cell(atoms):
    atoms, bonds = atoms
    return UnitCellFilter(atoms), bonds


@pytest.fixture()
def fixed_atoms(atoms):
    atoms, bonds = atoms
    atoms.set_constraint(FixAtoms(indices=[0]))
    return atoms, bonds


def check_assembly(precon, system):
    atoms, bonds = system
    kwargs = {}
    if precon == 'FF' or precon == 'Exp_FF':
        kwargs['bonds'] = bonds
    precon = make_precon(precon, atoms, **kwargs)
    assert isinstance(precon, Precon)
    # check its a symmetric positive definite matrix of expected size
    N = 3 * len(atoms)
    P = precon.asarray()
    assert P.shape == (N, N)
    assert np.abs(P - P.T).max() < 1e-6
    assert np.all(np.linalg.eigvalsh(P)) > 0


precons = [None, 'C1', 'Exp', 'Pfrommer', 'FF', 'Exp_FF']


@pytest.mark.parametrize('precon', precons)
def test_assembly_ref_atoms(precon, ref_atoms):
    check_assembly(precon, ref_atoms)


@pytest.mark.parametrize('precon', precons)
def test_assembly_atoms(precon, atoms):
    check_assembly(precon, atoms)


@pytest.mark.parametrize('precon', precons)
def test_assembly_var_cell(precon, var_cell):
    check_assembly(precon, var_cell)


@pytest.mark.parametrize('precon', precons)
def test_assembly_fixed_atoms(precon, fixed_atoms):
    check_assembly(precon, fixed_atoms)


def check_apply(precon, system):
    atoms, bonds = system
    kwargs = {}
    if precon == 'FF' or precon == 'Exp_FF':
        kwargs['bonds'] = bonds
    precon = make_precon(precon, atoms, **kwargs)
    forces = atoms.get_forces().reshape(-1)
    precon_forces, residual = precon.apply(forces, atoms)
    residual_P = np.linalg.norm(precon_forces, np.inf)
    print(f'|F| = {residual:.3f} '
          f'|F|_P = {np.linalg.norm(precon_forces, np.inf):.3f}')

    # force norm should not get much bigger when we precondition:
    # in this case all norms get smaller, but this will not be true in general
    assert residual_P <= residual

    # forces on any fixed atoms be zero, and remain zero after applying precon
    fixed_atoms = []
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            fixed_atoms.extend(list(constraint.index))
    if len(fixed_atoms) != 0:
        assert np.linalg.norm(forces[fixed_atoms], np.inf) < 1e-8
        assert np.linalg.norm(precon_forces[fixed_atoms], np.inf) < 1e-8


@pytest.mark.parametrize('precon', precons)
def test_apply_atoms(precon, atoms):
    check_apply(precon, atoms)


@pytest.mark.parametrize('precon', precons)
def test_apply_fixed_atoms(precon, fixed_atoms):
    check_apply(precon, fixed_atoms)
