import numpy as np
import pytest

from ase.build import molecule
from ase.constraints import FixCartesian


@pytest.fixture()
def atoms():
    return molecule('CH3CH2OH')


def test_fixcartesian_misc():
    mask = np.array([1, 1, 0], bool)
    indices = [2, 3]
    constraint = FixCartesian(indices, mask=mask)
    assert '3' in str(constraint)
    dct = constraint.todict()['kwargs']

    assert dct['a'] == indices
    assert all(dct['mask'] == mask)

    # 2 atoms x 2 directions == 4 DOFs constrained
    assert constraint.get_removed_dof(atoms=None) == 4  # XXX atoms


def test_fixcartesian_adjust(atoms):
    cart_mask = np.array([False, True, True])
    atom_index = [2, 3, 5, 6]  # Arbitrary subset of atoms

    fixmask = np.zeros((len(atoms), 3), bool)
    fixmask[atom_index] = cart_mask[None, :]

    oldpos = atoms.get_positions()
    constraint = FixCartesian(atom_index, mask=cart_mask)

    rng = np.random.RandomState(42)
    deviation = 1.0 + rng.random((len(atoms), 3))

    newpos = oldpos + deviation
    constraint.adjust_positions(atoms, newpos)

    newpos_expected = oldpos + deviation
    newpos_expected[fixmask] = oldpos[fixmask]

    assert newpos == pytest.approx(newpos_expected, abs=1e-14)

    oldforces = 1.0 + rng.random((len(atoms), 3))
    newforces = oldforces.copy()
    constraint.adjust_forces(atoms, newforces)

    newforces_expected = oldforces.copy()
    newforces_expected[fixmask] = 0.0
    assert newforces == pytest.approx(newforces_expected, abs=1e-14)

    nzeros = sum(abs(newforces_expected.ravel()) < 1e-14)
    ndof = constraint.get_removed_dof(atoms)
    assert nzeros == ndof == sum(cart_mask) * len(atom_index)
