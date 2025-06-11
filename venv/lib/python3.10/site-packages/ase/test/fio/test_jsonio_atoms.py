import numpy as np
import pytest

from ase import Atoms
from ase.build import bulk, molecule
from ase.constraints import FixAtoms, FixCartesian
from ase.io.jsonio import decode, encode


@pytest.fixture()
def silver_bulk() -> Atoms:
    return bulk('Ag', cubic=True)


def test_jsonio_atoms():

    def assert_equal(atoms1, atoms2):
        assert atoms1 == atoms2
        assert set(atoms1.arrays) == set(atoms2.arrays)
        for name in atoms1.arrays:
            assert np.array_equal(
                atoms1.arrays[name], atoms2.arrays[name]), name

    atoms = bulk('Ti')
    txt = encode(atoms)

    atoms1 = decode(txt)
    txt1 = encode(atoms1)
    assert txt == txt1
    assert_equal(atoms, atoms1)

    BeH = molecule('BeH')
    assert BeH.has('initial_magmoms')
    new_BeH = decode(encode(BeH))
    assert_equal(BeH, new_BeH)
    assert new_BeH.has('initial_magmoms')

    atoms = bulk('Ti')
    atoms.constraints = FixAtoms(indices=[0])
    new_atoms = decode(encode(atoms))
    c1 = atoms.constraints
    c2 = new_atoms.constraints
    assert len(c1) == len(c2) == 1
    # Can we check constraint equality somehow?
    # Would make sense for FixAtoms
    assert np.array_equal(c1[0].index, c2[0].index)


def test_jsonio_constraints_cartesian(silver_bulk):
    a = [0, 1]
    mask = [[False, False, True], [False, False, True]]

    silver_bulk.constraints = FixCartesian(a, mask=mask)
    new_atoms = decode(encode(silver_bulk))
    c1 = silver_bulk.constraints
    c2 = new_atoms.constraints
    assert len(c1) == len(c2) == 1
    assert np.array_equal(c1[0].index, c2[0].index)
    assert np.array_equal(c1[0].mask, c2[0].mask)


def test_jsonio_constraints_fix_atoms_empty(silver_bulk):
    a = np.empty(0, dtype=int)
    silver_bulk.set_constraint(FixAtoms(a))
    new_atoms: Atoms = decode(encode(silver_bulk))
    decoded_constraints = new_atoms.constraints[0]
    assert isinstance(decoded_constraints, FixAtoms)
    assert len(decoded_constraints.get_indices()) == 0
