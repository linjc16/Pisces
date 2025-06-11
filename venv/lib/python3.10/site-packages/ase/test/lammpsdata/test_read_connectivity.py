"""Testing lammpsdata reader."""

import re
from io import StringIO

import numpy as np
import pytest

from ase.io import read

CONTENTS = """
3 atoms
1 atom types
2 bonds
2 bond types
1 angles
1 angle types
1 dihedrals
1 dihedral types

0 10 xlo xhi
0 10 ylo yhi
0 10 zlo zhi

Masses

1 1

Atoms # full

3 1 1 0 2 0 0  # 3rd
1 1 1 0 0 0 0  # 1st
2 1 1 0 1 0 0  # 2nd

Bonds

1 1 1 2  # 1-2
2 2 2 3  # 2-3

Angles

1 1 1 2 3  # 1-2-3

Dihedrals

1 1 1 2 3 1  # 1-2-3-1
"""

SORTED = {
    True: np.array([0, 1, 2]),
    False: np.array([2, 0, 1]),
}

REFERENCE = {
    'positions': np.array([
        [0, 0, 0],
        [1, 0, 0],
        [2, 0, 0],
    ]),
    'cell': np.eye(3) * 10,
    'bonds': {
        'atoms': np.array([
            [0, 1],
            [1, 2],
        ]),
        'types': np.array([1, 2]),
    },
    'angles': {
        'atoms': np.array([
            [0, 1, 2],
        ]),
        'types': np.array([1]),
    },
    'dihedrals': {
        'atoms': np.array([
            [0, 1, 2, 0],
        ]),
        'types': np.array([1]),
    },
}


@pytest.fixture()
def fmt():
    return 'lammps-data'


@pytest.fixture(params=[True, False])
def sort_by_id(request):
    return request.param


@pytest.fixture()
def lammpsdata(fmt, sort_by_id):
    fd = StringIO(CONTENTS)
    return read(fd, format=fmt, sort_by_id=sort_by_id), SORTED[sort_by_id]


def parse_tuples(atoms, regex, permutation, label):
    """Parse connectivity strings stored in atoms."""
    all_tuples = np.zeros((0, len(permutation)), int)
    types = np.array([], int)

    tuples = atoms.arrays[label]
    bonded = np.where(tuples != '_')[0]

    for i, per_atom in zip(bonded, tuples[bonded]):
        per_atom = np.array(regex.findall(per_atom), int)
        new_tuples = np.array([
            np.full(per_atom.shape[0], i, int),
            *(per_atom[:, :-1].T)
        ])

        all_tuples = np.append(all_tuples,
                               new_tuples[permutation, :].T,
                               axis=0)
        types = np.append(types, per_atom[:, -1])

    return all_tuples, types


def test_positions(lammpsdata):
    atoms, sorted = lammpsdata
    assert pytest.approx(atoms.positions) == REFERENCE['positions'][sorted]


def test_cell(lammpsdata):
    atoms, _ = lammpsdata
    assert pytest.approx(atoms.cell.array) == REFERENCE['cell']


def test_connectivity(lammpsdata):
    atoms, sorted = lammpsdata

    parser_data = {
        'bonds': ((0, 1), re.compile(r'(\d+)\((\d+)\)')),
        'angles': ((1, 0, 2), re.compile(r'(\d+)-(\d+)\((\d+)\)')),
        'dihedrals': ((0, 1, 2, 3), re.compile(r'(\d+)-(\d+)-(\d+)\((\d+)\)')),
    }

    for k, v in parser_data.items():
        tuples, types = parse_tuples(atoms, v[1], v[0], k)
        tuples = sorted[tuples.flatten()].reshape(tuples.shape)
        assert np.all(tuples == REFERENCE[k]['atoms'])
        assert np.all(types == REFERENCE[k]['types'])
