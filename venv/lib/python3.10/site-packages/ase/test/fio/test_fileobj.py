# test reading and writing a file descriptor using its name

import pytest

import ase.io
from ase.build import bulk
from ase.calculators.calculator import compare_atoms


@pytest.fixture()
def at():
    return bulk('Si')


def test_write_to_obj_name(at):
    # compare writing to a file with a particular extension and writing to
    # a file object that has same extension
    ase.io.write('direct_to_file.xyz', at)
    with open('to_file_obj.xyz', 'w') as fout:
        ase.io.write(fout, at)

    with open('direct_to_file.xyz') as f1, open('to_file_obj.xyz') as f2:
        for l1, l2 in zip(f1, f2):
            assert l1 == l2

    # compare reading from a file with a particular extension and reading from
    # a file object that has same extension
    at1 = ase.io.read('direct_to_file.xyz')
    with open('to_file_obj.xyz') as fin:
        at2 = ase.io.read(fin)
    print('compare', compare_atoms(at1, at2, 1.0e-10))
    assert len(compare_atoms(at1, at2)) == 0

    # compare reading from a file with a particular extension and reading from
    # a file object that has same extension
    at1 = ase.io.iread('direct_to_file.xyz')
    with open('to_file_obj.xyz') as fin:
        at2 = ase.io.iread(fin)
        for a1, a2 in zip(at1, at2):
            assert len(compare_atoms(a1, a2)) == 0
