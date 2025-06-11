import filecmp

import pytest

from ase.build import molecule
from ase.calculators.calculator import compare_atoms
from ase.calculators.emt import EMT
from ase.io import read, write


def atoms_equal(atoms1, atoms2):
    # Check that the tolerance is compatible with the writer's precision
    return compare_atoms(atoms1, atoms2, tol=1e-8) == []


def test_single_write_and_read():
    # Test that `format='extxyz', plain=True` writes a standard xyz file
    # like the `format='xyz'`
    atoms = molecule('H2O')
    write('1.xyz', atoms, format='extxyz', plain=True)
    write('2.xyz', atoms, format='xyz', fmt='%16.8f')
    assert filecmp.cmp('1.xyz', '2.xyz', shallow=False), 'Files differ'

    # Test reading
    atoms1 = read('1.xyz', format='xyz')
    assert atoms_equal(atoms, atoms1), 'Read failed'


def test_single_write_with_forces():
    # Create atoms with forces and test that
    # those aren't written to the file
    atoms = molecule('CO')
    atoms.calc = EMT()
    atoms.get_forces()
    write('1.xyz', atoms, format='extxyz', plain=True)
    write('2.xyz', atoms, format='xyz', fmt='%16.8f')
    write('3.xyz', molecule('CO'), format='xyz', fmt='%16.8f')
    assert filecmp.cmp('1.xyz', '2.xyz', shallow=False), 'Files differ'
    assert filecmp.cmp('1.xyz', '3.xyz', shallow=False), 'Files differ'


def test_single_write_and_read_with_comment():
    # Test writing
    atoms = molecule('C6H6')
    comment = 'my comment'
    write('1.xyz', atoms, format='extxyz', plain=True, comment=comment)
    write('2.xyz', atoms, format='xyz', fmt='%16.8f', comment=comment)
    assert filecmp.cmp('1.xyz', '2.xyz', shallow=False), 'Files differ'

    # Test reading
    atoms1 = read('1.xyz', format='xyz')
    assert atoms_equal(atoms, atoms1), 'Read failed'


@pytest.mark.parametrize('format', ['xyz', 'extxyz'])
def test_single_write_with_newline_comment(format):
    # Test writing
    atoms = molecule('C6H6')
    comment = 'my comment\nnext line'
    try:
        write('atoms.xyz', atoms, format=format, comment=comment)
    except ValueError as e:
        assert 'comment line' in str(e).lower()
    else:
        raise RuntimeError('Write should fail for newlines in comment.')


def test_multiple_write_and_read():
    images = []
    for name in ['C6H6', 'H2O', 'CO']:
        images.append(molecule(name))
    write('1.xyz', images, format='extxyz', plain=True)
    write('2.xyz', images, format='xyz', fmt='%16.8f')
    assert filecmp.cmp('1.xyz', '2.xyz', shallow=False), 'Files differ'

    # Test reading
    images1 = read('1.xyz', format='xyz', index=':')
    assert len(images) == len(images1)
    for atoms, atoms1 in zip(images, images1):
        assert atoms_equal(atoms, atoms1), 'Read failed'
