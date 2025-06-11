import io

import numpy as np
import pytest

from ase.build import bulk
from ase.calculators.calculator import compare_atoms
from ase.io.vasp import parse_poscar_scaling_factor, read_vasp, write_vasp


@pytest.fixture(name="atoms")
def fixture_atoms():
    _atoms = bulk('NaCl', crystalstructure='rocksalt', a=4.1, cubic=True)
    _atoms.wrap()
    return _atoms


@pytest.mark.parametrize('filename', ['POSCAR', 'CONTCAR'])
@pytest.mark.parametrize('vasp5', [True, False])
def test_read_write_roundtrip(atoms, vasp5, filename):
    write_vasp(filename, atoms, vasp5=vasp5)
    atoms_loaded = read_vasp(filename)

    assert len(compare_atoms(atoms, atoms_loaded)) == 0


@pytest.mark.parametrize('filename', ['POSCAR', 'CONTCAR'])
@pytest.mark.parametrize('kwargs', [{}, {'vasp5': True}])
def test_write_vasp5(atoms, filename, kwargs):
    """Test that we write the symbols to the POSCAR/CONTCAR
    with vasp5=True (which should also be the default)"""
    write_vasp(filename, atoms, **kwargs)
    with open(filename) as file:
        lines = file.readlines()
    # Test the 5th line, which should be the symbols
    assert lines[5].strip().split() == list(atoms.symbols)


def test_scaling_factor_line_with_comment() -> None:
    """Test if the scaling-factor line with a comment can be parsed."""
    assert parse_poscar_scaling_factor('1.00 ! scaling factor\n') == 1.0


# The negative scaling factor -120 gives 2x2x2 expansion.
BUF_NEGATIVE = """\
 H
-120.00000000000000
     1.0000000000000000    0.0000000000000000    0.0000000000000000
     2.0000000000000000    3.0000000000000000    0.0000000000000000
    -3.0000000000000000    1.0000000000000000    5.0000000000000000
 H
   1
Cartesian
  0.1000000000000000  0.2000000000000000  0.3000000000000000
"""


def test_read_negative_scaling_factor():
    """Test negative scaling factor"""
    atoms = read_vasp(io.StringIO(BUF_NEGATIVE))
    cell_ref = [[2, 0, 0], [4, 6, 0], [-6, 2, 10]]
    positions_ref = [[0.2, 0.4, 0.6]]
    np.testing.assert_allclose(atoms.cell, cell_ref)
    np.testing.assert_allclose(atoms.positions, positions_ref)


BUF_MULTIPLE = """\
 H
 3.0000000000000000 2.0000000000000000 1.0000000000000000
     1.0000000000000000    0.0000000000000000    0.0000000000000000
     2.0000000000000000    3.0000000000000000    0.0000000000000000
    -3.0000000000000000    1.0000000000000000    5.0000000000000000
 H
   1
Cartesian
  0.1000000000000000  0.2000000000000000  0.3000000000000000
"""


def test_read_multiple_scaling_factors():
    """Test multiple scaling factors"""
    atoms = read_vasp(io.StringIO(BUF_MULTIPLE))
    cell_ref = [[3, 0, 0], [6, 6, 0], [-9, 2, 5]]
    positions_ref = [[0.3, 0.4, 0.3]]
    np.testing.assert_allclose(atoms.cell, cell_ref)
    np.testing.assert_allclose(atoms.positions, positions_ref)


def test_unwrapped_scaled_positions():
    """Test if `write_vasp` prints unwrapped scaled positions"""
    atoms_ref = bulk('Ge')
    # Shift atomic positions to get negative coordinates
    atoms_ref.wrap(center=(-1, -1, -1))

    buf = io.StringIO()
    write_vasp(buf, atoms_ref, direct=True)
    buf.seek(0)
    atoms = read_vasp(buf)
    np.testing.assert_allclose(atoms_ref.positions, atoms.positions)
