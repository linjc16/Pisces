# flake8: noqa
from io import StringIO

import numpy as np
import pytest

from ase.io.proteindatabank import read_proteindatabank

header1 = """
CRYST1    2.000    2.000    2.000  90.00  90.00  90.00 P 1
MODEL     1
"""

header2 = """
REMARK    Step 83, E = -55.02388312121
CRYST1    2.000    2.000    2.000  90.00  90.00  90.00 P 1
"""

body1 = """
ATOM      1    C MOL     1       0.443   1.409   1.905  1.00  0.00           C
ATOM      2    O MOL     1       1.837   1.409   1.373  1.00  0.00           O
"""

body2 = """
ATOM      1    C                 0.443   1.409   1.905  1.00  0.00           C
ATOM      2    O                 1.837   1.409   1.373  1.00  0.00           O
"""

body3 = """
ATOM                             0.443   1.409   1.905                       C
ATOM                             1.837   1.409   1.373                       O
"""

cellref = pytest.approx(np.array([[2., 0., 0.], [0., 2., 0.], [0., 0., 2.]]))
posref = pytest.approx(np.array([[0.443, 1.409, 1.905], [1.837, 1.409, 1.373]]))


@pytest.mark.parametrize('body', [body1, body2, body3])
def test_pdb_optional_tag_body_without_header(body):
    atoms = read_proteindatabank(StringIO(body))

    assert all(atoms.numbers == [6, 8])
    assert not any(atoms.pbc)
    assert atoms.get_positions() == posref


@pytest.mark.parametrize('body', [body1, body2, body3])
def test_pdb_optional_tag_body(body):
    txt = header1 + body
    atoms = read_proteindatabank(StringIO(txt))

    assert all(atoms.numbers == [6, 8])
    assert all(atoms.pbc)
    assert atoms.cell[:] == cellref
    assert atoms.get_positions() == posref


@pytest.mark.parametrize('header', [header1, header2])
def test_pdb_optional_heading(header):
    txt = header + body1
    atoms = read_proteindatabank(StringIO(txt))

    assert all(atoms.numbers == [6, 8])
    assert all(atoms.pbc)
    assert atoms.cell[:] == cellref
    assert atoms.get_positions() == posref


def test_pdb_filled_optional_fields():
    atoms = read_proteindatabank(StringIO(body1))

    assert all(atoms.get_array('occupancy') == np.array([1., 1.]))
    assert all(atoms.get_array('bfactor') == np.array([0., 0.]))
    assert all(atoms.get_array('atomtypes') == np.array(['C', 'O']))
    assert all(atoms.get_array('residuenames') == np.array(['MOL', 'MOL']))
    assert all(atoms.get_array('residuenumbers') == np.array([1, 1]))


def test_pdb_unfilled_optional_fields():
    atoms = read_proteindatabank(StringIO(body3))

    assert 'occupancy' not in atoms.__dict__['arrays']
    assert all(atoms.get_array('bfactor') == np.array([0., 0.]))
    assert all(atoms.get_array('atomtypes') == np.array(['', '']))
    assert all(atoms.get_array('residuenames') == np.array(['', '']))
    assert all(atoms.get_array('residuenumbers') == np.array([1, 1]))
