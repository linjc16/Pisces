# test read
# https://wiki.openchemistry.org/Chemical_JSON
import numpy as np
import pytest

from ase import io

ethane = """{
  "chemical json": 0,
  "name": "ethane",
  "inchi": "1/C2H6/c1-2/h1-2H3",
  "formula": "C 2 H 6",
  "atoms": {
    "elements": {
      "number": [  1,   6,   1,   1,   6,   1,   1,   1 ]
    },
    "coords": {
      "3d": [  1.185080, -0.003838,  0.987524,
               0.751621, -0.022441, -0.020839,
               1.166929,  0.833015, -0.569312,
               1.115519, -0.932892, -0.514525,
              -0.751587,  0.022496,  0.020891,
              -1.166882, -0.833372,  0.568699,
              -1.115691,  0.932608,  0.515082,
              -1.184988,  0.004424, -0.987522 ]
    }
  },
  "bonds": {
    "connections": {
      "index": [ 0, 1,
                 1, 2,
                 1, 3,
                 1, 4,
                 4, 5,
                 4, 6,
                 4, 7 ]
    },
    "order": [ 1, 1, 1, 1, 1, 1, 1 ]
  },
  "properties": {
    "molecular mass": 30.0690,
    "melting point": -172,
    "boiling point": -88
  }
}
"""

tio2 = """{
  "chemicalJson": 1,
  "name": "TiO2 rutile",
  "formula": "Ti 2 O 4",
  "unitCell": {
    "a": 2.95812,
    "b": 4.59373,
    "c": 4.59373,
    "alpha": 90.0,
    "beta":  90.0,
    "gamma": 90.0
  },
  "atoms": {
    "elements": {
      "number": [ 22, 22, 8, 8, 8, 8 ]
    },
    "coords": {
      "3dFractional": [ 0.00000, 0.00000, 0.00000,
                        0.50000, 0.50000, 0.50000,
                        0.00000, 0.30530, 0.30530,
                        0.00000, 0.69470, 0.69470,
                        0.50000, 0.19470, 0.80530,
                        0.50000, 0.80530, 0.19470 ]
    }
  }
}
"""


def test_ethane():
    fname = 'ethane.cjson'
    with open(fname, 'w') as fd:
        fd.write(ethane)

    atoms = io.read(fname)

    assert str(atoms.symbols) == 'HCH2CH3'


def test_rutile():
    fname = 'TiO2_rutile.cjson'
    with open(fname, 'w') as fd:
        fd.write(tio2)

    atoms = io.read(fname)

    assert atoms.pbc.all()
    cell = atoms.cell

    assert str(atoms.symbols) == 'Ti2O4'
    assert atoms[1].position == pytest.approx(cell.diagonal() / 2)

    assert cell[1, 1] == cell[2, 2]
    assert cell == pytest.approx(np.diag(cell.diagonal()))
