import io
import re

import numpy as np
import pytest

from ase.build import bulk
from ase.io import write
from ase.io.elk import parse_elk_eigval, read_elk
from ase.units import Bohr, Hartree


def test_elk_in():
    atoms = bulk('Si')
    buf = io.StringIO()
    write(buf, atoms, format='elk-in', parameters={'mockparameter': 17})
    text = buf.getvalue()
    print(text)
    assert 'avec' in text
    assert re.search(r'mockparameter\s+17\n', text, re.M)


mock_elk_eigval_out = """
2 : nkpt
3 : nstsv

1   0.0 0.0 0.0 : k-point, vkl
(state, eigenvalue and occupancy below)
1 -1.0 2.0
2 -0.5 1.5
3  1.0 0.0


2   0.0 0.1 0.2 : k-point, vkl
(state, <blah blah>)
1 1.0 1.9
2 1.1 1.8
3 1.2 1.7
"""


def test_parse_eigval():
    fd = io.StringIO(mock_elk_eigval_out)
    dct = dict(parse_elk_eigval(fd))
    eig = dct['eigenvalues'] / Hartree
    occ = dct['occupations']
    kpts = dct['ibz_kpoints']
    assert len(eig) == 1
    assert len(occ) == 1
    assert pytest.approx(eig[0]) == [[-1.0, -0.5, 1.0], [1.0, 1.1, 1.2]]
    assert pytest.approx(occ[0]) == [[2.0, 1.5, 0.0], [1.9, 1.8, 1.7]]
    assert pytest.approx(kpts) == [[0., 0., 0.], [0.0, 0.1, 0.2]]


elk_geometry_out = """
scale
 1.0

scale1
 1.0

scale2
 1.0

scale3
 1.0

avec
   1.0 0.1 0.2
   0.3 2.0 0.4
   0.5 0.6 3.0

atoms
   1                                    : nspecies
'Si.in'                                 : spfname
   2                                    : natoms; atpos, bfcmt below
    0.1 0.2 0.3    0.0 0.0 0.0
    0.4 0.5 0.6    0.0 0.0 0.0
"""


def test_read_elk():
    atoms = read_elk(io.StringIO(elk_geometry_out))
    assert str(atoms.symbols) == 'Si2'
    assert all(atoms.pbc)
    assert atoms.cell / Bohr == pytest.approx(np.array([
        [1.0, 0.1, 0.2],
        [0.3, 2.0, 0.4],
        [0.5, 0.6, 3.0],
    ]))
    assert atoms.get_scaled_positions() == pytest.approx(np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]))
