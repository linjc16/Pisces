from io import StringIO
from pathlib import Path

import numpy as np
import pytest

from ase.io import read
from ase.io.siesta import read_fdf, read_struct_out
from ase.units import Bohr

sample_struct_out = """\
  3.0   0.0   0.0
 -1.5   4.0   0.0
  0.0   0.0   5.0
        2
   1   45   0.0   0.0   0.0
   1   46   0.3   0.4   0.5
"""


def test_read_struct_out():
    atoms = read_struct_out(StringIO(sample_struct_out))
    assert all(atoms.numbers == [45, 46])
    assert atoms.get_scaled_positions() == pytest.approx(
        np.array([[0., 0., 0.], [.3, .4, .5]]))
    assert atoms.cell[:] == pytest.approx(np.array([[3.0, 0.0, 0.0],
                                                    [-1.5, 4.0, 0.0],
                                                    [0.0, 0.0, 5.0]]))
    assert all(atoms.pbc)


sample_fdf = """\
potatoes 5
COFFEE 6.5
%block spam
   1 2.5 hello
%endblock spam
"""


def test_read_fdf():
    dct = read_fdf(StringIO(sample_fdf))
    # This is a "raw" parser, no type conversion is done.
    ref = dict(potatoes=['5'],
               coffee=['6.5'],
               spam=[['1', '2.5', 'hello']])
    assert dct == ref


"""
"Hand-written" dummy file based on HCP Ti
"""
xv_file = """\
     5.6  0.0  0.0     0.0  0.0  0.0
    -2.8  4.8  0.0     0.0  0.0  0.0
     0.0  0.0  8.9     0.0  0.0  0.0
      2
  1    22  0.0  0.0  0.0     0.0  0.0  0.0
  1    22  0.0  3.2  4.4     0.0  0.0  0.0
"""


def test_read_xv():
    path = Path('tmp.XV')
    path.write_text(xv_file)
    atoms = read(path)

    assert str(atoms.symbols) == 'Ti2'
    pos = atoms.positions
    assert pos[0] == pytest.approx(0)
    assert pos[1] / Bohr == pytest.approx([0, 3.2, 4.4])
    assert all(atoms.pbc)
    assert atoms.cell / Bohr == pytest.approx(np.array(
        [[5.6, 0, 0], [-2.8, 4.8, 0], [0, 0, 8.9]]
    ))
