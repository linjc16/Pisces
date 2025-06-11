# flake8: noqa

"""Test the CP2K ASE calulator.

http://www.cp2k.org
Author: Ole Schuett <ole.schuett@mat.ethz.ch>
"""

from io import StringIO

import pytest

from ase.io.cp2k import read_cp2k_restart


@pytest.fixture
def inp():
    return StringIO("""\
 # Version information for this restart file
 &MOTION
   &CELL_OPT
     MAX_ITER  800
   &END CELL_OPT
 &FORCE_EVAL
   &DFT
     UKS  T
   &END DFT
   &SUBSYS
     &CELL
       A     1.4436982360095069E+01    0.0000000000000000E+00    0.0000000000000000E+00
       B    -1.2546391461121697E+01    8.0771799263415858E+00    0.0000000000000000E+00
       PERIODIC  XY
       MULTIPLE_UNIT_CELL  1 1 1
     &END CELL
     &COORD
C   -3.7242617044497828E+00    7.9038234645202037E+00    3.4613477913211641E+00
C1   -2.9068950543864061E+00    4.7668576748644087E+00    5.8444011777519380E+00
cu1    1.6455807102639135E+00    5.4728919446731368E+00    5.7625128629895181E+00
     &END COORD
   &END SUBSYS
   &PRINT
   &END PRINT
 &END FORCE_EVAL
""")


def test_restart(inp):
    mol = read_cp2k_restart(inp)
    assert len(mol) == 3
    assert (mol.get_pbc() == [True, True, False]).all()
