"""Test Wannier90 wout format."""
import io

from ase.io import read
from ase.io.wannier90 import read_wout_all

wout = """
                              Lattice Vectors (Ang)
                    a_1     5.740000   0.000000   0.000000
                    a_2     0.000000   5.000000   0.000000
                    a_3     0.000000   0.000000   5.000000

 *----------------------------------------------------------------------------*
 |   Site       Fractional Coordinate          Cartesian Coordinate (Ang)     |
 +----------------------------------------------------------------------------+
 | H    1   0.43554   0.50000   0.50000   |    2.50000   2.50000   2.50000    |
 | H    2   0.56446   0.50000   0.50000   |    3.24000   2.50000   2.50000    |
 *----------------------------------------------------------------------------*

 Final State
  WF centre and spread    1  (  2.870000,  2.500000,  2.500000 )     0.85842654
  Sum of centres and spreads (  2.870000,  2.500000,  2.500000 )     0.85842654

"""


def test_wout():
    file = io.StringIO(wout)
    hhx = read(file, format='wout')
    assert ''.join(hhx.symbols) == 'HHX'


def test_wout_all():
    """Check reading of extra stuff."""
    file = io.StringIO(wout)
    result = read_wout_all(file)
    assert result['spreads'][0] == 0.85842654
    assert abs(result['centers'] -
               result['atoms'].get_center_of_mass()).max() < 1e-5
