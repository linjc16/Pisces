import io

import pytest

from ase import Atoms
from ase.io.sdf import get_num_atoms_sdf_v2000, read_sdf

DIFFICULT_BUT_VALID_FIRST_LINE = '184192  0  0  0  0  0  0  0  0999 V2000'

VALID_SDF_V2000_COFFEE = '''
     RDKit          3D

 24  0  0  0  0  0  0  0  0  0999 V2000
    0.4700    2.5688    0.0006 O   0  0  0  0  0  0  0  0  0  0  0  0
   -3.1271   -0.4436   -0.0003 O   0  0  0  0  0  0  0  0  0  0  0  0
   -0.9686   -1.3125    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
    2.2182    0.1412   -0.0003 N   0  0  0  0  0  0  0  0  0  0  0  0
   -1.3477    1.0797   -0.0001 N   0  0  0  0  0  0  0  0  0  0  0  0
    1.4119   -1.9372    0.0002 N   0  0  0  0  0  0  0  0  0  0  0  0
    0.8579    0.2592   -0.0008 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.3897   -1.0264   -0.0004 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0307    1.4220   -0.0006 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.9061   -0.2495   -0.0004 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.5032   -1.1998    0.0003 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.4276   -2.6960    0.0008 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.1926    1.2061    0.0003 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.2969    2.1881    0.0007 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.5163   -1.5787    0.0008 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.0451   -3.1973   -0.8937 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.5186   -2.7596    0.0011 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.0447   -3.1963    0.8957 H   0  0  0  0  0  0  0  0  0  0  0  0
    4.1992    0.7801    0.0002 H   0  0  0  0  0  0  0  0  0  0  0  0
    3.0468    1.8092   -0.8992 H   0  0  0  0  0  0  0  0  0  0  0  0
    3.0466    1.8083    0.9004 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.8087    3.1651   -0.0003 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.9322    2.1027    0.8881 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.9346    2.1021   -0.8849 H   0  0  0  0  0  0  0  0  0  0  0  0
M  END
$$$$
'''


def test_read_sdf() -> None:
    """Test SDF V2000 reader."""
    with io.StringIO(VALID_SDF_V2000_COFFEE) as file_obj:
        atoms: Atoms = read_sdf(file_obj)

    assert len(atoms) == 24
    assert atoms.get_chemical_symbols() == ['O', 'O', 'N', 'N', 'N', 'N', 'C',
                                            'C', 'C', 'C', 'C',
                                            'C', 'C', 'C', 'H', 'H', 'H', 'H',
                                            'H', 'H', 'H', 'H',
                                            'H', 'H']
    assert atoms.get_positions()[0] == pytest.approx((0.4700, 2.5688, 0.0006))
    assert atoms.get_positions()[-1] == pytest.approx(
        (-2.9346, 2.1021, -0.8849))


def test_get_num_atoms_sdf_v2000() -> None:
    """Test the reading of the first line."""
    assert get_num_atoms_sdf_v2000(DIFFICULT_BUT_VALID_FIRST_LINE) == 184
