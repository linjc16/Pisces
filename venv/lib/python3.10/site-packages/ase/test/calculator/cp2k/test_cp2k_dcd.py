"""Test suit for the CP2K ASE calulator.

http://www.cp2k.org
Author: Ole Schuett <ole.schuett@mat.ethz.ch>
"""

import subprocess

import numpy as np
import pytest

from ase import io
from ase.build import molecule
from ase.calculators.calculator import compare_atoms
from ase.io.cp2k import iread_cp2k_dcd

inp = """\
&MOTION
  &PRINT
    &TRAJECTORY SILENT
      FORMAT DCD_ALIGNED_CELL
    &END TRAJECTORY
  &END PRINT
  &MD
    STEPS 5
  &END MD
&END MOTION
&GLOBAL
  RUN_TYPE MD
&END GLOBAL
"""


@pytest.fixture()
def cp2k_main(testconfig):
    try:
        return testconfig.parser['cp2k']['cp2k_main']
    except KeyError:
        pytest.skip("""\
Missing cp2k configuration.  Requires:

    [cp2k]
    cp2k_main = /path/to/cp2k
""")

# XXX multi-line skip messages are ugly since they mess up the
# pytest -r s listing.  Should we write a more terse message?
# Let's do it when calculator factory reporting is user friendly.


@pytest.mark.calculator_lite()
@pytest.mark.calculator('cp2k')
def test_dcd(factory, cp2k_main):
    calc = factory.calc(label='test_dcd', inp=inp)
    h2 = molecule('H2', calculator=calc)
    h2.center(vacuum=2.0)
    h2.set_pbc(True)
    energy = h2.get_potential_energy()
    assert energy is not None
    subprocess.check_call([cp2k_main, '-i', 'test_dcd.inp', '-o',
                           'test_dcd.out'])
    h2_end = io.read('test_dcd-pos-1.dcd')
    assert (h2_end.symbols == 'X').all()
    traj = io.read('test_dcd-pos-1.dcd', ref_atoms=h2,
                   index=slice(0, None), aligned=True)
    ioITraj = io.iread('test_dcd-pos-1.dcd', ref_atoms=h2,
                       index=slice(0, None), aligned=True)

    with open('test_dcd-pos-1.dcd', 'rb') as fd:
        itraj = iread_cp2k_dcd(fd, indices=slice(0, None),
                               ref_atoms=h2, aligned=True)
        for i, iMol in enumerate(itraj):
            ioIMol = next(ioITraj)
            assert compare_atoms(iMol, traj[i]) == []
            assert compare_atoms(iMol, ioIMol) == []
            assert iMol.get_pbc().all()

    traj = io.read('test_dcd-pos-1.dcd', ref_atoms=h2, index=slice(0, None))
    pbc = [mol.get_pbc() for mol in traj]
    assert not np.any(pbc)
