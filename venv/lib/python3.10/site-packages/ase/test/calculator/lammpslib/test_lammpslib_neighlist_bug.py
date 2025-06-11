import numpy as np
import pytest

from ase.atoms import Atoms


@pytest.mark.calculator_lite()
@pytest.mark.calculator("lammpslib")
def test_lammps_neighlist_buf(factory, testdir):
    # this is a perfectly symmetric FCC configurations, all forces
    # should be zero As of 6 May 2021, if lammpslib does wrap before
    # rotating into lammps coord system lammps messes up the neighbor
    # list.  This may or may not be fixed in lammps eventually, but
    # can also be worked around by having lammpslib do the wrap just
    # before passing coords to lammps

    atoms = Atoms('He', cell=[[2.045, 2.045, 0.0], [2.045, 0.0, 2.045], [
                  0.0, 2.045, 2.045]], pbc=[True] * 3)
    atoms *= 6

    calc = factory.calc(
        lmpcmds=['pair_style lj/cut 0.5995011000293092E+01',
                 'pair_coeff * * 3.0 3.0'],
        atom_types={'H': 1, 'He': 2}, log_file=None,
        keep_alive=True, lammps_header=['units metal', 'atom_style atomic',
                                        'atom_modify map array sort 0 0'])

    atoms.calc = calc
    f = atoms.get_forces()
    fmag = np.linalg.norm(f, axis=1)
    print(f'> 1e-6 f[{np.where(fmag > 1e-6)}] = {f[np.where(fmag > 1e-6)]}')
    print(f'max f[{np.argmax(fmag)}] = {np.max(fmag)}')
    assert len(np.where(fmag > 1e-10)[0]) == 0
