import subprocess

import numpy as np

from ase import Atoms
from ase.calculators.amber import Amber
from ase.io import write


def test_amber(factories):
    """Test that amber calculator works.

    This is conditional on the existence of the $AMBERHOME/bin/sander
    executable.
    """

    factories.require('amber')

    with open('mm.in', 'w') as outfile:
        outfile.write("""\
    zero step md to get energy and force
    &cntrl
    imin=0, nstlim=0,  ntx=1 !0 step md
    cut=100, ntb=0,          !non-periodic
    ntpr=1,ntwf=1,ntwe=1,ntwx=1 ! (output frequencies)
    &end
    END
    """)

    atoms = Atoms('OH2OH2',
                  [[-0.956, -0.121, 0],
                   [-1.308, 0.770, 0],
                   [0.000, 0.000, 0],
                   [3.903, 0.000, 0],
                   [4.215, -0.497, -0.759],
                   [4.215, -0.497, 0.759]])
    atoms.arrays['residuenames'] = ['WAT'] * 6
    atoms.arrays['residuenumbers'] = [1] * 3 + [2] * 3
    atoms.arrays['atomtypes'] = ['O', 'H1', 'H2'] * 2
    write('2h2o.pdb', atoms)

    with open('tleap.in', 'w') as outfile:
        outfile.write("""\
    source leaprc.protein.ff14SB
    source leaprc.gaff
    source leaprc.water.tip3p
    mol = loadpdb 2h2o.pdb
    saveamberparm mol 2h2o.top 2h2o.inpcrd
    quit
    """)

    subprocess.call('tleap -f tleap.in'.split())

    calc = Amber(amber_exe='sander -O ',
                 infile='mm.in',
                 outfile='mm.out',
                 topologyfile='2h2o.top',
                 incoordfile='mm.crd')
    calc.write_coordinates(atoms, 'mm.crd')
    atoms_check = atoms.copy()
    calc.read_coordinates(atoms_check, 'mm.crd')
    assert np.allclose(atoms_check.positions, atoms.positions)
    assert np.allclose(atoms_check.get_velocities(), atoms.get_velocities())

    atoms.calc = calc

    e = atoms.get_potential_energy()
    assert abs(e + 0.046799672) < 5e-3
