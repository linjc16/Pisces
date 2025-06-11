import numpy as np
import pytest

from ase import Atoms
from ase.filters import UnitCellFilter
from ase.io.trajectory import Trajectory
from ase.optimize import QuasiNewton
from ase.units import Ha, Ry
from ase.utils import tokenize_version


@pytest.mark.calculator('openmx')
def test_md(factory):
    # XXX ugly hack
    ver = factory.factory.version()
    if tokenize_version(ver) < tokenize_version('3.8'):
        pytest.skip('No stress tensor until openmx 3.8+')

    bud = Atoms('CH4', np.array([
        [0.000000, 0.000000, 0.100000],
        [0.682793, 0.682793, 0.682793],
        [-0.682793, -0.682793, 0.68279],
        [-0.682793, 0.682793, -0.682793],
        [0.682793, -0.682793, -0.682793]]),
        cell=[10, 10, 10])

    calc = factory.calc(
        label='ch4',
        xc='GGA',
        energy_cutoff=300 * Ry,
        convergence=1e-4 * Ha,
        # Use 'C_PBE19' and 'H_PBE19' for version 3.9
        definition_of_atomic_species=[['C', 'C5.0-s1p1', 'C_PBE13'],
                                      ['H', 'H5.0-s1', 'H_PBE13']],
        kpts=(1, 1, 1),
        eigensolver='Band'
    )

    bud.calc = calc
    with Trajectory('example.traj', 'w', bud) as traj:
        ucf = UnitCellFilter(
            bud, mask=[True, True, False, False, False, False])
        with QuasiNewton(ucf) as dyn:
            dyn.attach(traj.write)
            dyn.run(fmax=0.1)
            bud.get_potential_energy()
        # XXX maybe assert something?
