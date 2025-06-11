import numpy as np

from ase import units
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary


def test_langevin_com():
    """Check that the center of mass does not move during Langevin dynamics.

    In particular, test that this does not happen with atoms with different
    mass present (issue #1044).
    """
    # parameters
    size = 2
    T = 300
    dt = 0.01

    # setup
    atoms = bulk('CuAg', 'rocksalt', a=4.0).repeat(size)
    atoms.pbc = False
    atoms.calc = EMT()

    MaxwellBoltzmannDistribution(atoms, temperature_K=T)
    Stationary(atoms)

    mtot = atoms.get_momenta().sum(axis=0)
    print('initial momenta', mtot)
    print('initial forces', atoms.get_forces().sum(axis=0))

    # run NVT
    with Langevin(atoms, dt * units.fs, temperature_K=T, friction=0.02) as dyn:
        dyn.run(10)

    m2 = atoms.get_momenta().sum(axis=0)
    print('momenta', m2)
    print('forces', atoms.get_forces().sum(axis=0))
    print()

    assert np.linalg.norm(m2) < 1e-8
