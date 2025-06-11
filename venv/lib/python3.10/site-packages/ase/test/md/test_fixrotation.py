import numpy as np

from ase.build import bulk
from ase.md import Langevin
from ase.md.fix import FixRotation
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.units import fs
from ase.utils import seterr


def check_inertia(atoms):
    m, v = atoms.get_moments_of_inertia(vectors=True)
    print("Moments of inertia:")
    print(m)
    # There should be one vector in the z-direction
    n = 0
    delta = 1e-2
    for a in v:
        if (abs(a[0]) < delta
            and abs(a[1]) < delta
                and abs(abs(a[2]) - 1.0) < delta):

            print("Vector along z:", a)
            n += 1
        else:
            print("Vector not along z:", a)
    assert n == 1


def test_fixrotation_asap(asap3):
    rng = np.random.RandomState(123)

    with seterr(all='raise'):
        atoms = bulk('Au', cubic=True).repeat((3, 3, 10))
        atoms.pbc = False
        atoms.center(vacuum=5.0 + np.max(atoms.cell) / 2)
        print(atoms)
        atoms.calc = asap3.EMT()
        MaxwellBoltzmannDistribution(atoms, temperature_K=300, force_temp=True,
                                     rng=rng)
        Stationary(atoms)
        check_inertia(atoms)
        com = atoms.get_center_of_mass()
        with Langevin(
                atoms,
                timestep=20 * fs,
                temperature_K=300,
                friction=1e-3,
                logfile='-',
                loginterval=500,
                rng=rng
        ) as md:
            fx = FixRotation(atoms)
            md.attach(fx)
            md.run(steps=1000)
        check_inertia(atoms)
        # Test for issue #977 (it is free to do so here).
        delta = np.linalg.norm(atoms.get_center_of_mass() - com)
        print("Change in center of mass:", delta)
        assert delta < 1e-9
