'''These tests ensure that the potentiostat can keep a sysytem near the PEC'''

import numpy as np
import pytest

from ase.build import bulk
from ase.calculators.emt import EMT
from ase.md.contour_exploration import ContourExploration
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from .test_ce_curvature import Al_atom_pair


def Al_block():
    size = 2
    atoms = bulk('Al', 'fcc', cubic=True).repeat((size, size, size))
    atoms.calc = EMT()
    return atoms


bulk_Al_settings = {
    'maxstep': 1.0,
    'parallel_drift': 0.05,
    'remove_translation': True,
    'potentiostat_step_scale': None,
    'use_frenet_serret': True,
    'angle_limit': 20,
    'loginterval': 1}


def test_potentiostat(testdir):
    '''This is very realistic and stringent test of the potentiostatic accuracy
     with 32 atoms at ~235 meV/atom above the ground state.'''
    name = 'test_potentiostat'
    seed = 19460926

    atoms = Al_block()

    E0 = atoms.get_potential_energy()

    atoms.rattle(stdev=0.18, seed=seed)
    initial_energy = atoms.get_potential_energy()

    rng = np.random.RandomState(seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=300, rng=rng)
    with ContourExploration(
        atoms,
        **bulk_Al_settings,
        energy_target=initial_energy,
        rng=rng,
        trajectory=name + '.traj',
        logfile=name + '.log',
    ) as dyn:
        print("Energy Above Ground State: {: .4f} eV/atom".format(
            (initial_energy - E0) / len(atoms)))
        for _ in range(5):
            dyn.run(5)
            energy_error = (atoms.get_potential_energy() -
                            initial_energy) / len(atoms)
            print(f'Potentiostat Error {energy_error: .4f} eV/atom')
            assert 0 == pytest.approx(energy_error, abs=0.01)


def test_potentiostat_no_fs(testdir):
    '''This test ensures that the potentiostat is working even when curvature
    extrapolation (use_fs) is turned off.'''
    name = 'test_potentiostat_no_fs'

    atoms = Al_atom_pair()

    atoms.set_momenta([[0, -1, 0], [0, 1, 0]])

    initial_energy = atoms.get_potential_energy()
    with ContourExploration(
        atoms,
        maxstep=0.2,
        parallel_drift=0.0,
        remove_translation=False,
        energy_target=initial_energy,
        potentiostat_step_scale=None,
        use_frenet_serret=False,
        trajectory=name + '.traj',
        logfile=name + '.log',
    ) as dyn:
        for _ in range(5):
            dyn.run(10)
            energy_error = (atoms.get_potential_energy() -
                            initial_energy) / len(atoms)
            print(f'Potentiostat Error {energy_error: .4f} eV/atom')
            assert 0 == pytest.approx(energy_error, abs=0.01)
