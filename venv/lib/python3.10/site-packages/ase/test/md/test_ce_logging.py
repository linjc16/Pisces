"""This test ensures that logging to a text file and to the trajectory file are
reporting the same values as in the ContourExploration object."""

import numpy as np
import pytest

from ase import io
from ase.md.contour_exploration import ContourExploration

from .test_ce_potentiostat import Al_block, bulk_Al_settings


def test_logging(testdir):

    seed = 19460926

    atoms = Al_block()
    atoms.rattle(stdev=0.18, seed=seed)

    rng = np.random.RandomState(seed)

    initial_energy = atoms.get_potential_energy()

    name = 'test_logging'

    traj_name = name + '.traj'
    log_name = name + '.log'

    with ContourExploration(
            atoms,
            **bulk_Al_settings,
            rng=rng,
            trajectory=traj_name,
            logfile=log_name,
    ) as dyn:
        energy_target = initial_energy
        dev = (atoms.get_potential_energy() - energy_target) / len(atoms)
        energy_targets = [energy_target]
        curvatures = [dyn.curvature]
        stepsizes = [dyn.step_size]
        deviation_per_atom = [dev]

        # we shift the target_energy to ensure it's actaully being
        # logged when it changes.
        de = 0.001 * len(atoms)

        # these print statements, mirror the log file.
        # print(energy_target, dyn.curvature, dyn.step_size, dev)

        for i in range(5):
            energy_target = initial_energy + de * i

            dyn.energy_target = energy_target
            dyn.run(1)
            dev = (atoms.get_potential_energy() - energy_target) / len(atoms)
            # print(energy_target, dyn.curvature, dyn.step_size, dev)

            energy_targets.append(energy_target)
            curvatures.append(dyn.curvature)
            stepsizes.append(dyn.step_size)
            deviation_per_atom.append(dev)

    # Now we check the contents of the log file
    # assert log file has correct length
    with open(log_name) as fd:
        length = len(fd.readlines())
    assert length == 7, length

    with io.Trajectory(traj_name, 'r') as traj, open(log_name) as fd:
        # skip the first line because it's a small initialization step
        lines = fd.readlines()[1:]
        for i, (im, line) in enumerate(zip(traj, lines)):

            lineparts = [float(part) for part in line.split()]

            log_energy_target = lineparts[1]
            assert 0 == pytest.approx(
                log_energy_target - energy_targets[i], abs=1e-5)

            log_energy = lineparts[2]
            assert 0 == pytest.approx(
                log_energy - im.get_potential_energy(), abs=1e-5)

            log_curvature = lineparts[3]
            assert 0 == pytest.approx(log_curvature - curvatures[i], abs=1e-5)

            log_step_size = lineparts[4]
            assert 0 == pytest.approx(log_step_size - stepsizes[i], abs=1e-5)

            log_dev = lineparts[5]
            assert 0 == pytest.approx(log_dev - deviation_per_atom[i], abs=1e-5)
