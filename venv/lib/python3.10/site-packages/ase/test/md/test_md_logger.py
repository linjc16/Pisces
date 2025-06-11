"""Test to ensure that md logger and trajectory contain same data"""
import numpy as np
import pytest

import ase.units as u
from ase.calculators.tip3p import TIP3P
from ase.constraints import FixBondLengths
from ase.data import s22
from ase.io import Trajectory
from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet
from ase.optimize import BFGS, FIRE


@pytest.fixture()
def atoms():
    dimer = s22.create_s22_system("Water_dimer")
    dimer.constraints = FixBondLengths(
        [(3 * i + j, 3 * i + (j + 1) % 3) for i in range(2) for j in [0, 1, 2]]
    )
    return dimer


@pytest.fixture()
def calc():
    return TIP3P(rc=9.0)


def fmax(forces):
    return np.sqrt((forces ** 2).sum(axis=1).max())


md_cls_and_kwargs = [
    (VelocityVerlet, {}),
    (Langevin, {"temperature_K": 300, "friction": 0.02}),
]


@pytest.mark.parametrize('cls', [FIRE, BFGS])
def test_optimizer(cls, testdir, atoms, calc):
    """run optimization and verify that log and trajectory coincide"""

    opt_atoms = atoms.copy()
    opt_atoms.constraints = atoms.constraints
    logfile = 'opt.log'
    trajectory = 'opt.traj'
    opt_atoms.calc = calc

    with cls(opt_atoms, logfile=logfile, trajectory=trajectory) as opt:
        opt.run(0.2)
        opt.run(0.1)

    with Trajectory(trajectory) as traj, open(logfile) as fd:
        next(fd)
        for _, (a, line) in enumerate(zip(traj, fd)):
            fmax1 = float(line.split()[-1])
            fmax2 = fmax(a.get_forces())

            assert np.allclose(fmax1, fmax2, atol=0.01), (fmax1, fmax2)


@pytest.mark.parametrize('cls_and_kwargs', md_cls_and_kwargs)
def test_md(cls_and_kwargs, atoms, calc, testdir):
    """ run MD for 10 steps and verify that trajectory and log coincide """

    cls, kwargs = cls_and_kwargs
    if hasattr(atoms, "constraints"):
        del atoms.constraints

    atoms.calc = calc

    logfile = 'md.log'
    trajectory = 'md.traj'
    timestep = 1 * u.fs

    with cls(atoms, logfile=logfile, timestep=timestep,
             trajectory=trajectory, **kwargs) as md:
        md.run(steps=5)
        md.run(steps=5)

    # assert log file has correct length
    with open(logfile) as fd:
        length = len(fd.readlines())

    assert length == 12, length

    with Trajectory(trajectory) as traj, open(logfile) as fd:
        next(fd)
        for _, (a, line) in enumerate(zip(traj, fd)):
            Epot1, T1 = float(line.split()[-3]), float(line.split()[-1])
            Epot2, T2 = a.get_potential_energy(), a.get_temperature()

            assert np.allclose(T1, T2, atol=0.1), (T1, T2)
            assert np.allclose(Epot1, Epot2, atol=0.01), (Epot1, Epot2)
