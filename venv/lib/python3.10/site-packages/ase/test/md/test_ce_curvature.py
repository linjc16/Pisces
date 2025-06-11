'''These tests ensure that the computed PEC curvature matche the actual
geometries using a somewhat agressive angle_limit for each stepsize.'''
import numpy as np
import pytest

from ase import Atoms
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.md.contour_exploration import ContourExploration

pair_distance = 2.5


def Al_atom_pair(pair_distance=pair_distance):
    atoms = Atoms('AlAl', positions=[
        [-pair_distance / 2, 0, 0],
        [pair_distance / 2, 0, 0]])
    atoms.center(vacuum=10)
    atoms.calc = EMT()
    return atoms


def test_curvature1(testdir):
    '''This basic test has an atom spinning counter-clockwise around a fixed
    atom. The radius (1/curvature) must therefore be very
    close the pair_distance.'''
    name = 'test_curvature1'

    radius = pair_distance
    atoms = Al_atom_pair(pair_distance)

    atoms.set_constraint(FixAtoms(indices=[0]))
    atoms.set_velocities([[0, 0, 0], [0, 1, 0]])

    with ContourExploration(
        atoms,
        maxstep=1.5,
        parallel_drift=0.0,
        angle_limit=30,
        trajectory=name + '.traj',
        logfile=name + '.log',
    ) as dyn:
        print(f"Target Radius (1/curvature) {radius: .6f} Ang")
        for _ in range(5):
            dyn.run(30)
            print(f'Radius (1/curvature) {1 / dyn.curvature: .6f} Ang')
            assert radius == pytest.approx(1.0 / dyn.curvature, abs=2e-3)


def test_curvature2(testdir):
    '''This test has two atoms spinning counter-clockwise around eachother. the
    The radius (1/curvature) is less obviously pair_distance*sqrt(2)/2.
    This is the simplest multi-body analytic curvature test.'''
    name = 'test_curvature2'

    radius = pair_distance * np.sqrt(2) / 2
    atoms = Al_atom_pair(pair_distance)

    atoms.set_velocities([[0, -1, 0], [0, 1, 0]])

    with ContourExploration(
        atoms,
        maxstep=1.0,
        parallel_drift=0.0,
        angle_limit=30,
        trajectory=name + '.traj',
        logfile=name + '.log',
    ) as dyn:
        print(f"Target Radius (1/curvature) {radius: .6f} Ang")
        for _ in range(5):
            dyn.run(30)
            print(f'Radius (1/curvature) {1 / dyn.curvature: .6f} Ang')
            assert radius == pytest.approx(1.0 / dyn.curvature, abs=2e-3)
