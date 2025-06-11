import numpy as np
import pytest

from ase.calculators.combine_mm import CombineMM
from ase.calculators.tip3p import TIP3P, epsilon0, sigma0
from ase.constraints import FixBondLengths
from ase.data import s22
from ase.optimize import FIRE


def make_atoms():
    atoms = s22.create_s22_system('Water_dimer')
    # rotate down in x axis:
    center = atoms[0].position
    atoms.translate(-center)
    h = atoms[3].position[1] - atoms[0].position[1]
    L = np.linalg.norm(atoms[0].position - atoms[3].position)
    angle = np.degrees(np.arcsin(h / L))
    atoms.rotate(angle, '-z', center=center)
    return atoms


def make_4mer():
    atoms = make_atoms()
    atoms2 = make_atoms()
    atoms2.translate([0., 3, 0])
    atoms2.rotate(180, 'y')
    atoms2.translate([3, 0, 0])
    atoms += atoms2
    return atoms


@pytest.mark.slow()
def test_combine_mm2(testdir):
    # More biased initial positions for faster test. Set
    # to false for a slower, harder test.
    fast_test = True

    atoms = make_4mer()
    atoms.constraints = FixBondLengths([(3 * i + j, 3 * i + (j + 1) % 3)
                                        for i in range(int(len(atoms) // 3))
                                        for j in [0, 1, 2]])
    atoms.calc = TIP3P(np.inf)
    tag = '4mer_tip3_opt.'
    with FIRE(atoms, logfile=tag + 'log', trajectory=tag + 'traj') as opt:
        opt.run(fmax=0.05)
    tip3_pos = atoms.get_positions()

    sig = np.array([sigma0, 0, 0])
    eps = np.array([epsilon0, 0, 0])
    rc = np.inf
    idxes = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11],
             list(range(6)), list(range(9)), list(range(6, 12))]

    for ii, idx in enumerate(idxes):
        atoms = make_4mer()
        if fast_test:
            atoms.set_positions(tip3_pos)
        atoms.constraints = FixBondLengths([(3 * i + j, 3 * i + (j + 1) % 3)
                                            for i in range(len(atoms) // 3)
                                            for j in [0, 1, 2]])

        atoms.calc = CombineMM(idx, 3, 3, TIP3P(rc), TIP3P(rc),
                               sig, eps, sig, eps, rc=rc)

        tag = f'4mer_combtip3_opt_{ii:02d}.'
        with FIRE(atoms, logfile=tag + 'log', trajectory=tag + 'traj') as opt:
            opt.run(fmax=0.05)
        assert (abs(atoms.positions - tip3_pos) < 1e-8).all()
        print('{}: {!s:>28s}: Same Geometry as TIP3P'
              .format(atoms.calc.name, idx))
