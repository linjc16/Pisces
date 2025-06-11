from math import sqrt

import pytest

from ase import Atom, Atoms, io
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.mep import NEB
from ase.optimize import BFGS

# XXXXXXXX this is mostly a copy of COCu111 !!!  Grrrr!


@pytest.mark.optimize()
@pytest.mark.slow()
def test_COCu111_2(testdir):
    logfile = "-"  # supresses

    Optimizer = BFGS

    # Distance between Cu atoms on a (111) surface:
    a = 3.6
    d = a / sqrt(2)
    fcc111 = Atoms(symbols='Cu',
                   cell=[(d, 0, 0),
                         (d / 2, d * sqrt(3) / 2, 0),
                         (d / 2, d * sqrt(3) / 6, -a / sqrt(3))],
                   pbc=True)
    initial = fcc111 * (2, 2, 4)
    initial.set_cell([2 * d, d * sqrt(3), 1])
    initial.set_pbc((1, 1, 0))
    initial.calc = EMT()
    Z = initial.get_positions()[:, 2]
    indices = [i for i, z in enumerate(Z) if z < Z.mean()]
    constraint = FixAtoms(indices=indices)
    initial.set_constraint(constraint)

    with Optimizer(initial, logfile=logfile) as dyn:
        dyn.run(fmax=0.05)

    # relax initial image
    b = 1.2
    h = 1.5
    initial += Atom('C', (d / 2, -b / 2, h))
    initial += Atom('O', (d / 2, +b / 2, h))
    with Optimizer(initial, logfile=logfile) as dyn:
        dyn.run(fmax=0.05)

    # relax final image
    final = initial.copy()
    final.calc = EMT()
    final.set_constraint(constraint)
    final[-2].position = final[-1].position
    final[-1].x = d
    final[-1].y = d / sqrt(3)
    with Optimizer(final, logfile=logfile) as dyn:
        dyn.run(fmax=0.1)

    # Create neb with 2 intermediate steps
    neb = NEB([initial, initial.copy(), initial.copy(), final],
              allow_shared_calculator=True)
    # refine() removed, not implemented any more
    neb.interpolate()

    # Optimize neb using a single calculator
    neb.set_calculators(EMT())
    # refine() removed, not implemented any more
    with Optimizer(neb, maxstep=0.04, trajectory='mep_2coarse.traj',
                   logfile=logfile) as dyn:
        dyn.run(fmax=0.1)

    # Optimize neb using a many calculators
    neb = NEB([initial, initial.copy(), initial.copy(), final])
    neb.interpolate()
    neb.set_calculators([EMT() for _ in range(neb.nimages)])
    with Optimizer(neb, maxstep=0.04, trajectory='mep_2coarse.traj',
                   logfile=logfile) as dyn:
        dyn.run(fmax=0.1)

    # read from the trajectory
    neb = NEB(io.read('mep_2coarse.traj', index='-4:'),
              allow_shared_calculator=True)

    # refine() removed, not implemented any more
    neb.set_calculators(EMT())
    # Optimize refined neb using a single calculator
    with Optimizer(neb, maxstep=0.04, trajectory='mep_2fine.traj',
                   logfile=logfile) as dyn:
        dyn.run(fmax=0.1)
    assert len(neb.images) == 4
