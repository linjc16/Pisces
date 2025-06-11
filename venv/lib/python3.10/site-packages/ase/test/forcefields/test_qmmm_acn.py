import numpy as np

import ase.units as units
from ase import Atoms
from ase.calculators.acn import (
    ACN,
    epsilon_c,
    epsilon_me,
    epsilon_n,
    m_me,
    r_cn,
    r_mec,
    sigma_c,
    sigma_me,
    sigma_n,
)
from ase.calculators.qmmm import EIQMMM, LJInteractionsGeneral, SimpleQMMM
from ase.constraints import FixLinearTriatomic
from ase.optimize import BFGS


def test_qmmm_acn(testdir):
    # From https://www.sciencedirect.com/science/article/pii/S0166128099002079
    eref = 4.9 * units.kcal / units.mol
    dref = 3.368
    aref = 79.1

    sigma = np.array([sigma_me, sigma_c, sigma_n])
    epsilon = np.array([epsilon_me, epsilon_c, epsilon_n])
    inter = LJInteractionsGeneral(sigma, epsilon, sigma, epsilon, 3)

    for calc in [ACN(),
                 SimpleQMMM([0, 1, 2], ACN(), ACN(), ACN()),
                 SimpleQMMM([0, 1, 2], ACN(), ACN(), ACN(), vacuum=3.0),
                 EIQMMM([0, 1, 2], ACN(), ACN(), inter),
                 EIQMMM([0, 1, 2], ACN(), ACN(), inter, vacuum=3.0),
                 EIQMMM([3, 4, 5], ACN(), ACN(), inter, vacuum=3.0)]:
        dimer = Atoms('CCNCCN',
                      [(-r_mec, 0, 0),
                       (0, 0, 0),
                       (r_cn, 0, 0),
                       (r_mec, 3.7, 0),
                       (0, 3.7, 0),
                       (-r_cn, 3.7, 0)])

        masses = dimer.get_masses()
        masses[::3] = m_me
        dimer.set_masses(masses)

        dimer.calc = calc

        fixd = FixLinearTriatomic(triples=[(0, 1, 2), (3, 4, 5)])

        dimer.set_constraint(fixd)

        with BFGS(dimer, maxstep=0.04, trajectory=calc.name + '.traj',
                  logfile=calc.name + 'd.log') as opt:
            opt.run(0.001, steps=1000)

        e0 = dimer.get_potential_energy()
        d0 = dimer.get_distance(1, 4)
        a0 = dimer.get_angle(2, 1, 4)
        fmt = '{0:>25}: {1:.3f} {2:.3f} {3:.1f}'
        print(fmt.format(calc.name, -e0, d0, a0))
        assert abs(e0 + eref) < 0.013
        assert abs(d0 - dref) < 0.224
        assert abs(a0 - aref) < 2.9

    print(fmt.format('reference', eref, dref, aref))
