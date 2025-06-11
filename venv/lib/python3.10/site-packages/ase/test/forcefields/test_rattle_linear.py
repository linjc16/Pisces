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
from ase.md.verlet import VelocityVerlet


def test_rattle_linear(testdir):
    """Test RATTLE and QM/MM for rigid linear acetonitrile."""

    sigma = np.array([sigma_me, sigma_c, sigma_n])
    epsilon = np.array([epsilon_me, epsilon_c, epsilon_n])
    i = LJInteractionsGeneral(sigma, epsilon, sigma, epsilon, 3)

    for calc in [ACN(),
                 SimpleQMMM([0, 1, 2], ACN(), ACN(), ACN()),
                 EIQMMM([0, 1, 2], ACN(), ACN(), i)]:

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

        fixd = FixLinearTriatomic(triples=[(0, 1, 2), (3, 4, 5)])

        dimer.set_constraint(fixd)

        dimer.calc = calc

        d1 = dimer[:3].get_all_distances()
        d2 = dimer[3:].get_all_distances()
        e = dimer.get_potential_energy()

        with VelocityVerlet(dimer, 2.0 * units.fs,
                            trajectory=calc.name + '.traj',
                            logfile=calc.name + '.log',
                            loginterval=20) as md:
            md.run(100)

        de = dimer.get_potential_energy() - e

        assert np.all(abs(dimer[:3].get_all_distances() - d1) < 1e-10)
        assert np.all(abs(dimer[3:].get_all_distances() - d2) < 1e-10)
        assert abs(de - -0.005) < 0.001
