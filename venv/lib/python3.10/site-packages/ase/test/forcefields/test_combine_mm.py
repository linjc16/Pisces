from math import cos, pi, sin

import numpy as np

from ase import Atoms, units
from ase.calculators.combine_mm import CombineMM
from ase.calculators.counterions import AtomicCounterIon as ACI
from ase.calculators.fd import calculate_numerical_forces
from ase.calculators.qmmm import LJInteractionsGeneral
from ase.calculators.tip3p import TIP3P, angleHOH, rOH
from ase.calculators.tip3p import epsilon0 as eps3
from ase.calculators.tip3p import sigma0 as sig3
from ase.calculators.tip4p import TIP4P
from ase.calculators.tip4p import epsilon0 as eps4
from ase.calculators.tip4p import sigma0 as sig4


def test_combine_mm():
    """Test CombineMM forces by combining tip3p and tip4p with them selves, and
       by combining tip3p with tip4p and testing against numerical forces.

       Also test LJInterationsGeneral with CombineMM """

    def make_atoms():
        r = rOH
        a = angleHOH * pi / 180
        dimer = Atoms('H2OH2O',
                      [(r * cos(a), 0, r * sin(a)),
                       (r, 0, 0),
                       (0, 0, 0),
                       (r * cos(a / 2), r * sin(a / 2), 0),
                       (r * cos(a / 2), -r * sin(a / 2), 0),
                       (0, 0, 0)])

        dimer = dimer[[2, 0, 1, 5, 3, 4]]
        # put O-O distance in the cutoff range
        dimer.positions[3:, 0] += 2.8

        return dimer

    dimer = make_atoms()
    rc = 3.0
    for (TIPnP, (eps, sig), nm) in zip([TIP3P, TIP4P],
                                       ((eps3, sig3), (eps4, sig4)),
                                       [3, 3]):
        dimer.calc = TIPnP(rc=rc, width=1.0)
        F1 = dimer.get_forces()

        sigma = np.array([sig, 0, 0])
        epsilon = np.array([eps, 0, 0])

        dimer.calc = CombineMM([0, 1, 2], nm, nm,
                               TIPnP(rc=rc, width=1.0),
                               TIPnP(rc=rc, width=1.0),
                               sigma, epsilon, sigma, epsilon,
                               rc=rc, width=1.0)

        F2 = dimer.get_forces()
        dF = F1 - F2
        print(TIPnP)
        print(dF)
        assert abs(dF).max() < 1e-8

    # Also check a TIP3P/TIP4P combination against numerical forces:
    eps1 = np.array([eps3, 0, 0])
    sig1 = np.array([sig3, 0, 0])
    eps2 = np.array([eps4, 0, 0])
    sig2 = np.array([sig4, 0, 0])
    dimer.calc = CombineMM([0, 1, 2], 3, 3, TIP3P(rc, 1.0), TIP4P(rc, 1.0),
                           sig1, eps1, sig2, eps2, rc, 1.0)

    F2 = dimer.get_forces()
    Fn = calculate_numerical_forces(dimer, 1e-7)
    dF = F2 - Fn
    print('TIP3P/TIP4P')
    print(dF)
    assert abs(dF).max() < 1e-8

    # LJInteractionsGeneral with CombineMM.
    # As it is used within EIQMMM, but avoiding long calculations for tests.
    # Total system is a unit test system comprised of:
    #    2 Na ions playing the role of a 'QM' subsystem
    #    2 Na ions as the 'Counterions'
    #    2 Water molecules

    dimer.calc = []
    faux_qm = Atoms('2Na', positions=np.array([[1.4, -4, 0], [1.4, 4, 0]]))
    ions = Atoms('2Na', positions=np.array([[1.4, 0, -4], [1.4, 0, 4]]))

    mmatoms = ions + dimer

    sigNa = 1.868 * (1.0 / 2.0)**(1.0 / 6.0) * 10
    epsNa = 0.00277 * units.kcal / units.mol

    # ACI for atoms 0 and 1 of the MM subsystem (2 and 3 for the total system)
    # 1 atom 'per molecule'. The rest is TIP4P, 3 atoms per molecule:
    calc = CombineMM([0, 1], 1, 3,
                     ACI(1, epsNa, sigNa),  # calc 1
                     TIP4P(),  # calc 2
                     [sigNa], [epsNa],  # LJs for subsystem 1
                     sig2, eps2,  # arrays for TIP4P from earlier
                     rc=7.5)

    mmatoms.calc = calc
    mmatoms.calc.initialize(mmatoms)

    # LJ arrays for the 'QM' subsystem
    sig_qm = np.array([sigNa, sigNa])
    eps_qm = np.array([epsNa, epsNa])

    # For the MM subsystem, tuple of arrays (counterion, water)
    sig_mm = (np.array([sigNa]), sig2)
    eps_mm = (np.array([epsNa]), eps2)

    lj = LJInteractionsGeneral(sig_qm, eps_qm, sig_mm, eps_mm, 2)

    ecomb, _fcomb1, _fcomb2 = lj.calculate(faux_qm,
                                         mmatoms,
                                         np.array([0, 0, 0]))

    # This should give the same result as if not using CombineMM, on a sum
    # of these systems:
    # A: All the Na atoms in the 'QM' region
    # B: LJInteractions between the 'QM' Na and the 'MM' Na
    # C: -LJinteractions between the 'MM' Na and the 'MM' Water

    # A:
    mmatoms = dimer
    mmatoms.calc = []
    sig_qm = np.concatenate((sig_qm, sig_qm))
    eps_qm = np.concatenate((eps_qm, eps_qm))
    sig_mm = sig_mm[1]
    eps_mm = eps_mm[1]
    lj = LJInteractionsGeneral(sig_qm, eps_qm, sig_mm, eps_mm, 4)
    ea, _fa1, _fa2 = lj.calculate(faux_qm + ions, mmatoms, np.array([0, 0, 0]))

    # B:
    lj = LJInteractionsGeneral(sig_qm[:2], eps_qm[:2],
                               sig_qm[:2], eps_qm[:2], 2, 2)

    eb, _fb1, _fb2, = lj.calculate(faux_qm, ions, np.array([0, 0, 0]))

    # C:
    lj = LJInteractionsGeneral(sig_qm[:2], eps_qm[:2],
                               sig_mm, eps_mm,
                               2, 3)

    ec, _fc1, _fc2, = lj.calculate(ions, dimer, np.array([0, 0, 0]))

    assert ecomb - (ea + eb) + ec == 0
