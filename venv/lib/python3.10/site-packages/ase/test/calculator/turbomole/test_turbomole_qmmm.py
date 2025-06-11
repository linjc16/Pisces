# type: ignore
from math import cos, pi, sin

import numpy as np

from ase import Atoms
from ase.calculators.qmmm import EIQMMM, LJInteractions, SimpleQMMM
from ase.calculators.tip3p import TIP3P, angleHOH, epsilon0, rOH, sigma0
from ase.calculators.turbomole import Turbomole
from ase.constraints import FixInternals
from ase.optimize import BFGS


def test_turbomole_qmmm(turbomole_factory):
    """Test the Turbomole calculator in simple QMMM and
    explicit interaction QMMM simulations."""

    r = rOH
    a = angleHOH * pi / 180
    D = np.linspace(2.5, 3.5, 30)

    interaction = LJInteractions({('O', 'O'): (epsilon0, sigma0)})
    qm_par = {'esp fit': 'kollman', 'multiplicity': 1}

    calcs = [
        TIP3P(),
        SimpleQMMM([0, 1, 2], Turbomole(**qm_par), TIP3P(), TIP3P()),
        SimpleQMMM([0, 1, 2], Turbomole(**qm_par), TIP3P(), TIP3P(),
                   vacuum=3.0),
        EIQMMM([0, 1, 2], Turbomole(**qm_par), TIP3P(), interaction),
        EIQMMM([3, 4, 5], Turbomole(**qm_par), TIP3P(), interaction,
               vacuum=3.0),
        EIQMMM([0, 1, 2], Turbomole(**qm_par), TIP3P(), interaction,
               vacuum=3.0)]
    refs = [(0.269, 0.283, 2.748, 25.6),
            (2077.695, 2077.709, 2.749, 25.7),
            (2077.695, 2077.709, 2.749, 25.7),
            (2077.960, 2077.718, 2.701, 19.2),
            (2077.891, 2077.724, 2.724, 53.0),
            (2077.960, 2077.708, 2.725, 19.3)]
    for calc, ref in zip(calcs, refs):
        dimer = Atoms('H2OH2O',
                      [(r * cos(a), 0, r * sin(a)),
                       (r, 0, 0),
                       (0, 0, 0),
                       (r * cos(a / 2), r * sin(a / 2), 0),
                       (r * cos(a / 2), -r * sin(a / 2), 0),
                       (0, 0, 0)])
        dimer.calc = calc

        E = []
        F = []
        for d in D:
            dimer.positions[3:, 0] += d - dimer.positions[5, 0]
            E.append(dimer.get_potential_energy())
            F.append(dimer.get_forces())

        F = np.array(F)

        F1 = np.polyval(np.polyder(np.polyfit(D, E, 7)), D)
        F2 = F[:, :3, 0].sum(1)
        error = abs(F1 - F2).max()
        assert error < 0.9

        dimer.set_constraint(FixInternals(
            bonds=[(r, (0, 2)), (r, (1, 2)),
                   (r, (3, 5)), (r, (4, 5))],
            angles_deg=[(angleHOH, (0, 2, 1)), (angleHOH, (3, 5, 4))]))
        with BFGS(dimer,
                  trajectory=calc.name + '.traj',
                  logfile=calc.name + 'd.log') as opt:
            opt.run(0.01)

        e0 = dimer.get_potential_energy()
        d0 = dimer.get_distance(2, 5)
        R = dimer.positions
        v1 = R[1] - R[5]
        v2 = R[5] - (R[3] + R[4]) / 2
        a0 = np.arccos(np.dot(v1, v2) /
                       (np.dot(v1, v1) * np.dot(v2, v2))**0.5) / np.pi * 180
        # fmt = '{0:>20}: {1:.3f} {2:.3f} {3:.3f} {4:.1f}'
        # print(fmt.format(calc.name, -min(E), -e0, d0, a0))
        assert np.allclose([-min(E), -e0, d0, a0], ref, rtol=0.01)
