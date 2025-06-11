import numpy as np
import pytest

from ase import Atoms, io
from ase.calculators.lj import LennardJones
from ase.io import read
from ase.optimize.basin import BasinHopping
from ase.units import kB


@pytest.mark.optimize()
@pytest.mark.slow()
def test_basin(testdir):
    # Global minima from
    # Wales and Doye, J. Phys. Chem. A, vol 101 (1997) 5111-5116
    E_global = {
        4: -6.000000,
        5: -9.103852,
        6: -12.712062,
        7: -16.505384}
    N = 7
    R = N**(1. / 3.)
    rng = np.random.RandomState(42)
    pos = rng.uniform(-R, R, (N, 3))
    s = Atoms('He' + str(N),
              positions=pos)
    s.calc = LennardJones()
    original_positions = 1. * s.get_positions()

    ftraj = 'lowest.traj'

    with BasinHopping(s,
                      temperature=100 * kB,
                      dr=0.5,
                      trajectory=ftraj,
                      optimizer_logfile=None) as GlobalOptimizer:
        GlobalOptimizer.run(10)
        Emin, smin = GlobalOptimizer.get_minimum()
        print("N=", N, 'minimal energy found', Emin,
              ' global minimum:', E_global[N])

        # recalc energy
        smin.calc = LennardJones()
        E = smin.get_potential_energy()
        assert abs(E - Emin) < 1e-15
        other = read(ftraj)
        E2 = other.get_potential_energy()
        assert abs(E2 - Emin) < 1e-15

        # check that only minima were written
        last_energy = None
        for im in io.read(ftraj, index=':'):
            energy = im.get_potential_energy()
            if last_energy is not None:
                assert energy < last_energy
            last_energy = energy

        # reset positions
        s.set_positions(original_positions)
