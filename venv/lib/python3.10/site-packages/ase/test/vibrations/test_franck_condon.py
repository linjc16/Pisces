import sys
from math import factorial

import numpy as np
import pytest

from ase.build import molecule
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.vibrations import Vibrations
from ase.vibrations.franck_condon import (
    FranckCondon,
    FranckCondonOverlap,
    FranckCondonRecursive,
)


def equal(x, y, tolerance=0, fail=True, msg=''):
    """Compare x and y."""

    if not np.isfinite(x - y).any() or (np.abs(x - y) > tolerance).any():
        msg = (msg + '%s != %s (error: |%s| > %.9g)' %
               (x, y, x - y, tolerance))
        if fail:
            raise AssertionError(msg)
        else:
            sys.stderr.write('WARNING: %s\n' % msg)


@pytest.mark.optimize()
def test_franck_condon(testdir):
    # FCOverlap

    fco = FranckCondonOverlap()
    fcr = FranckCondonRecursive()

    # check factorial
    assert fco.factorial(8) == factorial(8)
    # the second test is useful according to the implementation
    assert fco.factorial(5) == factorial(5)
    assert fco.factorial.inv(5) == 1. / factorial(5)

    # check T=0 and n=0 equality
    S = np.array([1, 2.1, 34])
    m = 5
    assert ((fco.directT0(m, S) - fco.direct(0, m, S)) / fco.directT0(m, S) <
            1e-15).all()

    # check symmetry
    S = 2
    n = 3
    assert fco.direct(n, m, S) == fco.direct(m, n, S)

    # ---------------------------
    # specials
    S = np.array([0, 1.5])
    delta = np.sqrt(2 * S)
    for m in [2, 7]:
        equal(fco.direct0mm1(m, S)**2,
              fco.direct(1, m, S) * fco.direct(m, 0, S), 1.e-17)
        equal(fco.direct0mm1(m, S), fcr.ov0mm1(m, delta), 1.e-15)
        equal(fcr.ov0mm1(m, delta),
              fcr.ov0m(m, delta) * fcr.ov1m(m, delta), 1.e-15)
        equal(fcr.ov0mm1(m, -delta), fcr.direct0mm1(m, -delta), 1.e-15)
        equal(fcr.ov0mm1(m, delta), - fcr.direct0mm1(m, -delta), 1.e-15)

        equal(fco.direct0mm2(m, S)**2,
              fco.direct(2, m, S) * fco.direct(m, 0, S), 1.e-17)
        equal(fco.direct0mm2(m, S), fcr.ov0mm2(m, delta), 1.e-15)
        equal(fcr.ov0mm2(m, delta),
              fcr.ov0m(m, delta) * fcr.ov2m(m, delta), 1.e-15)
        equal(fco.direct0mm2(m, S), fcr.direct0mm2(m, delta), 1.e-15)

        equal(fcr.direct0mm3(m, delta),
              fcr.ov0m(m, delta) * fcr.ov3m(m, delta), 1.e-15)

        equal(fcr.ov1mm2(m, delta),
              fcr.ov1m(m, delta) * fcr.ov2m(m, delta), 1.e-15)
        equal(fcr.direct1mm2(m, delta), fcr.ov1mm2(m, delta), 1.e-15)


@pytest.fixture(scope='module')
def unrelaxed():
    atoms = molecule('CH4')
    atoms.calc = EMT()
    return atoms


@pytest.fixture(scope='module')
def forces_a(unrelaxed):
    # evaluate forces in this configuration
    return unrelaxed.get_forces()


@pytest.fixture(scope='module')
def relaxed(unrelaxed):
    atoms = unrelaxed.copy()
    atoms.calc = unrelaxed.calc
    with BFGS(atoms, logfile=None) as opt:
        opt.run(fmax=0.01)
    return atoms


@pytest.fixture()
def vibname(testdir, relaxed):
    atoms = relaxed.copy()
    atoms.calc = relaxed.calc
    name = 'vib'
    vib = Vibrations(atoms, name=name)
    vib.run()
    return name


@pytest.mark.optimize()
def test_ch4_all(forces_a, relaxed, vibname):
    """Evaluate Franck-Condon overlaps in
    a molecule suddenly exposed to a different potential"""

    # FC factor for all frequencies
    fc = FranckCondon(relaxed, vibname)
    ndof = 3 * len(relaxed)

    # by symmetry only one frequency has a non-vanishing contribution
    HR_a, _f_a = fc.get_Huang_Rhys_factors(forces_a)
    assert len(HR_a) == ndof
    assert HR_a[:-1] == pytest.approx(0, abs=1e-10)
    assert HR_a[-1] == pytest.approx(0.859989171)

    FC, freq = fc.get_Franck_Condon_factors(293, forces_a)
    assert len(FC[0]) == 2 * ndof + 1
    assert len(freq[0]) == 2 * ndof + 1


@pytest.mark.optimize()
def test_ch4_minfreq(forces_a, relaxed, vibname):
    # FC factor for relevant frequencies only
    fc = FranckCondon(relaxed, vibname, minfreq=2000)
    nrel = 4

    # single excitations
    FC, freq = fc.get_Franck_Condon_factors(293, forces_a)
    assert len(FC[0]) == 2 * nrel + 1
    assert len(freq[0]) == 2 * nrel + 1

    # include double excitations
    FC, freq = fc.get_Franck_Condon_factors(293, forces_a, 2)
    assert len(FC[1]) == 2 * nrel
    # assert len(FC[2]) == 22  # XXX why? - gives 20 in oldlibs???
    for i in range(3):
        assert len(freq[i]) == len(FC[i])
