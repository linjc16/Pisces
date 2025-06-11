import numpy as np
import pytest

from ase.build import molecule
from ase.calculators.fd import calculate_numerical_forces
from ase.calculators.gamess_us import GAMESSUS


@pytest.fixture()
def water():
    return molecule('H2O')


kwargs = [dict(label='noargs'),
          dict(label='xc', xc='PBE'),
          dict(label='dfttyp', contrl=dict(dfttyp='PBE')),
          dict(label='mp2', contrl=dict(mplevl=2), mp2=dict(mp2prp=True)),
          dict(label='ccsdt', contrl=dict(cctyp='CCSD(T)'))]

erefs = [-2056.7877424926373,
         -2064.9141313969094,
         -2064.9141313969094,
         -2060.091817423073,
         -2060.3341175255055]


grad = [True, True, True, True, False]


@pytest.mark.parametrize('kwargs, eref, grad', zip(kwargs, erefs, grad))
def test_gamess(water, kwargs, eref, grad, gamess_us_factory):
    water.calc = GAMESSUS(**kwargs)
    e = water.get_potential_energy()
    if eref is not None:
        assert abs(eref - e) < 1e-3
    if grad:
        f = water.get_forces()
        f_numer = calculate_numerical_forces(water, 1e-4)
        np.testing.assert_allclose(f, f_numer, atol=1e-3, rtol=1e-3)
