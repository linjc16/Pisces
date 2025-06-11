import pytest

from ase.calculators.h2morse import (
    H2Morse,
    H2MorseExcitedStates,
    H2MorseExcitedStatesCalculator,
)
from ase.vibrations.albrecht import Albrecht
from ase.vibrations.resonant_raman import ResonantRamanCalculator


@pytest.fixture()
def atoms():
    return H2Morse()


@pytest.fixture()
def rrname(atoms):
    """Prepare the Resonant Raman calculation"""
    name = 'rrmorse'
    with ResonantRamanCalculator(atoms, H2MorseExcitedStatesCalculator,
                                 overlap=lambda x, y: x.overlap(y),
                                 name=name, txt='-') as rmc:
        rmc.run()
    return name


def test_one_state(testdir, rrname, atoms):
    om = 1
    gam = 0.1

    with Albrecht(atoms, H2MorseExcitedStates,
                  exkwargs={'nstates': 1},
                  name=rrname, overlap=True,
                  approximation='Albrecht A', txt=None) as ao:
        aoi = ao.get_absolute_intensities(omega=om, gamma=gam)[-1]

    with Albrecht(atoms, H2MorseExcitedStates,
                  exkwargs={'nstates': 1},
                  name=rrname, approximation='Albrecht A', txt=None) as al:
        ali = al.get_absolute_intensities(omega=om, gamma=gam)[-1]
    assert ali == pytest.approx(aoi, 1e-9)


def test_all_states(testdir, rrname, atoms):
    """Include degenerate states"""
    om = 1
    gam = 0.1

    with Albrecht(atoms, H2MorseExcitedStates,
                  name=rrname, overlap=True,
                  approximation='Albrecht A', txt=None) as ao:
        aoi = ao.get_absolute_intensities(omega=om, gamma=gam)[-1]

    with Albrecht(atoms, H2MorseExcitedStates,
                  name=rrname, approximation='Albrecht A', txt=None) as al:
        ali = al.get_absolute_intensities(omega=om, gamma=gam)[-1]
    assert ali == pytest.approx(aoi, 1e-5)


def test_multiples(testdir, rrname, atoms):
    """Run multiple vibrational excitations"""
    om = 1
    gam = 0.1

    with Albrecht(atoms, H2MorseExcitedStates,
                  name=rrname, overlap=True, combinations=2,
                  approximation='Albrecht A', txt=None) as ao:
        aoi = ao.intensity(omega=om, gamma=gam)
    assert len(aoi) == 27


def test_summary(testdir, rrname, atoms):
    om = 1
    gam = 0.1

    with Albrecht(atoms, H2MorseExcitedStates,
                  name=rrname, overlap=True,
                  approximation='Albrecht B', txt=None) as ao:
        ao.summary(om, gam)

    with Albrecht(atoms, H2MorseExcitedStates,
                  name=rrname, overlap=True, combinations=2,
                  approximation='Albrecht A', txt=None) as ao:
        ao.extended_summary(om, gam)
