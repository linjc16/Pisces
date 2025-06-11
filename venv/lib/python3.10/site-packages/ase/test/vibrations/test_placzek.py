"""
Test Placzek type resonant Raman implementations
"""
from pathlib import Path

import pytest

from ase.calculators.h2morse import (
    H2Morse,
    H2MorseExcitedStates,
    H2MorseExcitedStatesCalculator,
)
from ase.parallel import parprint, world
from ase.vibrations.placzek import Placzek, Profeta
from ase.vibrations.resonant_raman import ResonantRamanCalculator
from ase.vibrations.vibrations import Vibrations


def test_summary(testdir):
    atoms = H2Morse()
    rmc = ResonantRamanCalculator(atoms, H2MorseExcitedStatesCalculator)
    rmc.run()

    pz = Placzek(atoms, H2MorseExcitedStates)
    pz.summary(1.)


def test_names(testdir):
    """Test different gs vs excited name. Tests also default names."""
    # do a Vibrations calculation first
    atoms = H2Morse()
    vib = Vibrations(atoms)
    vib.run()
    assert '0x-' in vib.cache

    # do a Resonant Raman calculation
    rmc = ResonantRamanCalculator(atoms, H2MorseExcitedStatesCalculator,
                                  verbose=True)
    rmc.run()

    # excitation files should reside in the same directory as cache files
    assert (Path(rmc.name) / ('ex.eq' + rmc.exext)).is_file()

    # XXX does this still make sense?
    # remove the corresponding pickle file,
    # then Placzek can not anymore use it for vibrational properties
    key = '0x-'
    assert key in rmc.cache
    del rmc.cache[key]  # make sure this is not used

    om = 1
    gam = 0.1
    pz = Placzek(atoms, H2MorseExcitedStates,
                 name='vib', exname='raman')
    pzi = pz.get_absolute_intensities(omega=om, gamma=gam)[-1]
    parprint(pzi, 'Placzek')

    # check that work was distributed correctly
    assert len(pz.myindices) <= -(-6 // world.size)


def test_overlap(testdir):
    """Test equality with and without overlap"""
    atoms = H2Morse()
    name = 'rrmorse'
    nstates = 3
    with ResonantRamanCalculator(atoms, H2MorseExcitedStatesCalculator,
                                 exkwargs={'nstates': nstates},
                                 overlap=lambda x, y: x.overlap(y),
                                 name=name, txt='-') as rmc:
        rmc.run()

    om = 1
    gam = 0.1

    with Profeta(atoms, H2MorseExcitedStates,
                 exkwargs={'nstates': nstates}, approximation='Placzek',
                 overlap=True, name=name, txt='-') as po:
        poi = po.get_absolute_intensities(omega=om, gamma=gam)[-1]

    with Profeta(atoms, H2MorseExcitedStates,
                 exkwargs={'nstates': nstates}, approximation='Placzek',
                 name=name, txt=None) as pr:
        pri = pr.get_absolute_intensities(omega=om, gamma=gam)[-1]

    print('overlap', pri, poi, poi / pri)
    assert pri == pytest.approx(poi, 1e-4)


def test_compare_placzek_implementation_intensities(testdir):
    """Intensities of different Placzek implementations
    should be similar"""
    atoms = H2Morse()
    name = 'placzek'
    with ResonantRamanCalculator(atoms, H2MorseExcitedStatesCalculator,
                                 overlap=lambda x, y: x.overlap(y),
                                 name=name, txt='-') as rmc:
        rmc.run()

    om = 1
    gam = 0.1

    with Placzek(atoms, H2MorseExcitedStates,
                 name=name, txt=None) as pz:
        pzi = pz.get_absolute_intensities(omega=om, gamma=gam)[-1]
    print(pzi, 'Placzek')

    # Profeta using frozenset
    with Profeta(atoms, H2MorseExcitedStates,
                 approximation='Placzek',
                 name=name, txt=None) as pr:
        pri = pr.get_absolute_intensities(omega=om, gamma=gam)[-1]
    print(pri, 'Profeta using frozenset')
    assert pzi == pytest.approx(pri, 1e-3)

    # Profeta using overlap
    with Profeta(atoms, H2MorseExcitedStates,
                 approximation='Placzek', overlap=True,
                 name=name, txt=None) as pr:
        pro = pr.get_absolute_intensities(omega=om, gamma=gam)[-1]
    print(pro, 'Profeta using overlap')
    assert pro == pytest.approx(pri, 1e-3)
