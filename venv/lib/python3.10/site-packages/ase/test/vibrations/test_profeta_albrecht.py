"""
Test Placzek and Albrecht resonant Raman implementations
"""
from pathlib import Path

import pytest

from ase.calculators.h2morse import (
    H2Morse,
    H2MorseExcitedStates,
    H2MorseExcitedStatesCalculator,
)
from ase.vibrations.albrecht import Albrecht
from ase.vibrations.placzek import Profeta
from ase.vibrations.resonant_raman import ResonantRamanCalculator


def test_compare_placzek_albrecht_intensities(testdir):
    atoms = H2Morse()
    name = 'rrmorse'
    with ResonantRamanCalculator(atoms, H2MorseExcitedStatesCalculator,
                                 overlap=lambda x, y: x.overlap(y),
                                 name=name, txt='-') as rmc:
        rmc.run()

        # check overlap files to be at the correct place
        assert (Path(name) / 'eq.ov.npy').is_file()
        # check that there are no leftover files
        assert len([x for x in Path(name).parent.iterdir()]) == 1

    om = 1
    gam = 0.1
    pri, ali = 0, 0

    """Albrecht A and P-P are approximately equal"""

    with Profeta(atoms, H2MorseExcitedStates,
                 name=name, overlap=True,
                 approximation='p-p', txt=None) as pr:
        pri = pr.get_absolute_intensities(omega=om, gamma=gam)[-1]
    with Albrecht(atoms, H2MorseExcitedStates,
                  name=name, overlap=True,
                  approximation='Albrecht A', txt=None) as al:
        ali = al.get_absolute_intensities(omega=om, gamma=gam)[-1]
    print('pri, ali', pri, ali)
    assert pri == pytest.approx(ali, 1e-2)

    """Albrecht B+C and Profeta are approximately equal"""

    pr.approximation = 'Profeta'
    pri = pr.get_absolute_intensities(omega=om, gamma=gam)[-1]
    al.approximation = 'Albrecht BC'
    ali = al.get_absolute_intensities(omega=om, gamma=gam)[-1]
    print('pri, ali', pri, ali)
    assert pri == pytest.approx(ali, 1e-2)

    """Albrecht and Placzek are approximately equal"""

    pr.approximation = 'Placzek'
    pri = pr.get_absolute_intensities(omega=om, gamma=gam)[-1]
    al.approximation = 'Albrecht'
    ali = al.get_absolute_intensities(omega=om, gamma=gam)[-1]
    print('pri, ali', pri, ali)
    assert pri == pytest.approx(ali, 1e-2)
