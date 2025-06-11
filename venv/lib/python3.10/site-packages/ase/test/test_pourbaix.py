"""Test Pourbaix diagram."""
import numpy as np
import pytest

from ase.phasediagram import Pourbaix as Pourbaix_old
from ase.phasediagram import solvated
from ase.pourbaix import (
    PREDEF_ENERGIES,
    U_STD_AGCL,
    U_STD_SCE,
    Pourbaix,
    RedOx,
    Species,
    get_main_products,
)


@pytest.fixture
def refs():
    return {
        'Zn': 0.0,
        'ZnO': -3.336021896,
        'Zn++(aq)': -1.525613424,
        'ZnOH+(aq)': -3.4125107,
        'HZnO2-(aq)': -4.8087349,
        'ZnO2--(aq)': -4.03387383
    }


@pytest.fixture
def pbx(refs):
    return Pourbaix('Zn', refs)


def test_old_pourbaix():
    """Test ZnO system from docs."""
    refs_old = solvated('Zn')
    refs_old += [('Zn', 0.0), ('ZnO', -3.323), ('ZnO2(aq)', -2.921)]
    with pytest.warns(FutureWarning):
        pb = Pourbaix_old(refs_old, formula='ZnO')

    _, e = pb.decompose(-1.0, 7.0)
    assert e == pytest.approx(-3.625, abs=0.001)

    U = np.linspace(-2, 2, 3)
    pH = np.linspace(6, 16, 11)
    d, names, _ = pb.diagram(U, pH, plot=False)
    assert d.shape == (3, 11)
    assert np.ptp(d) == 6
    assert names == ['Zn', 'ZnO2(aq)', 'Zn++(aq)', 'HZnO2-(aq)',
                     'ZnOH+(aq)', 'ZnO', 'ZnO2--(aq)']


def test_Zn_diagram(pbx):
    """Test module against Zn Pourbaix diagram from the Atlas"""

    U = np.linspace(-2, 2, 5)
    pH = np.linspace(0, 14, 8)
    diagram = pbx.diagram(U, pH)

    # Verify that the stability domains are the expected ones
    names = [get_main_products(txt[2])[0] for txt in diagram.text]
    for name in ['Zn', 'ZnO', 'Zn++(aq)', 'HZnO2-(aq)', 'ZnO2--(aq)']:
        assert name in names
    assert 'ZnOH+(aq)' not in names

    # Verify that Zn is stable at U=-2, pH=0
    assert diagram.meta[0, 0] <= 0

    # Verify that Zn++ is the stable phase at U=0, pH=6
    i0 = int(diagram.pour[2, 3])
    phase0 = pbx.phases[i0]
    assert 'Zn++(aq)' in phase0.count

    # Verify that the pourbaix energy at U=1, pH=7 is the expected one
    # and similarly for U=-2, pH=0
    Epbx1 = pbx.get_pourbaix_energy(1.0, 7.0, verbose=True)[0]
    assert Epbx1 == pytest.approx(3.880, abs=0.001)
    Epbx2 = pbx.get_pourbaix_energy(-2.0, 0.0, verbose=True)[0]
    assert Epbx2 == pytest.approx(-2.119, abs=0.001)


def test_plotting(pbx, figure):
    """Test all the plotting functionalities and options"""

    diagram = pbx.diagram(U=np.linspace(-2, 2, 10), pH=np.linspace(0, 14, 10))

    ax = figure.gca()

    args = {'include_text': True,
            'include_water': True,
            'labeltype': 'phases',
            'cap': 1.0,
            # 'figsize': [12, 6],
            'cmap': "RdYlGn_r",
            'normalize': True,
            'show': False,
            'filename': None,
            'ax': ax}

    diagram.plot(**args)

    args.update({'include_text': False,
                 'include_water': False,
                 'labeltype': 'numbers',
                 'normalize': False,
                 'cap': [0, 1]})

    diagram.plot(**args)

    args.update({'labeltype': 'askjglkjh'})

    with pytest.raises(ValueError):
        diagram.plot(**args)


def test_redox():
    """Test different reference electrode corrections
       Plus other unused RedOx methods

    Reaction:
        Zn + H2O + e-  ➜  H+ + HZnO--(aq)
    """
    reactant = Species.from_string('Zn', 0.0)
    products = [Species.from_string('HZnO--(aq)', 0.0)]
    products_noredox = [Species.from_string('Pt', 0.0)]

    reaction = RedOx.from_species(reactant, products)
    noreaction = RedOx.from_species(reactant, products_noredox)
    assert reaction is not None
    assert noreaction is None

    corr = []
    for reference in ['SHE', 'RHE', 'Pt', 'AgCl', 'SCE']:
        corr.append(reaction.get_ref_correction(reference, alpha=1.0))

    assert corr[0][0] == pytest.approx(corr[0][1]) == pytest.approx(0.0)
    assert corr[1][1] == pytest.approx(-1.0)
    assert corr[2][0] == pytest.approx(-0.5 * PREDEF_ENERGIES['H2O'])
    assert corr[3][0] == pytest.approx(U_STD_AGCL)
    assert corr[4][0] == pytest.approx(U_STD_SCE)

    assert reaction.equation()

    G = reaction.get_free_energy(1.0, 1.0)
    assert G == pytest.approx(3.044, abs=0.001)


def test_species_extras():
    """Test some methods of Species not used by Pourbaix"""
    s = Species.from_string('H2O', 0.0)
    chemsys = s.get_chemsys()
    assert len(chemsys) == 3
    frac = s.get_fractional_composition('H')
    assert frac == pytest.approx(2 / 3)


def test_trigger_phases_error():
    """Produce an error when provided refs don't produce valid reactions"""
    refs = {
        'Zn': 0.0,
        'Mn': 0.0
    }

    with pytest.raises(ValueError):
        Pourbaix('Zn', refs)
