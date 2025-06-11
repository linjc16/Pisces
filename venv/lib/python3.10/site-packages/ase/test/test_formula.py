import numpy as np
import pytest

from ase import Atoms
from ase.formula import Formula


def test_formula_things():
    assert Formula('A3B2C2D').format('abc') == 'DB2C2A3'
    assert Formula('A3B2C2D').format('ab2') == 'DB2C2A3'
    assert Formula('A3B2C2D').format('a2b') == 'A3B2C2D'
    assert str(Formula('HHOOO', format='reduce')) == 'H2O3'
    assert Formula('HHOOOUO').format('reduce') == 'H2O3UO'


@pytest.mark.parametrize('f',
                         ['SiC', 'MoS2', 'GaAs', 'CO', 'NH3'])
def test_periodic(f):
    for fmt in ['ab2', 'a2b', 'periodic']:
        assert Formula(f, format=fmt).format('periodic') == f


def test_atoms_formula_things():
    assert Atoms('MoS2').get_chemical_formula() == 'MoS2'
    assert Atoms('SnO2').get_chemical_formula(mode='metal') == 'SnO2'


def test_h0c1():
    f = Formula.from_dict({'H': 0, 'C': 1})
    assert f.format('hill') == 'C'
    with pytest.raises(ValueError):
        Formula.from_dict({'H': -1})
    with pytest.raises(ValueError):
        Formula.from_dict({'H': 1.5})
    with pytest.raises(ValueError):
        Formula.from_dict({7: 1})


def test_formula():
    for sym in ['', 'Pu', 'Pu2', 'U2Pu2', 'U2((Pu2)2H)']:
        for mode in ['all', 'reduce', 'hill', 'metal']:
            for empirical in [False, True]:
                if empirical and mode in ['all', 'reduce']:
                    continue
                atoms = Atoms(sym)
                formula = atoms.get_chemical_formula(mode=mode,
                                                     empirical=empirical)
                atoms2 = Atoms(formula)
                print(repr(sym), '->', repr(formula))
                n1 = np.sort(atoms.numbers)
                n2 = np.sort(atoms2.numbers)
                if empirical and len(atoms) > 0:
                    reduction = len(n1) // len(n2)
                    n2 = np.repeat(n2, reduction)
                assert (n1 == n2).all()


def test_convert():
    assert str(Formula('AgAg').convert('hill')) == 'Ag2'


def test_formula_on_formula():
    formula = Formula('CH3CH2OH')
    assert formula == Formula(formula)


@pytest.mark.parametrize(
    'x',
    ['H2O', '10H2O', '2(CuO2(H2O)2)10', 'Cu20+H2', 'H' * 15, 'AuBC2', ''])
def test_formulas(x):
    f = Formula(x)
    y = str(f)
    assert y == x
    print(f.count(), f'{f:latex}')
    a, b = divmod(f, 'H2O')
    assert a * Formula('H2O') + b == f
    assert f != 117  # check that formula can be compared to non-formula object
