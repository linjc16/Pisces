import pytest

from ase.build import molecule
from ase.calculators.emt import EMT

# Careful testing since these deprecated functions will otherwise be untested.


@pytest.fixture()
def atoms():
    return molecule('H2O')


def test_set_calculator(atoms):
    calc = EMT()
    with pytest.warns(FutureWarning):
        atoms.set_calculator(calc)
    assert atoms.calc is calc


def test_get_calculator(atoms):
    with pytest.warns(FutureWarning):
        assert atoms.get_calculator() is None


def test_del_calculator(atoms):
    atoms.calc = EMT()
    with pytest.warns(FutureWarning):
        del atoms.calc
    assert atoms.calc is None
