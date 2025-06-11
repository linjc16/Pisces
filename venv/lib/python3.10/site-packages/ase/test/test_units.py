"""This test cross-checks our implementation of CODATA against the
implementation that SciPy brings with it.
"""

import pytest

import ase.units
from ase.units import create_units


def test_create_units():
    """Check that units are created and allow attribute access."""

    # just use current CODATA version
    new_units = ase.units.create_units(ase.units.__codata_version__)
    assert new_units.eV == new_units['eV'] == ase.units.eV
    for unit_name in new_units:
        assert getattr(new_units, unit_name) == getattr(ase.units, unit_name)
        assert new_units[unit_name] == getattr(ase.units, unit_name)


def test_bad_codata():
    name = 'my_bad_codata_version'
    with pytest.raises(NotImplementedError, match=name):
        create_units(name)
