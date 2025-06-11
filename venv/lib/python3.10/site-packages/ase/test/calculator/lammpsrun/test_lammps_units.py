from math import pi

import pytest
from numpy.testing import assert_allclose

from ase import Atoms
from ase.units import _e, _eps0  # for reference values only


@pytest.mark.calculator_lite()
@pytest.mark.calculator('lammpsrun')
def test_lammps_units_conversions(factory):
    distance = 1.5  # Angstr.
    ref_energy = -2 * _e * 1e10 / (4 * pi * _eps0 * distance)
    ref_force = 2 * _e * 1e10 / (4 * pi * _eps0 * distance**2)

    atoms = Atoms(['H', 'O'], [(0.1, 0.1, 0.1),
                               (0.1, 0.1, 0.1 + distance)])
    atoms.set_initial_charges([1, -2])
    atoms.center(10.1)

    for units in ['real', 'metal', 'electron', 'nano']:
        with factory.calc(
                specorder=['H', 'O'],
                pair_style='coul/cut 10.0',
                pair_coeff=['* *'],
                atom_style='charge',
                units=units
        ) as calc:
            atoms.calc = calc
            energy = atoms.get_potential_energy()
            force = atoms.get_forces()[0, 2]
            assert_allclose(energy, ref_energy, atol=1e-4, rtol=1e-4)
            assert_allclose(force, ref_force, atol=1e-4, rtol=1e-4)
