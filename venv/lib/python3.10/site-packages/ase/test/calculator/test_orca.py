import re

import numpy as np
import pytest

from ase.atoms import Atoms
from ase.calculators.orca import get_version_from_orca_header
from ase.optimize import BFGS
from ase.units import Hartree

calc = pytest.mark.calculator


@pytest.fixture()
def txt1():
    return '               Program Version 4.1.2  - RELEASE  -'


@pytest.fixture()
def ref1():
    return '4.1.2'


def test_orca_version_from_string(txt1, ref1):
    version = get_version_from_orca_header(txt1)
    assert version == ref1


# @calc('orca') #somehow this test only works with static factory fixture
def test_orca_version_from_executable(orca_factory):
    # only check the format to be compatible with future versions
    version_regexp = re.compile(r'\d+.\d+.\d+')
    version = orca_factory.version()

    assert version_regexp.match(version)


@calc('orca')
def test_ohh(factory):
    atoms = Atoms('OHH',
                  positions=[(0, 0, 0), (1, 0, 0), (0, 1, 0)])

    atoms.calc = factory.calc(orcasimpleinput='BLYP def2-SVP')


@pytest.fixture()
def water():
    return Atoms('OHH', positions=[(0, 0, 0), (1, 0, 0), (0, 1, 0)])


@calc('orca')
def test_orca(water, factory):
    water.calc = factory.calc(label='water',
                              orcasimpleinput='BLYP def2-SVP Engrad')

    with BFGS(water) as opt:
        opt.run(fmax=0.05)

    final_energy = water.get_potential_energy()
    final_dipole = water.get_dipole_moment()

    np.testing.assert_almost_equal(final_energy, -2077.24420, decimal=0)
    np.testing.assert_almost_equal(
        final_dipole, [0.28669234, 0.28669234, 0.], decimal=6)


@pytest.mark.parametrize('charge', [-2, 0])
@calc('orca')
def test_orca_charged_dipole(water, factory, charge):
    # Make sure that dipole obeys the correct translation symmetry
    water.calc = factory.calc(label='water',
                              charge=charge,
                              orcasimpleinput='BLYP def2-SVP Engrad')

    displacement = np.array([0.2, 1.5, -4.2])
    dipole = water.get_dipole_moment()
    water.translate(displacement)
    new_dipole = water.get_dipole_moment()

    np.testing.assert_almost_equal(
        dipole + charge * displacement, new_dipole, decimal=4)


@calc('orca')
def test_orca_sp(water, factory):
    water.calc = factory.calc(label='water', orcasimpleinput='BLYP def2-SVP',
                              task="SP")

    final_energy = water.get_potential_energy()
    np.testing.assert_almost_equal(final_energy, -2077.24420, decimal=0)


@calc('orca')
def test_orca_use_last_energy(water, factory):
    water.calc = factory.calc(
        label='water',
        orcasimpleinput='PBE def2-SVP Opt TightOpt')
    energy = water.get_potential_energy() / Hartree

    np.testing.assert_almost_equal(energy, -76.272686944630, decimal=6)
