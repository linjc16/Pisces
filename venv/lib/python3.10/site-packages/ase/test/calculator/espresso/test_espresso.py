import pytest

from ase.build import bulk, molecule
from ase.calculators.espresso import Espresso, EspressoProfile

espresso_versions = [
    ('6.4.1', """
Program PWSCF v.6.4.1 starts on  5Aug2021 at 11: 2:26

This program is part of the open-source Quantum ESPRESSO suite
"""),
    ('6.7MaX', """

Program PWSCF v.6.7MaX starts on  1Oct2022 at 16:26:59

This program is part of the open-source Quantum ESPRESSO suite
""")]


@pytest.mark.parametrize('version, txt', espresso_versions)
def test_version(version, txt):
    assert EspressoProfile.parse_version(txt) == version


def test_version_integration(espresso_factory):
    version = espresso_factory.profile.version()
    assert version[0].isdigit()


def verify(calc):
    assert calc.get_fermi_level() is not None
    assert calc.get_ibz_k_points() is not None
    assert calc.get_eigenvalues(spin=0, kpt=0) is not None
    assert calc.get_number_of_spins() is not None
    assert calc.get_k_point_weights() is not None


@pytest.mark.calculator_lite()
def test_main(espresso_factory):
    atoms = bulk('Si')
    atoms.calc = espresso_factory.calc()
    atoms.get_potential_energy()
    verify(atoms.calc)


@pytest.mark.calculator_lite()
def test_smearing(espresso_factory):
    atoms = bulk('Cu')
    input_data = {'system': {'occupations': 'smearing',
                             'smearing': 'fermi-dirac',
                             'degauss': 0.02}}
    atoms.calc = espresso_factory.calc(input_data=input_data)
    atoms.get_potential_energy()
    verify(atoms.calc)


@pytest.mark.calculator_lite()
def test_dipole(espresso_factory):
    atoms = molecule('H2O', cell=[10, 10, 10])
    atoms.center()
    input_data = {'control': {'tefield': True,
                              'dipfield': True},
                  'system': {'occupations': 'smearing',
                             'smearing': 'fermi-dirac',
                             'degauss': 0.02,
                             'edir': 3,
                             'eamp': 0.00,
                             'eopreg': 0.0001,
                             'emaxpos': 0.0001}}
    atoms.calc = espresso_factory.calc(input_data=input_data)
    atoms.get_potential_energy()
    verify(atoms.calc)
    dipol_arr = atoms.get_dipole_moment().tolist()
    expected_dipol_arr = [0, 0, -0.36991972]
    assert dipol_arr == pytest.approx(expected_dipol_arr, abs=0.02)


def test_warn_label(config_file):
    with pytest.warns(FutureWarning):
        Espresso(label='hello')


def test_error_command():
    with pytest.raises(RuntimeError):
        Espresso(command='hello')
