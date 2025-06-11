# from os.path import join
from unittest import mock

import pytest

from ase.atoms import Atoms
from ase.calculators.vasp.create_input import GenerateVaspInput


@pytest.fixture()
def vaspinput_factory():
    """Factory for GenerateVaspInput class, which mocks the generation of
    pseudopotentials."""

    def _vaspinput_factory(**kwargs) -> GenerateVaspInput:
        mocker = mock.Mock()
        inputs = GenerateVaspInput()
        inputs.set(**kwargs)
        inputs._build_pp_list = mocker(  # type: ignore[method-assign]
            return_value=None
        )
        return inputs

    return _vaspinput_factory


def check_written_incar(
    parameters, expected_output, vaspinput_factory, tmpdir
):
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 1.2]])
    calc_factory = vaspinput_factory(**parameters)
    calc_factory.initialize(atoms)
    calc_factory.write_incar(atoms, tmpdir)
    with open(tmpdir / 'INCAR', 'r') as written_incar:
        assert written_incar.read() == expected_output


def test_str_key(vaspinput_factory, tmpdir):
    parameters = {"prec": "Low"}
    expected_output = "PREC = Low\n"
    check_written_incar(
        parameters, expected_output, vaspinput_factory, tmpdir
    )


def test_special_str_key(vaspinput_factory, tmpdir):
    parameters = {"xc": "PBE"}
    expected_output = "GGA = PE\n"
    check_written_incar(
        parameters, expected_output, vaspinput_factory, tmpdir
    )


def test_float_key(vaspinput_factory, tmpdir):
    parameters = {"encut": 400}
    expected_output = "ENCUT = 400.000000\n"
    check_written_incar(
        parameters, expected_output, vaspinput_factory, tmpdir
    )


def test_exp_key(vaspinput_factory, tmpdir):
    parameters = {"ediff": 1e-6}
    expected_output = "EDIFF = 1.00e-06\n"
    check_written_incar(
        parameters, expected_output, vaspinput_factory, tmpdir
    )


def test_int_key(vaspinput_factory, tmpdir):
    parameters = {"ibrion": 2}
    expected_output = "IBRION = 2\n"
    check_written_incar(
        parameters, expected_output, vaspinput_factory, tmpdir
    )


def test_list_bool_key(vaspinput_factory, tmpdir):
    parameters = {"lattice_constraints": [False, True, False]}
    expected_output = (
        "LATTICE_CONSTRAINTS = .FALSE. .TRUE. .FALSE.\n"
    )
    check_written_incar(
        parameters, expected_output, vaspinput_factory, tmpdir
    )


def test_bool_key(vaspinput_factory, tmpdir):
    parameters = {"lhfcalc": True}
    expected_output = "LHFCALC = .TRUE.\n"
    check_written_incar(
        parameters, expected_output, vaspinput_factory, tmpdir
    )


def test_special_key(vaspinput_factory, tmpdir):
    parameters = {"lreal": True}
    expected_output = "LREAL = .TRUE.\n"
    check_written_incar(
        parameters, expected_output, vaspinput_factory, tmpdir
    )


def test_list_float_key(vaspinput_factory, tmpdir):
    parameters = {"magmom": [0.5, 1.5]}
    expected_output = (
        "MAGMOM = 1*0.5000 1*1.5000\n"
        "ISPIN = 2\n"
    )  # Writer uses :.4f
    check_written_incar(
        parameters, expected_output, vaspinput_factory, tmpdir
    )


def test_dict_key(
    vaspinput_factory, tmpdir
):  # dict key. Current writer uses %.3f
    parameters = {"ldau_luj": {"H": {"L": 2, "U": 4.0, "J": 0.0}}}
    expected_output = (
        "LDAU = .TRUE.\n"
        "LDAUL = 2\n"
        "LDAUU = 4.000\n"
        "LDAUJ = 0.000\n"
    )
    check_written_incar(
        parameters, expected_output, vaspinput_factory, tmpdir
    )
