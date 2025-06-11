from unittest.mock import mock_open, patch

import numpy as np

from ase.io.vasp_parsers.incar_writer import write_incar


def test_write_string_to_incar():
    parameters = {"INCAR_TAG": "string"}
    expected_output = "INCAR_TAG = string\n"
    check_write_incar_file(parameters, expected_output)


def check_write_incar_file(parameters, expected_output):
    mock = mock_open()
    with patch("ase.io.vasp_parsers.incar_writer.open", mock):
        write_incar("directory", parameters)
        mock.assert_called_once_with("directory/INCAR", "w")
        incar = mock()
        incar.write.assert_called_once_with(expected_output)


def test_write_integer_to_incar():
    parameters = {"INCAR_TAG": 5}
    expected_output = "INCAR_TAG = 5\n"
    check_write_incar_file(parameters, expected_output)


def test_write_bool_to_incar():
    parameters = {"INCAR_TAG": True}
    expected_output = "INCAR_TAG = True\n"
    check_write_incar_file(parameters, expected_output)
    parameters = {"INCAR_TAG": False}
    expected_output = "INCAR_TAG = False\n"
    check_write_incar_file(parameters, expected_output)


def test_write_float_to_incar():
    parameters = {"INCAR_TAG": 1.234}
    expected_output = "INCAR_TAG = 1.234\n"
    check_write_incar_file(parameters, expected_output)
    parameters = {"INCAR_TAG": 1e-15}
    expected_output = "INCAR_TAG = 1e-15\n"
    check_write_incar_file(parameters, expected_output)


def test_write_list_to_incar():
    parameters = {"INCAR_TAG": [1, 2, 3]}
    expected_output = "INCAR_TAG = 1 2 3\n"
    check_write_incar_file(parameters, expected_output)


def test_write_tuple_to_incar():
    parameters = {"INCAR_TAG": (1, 2, 3)}
    expected_output = "INCAR_TAG = 1 2 3\n"
    check_write_incar_file(parameters, expected_output)


def test_write_array_to_incar():
    parameters = {"INCAR_TAG": np.arange(3)}
    expected_output = "INCAR_TAG = 0 1 2\n"
    check_write_incar_file(parameters, expected_output)


def test_write_multiple_text_to_incar():
    parameters = {"INCAR_TAG": "hello", "INCAR_TAG2": "world"}
    expected_output = "INCAR_TAG = hello\nINCAR_TAG2 = world\n"
    check_write_incar_file(parameters, expected_output)


def test_write_multiline_string_to_incar():
    parameters = {"INCAR_TAG": "hello\nworld"}
    expected_output = 'INCAR_TAG = "hello\nworld"\n'
    check_write_incar_file(parameters, expected_output)


def test_write_dictionary_to_incar():
    parameters = {"OUTER": {"INNER": "value"}}
    expected_output = """OUTER {
    INNER = value
}
"""
    check_write_incar_file(parameters, expected_output)


def test_write_complex_dictionary_to_incar():
    parameters = {"one": {"two": {"integer": 1, "list": [2, 3, 4]},
                          "string": "value"}}
    expected_output = """ONE {
    TWO {
        INTEGER = 1
        LIST = 2 3 4
    }
    STRING = value
}
"""
    check_write_incar_file(parameters, expected_output)


def test_write_incar_directly():
    expected_output = """SYSTEM = test INCAR file
ENCUT = 600
ISMEAR = 0
SIGMA = 0.05
"""
    check_write_incar_file(expected_output, expected_output)


def test_write_incar_no_parameters():
    check_write_incar_file(None, "")
