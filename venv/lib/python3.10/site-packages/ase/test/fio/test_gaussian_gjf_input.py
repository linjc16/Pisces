"""
test_gaussian_gjf_input.py, Geoffrey Weal, 23/5/24

This will test the read input for gjf files

Notes
-----
* 23/5/24 -> Wrote test for testing the charge and mutliplicity inputs.
"""
import warnings
from io import StringIO

import pytest

from ase.calculators.calculator import compare_atoms
from ase.io import ParseError, read

# -----------------------------------------------------------
# The following strings give the components of an example gif file
# for two methane molecules.


# The input line of the gjf file.
ORIGINAL_GJF_FILE_PREFIX = """%chk=Preview_s6ozz.chk
# opt freq wb97xd/cc-pvqz geom=connectivity

Title Card Required

"""


# Provide an example of two methane molecules.
ORIGINAL_GJF_FILE_SUFFIX = """
 C                  0.96604221    0.36299765    0.00000000
 H                  1.32269663   -0.64581235    0.00000000
 H                  1.32271505    0.86739584    0.87365150
 H                  1.32271505    0.86739584   -0.87365150
 H                 -0.10395779    0.36301084    0.00000000
 C                 -4.67798621   -1.41686181    0.00000000
 H                 -4.32133178   -2.42567181    0.00000000
 H                 -4.32131337   -0.91246362    0.87365150
 H                 -4.32131337   -0.91246362   -0.87365150
 H                 -5.74798621   -1.41684862    0.00000000"""


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# Provide an example of two methane molecules, which have been identified
# as different fragment.
FRAGMENT_GJF_FILE_SUFFIX = """
 C(Fragment=1)                  0.96604221    0.36299765    0.00000000
 H(Fragment=1)                  1.32269663   -0.64581235    0.00000000
 H(Fragment=1)                  1.32271505    0.86739584    0.87365150
 H(Fragment=1)                  1.32271505    0.86739584   -0.87365150
 H(Fragment=1)                 -0.10395779    0.36301084    0.00000000
 C(Fragment=2)                 -4.67798621   -1.41686181    0.00000000
 H(Fragment=2)                 -4.32133178   -2.42567181    0.00000000
 H(Fragment=2)                 -4.32131337   -0.91246362    0.87365150
 H(Fragment=2)                 -4.32131337   -0.91246362   -0.87365150
 H(Fragment=2)                 -5.74798621   -1.41684862    0.00000000"""


# -----------------------------------------------------------


# This is the original charge and multiplicity format.
ORIGINAL_CHARGE_MULTIPLICITY = "0 1"


# This is the fragment charge and multiplicity format that should work.
FRAGMENT_CHARGE_MULTIPLICITY_SUCCESS_1 = '0 1'
FRAGMENT_CHARGE_MULTIPLICITY_SUCCESS_2 = '0 1 0 1 0 1'


# This is the fragment charge and multiplicity format that should work, even
# if it wont make sense to Gaussian. But for ASE, this is fine, as ASE
# currently only reads in the total charge and the total multiplicity for
# the total system, which is given by the first two digits in this line.
FRAGMENT_CHARGE_MULTIPLICITY_SUCCESS_3 = '0 1 0 1'
FRAGMENT_CHARGE_MULTIPLICITY_SUCCESS_4 = '0 1 0 1 0 1 0 1'


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# This is the original Gaussian file that should fail.
ORIGINAL_CHARGE_MULTIPLICITY_FAIL_1 = ' '
ORIGINAL_CHARGE_MULTIPLICITY_FAIL_2 = '0 '


# These are the fragment charge and multiplcities that should fail.
FRAGMENT_CHARGE_MULTIPLICITY_FAIL_1 = '0 '
FRAGMENT_CHARGE_MULTIPLICITY_FAIL_2 = '0 1 0'
FRAGMENT_CHARGE_MULTIPLICITY_FAIL_3 = '0 1 0 1 0'
FRAGMENT_CHARGE_MULTIPLICITY_FAIL_4 = '0 1 0 1 0 1 0'


# -----------------------------------------------------------


def _quiet_parse(input_string):
    """Contains the code for reading in the Gaussian input file into ASE."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        return read(StringIO(input_string), format="gaussian-in")


@pytest.fixture()
def reference_system():
    """This is the reference system to compare tests to."""
    return _quiet_parse(ORIGINAL_GJF_FILE_PREFIX +
                        ORIGINAL_CHARGE_MULTIPLICITY +
                        ORIGINAL_GJF_FILE_SUFFIX)


# -----------------------------------------------------------


def test_original_gjf_success(reference_system):
    """Test to make sure an example gjf file that has been written
    correctly is being read in correctly."""
    atoms = _quiet_parse(ORIGINAL_GJF_FILE_PREFIX +
                         ORIGINAL_CHARGE_MULTIPLICITY +
                         ORIGINAL_GJF_FILE_SUFFIX)
    assert compare_atoms(atoms, reference_system) == []


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def test_fragment_gjf_success_1(reference_system):
    """Test to make sure the first example gjf file that has been written
    correctly and contains fragments is being read in correctly."""
    atoms = _quiet_parse(ORIGINAL_GJF_FILE_PREFIX +
                         FRAGMENT_CHARGE_MULTIPLICITY_SUCCESS_1 +
                         FRAGMENT_GJF_FILE_SUFFIX)
    assert compare_atoms(atoms, reference_system) == []


def test_fragment_gjf_success_2(reference_system):
    """Test to make sure the second example gjf file that has been written
    correctly and contains fragments is being read in correctly."""
    atoms = _quiet_parse(ORIGINAL_GJF_FILE_PREFIX +
                         FRAGMENT_CHARGE_MULTIPLICITY_SUCCESS_2 +
                         FRAGMENT_GJF_FILE_SUFFIX)
    assert compare_atoms(atoms, reference_system) == []


def test_fragment_gjf_success_3(reference_system):
    """Test to make sure the third example gjf file that has been written
    correctly and contains fragments is being read in correctly."""
    atoms = _quiet_parse(ORIGINAL_GJF_FILE_PREFIX +
                         FRAGMENT_CHARGE_MULTIPLICITY_SUCCESS_3 +
                         FRAGMENT_GJF_FILE_SUFFIX)
    assert compare_atoms(atoms, reference_system) == []


def test_fragment_gjf_success_4(reference_system):
    """Test to make sure the fourth example gjf file that has been written
    correctly and contains fragments is being read in correctly."""
    atoms = _quiet_parse(ORIGINAL_GJF_FILE_PREFIX +
                         FRAGMENT_CHARGE_MULTIPLICITY_SUCCESS_4 +
                         FRAGMENT_GJF_FILE_SUFFIX)
    assert compare_atoms(atoms, reference_system) == []


# -----------------------------------------------------------


def test_original_gjf_fail_1():
    """Test to make sure the first example gjf file that is expected to fail
    does infact fail based on the ParseError."""
    with pytest.raises(ParseError):
        _quiet_parse(ORIGINAL_GJF_FILE_PREFIX +
                     ORIGINAL_CHARGE_MULTIPLICITY_FAIL_1 +
                     ORIGINAL_GJF_FILE_SUFFIX)


def test_original_gjf_fail_2():
    """Test to make sure the second example gjf file that is expected to fail
    does infact fail based on the ParseError."""
    with pytest.raises(ParseError):
        _quiet_parse(ORIGINAL_GJF_FILE_PREFIX +
                     ORIGINAL_CHARGE_MULTIPLICITY_FAIL_2 +
                     ORIGINAL_GJF_FILE_SUFFIX)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def test_fragment_gjf_fail_1():
    """Test to make sure the first example gjf file that is expected to fail
    and contains fragments does infact fail based on the ParseError."""
    with pytest.raises(ParseError):
        _quiet_parse(ORIGINAL_GJF_FILE_PREFIX +
                     FRAGMENT_CHARGE_MULTIPLICITY_FAIL_1 +
                     FRAGMENT_GJF_FILE_SUFFIX)


def test_fragment_gjf_fail_2():
    """Test to make sure the second example gjf file that is expected to fail
    and contains fragments does infact fail based on the ParseError."""
    with pytest.raises(ParseError):
        _quiet_parse(ORIGINAL_GJF_FILE_PREFIX +
                     FRAGMENT_CHARGE_MULTIPLICITY_FAIL_2 +
                     FRAGMENT_GJF_FILE_SUFFIX)


def test_fragment_gjf_fail_3():
    """Test to make sure the third example gjf file that is expected to fail
    and contains fragments does infact fail based on the ParseError."""
    with pytest.raises(ParseError):
        _quiet_parse(ORIGINAL_GJF_FILE_PREFIX +
                     FRAGMENT_CHARGE_MULTIPLICITY_FAIL_3 +
                     FRAGMENT_GJF_FILE_SUFFIX)


def test_fragment_gjf_fail_4():
    """Test to make sure the fourth example gjf file that is expected to fail
    and contains fragments does infact fail based on the ParseError."""
    with pytest.raises(ParseError):
        _quiet_parse(ORIGINAL_GJF_FILE_PREFIX +
                     FRAGMENT_CHARGE_MULTIPLICITY_FAIL_4 +
                     FRAGMENT_GJF_FILE_SUFFIX)


# -----------------------------------------------------------
