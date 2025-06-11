"""
Check the many ways of specifying KPOINTS
"""
import os

import pytest

from ase.build import bulk
from ase.calculators.vasp.create_input import format_kpoints

from .filecmp_ignore_whitespace import filecmp_ignore_whitespace

calc = pytest.mark.calculator


@pytest.fixture()
def atoms():
    return bulk('Al', 'fcc', a=4.5, cubic=True)


def check_kpoints_line(n, contents):
    """Assert the contents of a line"""
    with open('KPOINTS') as fd:
        lines = fd.readlines()
    assert lines[n].strip() == contents


@pytest.fixture()
def write_kpoints(atoms):
    """Helper fixture to write the input kpoints file"""
    def _write_kpoints(factory, **kwargs):
        calc = factory.calc(**kwargs)
        calc.initialize(atoms)
        calc.write_kpoints(atoms=atoms)
        return atoms, calc

    return _write_kpoints


def test_vasp_kpoints_111(atoms):
    # Default to (1 1 1)
    string = format_kpoints(gamma=True, atoms=atoms, kpts=(1, 1, 1))
    check_kpoints_string(string, 2, 'Gamma')
    check_kpoints_string(string, 3, '1 1 1')


def test_vasp_kpoints_3_tuple(atoms):
    string = format_kpoints(gamma=False, kpts=(4, 4, 4), atoms=atoms)
    lines = string.split('\n')
    assert lines[1] == '0'
    assert lines[2] == 'Monkhorst-Pack'
    assert lines[3] == '4 4 4'


def check_kpoints_string(string, lineno, value):
    assert string.splitlines()[lineno].strip() == value


def test_vasp_kpoints_auto(atoms):
    string = format_kpoints(atoms=atoms, kpts=20)
    check_kpoints_string(string, 1, '0')
    check_kpoints_string(string, 2, 'Auto')
    check_kpoints_string(string, 3, '20')


def test_vasp_kpoints_1_element_list_gamma(atoms):
    # 1-element list ok, Gamma ok
    string = format_kpoints(atoms=atoms, kpts=[20], gamma=True)
    check_kpoints_string(string, 1, '0')
    check_kpoints_string(string, 2, 'Auto')
    check_kpoints_string(string, 3, '20')


@calc('vasp')
def test_kspacing_supress_kpoints_file(factory, write_kpoints):
    # KSPACING suppresses KPOINTS file
    Al, calc = write_kpoints(factory, kspacing=0.23)
    calc.write_incar(Al)
    assert not os.path.isfile('KPOINTS')
    with open('INCAR') as fd:
        assert 'KSPACING = 0.230000\n' in fd.readlines()


@calc('vasp')
def test_negative_kspacing_error(factory, write_kpoints):
    # Negative KSPACING raises an error
    with pytest.raises(ValueError):
        write_kpoints(factory, kspacing=-0.5)


def test_weighted(atoms, testdir):
    # Explicit weighted points with nested lists, Cartesian if not specified
    string = format_kpoints(
        atoms=atoms,
        kpts=[[0.1, 0.2, 0.3, 2], [0.0, 0.0, 0.0, 1],
              [0.0, 0.5, 0.5, 2]])

    with open('KPOINTS', 'w') as fd:
        fd.write(string)

    with open('KPOINTS.ref', 'w') as fd:
        fd.write("""KPOINTS created by Atomic Simulation Environment
    3 \n\
    Cartesian
    0.100000 0.200000 0.300000 2.000000 \n\
    0.000000 0.000000 0.000000 1.000000 \n\
    0.000000 0.500000 0.500000 2.000000 \n\
    """)

    assert filecmp_ignore_whitespace('KPOINTS', 'KPOINTS.ref')


def test_explicit_auto_weight(atoms, testdir):
    # Explicit points as list of tuples, automatic weighting = 1.
    string = format_kpoints(
        atoms=atoms,
        kpts=[(0.1, 0.2, 0.3), (0.0, 0.0, 0.0), (0.0, 0.5, 0.5)],
        reciprocal=True)

    with open('KPOINTS', 'w') as fd:
        fd.write(string)

    with open('KPOINTS.ref', 'w') as fd:
        fd.write("""KPOINTS created by Atomic Simulation Environment
    3 \n\
    Reciprocal
    0.100000 0.200000 0.300000 1.0 \n\
    0.000000 0.000000 0.000000 1.0 \n\
    0.000000 0.500000 0.500000 1.0 \n\
    """)

    assert filecmp_ignore_whitespace('KPOINTS', 'KPOINTS.ref')


def test_bandpath(atoms):
    bandpath = atoms.cell.bandpath('GXMGRX,MR', npoints=100)
    string = format_kpoints(atoms=atoms, kpts=bandpath.kpts, reciprocal=True)
    check_kpoints_string(string, 1, '100')
    check_kpoints_string(string, 2, 'Reciprocal')
    check_kpoints_string(string, 3, '0.000000 0.000000 0.000000 1.0')
    check_kpoints_string(string, 102, '0.500000 0.500000 0.500000 1.0')
