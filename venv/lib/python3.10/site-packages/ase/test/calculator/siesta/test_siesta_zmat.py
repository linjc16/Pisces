"""Tests for Zmatrix"""
import os

import pytest

from ase import Atoms
from ase.constraints import FixAtoms, FixCartesian, FixedLine, FixedPlane


@pytest.fixture(name="atoms")
def fixture_atoms():
    """methane molecule"""
    positions = [
        (0.0, 0.0, 0.0),
        (+0.629118, +0.629118, +0.629118),
        (-0.629118, -0.629118, +0.629118),
        (+0.629118, -0.629118, -0.629118),
        (-0.629118, +0.629118, -0.629118),
    ]
    return Atoms('CH4', positions)


def test_siesta_zmat(siesta_factory, atoms: Atoms):
    """Test if the Zmatrix block (with constraints) is written properly."""
    c1 = FixAtoms(indices=[0])
    c2 = FixedLine(1, [0.0, 1.0, 0.0])
    c3 = FixedPlane(2, [1.0, 0.0, 0.0])
    c4 = FixCartesian(3, (True, True, False))

    atoms.set_constraint([c1, c2, c3, c4])

    custom_dir = './dir1/'

    # Test simple fdf-argument case.
    fdf_arguments = {'MD.TypeOfRun': 'CG', 'MD.NumCGsteps': 1000}
    siesta = siesta_factory.calc(
        label=custom_dir + 'test_label',
        symlink_pseudos=False,
        atomic_coord_format='zmatrix',
        fdf_arguments=fdf_arguments,
    )

    atoms.calc = siesta
    siesta.write_input(atoms, properties=['energy'])

    file = os.path.join(custom_dir, 'test_label.fdf')
    with open(file, encoding='utf-8') as fd:
        lines = fd.readlines()
    lsl = [line.split() for line in lines]
    assert ['cartesian'] in lsl
    assert ['%block', 'Zmatrix'] in lsl
    assert ['%endblock', 'Zmatrix'] in lsl
    assert ['MD.TypeOfRun', 'CG'] in lsl

    assert any(line.split()[4:9] == ['0', '0', '0', '1', 'C'] for line in lines)
    assert any(line.split()[4:9] == ['0', '1', '0', '2', 'H'] for line in lines)
    assert any(line.split()[4:9] == ['0', '1', '1', '3', 'H'] for line in lines)
    assert any(line.split()[4:9] == ['0', '0', '1', '4', 'H'] for line in lines)


@pytest.mark.parametrize("constraint_class", [FixedLine, FixedPlane])
def test_invalid_constraint(siesta_factory, atoms: Atoms, constraint_class):
    """Test if an invalid constraint return RuntimeError."""

    # constraint with an invalid direction (not along Carterian)
    constraint = constraint_class(indices=[0], direction=(1, 1, 0))

    atoms.set_constraint(constraint)

    custom_dir = './dir2/'

    # Test simple fdf-argument case.
    fdf_arguments = {'MD.TypeOfRun': 'CG', 'MD.NumCGsteps': 1000}
    siesta = siesta_factory.calc(
        label=custom_dir + 'test_label',
        symlink_pseudos=False,
        atomic_coord_format='zmatrix',
        fdf_arguments=fdf_arguments,
    )
    with pytest.raises(RuntimeError):  # from make_xyz_constraints
        siesta.write_input(atoms, properties=['energy'])
