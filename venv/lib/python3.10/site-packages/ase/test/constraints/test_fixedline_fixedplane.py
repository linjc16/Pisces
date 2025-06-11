import numpy as np
import pytest

from ase.build import molecule
from ase.calculators.emt import EMT
from ase.constraints import FixedLine, FixedPlane
from ase.optimize import BFGS


@pytest.fixture(params=[FixedLine, FixedPlane])
def fixture_test_class(request):
    return request.param


@pytest.mark.parametrize(
    'indices', [
        0,
        [0],
        [0, 1],
        np.array([0, 1], dtype=np.int64),
    ]
)
def test_valid_inputs_indices(fixture_test_class, indices):
    _ = fixture_test_class(indices, [1, 0, 0])


@pytest.mark.parametrize(
    'indices', [
        [0, 1, 1],
        [[0, 1], [0, 1]],
    ]
)
def test_invalid_inputs_indices(fixture_test_class, indices):
    with pytest.raises(ValueError) as _:
        _ = fixture_test_class(indices, [1, 0, 0])


@pytest.mark.parametrize('direction', [[0, 0, 1], (0, 0, 1)])
def test_valid_inputs_direction(fixture_test_class, direction):
    _ = fixture_test_class(0, direction)


@pytest.mark.parametrize('direction', [[0, 1], None, "42"])
def test_invalid_inputs_direction(fixture_test_class, direction):
    with pytest.raises(Exception) as _:
        _ = FixedLine(0, direction)


def _check_simple_constraints(constraints, indices):
    mol = molecule("butadiene")
    mol.set_constraint(constraints)

    assert len(mol.constraints) == 1
    assert isinstance(constraints.dir, np.ndarray)
    assert (np.asarray([1, 0, 0]) == constraints.dir).all()

    mol.calc = EMT()

    cold_positions = mol[indices].positions.copy()
    opt = BFGS(mol)
    opt.run(steps=5)
    cnew_positions = mol[indices].positions.copy()

    return cold_positions, cnew_positions


@pytest.mark.parametrize('indices', [0, [0], [0, 1]])
def test_repr_fixedline(fixture_test_class, indices):
    repr(FixedLine(indices, [1, 0, 0])) == (
        "<FixedLine: "
        "{'indices': " + str(indices) + ", 'direction': [1. 0. 0.]}>"
    )


@pytest.mark.parametrize(
    'indices,expected', [
        (0, 2),
        ([0], 2),
        ([0, 1], 4),
    ]
)
def test_removed_dof_fixedline(indices, expected):
    mol = molecule("butadiene")  # `get_removed_dof` requires an `Atoms` object
    constraints = FixedLine(indices, direction=[1, 0, 0])
    assert constraints.get_removed_dof(atoms=mol) == expected


@pytest.mark.optimize()
@pytest.mark.parametrize('indices', [[0], [0, 1]])
def test_constrained_optimization_fixedline(indices):
    """
    A single int is not tested as that changes the call from Atoms.positions
    to Atom.position
    """
    constraints = FixedLine(indices, [1, 0, 0])

    cold_positions, cnew_positions = _check_simple_constraints(
        constraints, indices
    )
    assert np.max(np.abs(cnew_positions[:, 1:] - cold_positions[:, 1:])) < 1e-8
    assert np.max(np.abs(cnew_positions[:, 0] - cold_positions[:, 0])) > 1e-8


@pytest.mark.parametrize('indices', [0, [0], [0, 1]])
def test_repr_fixedplane(fixture_test_class, indices):
    repr(FixedPlane(indices, [1, 0, 0])) == (
        "<FixedPlane: "
        "{'indices': " + str(indices) + ", 'direction': [1. 0. 0.]}>"
    )


@pytest.mark.parametrize(
    'indices,expected', [
        (0, 1),
        ([0], 1),
        ([0, 1], 2),
    ]
)
def test_removed_dof_fixedplane(indices, expected):
    mol = molecule("butadiene")  # `get_removed_dof` requires an `Atoms` object
    constraints = FixedPlane(indices, direction=[1, 0, 0])
    assert constraints.get_removed_dof(atoms=mol) == expected


@pytest.mark.optimize()
@pytest.mark.parametrize('indices', [[0], [0, 1]])
def test_constrained_optimization_fixedplane(indices):
    """
    A single int is not tested as that changes the call from Atoms.positions
    to Atom.position
    """
    constraints = FixedPlane(indices, [1, 0, 0])

    cold_positions, cnew_positions = _check_simple_constraints(
        constraints, indices
    )
    assert np.max(np.abs(cnew_positions[:, 1:] - cold_positions[:, 1:])) > 1e-8
    assert np.max(np.abs(cnew_positions[:, 0] - cold_positions[:, 0])) < 1e-8
