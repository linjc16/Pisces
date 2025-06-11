import numpy as np

from ase.build import bulk
from ase.calculators.emt import EMT
from ase.constraints import (
    FixCartesianParametricRelations,
    FixScaledParametricRelations,
    dict2constraint,
)


def test_parameteric_constr():

    # Build the atoms object and attach a calculator
    a = bulk("Ni", cubic=True)
    a.calc = EMT()

    # Get adjusted cell
    cell = a.cell + 0.01

    # Generate lattice constraint
    param_lat = ["a"]
    expr_lat = [
        "a", "0", "0",
        "0", "a", "0",
        "0", "0", "a",
    ]
    constr_lat = FixCartesianParametricRelations.from_expressions(
        indices=[0, 1, 2],
        params=param_lat,
        expressions=expr_lat,
        use_cell=True,
    )

    # Check expression generator
    for const_expr, passed_expr in zip(
            constr_lat.expressions.flatten(), expr_lat):
        assert const_expr == passed_expr

    # Check adjust_cell
    constr_lat.adjust_cell(a, cell)

    # Check serialization and construction from dict
    constr_lat_dict = constr_lat.todict()
    dict2constraint(constr_lat_dict)

    cell_diff = (cell - a.cell).flatten()
    expected_cell_diff = np.array(
        [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01])
    assert np.max(np.abs(cell_diff - expected_cell_diff)) < 1e-12

    # Check adjust_stress
    a.cell += 0.01
    stress = a.get_stress().copy()
    constr_lat.adjust_stress(a, stress)
    stress_rat = stress / a.get_stress()

    assert np.max(
        np.abs(stress_rat - np.array([1., 1., 1., 0., 0., 0.]))) < 1e-12

    # Reset cell
    a.cell -= 0.01

    # Get adjusted cell/positions for the system
    pos = a.get_positions().copy() + 0.01

    # Generate proper atomic constraints
    constr_atom = FixScaledParametricRelations(
        [0, 1, 2, 3],
        np.ndarray((12, 0)),
        a.get_scaled_positions().flatten(),
    )

    # Check serialization and construction from dict
    constr_atom_dict = constr_atom.todict()
    dict2constraint(constr_atom_dict)

    # Check adjust_positions
    constr_atom.adjust_positions(a, pos)
    assert np.max(np.abs(a.get_positions() - pos)) < 1e-12

    # Check adjust_forces
    assert np.max(np.abs(a.get_forces())) < 1e-12

    # Check non-empty constraint
    param_atom = ["dis"]
    expr_atom = [
        "dis", "dis", "dis",
        "dis", "-0.5", "0.5",
        "0.5", "dis", "0.5",
        "0.5", "0.5", "dis",
    ]

    constr_atom = FixScaledParametricRelations.from_expressions(
        indices=[0, 1, 2, 3],
        params=param_atom,
        expressions=expr_atom,
    )

    # Restart position adjustment
    pos += 0.01 * a.cell[0, 0]

    # Check adjust_positions
    constr_atom.adjust_positions(a, pos)
    scaled_pos = a.cell.scaled_positions(pos)
    pos_diff = (scaled_pos - a.get_scaled_positions()).flatten()
    expected_pos_diff = np.array(
        [0.01, 0.01, 0.01, 0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01]
    )
    assert np.max(np.abs(pos_diff - expected_pos_diff)) < 1e-12

    # Check adjust_forces
    a.set_positions(pos + 0.3)
    forces = a.get_forces()
    constr_atom.adjust_forces(a, forces)
    forces_rat = forces / a.get_forces()

    assert np.max(np.abs(forces_rat.flatten() /
                  100.0 - expected_pos_diff)) < 1e-12

    # Check auto-remapping/expression generation, the -0.5 should now be 0.5
    expr_atom[4] = "0.5"
    current_expression = constr_atom.expressions.flatten()
    for const_expr, passed_expr in zip(current_expression, expr_atom):
        assert const_expr == passed_expr

    # Check with Cartesian parametric constraints now
    expr_atom = [
        "dis", "dis", "dis",
        "dis", "1.76", "1.76",
        "1.76", "dis", "1.76",
        "1.76", "1.76", "dis",
    ]
    constr_atom = FixCartesianParametricRelations.from_expressions(
        indices=[0, 1, 2, 3],
        params=param_atom,
        expressions=expr_atom,
    )

    # Restart position adjustment
    a.set_positions(pos)
    pos += 0.01
    # Check adjust_positions
    constr_atom.adjust_positions(a, pos)
    pos_diff = (pos - a.get_positions()).flatten()
    expected_pos_diff = np.array(
        [0.01, 0.01, 0.01, 0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01]
    )
    assert np.max(np.abs(pos_diff - expected_pos_diff)) < 1e-12

    # Check adjust_forces
    a.set_positions(pos + 0.3)
    forces = a.get_forces()
    constr_atom.adjust_forces(a, forces)
    forces_rat = forces / a.get_forces()

    assert np.max(np.abs(forces_rat.flatten() /
                  100.0 - expected_pos_diff)) < 1e-12
