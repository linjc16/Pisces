import copy

import pytest

from ase.build import molecule
from ase.calculators.emt import EMT
from ase.constraints import FixInternals
from ase.optimize.bfgs import BFGS


def setup_atoms():
    atoms = molecule('CH3CH2OH', vacuum=5.0)
    atoms.rattle(stdev=0.3)
    return atoms


def setup_fixinternals():
    atoms = setup_atoms()

    # Angles, Bonds, Dihedrals are built up with pairs of constraint
    # value and indices defining the constraint
    # Linear combinations of bond lengths are built up similarly with the
    # coefficients appended to the indices defining the constraint

    # Fix bond between atoms 1 and 2 to 1.4
    bond_def = [1, 2]
    target_bond = 1.4

    # Fix angle to whatever it was from the start
    angle_def = [6, 0, 1]
    target_angle = atoms.get_angle(*angle_def)

    # Fix this dihedral angle to whatever it was from the start
    dihedral_def = [6, 0, 1, 2]
    target_dihedral = atoms.get_dihedral(*dihedral_def)

    # Initialize constraint
    constr = FixInternals(bonds=[(target_bond, bond_def)],
                          angles_deg=[(target_angle, angle_def)],
                          dihedrals_deg=[(target_dihedral, dihedral_def)],
                          epsilon=1e-10)
    return (atoms, constr, bond_def, target_bond, angle_def, target_angle,
            dihedral_def, target_dihedral)


@pytest.mark.optimize()
def test_fixinternals():
    (atoms, constr, bond_def, target_bond, angle_def, _target_angle,
     dihedral_def, _target_dihedral) = setup_fixinternals()

    opt = BFGS(atoms)

    previous_angle = atoms.get_angle(*angle_def)
    previous_dihedral = atoms.get_dihedral(*dihedral_def)

    print('angle before', previous_angle)
    print('dihedral before', previous_dihedral)
    print('bond length before', atoms.get_distance(*bond_def))
    print('target bondlength', target_bond)

    atoms.calc = EMT()
    atoms.set_constraint(constr)
    print('-----Optimization-----')
    opt.run(fmax=0.01)

    new_angle = atoms.get_angle(*angle_def)
    new_dihedral = atoms.get_dihedral(*dihedral_def)
    new_bondlength = atoms.get_distance(*bond_def)

    print('angle after', new_angle)
    print('dihedral after', new_dihedral)
    print('bondlength after', new_bondlength)

    err1 = new_angle - previous_angle
    err2 = new_dihedral - previous_dihedral
    err3 = new_bondlength - target_bond

    print('error in angle', repr(err1))
    print('error in dihedral', repr(err2))
    print('error in bondlength', repr(err3))

    for err in [err1, err2, err3]:
        assert abs(err) < 1e-11


def setup_combos():
    atoms = setup_atoms()

    # Fix linear combination of two bond lengths with atom indices 0-8 and
    # 0-6 with weighting coefficients 1.0 and -1.0 to the current value.
    # In other words, fulfil the following constraint:
    # 1.0 * atoms.get_distance(2, 1) + -1.0 * atoms.get_distance(2, 3) = const.
    bondcombo_def = [[2, 1, 1.0], [2, 3, -1.0]]
    target_bondcombo = FixInternals.get_bondcombo(atoms, bondcombo_def)

    # Initialize constraint; 'None' value should be converted to current value
    constr = FixInternals(bondcombos=[(None, bondcombo_def)], epsilon=1e-10)
    return atoms, constr, bondcombo_def, target_bondcombo,


@pytest.mark.optimize()
def test_combos():
    atoms, constr, bondcombo_def, target_bondcombo = setup_combos()

    ref_bondcombo = FixInternals.get_bondcombo(atoms, bondcombo_def)

    atoms.calc = EMT()
    atoms.set_constraint(constr)

    atoms2 = atoms.copy()  # check if 'None' value converts to current value
    atoms2.set_positions(atoms2.get_positions())
    checked_bondcombo = False
    for subconstr in atoms2.constraints[0].constraints:
        if repr(subconstr).startswith('FixBondCombo'):
            assert subconstr.targetvalue == target_bondcombo
            checked_bondcombo = True
    assert checked_bondcombo

    opt = BFGS(atoms)
    opt.run(fmax=0.01)

    new_bondcombo = FixInternals.get_bondcombo(atoms, bondcombo_def)
    err_bondcombo = new_bondcombo - ref_bondcombo

    print('error in bondcombo:', repr(err_bondcombo))
    assert abs(err_bondcombo) < 1e-11


def test_index_shuffle():
    (atoms, constr, _bond_def, _target_bond, _angle_def, _target_angle,
     _dihedral_def, _target_dihedral) = setup_fixinternals()

    constr2 = copy.deepcopy(constr)

    # test no change, test constr.get_indices()
    assert all(a == b for a, b in zip(constr.get_indices(), (0, 1, 2, 6, 8)))
    constr.index_shuffle(atoms, range(len(atoms)))
    assert all(a == b for a, b in zip(constr.get_indices(), (0, 1, 2, 6, 8)))

    # test full constraint is not part of new slice
    with pytest.raises(IndexError):
        constr.index_shuffle(atoms, [0])

    # test correct shuffling
    constr2.index_shuffle(atoms, [1, 2, 0, 6])
    assert constr2.bonds[0][1] == [0, 1]
    assert constr2.angles[0][1] == [3, 2, 0]
    assert constr2.dihedrals[0][1] == [3, 2, 0, 1]


def test_combo_index_shuffle():
    atoms, constr, _bondcombo_def, _target_bondcombo = setup_combos()

    # test no change, test constr.get_indices()
    answer = (1, 2, 3)
    assert all(a == b for a, b in zip(constr.get_indices(), answer))
    constr.index_shuffle(atoms, range(len(atoms)))
    assert all(a == b for a, b in zip(constr.get_indices(), answer))


@pytest.mark.optimize()
def test_zero_distance_error():
    """Zero distances cannot be fixed due to a singularity in the derivative.
    """
    atoms = setup_atoms()
    constr = FixInternals(bonds=[(0.0, [1, 2])])
    atoms.calc = EMT()
    atoms.set_constraint(constr)
    opt = BFGS(atoms)
    with pytest.raises(ZeroDivisionError):
        for _ in opt.irun():
            atoms.get_distance(1, 2)


@pytest.mark.optimize()
def test_planar_angle_error():
    """Support for planar angles could be added in the future using
       dummy/ghost atoms. See issue #868."""
    atoms = setup_atoms()
    constr = FixInternals(angles_deg=[(180, [6, 0, 1])])
    atoms.calc = EMT()
    atoms.set_constraint(constr)
    opt = BFGS(atoms)
    with pytest.raises(ZeroDivisionError):
        opt.run()


@pytest.mark.optimize()
def test_undefined_dihedral_error():
    atoms = setup_atoms()
    pos = atoms.get_positions()
    pos[0:3] = [[8, 5, 5], [7, 5, 5], [6, 5, 5]]
    atoms.set_positions(pos)  # with undefined dihedral
    with pytest.raises(ZeroDivisionError):
        atoms.get_dihedral(6, 0, 1, 2)
    constr = FixInternals(dihedrals_deg=[(20., [6, 0, 1, 2])])
    atoms.calc = EMT()
    atoms.set_constraint(constr)
    opt = BFGS(atoms)
    with pytest.raises(ZeroDivisionError):
        opt.run()
