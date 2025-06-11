import numpy as np

from ase.build import bulk, minimize_rotation_and_translation, molecule


def test_with_pbc():

    atoms_start = bulk('Cu', 'fcc', a=3.6, cubic=True)
    # regardless of the structure, we want the test
    # contained in the first cell for the position test at the end
    atoms_start.wrap()
    atoms_end = atoms_start.copy()

    # ensures atoms cross the PBC
    shift = np.array([-1.6, -1.6, -1.6]) @ atoms_start.get_cell()
    atoms_end.translate(shift)

    minimize_rotation_and_translation(atoms_start, atoms_end)
    # since minimize_rotation_and_translation adds the displacement only,
    # we should to rewrap to be certain.
    atoms_end.wrap()

    assert np.allclose(atoms_end.get_positions(), atoms_start.get_positions())


def test_without_pbc():

    atoms_start = molecule('NH3')
    atoms_end = atoms_start.copy()

    # very well rotated
    atoms_end.rotate(a=88, v='x', center='COP')
    atoms_end.rotate(a=66, v='y', center='COP')
    atoms_end.rotate(a=44, v='z', center='COP')

    shift = [1.0, 2.0, 3.0]
    atoms_end.translate(shift)

    minimize_rotation_and_translation(atoms_start, atoms_end)
    assert np.allclose(atoms_end.get_positions(), atoms_start.get_positions())
