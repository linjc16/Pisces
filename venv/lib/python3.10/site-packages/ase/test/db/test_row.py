from ase import Atoms
from ase.db.row import AtomsRow


def test_row():
    atoms = Atoms('H')
    row = AtomsRow(atoms)
    assert row.charge == 0.0
    atoms.set_initial_charges([1.0])
    row = AtomsRow(atoms)
    assert row.charge == 1.0
