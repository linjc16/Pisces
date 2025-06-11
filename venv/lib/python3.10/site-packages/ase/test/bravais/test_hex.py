import numpy as np

from ase.build import bulk


def test_bravais_hex():
    atoms = bulk('Ti')
    assert np.allclose(atoms.cell.angles(), [90, 90, 120])
    atoms.cell.get_bravais_lattice().name == 'HEX'
    cell = atoms.cell.copy()
    cell[0] += cell[1]

    assert np.allclose(cell.angles(), [90, 90, 60])
    lat = cell.get_bravais_lattice()
    assert lat.name == 'HEX'
    assert np.allclose(lat.tocell().angles(), [90, 90, 120])
