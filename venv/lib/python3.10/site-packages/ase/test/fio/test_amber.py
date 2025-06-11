import numpy as np
import pytest

from ase.build import bulk
from ase.calculators.calculator import compare_atoms
from ase.io.amber import read_amber_coordinates, write_amber_coordinates


@pytest.mark.parametrize('with_velocities', [False, True])
def test_io_amber_coordinates(with_velocities):
    atoms = bulk('Au', orthorhombic=True)
    filename = 'amber.netcdf'

    if with_velocities:
        atoms.set_velocities(np.random.default_rng(42).random((len(atoms), 3)))

    write_amber_coordinates(atoms, filename)
    atoms2 = read_amber_coordinates(filename)

    # The format does not save the species so they revert to 'X'
    assert all(atoms2.symbols == 'X')
    assert compare_atoms(atoms, atoms2) == ['numbers']
    assert atoms.get_velocities() == pytest.approx(atoms2.get_velocities())


def test_cannot_write_nonorthorhombic():
    atoms = bulk('Ti')
    with pytest.raises(ValueError, match='Non-orthorhombic'):
        write_amber_coordinates(atoms, 'xxx')
