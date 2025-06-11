import numpy as np
import pytest

from ase import Atoms
from ase.build import bulk
from ase.io import read

from .test_mustem import make_STO_atoms


def test_write_read_cycle_xyz_prismatic():
    """Check writing and reading a xtl mustem file."""
    # Reproduce the SI100.XYZ file distributed with prismatic
    atoms = bulk('Si', cubic=True)
    atoms.set_array('occupancies', np.ones(len(atoms)))
    rng = np.random.RandomState(42)
    atoms.set_array('debye_waller_factors', 0.62 + 0.1 * rng.random(len(atoms)))

    filename = 'SI100.XYZ'
    atoms.write(filename=filename, format='prismatic',
                comments='one unit cell of 100 silicon')

    atoms_loaded = read(filename=filename, format='prismatic')

    np.testing.assert_allclose(atoms.positions, atoms_loaded.positions)
    np.testing.assert_allclose(atoms.cell, atoms_loaded.cell)
    np.testing.assert_allclose(atoms.get_array('occupancies'),
                               atoms_loaded.get_array('occupancies'))
    np.testing.assert_allclose(atoms.get_array('debye_waller_factors'),
                               atoms_loaded.get_array('debye_waller_factors'),
                               rtol=1E-5)


def test_write_error():
    """Check missing parameter when writing xyz prismatic file."""
    atoms_Si100 = bulk('Si', cubic=True)
    atoms_STO = make_STO_atoms()
    filename = 'SI100.XYZ'

    with pytest.raises(ValueError):
        # DW not provided
        atoms_Si100.write(filename, format='prismatic')

    # Write file with DW provided as scalar
    atoms_Si100.write(filename, format='prismatic',
                      debye_waller_factors=0.076)

    # Write file with DW provided as dict
    atoms_Si100.write(filename, format='prismatic',
                      debye_waller_factors={'Si': 0.076})

    STO_DW_dict = {'Sr': 0.62, 'O': 0.73, 'Ti': 0.43}
    STO_DW_dict_Ti_missing = {key: STO_DW_dict[key] for key in ['Sr', 'O']}

    with pytest.raises(ValueError):
        # DW missing keys
        atoms_STO.write(filename, format='prismatic',
                        debye_waller_factors=STO_DW_dict_Ti_missing)

    atoms_STO.write(filename, format='prismatic',
                    debye_waller_factors=STO_DW_dict)

    with pytest.raises(ValueError):
        # Raise an error if the unit cell is not defined.
        atoms4 = Atoms(['Sr', 'Ti', 'O', 'O', 'O'],
                       positions=[[0, 0, 0],
                                  [0.5, 0.5, 0.5],
                                  [0.5, 0.5, 0],
                                  [0.5, 0, 0.5],
                                  [0, 0.5, 0.5]])
        atoms4.write(filename, format='prismatic',
                     debye_waller_factors=STO_DW_dict)
