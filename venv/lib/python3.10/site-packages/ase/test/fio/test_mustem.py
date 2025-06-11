import numpy as np
import pytest

from ase import Atoms
from ase.atoms import symbols2numbers
from ase.build import bulk
from ase.io import read


def make_STO_atoms():
    atoms = Atoms(['O', 'O', 'O', 'Sr', 'Ti'],
                  scaled_positions=[[0.5, 0.5, 0],
                                    [0.5, 0, 0.5],
                                    [0, 0.5, 0.5],
                                    [0, 0, 0],
                                    [0.5, 0.5, 0.5]],
                  cell=[3.905, 3.905, 3.905],
                  pbc=True)

    return atoms


def test_mustem_several_elements():
    """Check writing and reading a xtl mustem file."""
    # Reproduce the sto xtl file distributed with muSTEM
    atoms = make_STO_atoms()

    filename = 'sto_mustem.xtl'
    STO_DW_dict = {'Sr': 0.62, 'O': 0.73, 'Ti': 0.43}
    STO_DW_dict_Ti_missing = {key: STO_DW_dict[key] for key in ['Sr', 'O']}

    with pytest.raises(TypeError):
        atoms.write(filename)

    with pytest.raises(ValueError):
        atoms.write(filename, keV=300)

    with pytest.raises(TypeError):
        atoms.write(filename,
                    debye_waller_factors=STO_DW_dict)

    atoms.write(filename, keV=300,
                debye_waller_factors=STO_DW_dict)

    atoms2 = read(filename, format='mustem')
    atoms3 = read(filename)

    for _atoms in [atoms2, atoms3]:
        assert atoms.positions == pytest.approx(_atoms.positions)
        np.testing.assert_allclose(atoms.cell, _atoms.cell)

    with pytest.raises(ValueError):
        # Raise an error if there is a missing key.
        atoms.write(filename, keV=300,
                    debye_waller_factors=STO_DW_dict_Ti_missing)

    atoms.write(filename, keV=300,
                debye_waller_factors=STO_DW_dict,
                occupancies={'Sr': 1.0, 'O': 0.5, 'Ti': 0.9})

    with pytest.raises(ValueError):
        # Raise an error if there is a missing key.
        atoms.write(filename, keV=300,
                    debye_waller_factors=STO_DW_dict,
                    occupancies={'O': 0.5, 'Ti': 0.9})

    with pytest.raises(ValueError):
        # Raise an error if the unit cell is not defined.
        atoms4 = Atoms(['Sr', 'Ti', 'O', 'O', 'O'],
                       positions=[[0, 0, 0],
                                  [0.5, 0.5, 0.5],
                                  [0.5, 0.5, 0],
                                  [0.5, 0, 0.5],
                                  [0, 0.5, 0.5]])
        atoms4.write(filename, keV=300,
                     debye_waller_factors=STO_DW_dict)

    atoms5 = make_STO_atoms()
    atoms5.set_array('occupancies', np.ones(5))
    atoms5.arrays['occupancies'][atoms5.numbers == symbols2numbers('Sr')] = 0.9
    # element 0 is Sr and there is onlye one Sr in the cell: this is a valid
    # cell to export to xtl file
    atoms5.write(filename, keV=300, debye_waller_factors=STO_DW_dict)
    atoms6 = read(filename)
    condition = atoms6.numbers == symbols2numbers('Sr')
    np.testing.assert_allclose(atoms6.arrays['occupancies'][condition], 0.9)

    atoms5.arrays['occupancies'][0] = 0.8
    with pytest.raises(ValueError):
        atoms5.write(filename, keV=300, debye_waller_factors=STO_DW_dict)

    atoms7 = make_STO_atoms()
    debye_waller_factors = np.array([0.73, 0.73, 0.73, 0.62, 0.43])
    atoms7.set_array('debye_waller_factors', debye_waller_factors)
    # element 0 is Sr and there is onlye one Sr in the cell: this is a valid
    # cell to export to xtl file
    atoms7.write(filename, keV=300)
    atoms8 = read(filename)
    for element in ['Sr', 'Ti', 'O']:
        number = symbols2numbers(element)
        np.testing.assert_allclose(
            atoms7.arrays['debye_waller_factors'][atoms7.numbers == number],
            atoms8.arrays['debye_waller_factors'][atoms8.numbers == number],
            rtol=1e-2
        )


def test_mustem_single_elements():
    # Setting Debye-Waller factor as float.
    Si_atoms = bulk('Si', cubic=True)

    filename = 'Si100.xtl'
    DW = 0.62
    Si_atoms.write(filename, keV=300, debye_waller_factors=DW)
    Si_atoms2 = read(filename)

    np.testing.assert_allclose(Si_atoms.positions, Si_atoms2.positions)
    np.testing.assert_allclose(Si_atoms.cell, Si_atoms2.cell)
    np.testing.assert_allclose(Si_atoms2.arrays['occupancies'], np.ones(8))
    np.testing.assert_allclose(Si_atoms2.arrays['debye_waller_factors'],
                               np.ones(8) * DW, rtol=1e-2)

    Si_atoms3 = bulk('Si', cubic=True)
    Si_atoms3.set_array('occupancies', np.ones(8) * 0.9)
    Si_atoms3.set_array('debye_waller_factors', np.ones(8) * DW)
    Si_atoms3.write(filename, keV=300)

    Si_atoms4 = read(filename)
    np.testing.assert_allclose(Si_atoms3.positions, Si_atoms4.positions)
    np.testing.assert_allclose(Si_atoms3.cell, Si_atoms4.cell)
    np.testing.assert_allclose(Si_atoms3.arrays['occupancies'],
                               Si_atoms4.arrays['occupancies'])
    np.testing.assert_allclose(Si_atoms3.arrays['debye_waller_factors'],
                               Si_atoms4.arrays['debye_waller_factors'],
                               rtol=1e-2)

    Si_atoms5 = bulk('Si', cubic=True)
    debye_waller_factors = np.ones(8) * DW
    debye_waller_factors[0] = debye_waller_factors[0] / 2
    Si_atoms5.set_array('debye_waller_factors', debye_waller_factors)

    with pytest.raises(ValueError):
        # Raise an error if one Debye-Waller factor is different.
        Si_atoms5.write(filename, keV=300)
