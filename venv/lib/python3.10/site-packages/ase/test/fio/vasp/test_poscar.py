# import inspect
from shutil import copyfile

import numpy as np
import pytest

# from ase import Atoms
from ase.io import read  # , iread


@pytest.fixture()
def outcar(datadir):
    return datadir / 'vasp' / 'OUTCAR_example_1'


@pytest.fixture()
def poscar_no_species(datadir):
    return datadir / 'vasp' / 'POSCAR_example_1'


def test_read_poscar_no_species(outcar, poscar_no_species, tmp_path):
    copyfile(outcar, tmp_path / 'OUTCAR')
    copyfile(poscar_no_species, tmp_path / 'POSCAR')

    at_outcar = read(outcar)
    at_poscar = read(tmp_path / 'POSCAR')

    assert len(at_outcar) == len(at_poscar)
    assert np.all(np.isclose(at_outcar.cell, at_poscar.cell))
    assert np.all(np.isclose(at_outcar.positions, at_poscar.positions))
    assert np.all(at_outcar.numbers == at_poscar.numbers)
