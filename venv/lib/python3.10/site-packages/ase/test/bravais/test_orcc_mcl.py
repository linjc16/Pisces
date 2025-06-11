import numpy as np

from ase import Atoms
from ase.calculators.emt import EMT
from ase.cell import Cell


def test_bravais_orcc_mcl():

    def get_e(cell):
        atoms = Atoms('Au', cell=cell, pbc=1)
        atoms.calc = EMT()
        return atoms.get_potential_energy()

    cell = Cell.new([[1, 0, 0], [0, 2, 0], [0.5, 0, 3]])

    lat = cell.get_bravais_lattice()
    assert lat.name == 'ORCC'

    cell2 = lat.tocell()
    e1 = get_e(cell)
    e2 = get_e(cell2)
    print(e1, e2)
    assert abs(e2 - e1) < 1e-12

    cp1 = cell.niggli_reduce()[0].cellpar()
    cp2 = lat.tocell().niggli_reduce()[0].cellpar()
    print('cellpar1', cp1)
    print('cellpar2', cp2)
    assert np.abs(cp2 - cp1).max() < 1e-12

    mcl_cell = Cell.new([[1, 0, 0], [0, 2, 0], [0.5 - 1e-3, 0, 3]])
    mcl_lat = mcl_cell.get_bravais_lattice()
    assert mcl_lat.name == 'MCL'
    e1 = get_e(mcl_cell)
    e2 = get_e(mcl_lat.tocell())
    assert abs(e2 - e1) < 1e-11, abs(e2 - e1)  # (Error is actually 1e-12)
    cp1 = mcl_cell.niggli_reduce()[0].cellpar()
    cp2 = mcl_lat.tocell().niggli_reduce()[0].cellpar()
    print(cp1)
    print(cp2)
    assert np.abs(cp2 - cp1).max() < 1e-12
