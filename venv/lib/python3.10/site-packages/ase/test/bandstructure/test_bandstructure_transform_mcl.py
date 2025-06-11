import numpy as np

from ase import Atoms
from ase.calculators.test import FreeElectrons
from ase.cell import Cell
from ase.spectrum.band_structure import calculate_band_structure
from ase.utils import workdir


def test_bandstructure_transform_mcl(testdir):
    # Test that bandpath() correctly transforms the band path from
    # reference (canonical) cell to actual cell provided by user.

    def _atoms(cell):
        atoms = Atoms(cell=cell, pbc=True)
        atoms.calc = FreeElectrons()
        return atoms

    # MCL with beta > 90, which is a common convention -- but ours is
    # alpha < 90.  We want the bandpath returned by that cell to yield the
    # exact same band structure as our own (alpha < 90) version of the
    # same cell.
    cell = Cell.new([3., 5., 4., 90., 110., 90.])
    lat = cell.get_bravais_lattice()

    density = 10.0
    cell0 = lat.tocell()
    path0 = lat.bandpath(density=density)

    print(cell.cellpar().round(3))
    print(cell0.cellpar().round(3))

    with workdir('files', mkdir=True):
        bs = calculate_band_structure(_atoms(cell),
                                      cell.bandpath(density=density))
        bs.write('bs.json')
        # bs.plot(emin=0, emax=20, filename='fig.bs.svg')

        bs0 = calculate_band_structure(_atoms(cell0), path0)
        bs0.write('bs0.json')
        # bs0.plot(emin=0, emax=20, filename='fig.bs0.svg')

    maxerr = np.abs(bs.energies - bs0.energies).max()
    assert maxerr < 1e-12, maxerr
