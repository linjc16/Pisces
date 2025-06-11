from ase import Atoms
from ase.calculators.test import FreeElectrons
from ase.cell import Cell


def test_monoclinic():
    """Test band structure from different variations of hexagonal cells."""
    mc1 = Cell([[1, 0, 0], [0, 1, 0], [0, 0.2, 1]])
    par = mc1.cellpar()
    mc2 = Cell.new(par)
    mc3 = Cell([[1, 0, 0], [0, 1, 0], [-0.2, 0, 1]])
    mc4 = Cell([[1, 0, 0], [-0.2, 1, 0], [0, 0, 1]])
    path = 'GYHCEM1AXH1'

    firsttime = True
    for cell in [mc1, mc2, mc3, mc4]:
        a = Atoms(cell=cell, pbc=True)
        a.cell *= 3
        a.calc = FreeElectrons(nvalence=1, kpts={'path': path})

        lat = a.cell.get_bravais_lattice()
        assert lat.name == 'MCL'
        a.get_potential_energy()
        bs = a.calc.band_structure()
        coords, labelcoords, labels = bs.get_labels()
        assert ''.join(labels) == path
        e_skn = bs.energies

        if firsttime:
            coords1 = coords
            labelcoords1 = labelcoords
            e_skn1 = e_skn
            firsttime = False
        else:
            for d in [coords - coords1,
                      labelcoords - labelcoords1,
                      e_skn - e_skn1]:
                print(abs(d).max())
                assert abs(d).max() < 1e-13, d
