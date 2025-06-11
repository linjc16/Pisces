from pathlib import Path

from ase.build import molecule
from ase.io.bader import attach_charges


def test_bader(testdir):
    fname = 'ACF.dat'
    Path(fname).write_text("""
       #         X           Y           Z        CHARGE     MIN DIST
     ----------------------------------------------------------------
       1      7.0865      8.5038      9.0672      9.0852      1.3250
       2      7.0865      9.9461      7.9403      0.4574      0.3159
       3      7.0865      7.0615      7.9403      0.4574      0.3159
     ----------------------------------------------------------------
      NUMBER OF ELECTRONS:        9.99999
    """)

    atoms = molecule('H2O')
    atoms.set_cell([7.5, 9, 9])
    atoms.center()

    attach_charges(atoms)
    attach_charges(atoms, fname)

    for atom in atoms:
        print('Atom', atom.symbol, 'Bader charge', atom.charge)

    # O is negatively charged
    assert atoms[0].charge < -1 and atoms[0].charge > -2
