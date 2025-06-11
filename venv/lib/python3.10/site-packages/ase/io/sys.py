# fmt: off

"""
IO support for the qb@ll sys format.

The positions and cell dimensions are in Bohrs.

Contributed by Rafi Ullah <rraffiu@gmail.com>
"""

from re import compile

from ase.atoms import Atoms
from ase.units import Bohr

__all__ = ['read_sys', 'write_sys']


def read_sys(fileobj):
    """
    Function to read a qb@ll sys file.

    fileobj: file (object) to read from.
    """
    a1, a2, a3, b1, b2, b3, c1, c2, c3 = fileobj.readline().split()[2:11]
    cell = []
    cell.append([float(a1) * Bohr, float(a2) * Bohr, float(a3) * Bohr])
    cell.append([float(b1) * Bohr, float(b2) * Bohr, float(b3) * Bohr])
    cell.append([float(c1) * Bohr, float(c2) * Bohr, float(c3) * Bohr])
    positions = []
    symbols = []
    reg = compile(r'(\d+|\s+)')
    line = fileobj.readline()
    while 'species' in line:
        line = fileobj.readline()
    while line:
        # The units column is ignored.
        _a, symlabel, _spec, x, y, z = line.split()[0:6]
        positions.append([float(x) * Bohr, float(y) * Bohr,
                         float(z) * Bohr])
        sym = reg.split(str(symlabel))
        symbols.append(sym[0])
        line = fileobj.readline()
    atoms = Atoms(symbols=symbols, cell=cell, positions=positions)
    return atoms


def write_sys(fileobj, atoms):
    """
    Function to write a sys file.

    fileobj: file object
        File to which output is written.
    atoms: Atoms object
        Atoms object specifying the atomic configuration.
    """
    fileobj.write('set cell')
    for i in range(3):
        d = atoms.cell[i] / Bohr
        fileobj.write((' {:6f}  {:6f}  {:6f}').format(*d))
    fileobj.write('  bohr\n')

    ch_sym = atoms.get_chemical_symbols()
    atm_nm = atoms.numbers
    a_pos = atoms.positions
    an = list(set(atm_nm))

    for i, s in enumerate(set(ch_sym)):
        fileobj.write(('species {}{} {}.xml \n').format(s, an[i], s))
    for i, (S, Z, (x, y, z)) in enumerate(zip(ch_sym, atm_nm, a_pos)):
        fileobj.write(('atom {:5} {:5}  {:12.6f} {:12.6f} {:12.6f}\
        bohr\n').format(S + str(i + 1), S + str(Z), x / Bohr, y / Bohr,
                        z / Bohr))
