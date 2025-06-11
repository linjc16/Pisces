# fmt: off

"""Reads chemical data in SDF format (wraps the molfile format).

See https://en.wikipedia.org/wiki/Chemical_table_file#SDF
"""
from typing import TextIO

from ase.atoms import Atoms
from ase.utils import reader


def get_num_atoms_sdf_v2000(first_line: str) -> int:
    """Parse the first line extracting the number of atoms.

    The V2000 dialect uses a fixed field length of 3, which means there
    won't be space between the numbers if there are 100+ atoms, and
    the format doesn't support 1000+ atoms at all.

    http://biotech.fyicenter.com/1000024_SDF_File_Format_Specification.html
    """
    return int(first_line[0:3])  # first three characters


@reader
def read_sdf(file_obj: TextIO) -> Atoms:
    """Read the sdf data and compose the corresponding Atoms object."""
    lines = file_obj.readlines()
    # first three lines header
    del lines[:3]

    num_atoms = get_num_atoms_sdf_v2000(lines.pop(0))
    positions = []
    symbols = []
    for line in lines[:num_atoms]:
        x, y, z, symbol = line.split()[:4]
        symbols.append(symbol)
        positions.append((float(x), float(y), float(z)))
    return Atoms(symbols=symbols, positions=positions)
