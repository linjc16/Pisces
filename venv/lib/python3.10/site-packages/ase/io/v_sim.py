# fmt: off

"""
This module contains functionality for reading and writing an ASE
Atoms object in V_Sim 3.5+ ascii format.

"""

import numpy as np

from ase.utils import reader, writer


@reader
def read_v_sim(fd):
    """Import V_Sim input file.

    Reads cell, atom positions, etc. from v_sim ascii file.
    V_sim format is specified here:
    https://l_sim.gitlab.io/v_sim/sample.html#sample_ascii
    """

    import re

    from ase import Atoms, units
    from ase.geometry import cellpar_to_cell

    # Read comment:
    fd.readline()

    line = fd.readline() + ' ' + fd.readline()
    box = line.split()
    for i in range(len(box)):
        box[i] = float(box[i])

    keywords = []
    positions = []
    symbols = []
    unit = 1.0

    re_comment = re.compile(r'^\s*[#!]')
    re_node = re.compile(r'^\s*\S+\s+\S+\s+\S+\s+\S+')

    while True:
        line = fd.readline()

        if line == '':
            break  # EOF

        p = re_comment.match(line)
        if p is not None:
            # remove comment character at the beginning of line
            line = line[p.end():].replace(',', ' ').lower()
            if line[:8] == "keyword:":
                keywords.extend(line[8:].split())

        elif re_node.match(line):
            unit = 1.0
            if "reduced" not in keywords:
                if (("bohr" in keywords) or ("bohrd0" in keywords) or
                        ("atomic" in keywords) or ("atomicd0" in keywords)):
                    unit = units.Bohr

            fields = line.split()
            positions.append([unit * float(fields[0]),
                              unit * float(fields[1]),
                              unit * float(fields[2])])
            symbols.append(fields[3])

    # create atoms object based on the information
    if "angdeg" in keywords:
        cell = cellpar_to_cell(box)
    else:
        unit = 1.0
        if (("bohr" in keywords) or ("bohrd0" in keywords) or
                ("atomic" in keywords) or ("atomicd0" in keywords)):
            unit = units.Bohr
        cell = np.zeros((3, 3))
        cell.flat[[0, 3, 4, 6, 7, 8]] = box[:6]
        cell *= unit

    if "reduced" in keywords:
        atoms = Atoms(cell=cell, scaled_positions=positions)
    else:
        atoms = Atoms(cell=cell, positions=positions)

    if "periodic" in keywords:
        atoms.pbc = [True, True, True]
    elif "freebc" in keywords:
        atoms.pbc = [False, False, False]
    elif "surface" in keywords:
        atoms.pbc = [True, False, True]
    else:  # default is periodic boundary conditions
        atoms.pbc = [True, True, True]

    atoms.set_chemical_symbols(symbols)
    return atoms


@writer
def write_v_sim(fd, atoms):
    """Write V_Sim input file.

    Writes the atom positions and unit cell.
    V_sim format is specified here:
    https://l_sim.gitlab.io/v_sim/sample.html#sample_ascii
    """
    from ase.geometry import cell_to_cellpar, cellpar_to_cell

    # Convert the lattice vectors to triangular matrix by converting
    #   to and from a set of lengths and angles
    cell = cellpar_to_cell(cell_to_cellpar(atoms.cell))
    dxx = cell[0, 0]
    dyx, dyy = cell[1, 0:2]
    dzx, dzy, dzz = cell[2, 0:3]

    fd.write('===== v_sim input file created using the'
             ' Atomic Simulation Environment (ASE) ====\n')
    fd.write(f'{dxx} {dyx} {dyy}\n')
    fd.write(f'{dzx} {dzy} {dzz}\n')

    # Use v_sim 3.5 keywords to indicate scaled positions, etc.
    fd.write('#keyword: reduced\n')
    fd.write('#keyword: angstroem\n')
    if np.all(atoms.pbc):
        fd.write('#keyword: periodic\n')
    elif not np.any(atoms.pbc):
        fd.write('#keyword: freeBC\n')
    elif np.array_equiv(atoms.pbc, [True, False, True]):
        fd.write('#keyword: surface\n')
    else:
        raise Exception(
            'Only supported boundary conditions are full PBC,'
            ' no periodic boundary, and surface which is free in y direction'
            ' (i.e. Atoms.pbc = [True, False, True]).')

    # Add atoms (scaled positions)
    for position, symbol in zip(atoms.get_scaled_positions(),
                                atoms.get_chemical_symbols()):
        fd.write('{} {} {} {}\n'.format(
            position[0], position[1], position[2], symbol))
