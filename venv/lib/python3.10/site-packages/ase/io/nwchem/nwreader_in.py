# fmt: off

import re

import numpy as np

from ase import Atoms
from ase.geometry import cellpar_to_cell

from .parser import _define_pattern

# Geometry block parser
_geom = _define_pattern(
    r'^[ \t]*geometry[ \t\S]*\n'
    r'((?:^[ \t]*[\S]+[ \t\S]*\n)+)'
    r'^[ \t]*end\n\n',
    """\
geometry units angstrom nocenter noautosym noautoz
  system crystal units angstrom
    lattice_vectors
      4.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00
      0.0000000000000000e+00 5.5264780000000000e+00 0.0000000000000000e+00
      0.0000000000000000e+00 0.0000000000000000e+00 4.5963089999999998e+00
  end
   O 5.0000000000000000e-01 5.0000000000000011e-01 5.6486824536818558e-01
   H 5.0000000000000000e-01 6.3810586054988372e-01 4.3513175463181430e-01
   H 5.0000000000000000e-01 3.6189413945011639e-01 4.3513175463181430e-01
end

""", re.M)

# Finds crystal specification
_crystal = _define_pattern(
    r'^[ \t]*system crystal[ \t\S]*\n'
    r'((?:[ \t]*[\S]+[ \t\S]*\n)+?)'
    r'^[ \t]*end[ \t]*\n',
    """\
  system crystal units angstrom
    lattice_vectors
      4.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00
      0.0000000000000000e+00 5.5264780000000000e+00 0.0000000000000000e+00
      0.0000000000000000e+00 0.0000000000000000e+00 4.5963089999999998e+00
  end
""", re.M)

# Finds 3d-periodic unit cell
_cell_3d = _define_pattern(
    r'^[ \t]*lattice_vectors[ \t]*\n'
    r'^((?:(?:[ \t]+[\S]+){3}\n){3})',
    """\
    lattice_vectors
      4.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00
      0.0000000000000000e+00 5.5264780000000000e+00 0.0000000000000000e+00
      0.0000000000000000e+00 0.0000000000000000e+00 4.5963089999999998e+00
""", re.M)

# Extracts chemical species from a geometry block
_species = _define_pattern(
    r'^[ \t]*[A-Z][a-z]?(?:[ \t]+[\S]+){3}\n',
    "   O 0.0 0.0 0.0\n", re.M,
)


def read_nwchem_in(fobj, index=-1):
    text = ''.join(fobj.readlines())
    atomslist = []
    for match in _geom.findall(text):
        symbols = []
        positions = []
        for atom in _species.findall(match):
            atom = atom.split()
            symbols.append(atom[0])
            positions.append([float(x) for x in atom[1:]])
        positions = np.array(positions)
        atoms = Atoms(symbols)
        cell, pbc = _get_cell(text)
        pos = np.zeros_like(positions)
        for dim, ipbc in enumerate(pbc):
            if ipbc:
                pos += np.outer(positions[:, dim], cell[dim, :])
            else:
                pos[:, dim] = positions[:, dim]
        atoms.set_cell(cell)
        atoms.pbc = pbc
        atoms.set_positions(pos)
        atomslist.append(atoms)
    return atomslist[index]


def _get_cell(text):
    # first check whether there is a lattice definition
    cell = np.zeros((3, 3))
    lattice = _cell_3d.findall(text)
    if lattice:
        pbc = [True, True, True]
        for i, row in enumerate(lattice[0].strip().split('\n')):
            cell[i] = [float(x) for x in row.split()]
        return cell, pbc
    pbc = [False, False, False]
    lengths = [None, None, None]
    angles = [None, None, None]
    for row in text.strip().split('\n'):
        row = row.strip().lower()
        for dim, vecname in enumerate(['a', 'b', 'c']):
            if row.startswith(f'lat_{vecname}'):
                pbc[dim] = True
                lengths[dim] = float(row.split()[1])
        for i, angle in enumerate(['alpha', 'beta', 'gamma']):
            if row.startswith(angle):
                angles[i] = float(row.split()[1])

    if not np.any(pbc):
        return None, pbc

    for i in range(3):
        a, b, c = np.roll(np.array([0, 1, 2]), i)
        if pbc[a] and pbc[b]:
            assert angles[c] is not None
        if angles[c] is not None:
            assert pbc[a] and pbc[b]

    # The easiest case: all three lattice vectors and angles are specified
    if np.all(pbc):
        return cellpar_to_cell(lengths + angles), pbc

    # Next easiest case: exactly one lattice vector has been specified
    if np.sum(pbc) == 1:
        dim = np.argmax(pbc)
        cell[dim, dim] = lengths[dim]
        return cell, pbc

    # Hardest case: two lattice vectors are specified.
    dim1, dim2 = (dim for dim, ipbc in enumerate(pbc) if ipbc)
    angledim = np.argmin(pbc)
    cell[dim1, dim1] = lengths[dim1]
    cell[dim2, dim2] = lengths[dim2] * np.sin(angles[angledim])
    cell[dim2, dim1] = lengths[dim2] * np.cos(angles[angledim])
    return cell, pbc
