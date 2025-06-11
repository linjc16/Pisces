# fmt: off

"""Read Wannier90 wout format."""
from typing import IO, Any, Dict

import numpy as np

from ase import Atoms


def read_wout_all(fileobj: IO[str]) -> Dict[str, Any]:
    """Read atoms, wannier function centers and spreads."""
    lines = fileobj.readlines()

    for n, line in enumerate(lines):
        if line.strip().lower().startswith('lattice vectors (ang)'):
            break
    else:
        raise ValueError('Could not fine lattice vectors')

    cell = [[float(x) for x in line.split()[-3:]]
            for line in lines[n + 1:n + 4]]

    for n, line in enumerate(lines):
        if 'cartesian coordinate (ang)' in line.lower():
            break
    else:
        raise ValueError('Could not find coordinates')

    positions = []
    symbols = []
    n += 2
    while True:
        words = lines[n].split()
        if len(words) == 1:
            break
        positions.append([float(x) for x in words[-4:-1]])
        symbols.append(words[1])
        n += 1

    atoms = Atoms(symbols, positions, cell=cell, pbc=True)

    n = len(lines) - 1
    while n > 0:
        if lines[n].strip().lower().startswith('final state'):
            break
        n -= 1
    else:
        return {'atoms': atoms,
                'centers': np.zeros((0, 3)),
                'spreads': np.zeros((0,))}

    n += 1
    centers = []
    spreads = []
    while True:
        line = lines[n].strip()
        if line.startswith('WF'):
            centers.append([float(x)
                            for x in
                            line.split('(')[1].split(')')[0].split(',')])
            spreads.append(float(line.split()[-1]))
            n += 1
        else:
            break

    return {'atoms': atoms,
            'centers': np.array(centers),
            'spreads': np.array(spreads)}


def read_wout(fileobj: IO[str],
              include_wannier_function_centers: bool = True) -> Atoms:
    """Read atoms and wannier function centers (as symbol X)."""
    dct = read_wout_all(fileobj)
    atoms = dct['atoms']
    if include_wannier_function_centers:
        centers = dct['centers']
        atoms += Atoms(f'X{len(centers)}', centers)
    return atoms
