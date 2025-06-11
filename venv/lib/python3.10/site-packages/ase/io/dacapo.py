# fmt: off

import numpy as np

from ase.atom import Atom
from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.utils import reader


@reader
def read_dacapo_text(fd):
    lines = fd.readlines()
    i = lines.index(' Structure:             A1           A2            A3\n')
    cell = np.array([[float(w) for w in line.split()[2:5]]
                     for line in lines[i + 1:i + 4]]).transpose()
    i = lines.index(' Structure:  >>         Ionic positions/velocities ' +
                    'in cartesian coordinates       <<\n')
    atoms = []
    for line in lines[i + 4:]:
        words = line.split()
        if len(words) != 9:
            break
        Z, x, y, z = words[2:6]
        atoms.append(Atom(int(Z), [float(x), float(y), float(z)]))

    atoms = Atoms(atoms, cell=cell.tolist())

    try:
        i = lines.index(
            ' DFT:  CPU time                           Total energy\n')
    except ValueError:
        pass
    else:
        column = lines[i + 3].split().index('selfcons') - 1
        try:
            i2 = lines.index(' ANALYSIS PART OF CODE\n', i)
        except ValueError:
            pass
        else:
            while i2 > i:
                if lines[i2].startswith(' DFT:'):
                    break
                i2 -= 1
            energy = float(lines[i2].split()[column])
            atoms.calc = SinglePointCalculator(atoms, energy=energy)

    return atoms
