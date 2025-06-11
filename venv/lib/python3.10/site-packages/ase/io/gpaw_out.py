# fmt: off

import re
from typing import List, Tuple, Union

import numpy as np

from ase.atoms import Atoms
from ase.calculators.singlepoint import (
    SinglePointDFTCalculator,
    SinglePointKPoint,
)


def index_startswith(lines: List[str], string: str) -> int:
    for i, line in enumerate(lines):
        if line.strip().startswith(string):
            return i
    raise ValueError


def index_pattern(lines: List[str], pattern: str) -> int:
    repat = re.compile(pattern)
    for i, line in enumerate(lines):
        if repat.match(line):
            return i
    raise ValueError


def read_forces(lines: List[str],
                ii: int,
                atoms: Atoms) -> Tuple[List[Tuple[float, float, float]], int]:
    f = []
    for i in range(ii + 1, ii + 1 + len(atoms)):
        try:
            x, y, z = lines[i].split()[-3:]
            f.append((float(x), float(y), float(z)))
        except (ValueError, IndexError) as m:
            raise OSError(f'Malformed GPAW log file: {m}')
    return f, i


def read_stresses(lines: List[str],
                  ii: int,) -> Tuple[List[Tuple[float, float, float]], int]:
    s = []
    for i in range(ii + 1, ii + 4):
        try:
            x, y, z = lines[i].split()[-3:]
            s.append((float(x), float(y), float(z)))
        except (ValueError, IndexError) as m:
            raise OSError(f'Malformed GPAW log file: {m}')
    return s, i


def read_gpaw_out(fileobj, index):  # -> Union[Atoms, List[Atoms]]:
    """Read text output from GPAW calculation."""
    lines = [line.lower() for line in fileobj.readlines()]

    # read charge
    try:
        ii = index_startswith(lines, 'total charge:')
    except ValueError:
        q = None
    else:
        q = float(lines[ii].split()[2])

    blocks = []
    i1 = 0
    for i2, line in enumerate(lines):
        if line == 'positions:\n':
            if i1 > 0:
                blocks.append(lines[i1:i2])
            i1 = i2
    blocks.append(lines[i1:])

    images: List[Atoms] = []
    for lines in blocks:
        try:
            i = lines.index('unit cell:\n')
        except ValueError:
            pass
        else:
            if lines[i + 2].startswith('  -'):
                del lines[i + 2]  # old format
            cell: List[Union[float, List[float]]] = []
            pbc = []
            for line in lines[i + 2:i + 5]:
                words = line.split()
                if len(words) == 5:  # old format
                    cell.append(float(words[2]))
                    pbc.append(words[1] == 'yes')
                else:  # new format with GUC
                    cell.append([float(word) for word in words[3:6]])
                    pbc.append(words[2] == 'yes')

        symbols = []
        positions = []
        magmoms = []
        for line in lines[1:]:
            words = line.split()
            if len(words) < 5:
                break
            n, symbol, x, y, z = words[:5]
            symbols.append(symbol.split('.')[0].title())
            positions.append([float(x), float(y), float(z)])
            if len(words) > 5:
                mom = float(words[-1].rstrip(')'))
                magmoms.append(mom)
        if len(symbols):
            atoms = Atoms(symbols=symbols, positions=positions,
                          cell=cell, pbc=pbc)
        else:
            atoms = Atoms(cell=cell, pbc=pbc)

        try:
            ii = index_pattern(lines, '\\d+ k-point')
            word = lines[ii].split()
            kx = int(word[2])
            ky = int(word[4])
            kz = int(word[6])
            bz_kpts = (kx, ky, kz)
            ibz_kpts = int(lines[ii + 1].split()[0])
        except (ValueError, TypeError, IndexError):
            bz_kpts = None
            ibz_kpts = None

        try:
            i = index_startswith(lines, 'energy contributions relative to')
        except ValueError:
            e = energy_contributions = None
        else:
            energy_contributions = {}
            for line in lines[i + 2:i + 13]:
                fields = line.split(':')
                if len(fields) == 2:
                    name = fields[0]
                    energy = float(fields[1])
                    energy_contributions[name] = energy
                    if name in ['zero kelvin', 'extrapolated']:
                        e = energy
                        break
            else:  # no break
                raise ValueError

        try:
            ii = index_pattern(lines, '(fixed )?fermi level(s)?:')
        except ValueError:
            eFermi = None
        else:
            fields = lines[ii].split()
            try:
                def strip(string):
                    for rubbish in '[],':
                        string = string.replace(rubbish, '')
                    return string
                eFermi = [float(strip(fields[-2])),
                          float(strip(fields[-1]))]
            except ValueError:
                eFermi = float(fields[-1])

        # read Eigenvalues and occupations
        ii1 = ii2 = 1e32
        try:
            ii1 = index_startswith(lines, 'band   eigenvalues  occupancy')
        except ValueError:
            pass
        try:
            ii2 = index_startswith(lines, 'band  eigenvalues  occupancy')
        except ValueError:
            pass
        ii = min(ii1, ii2)
        if ii == 1e32:
            kpts = None
        else:
            ii += 1
            words = lines[ii].split()
            vals = []
            while len(words) > 2:
                vals.append([float(w) for w in words])
                ii += 1
                words = lines[ii].split()
            vals = np.array(vals).transpose()
            kpts = [SinglePointKPoint(1, 0, 0)]
            kpts[0].eps_n = vals[1]
            kpts[0].f_n = vals[2]
            if vals.shape[0] > 3:
                kpts.append(SinglePointKPoint(1, 1, 0))
                kpts[1].eps_n = vals[3]
                kpts[1].f_n = vals[4]
        # read dipole moment
        try:
            ii = index_startswith(lines, 'dipole moment:')
        except ValueError:
            dipole = None
        else:
            line = lines[ii]
            for x in '()[],':
                line = line.replace(x, '')
            dipole = np.array([float(c) for c in line.split()[2:5]])

        try:
            ii = index_startswith(lines, 'local magnetic moments')
        except ValueError:
            pass
        else:
            magmoms = []
            for j in range(ii + 1, ii + 1 + len(atoms)):
                line = lines[j]
                if '#' in line:  # new GPAW format
                    magmom = line.split()[-4].split(']')[0]
                else:
                    magmom = line.split()[-1].rstrip(')')
                magmoms.append(float(magmom))

        try:
            ii = lines.index('forces in ev/ang:\n')
        except ValueError:
            f = None
        else:
            f, i = read_forces(lines, ii, atoms)

        try:
            ii = lines.index('stress tensor:\n')
        except ValueError:
            stress_tensor = None
        else:
            stress_tensor, i = read_stresses(lines, ii)

        try:
            parameters = {}
            ii = index_startswith(lines, 'vdw correction:')
        except ValueError:
            name = 'gpaw'
        else:
            name = lines[ii - 1].strip()
            # save uncorrected values
            parameters.update({
                'calculator': 'gpaw',
                'uncorrected_energy': e,
            })
            line = lines[ii + 1]
            assert line.startswith('energy:')
            e = float(line.split()[-1])
            f, i = read_forces(lines, ii + 3, atoms)

        if len(images) > 0 and e is None:
            break

        if q is not None and len(atoms) > 0:
            n = len(atoms)
            atoms.set_initial_charges([q / n] * n)

        if magmoms:
            atoms.set_initial_magnetic_moments(magmoms)
        else:
            magmoms = None
        if e is not None or f is not None:
            calc = SinglePointDFTCalculator(atoms, energy=e, forces=f,
                                            dipole=dipole, magmoms=magmoms,
                                            efermi=eFermi,
                                            bzkpts=bz_kpts, ibzkpts=ibz_kpts,
                                            stress=stress_tensor)
            calc.name = name
            calc.parameters = parameters
            if energy_contributions is not None:
                calc.energy_contributions = energy_contributions
            if kpts is not None:
                calc.kpts = kpts
            atoms.calc = calc

        images.append(atoms)

    if len(images) == 0:
        raise OSError('Corrupted GPAW-text file!')

    return images[index]
