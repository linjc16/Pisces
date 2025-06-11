# fmt: off

import os
import re
from copy import deepcopy
from subprocess import TimeoutExpired, call

import numpy as np

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.units import Bohr, Debye, Hartree
from ase.utils import workdir


def _format_value(val):
    if isinstance(val, bool):
        return '.t.' if val else '.f.'
    return str(val).upper()


def _write_block(name, args):
    out = [f' ${name.upper()}']
    for key, val in args.items():
        out.append(f'  {key.upper()}={_format_value(val)}')
    out.append(' $END')
    return '\n'.join(out)


def _write_geom(atoms, basis_spec):
    out = [' $DATA', atoms.get_chemical_formula(), 'C1']
    for i, atom in enumerate(atoms):
        out.append('{:<3} {:>3} {:20.13e} {:20.13e} {:20.13e}'
                   .format(atom.symbol, atom.number, *atom.position))
        if basis_spec is not None:
            basis = basis_spec.get(i)
            if basis is None:
                basis = basis_spec.get(atom.symbol)
            if basis is None:
                raise ValueError('Could not find an appropriate basis set '
                                 'for atom number {}!'.format(i))
            out += [basis, '']
    out.append(' $END')
    return '\n'.join(out)


def _write_ecp(atoms, ecp):
    out = [' $ECP']
    for i, symbol in enumerate(atoms.symbols):
        if i in ecp:
            out.append(ecp[i])
        elif symbol in ecp:
            out.append(ecp[symbol])
        else:
            raise ValueError('Could not find an appropriate ECP for '
                             'atom number {}!'.format(i))
    out.append(' $END')
    return '\n'.join(out)


_xc = dict(LDA='SVWN')


def write_gamess_us_in(fd, atoms, properties=None, **params):
    params = deepcopy(params)

    if properties is None:
        properties = ['energy']

    # set RUNTYP from properties iff value not provided by the user
    contrl = params.pop('contrl', {})
    if 'runtyp' not in contrl:
        if 'forces' in properties:
            contrl['runtyp'] = 'gradient'
        else:
            contrl['runtyp'] = 'energy'

    # Support xc keyword for functional specification
    xc = params.pop('xc', None)
    if xc is not None and 'dfttyp' not in contrl:
        contrl['dfttyp'] = _xc.get(xc.upper(), xc.upper())

    # Automatically determine multiplicity from magnetic moment
    magmom_tot = int(round(atoms.get_initial_magnetic_moments().sum()))
    if 'mult' not in contrl:
        contrl['mult'] = abs(magmom_tot) + 1

    # Since we're automatically determining multiplicity, we also
    # need to automatically switch to UHF when the multiplicity
    # is not 1
    if 'scftyp' not in contrl:
        contrl['scftyp'] = 'rhf' if contrl['mult'] == 1 else 'uhf'

    # effective core potentials
    ecp = params.pop('ecp', None)
    if ecp is not None and 'pp' not in contrl:
        contrl['pp'] = 'READ'

    # If no basis set is provided, use 3-21G by default.
    basis_spec = None
    if 'basis' not in params:
        params['basis'] = dict(gbasis='N21', ngauss=3)
    else:
        keys = set(params['basis'])
        # Check if the user is specifying a literal per-atom basis.
        # We assume they are passing a per-atom basis if the keys of the
        # basis dict are atom symbols, or if they are atom indices, or
        # a mixture of both.
        if (keys.intersection(set(atoms.symbols))
                or any(map(lambda x: isinstance(x, int), keys))):
            basis_spec = params.pop('basis')

    out = [_write_block('contrl', contrl)]
    out += [_write_block(*item) for item in params.items()]
    out.append(_write_geom(atoms, basis_spec))
    if ecp is not None:
        out.append(_write_ecp(atoms, ecp))
    fd.write('\n\n'.join(out))


_geom_re = re.compile(r'^\s*ATOM\s+ATOMIC\s+COORDINATES')
_atom_re = re.compile(r'^\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*\n')
_energy_re = re.compile(r'^\s*FINAL [\S\s]+ ENERGY IS\s+(\S+) AFTER')
_grad_re = re.compile(r'^\s*GRADIENT OF THE ENERGY\s*')
_dipole_re = re.compile(r'^\s+DX\s+DY\s+DZ\s+\/D\/\s+\(DEBYE\)')


def read_gamess_us_out(fd):
    atoms = None
    energy = None
    forces = None
    dipole = None
    for line in fd:
        # Geometry
        if _geom_re.match(line):
            fd.readline()
            symbols = []
            pos = []
            while True:
                atom = _atom_re.match(fd.readline())
                if atom is None:
                    break
                symbol, _, x, y, z = atom.groups()
                symbols.append(symbol.capitalize())
                pos.append(list(map(float, [x, y, z])))
            atoms = Atoms(symbols, np.array(pos) * Bohr)
            continue

        # Energy
        ematch = _energy_re.match(line)
        if ematch is not None:
            energy = float(ematch.group(1)) * Hartree

        # MPn energy. Supplants energy parsed above.
        elif line.strip().startswith('TOTAL ENERGY'):
            energy = float(line.strip().split()[-1]) * Hartree

        # Higher-level energy (e.g. coupled cluster)
        # Supplants energies parsed above.
        elif line.strip().startswith('THE FOLLOWING METHOD AND ENERGY'):
            energy = float(fd.readline().strip().split()[-1]) * Hartree

        # Gradients
        elif _grad_re.match(line):
            for _ in range(3):
                fd.readline()
            grad = []
            while True:
                atom = _atom_re.match(fd.readline())
                if atom is None:
                    break
                grad.append(list(map(float, atom.groups()[2:])))
            forces = -np.array(grad) * Hartree / Bohr
        elif _dipole_re.match(line):
            dipole = np.array(list(map(float, fd.readline().split()[:3])))
            dipole *= Debye

    atoms.calc = SinglePointCalculator(atoms, energy=energy,
                                       forces=forces, dipole=dipole)
    return atoms


def read_gamess_us_punch(fd):
    atoms = None
    energy = None
    forces = None
    dipole = None
    for line in fd:
        if line.strip() == '$DATA':
            symbols = []
            pos = []
            while line.strip() != '$END':
                line = fd.readline()
                atom = _atom_re.match(line)
                if atom is None:
                    # The basis set specification is interlaced with the
                    # molecular geometry. We don't care about the basis
                    # set, so ignore lines that don't match the pattern.
                    continue
                symbols.append(atom.group(1).capitalize())
                pos.append(list(map(float, atom.group(3, 4, 5))))
            atoms = Atoms(symbols, np.array(pos))
        elif line.startswith('E('):
            energy = float(line.split()[1][:-1]) * Hartree
        elif line.strip().startswith('DIPOLE'):
            dipole = np.array(list(map(float, line.split()[1:]))) * Debye
        elif line.strip() == '$GRAD':
            # The gradient block also contains the energy, which we prefer
            # over the energy obtained above because it is more likely to
            # be consistent with the gradients. It probably doesn't actually
            # make a difference though.
            energy = float(fd.readline().split()[1]) * Hartree
            grad = []
            while line.strip() != '$END':
                line = fd.readline()
                atom = _atom_re.match(line)
                if atom is None:
                    continue
                grad.append(list(map(float, atom.group(3, 4, 5))))
            forces = -np.array(grad) * Hartree / Bohr

    atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces,
                                       dipole=dipole)

    return atoms


def clean_userscr(userscr, prefix):
    for fname in os.listdir(userscr):
        tokens = fname.split('.')
        if tokens[0] == prefix and tokens[-1] != 'bak':
            fold = os.path.join(userscr, fname)
            os.rename(fold, fold + '.bak')


def get_userscr(prefix, command):
    prefix_test = prefix + '_test'
    command = command.replace('PREFIX', prefix_test)
    with workdir(prefix_test, mkdir=True):
        try:
            call(command, shell=True, timeout=2)
        except TimeoutExpired:
            pass

        try:
            with open(prefix_test + '.log') as fd:
                for line in fd:
                    if line.startswith('GAMESS supplementary output files'):
                        return ' '.join(line.split(' ')[8:]).strip()
        except FileNotFoundError:
            return None

    return None
