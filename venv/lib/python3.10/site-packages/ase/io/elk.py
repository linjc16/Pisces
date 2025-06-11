# fmt: off

import collections
from pathlib import Path

import numpy as np

from ase import Atoms
from ase.units import Bohr, Hartree
from ase.utils import reader, writer

elk_parameters = {'swidth': Hartree}


@reader
def read_elk(fd):
    """Import ELK atoms definition.

    Reads unitcell, atom positions, magmoms from elk.in/GEOMETRY.OUT file.
    """

    lines = fd.readlines()

    scale = np.ones(4)  # unit cell scale
    positions = []
    cell = []
    symbols = []
    magmoms = []

    # find cell scale
    for n, line in enumerate(lines):
        if line.split() == []:
            continue
        if line.strip() == 'scale':
            scale[0] = float(lines[n + 1])
        elif line.startswith('scale'):
            scale[int(line.strip()[-1])] = float(lines[n + 1])
    for n, line in enumerate(lines):
        if line.split() == []:
            continue
        if line.startswith('avec'):
            cell = np.array(
                [[float(v) * scale[1] for v in lines[n + 1].split()],
                 [float(v) * scale[2] for v in lines[n + 2].split()],
                 [float(v) * scale[3] for v in lines[n + 3].split()]])
        if line.startswith('atoms'):
            lines1 = lines[n + 1:]  # start subsearch
            spfname = []
            natoms = []
            atpos = []
            bfcmt = []
            for n1, line1 in enumerate(lines1):
                if line1.split() == []:
                    continue
                if 'spfname' in line1:
                    spfnamenow = lines1[n1].split()[0]
                    spfname.append(spfnamenow)
                    natomsnow = int(lines1[n1 + 1].split()[0])
                    natoms.append(natomsnow)
                    atposnow = []
                    bfcmtnow = []
                    for line in lines1[n1 + 2:n1 + 2 + natomsnow]:
                        atposnow.append([float(v) for v in line.split()[0:3]])
                        if len(line.split()) == 6:  # bfcmt present
                            bfcmtnow.append(
                                [float(v) for v in line.split()[3:]])
                    atpos.append(atposnow)
                    bfcmt.append(bfcmtnow)
    # symbols, positions, magmoms based on ELK spfname, atpos, and bfcmt
    symbols = ''
    positions = []
    magmoms = []
    for n, s in enumerate(spfname):
        symbols += str(s[1:].split('.')[0]) * natoms[n]
        positions += atpos[n]  # assumes fractional coordinates
        if len(bfcmt[n]) > 0:
            # how to handle cases of magmoms being one or three dim array?
            magmoms += [m[-1] for m in bfcmt[n]]
    atoms = Atoms(symbols, scaled_positions=positions, cell=[1, 1, 1])
    if len(magmoms) > 0:
        atoms.set_initial_magnetic_moments(magmoms)
    # final cell scale
    cell = cell * scale[0] * Bohr

    atoms.set_cell(cell, scale_atoms=True)
    atoms.pbc = True
    return atoms


@writer
def write_elk_in(fd, atoms, parameters=None):
    if parameters is None:
        parameters = {}

    parameters = dict(parameters)
    species_path = parameters.pop('species_dir', None)

    if parameters.get('spinpol') is None:
        if atoms.get_initial_magnetic_moments().any():
            parameters['spinpol'] = True

    if 'xctype' in parameters:
        if 'xc' in parameters:
            raise RuntimeError("You can't use both 'xctype' and 'xc'!")

    if parameters.get('autokpt'):
        if 'kpts' in parameters:
            raise RuntimeError("You can't use both 'autokpt' and 'kpts'!")
        if 'ngridk' in parameters:
            raise RuntimeError(
                "You can't use both 'autokpt' and 'ngridk'!")
    if 'ngridk' in parameters:
        if 'kpts' in parameters:
            raise RuntimeError("You can't use both 'ngridk' and 'kpts'!")

    if parameters.get('autoswidth'):
        if 'smearing' in parameters:
            raise RuntimeError(
                "You can't use both 'autoswidth' and 'smearing'!")
        if 'swidth' in parameters:
            raise RuntimeError(
                "You can't use both 'autoswidth' and 'swidth'!")

    inp = {}
    inp.update(parameters)

    if 'xc' in parameters:
        xctype = {'LDA': 3,  # PW92
                  'PBE': 20,
                  'REVPBE': 21,
                  'PBESOL': 22,
                  'WC06': 26,
                  'AM05': 30,
                  'mBJLDA': (100, 208, 12)}[parameters['xc']]
        inp['xctype'] = xctype
        del inp['xc']

    if 'kpts' in parameters:
        # XXX should generalize kpts handling.
        from ase.calculators.calculator import kpts2mp
        mp = kpts2mp(atoms, parameters['kpts'])
        inp['ngridk'] = tuple(mp)
        vkloff = []  # is this below correct?
        for nk in mp:
            if nk % 2 == 0:  # shift kpoint away from gamma point
                vkloff.append(0.5)
            else:
                vkloff.append(0)
        inp['vkloff'] = vkloff
        del inp['kpts']

    if 'smearing' in parameters:
        name = parameters.smearing[0].lower()
        if name == 'methfessel-paxton':
            stype = parameters.smearing[2]
        else:
            stype = {'gaussian': 0,
                     'fermi-dirac': 3,
                     }[name]
        inp['stype'] = stype
        inp['swidth'] = parameters.smearing[1]
        del inp['smearing']

    # convert keys to ELK units
    for key, value in inp.items():
        if key in elk_parameters:
            inp[key] /= elk_parameters[key]

    # write all keys
    for key, value in inp.items():
        fd.write(f'{key}\n')
        if isinstance(value, bool):
            fd.write(f'.{("false", "true")[value]}.\n\n')
        elif isinstance(value, (int, float)):
            fd.write(f'{value}\n\n')
        else:
            fd.write('%s\n\n' % ' '.join([str(x) for x in value]))

    # cell
    fd.write('avec\n')
    for vec in atoms.cell:
        fd.write('%.14f %.14f %.14f\n' % tuple(vec / Bohr))
    fd.write('\n')

    # atoms
    species = {}
    symbols = []
    for a, (symbol, m) in enumerate(
        zip(atoms.get_chemical_symbols(),
            atoms.get_initial_magnetic_moments())):
        if symbol in species:
            species[symbol].append((a, m))
        else:
            species[symbol] = [(a, m)]
            symbols.append(symbol)
    fd.write('atoms\n%d\n' % len(species))
    # scaled = atoms.get_scaled_positions(wrap=False)
    scaled = np.linalg.solve(atoms.cell.T, atoms.positions.T).T
    for symbol in symbols:
        fd.write(f"'{symbol}.in' : spfname\n")
        fd.write('%d\n' % len(species[symbol]))
        for a, m in species[symbol]:
            fd.write('%.14f %.14f %.14f 0.0 0.0 %.14f\n' %
                     (tuple(scaled[a]) + (m,)))

    # if sppath is present in elk.in it overwrites species blocks!

    # Elk seems to concatenate path and filename in such a way
    # that we must put a / at the end:
    if species_path is not None:
        fd.write(f"sppath\n'{species_path}/'\n\n")


class ElkReader:
    def __init__(self, path):
        self.path = Path(path)

    def _read_everything(self):
        yield from self._read_energy()

        with (self.path / 'INFO.OUT').open() as fd:
            yield from parse_elk_info(fd)

        with (self.path / 'EIGVAL.OUT').open() as fd:
            yield from parse_elk_eigval(fd)

        with (self.path / 'KPOINTS.OUT').open() as fd:
            yield from parse_elk_kpoints(fd)

    def read_everything(self):
        dct = dict(self._read_everything())

        # The eigenvalue/occupation tables do not say whether there are
        # two spins, so we have to reshape them from 1 x K x SB to S x K x B:
        spinpol = dct.pop('spinpol')
        if spinpol:
            for name in 'eigenvalues', 'occupations':
                array = dct[name]
                _, nkpts, nbands_double = array.shape
                assert _ == 1
                assert nbands_double % 2 == 0
                nbands = nbands_double // 2
                newarray = np.empty((2, nkpts, nbands))
                newarray[0, :, :] = array[0, :, :nbands]
                newarray[1, :, :] = array[0, :, nbands:]
                if name == 'eigenvalues':
                    # Verify that eigenvalues are still sorted:
                    diffs = np.diff(newarray, axis=2)
                    assert all(diffs.flat[:] > 0)
                dct[name] = newarray
        return dct

    def _read_energy(self):
        txt = (self.path / 'TOTENERGY.OUT').read_text()
        tokens = txt.split()
        energy = float(tokens[-1]) * Hartree
        yield 'free_energy', energy
        yield 'energy', energy


def parse_elk_kpoints(fd):
    header = next(fd)
    lhs, rhs = header.split(':', 1)
    assert 'k-point, vkl, wkpt' in rhs, header
    nkpts = int(lhs.strip())

    kpts = np.empty((nkpts, 3))
    weights = np.empty(nkpts)

    for ikpt in range(nkpts):
        line = next(fd)
        tokens = line.split()
        kpts[ikpt] = np.array(tokens[1:4]).astype(float)
        weights[ikpt] = float(tokens[4])
    yield 'ibz_kpoints', kpts
    yield 'kpoint_weights', weights


def parse_elk_info(fd):
    dct = collections.defaultdict(list)
    fd = iter(fd)

    spinpol = None
    converged = False
    actually_did_not_converge = False
    # Legacy code kept track of both these things, which is strange.
    # Why could a file both claim to converge *and* not converge?

    # We loop over all lines and extract also data that occurs
    # multiple times (e.g. in multiple SCF steps)
    for line in fd:
        # "name of quantity  :   1 2 3"
        tokens = line.split(':', 1)
        if len(tokens) == 2:
            lhs, rhs = tokens
            dct[lhs.strip()].append(rhs.strip())

        elif line.startswith('Convergence targets achieved'):
            converged = True
        elif 'reached self-consistent loops maximum' in line.lower():
            actually_did_not_converge = True

        if 'Spin treatment' in line:
            # (Somewhat brittle doing multi-line stuff here.)
            line = next(fd)
            spinpol = line.strip() == 'spin-polarised'

    yield 'converged', converged and not actually_did_not_converge
    if spinpol is None:
        raise RuntimeError('Could not determine spin treatment')
    yield 'spinpol', spinpol

    if 'Fermi' in dct:
        yield 'fermi_level', float(dct['Fermi'][-1]) * Hartree

    if 'total force' in dct:
        forces = []
        for line in dct['total force']:
            forces.append(line.split())
        yield 'forces', np.array(forces, float) * (Hartree / Bohr)


def parse_elk_eigval(fd):

    def match_int(line, word):
        number, colon, word1 = line.split()
        assert word1 == word
        assert colon == ':'
        return int(number)

    def skip_spaces(line=''):
        while not line.strip():
            line = next(fd)
        return line

    line = skip_spaces()
    nkpts = match_int(line, 'nkpt')  # 10 : nkpts
    line = next(fd)
    nbands = match_int(line, 'nstsv')  # 15 : nstsv

    eigenvalues = np.empty((nkpts, nbands))
    occupations = np.empty((nkpts, nbands))
    kpts = np.empty((nkpts, 3))

    for ikpt in range(nkpts):
        line = skip_spaces()
        tokens = line.split()
        assert tokens[-1] == 'vkl', tokens
        assert ikpt + 1 == int(tokens[0])
        kpts[ikpt] = np.array(tokens[1:4]).astype(float)

        line = next(fd)  # "(state, eigenvalue and occupancy below)"
        assert line.strip().startswith('(state,'), line
        for iband in range(nbands):
            line = next(fd)
            tokens = line.split()  # (band number, eigenval, occ)
            assert iband + 1 == int(tokens[0])
            eigenvalues[ikpt, iband] = float(tokens[1])
            occupations[ikpt, iband] = float(tokens[2])

    yield 'ibz_kpoints', kpts
    yield 'eigenvalues', eigenvalues[None] * Hartree
    yield 'occupations', occupations[None]
