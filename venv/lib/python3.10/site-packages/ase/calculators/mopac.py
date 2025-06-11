# fmt: off

"""This module defines an ASE interface to MOPAC.

Set $ASE_MOPAC_COMMAND to something like::

    LD_LIBRARY_PATH=/path/to/lib/ \
    MOPAC_LICENSE=/path/to/license \
    /path/to/MOPAC2012.exe PREFIX.mop 2> /dev/null

"""
import os
import re
from typing import Sequence
from warnings import warn

import numpy as np
from packaging import version

from ase import Atoms
from ase.calculators.calculator import FileIOCalculator, Parameters, ReadError
from ase.units import Debye, kcal, mol


def get_version_number(lines: Sequence[str]):
    pattern1 = r'MOPAC\s+v(\S+)'
    pattern2 = r'MOPAC2016, Version:\s+([^,]+)'

    for line in lines[:10]:
        match = re.search(pattern1, line) or re.search(pattern2, line)
        if match:
            return match.group(1)
    raise ValueError('Version number was not found in MOPAC output')


class MOPAC(FileIOCalculator):
    implemented_properties = ['energy', 'forces', 'dipole',
                              'magmom', 'free_energy']
    _legacy_default_command = 'mopac PREFIX.mop 2> /dev/null'
    discard_results_on_any_change = True

    default_parameters = dict(
        method='PM7',
        task='1SCF GRADIENTS',
        charge=None,
        relscf=0.0001)

    methods = ['AM1', 'MNDO', 'MNDOD', 'PM3', 'PM6', 'PM6-D3', 'PM6-DH+',
               'PM6-DH2', 'PM6-DH2X', 'PM6-D3H4', 'PM6-D3H4X', 'PMEP', 'PM7',
               'PM7-TS', 'RM1']

    fileio_rules = FileIOCalculator.ruleset(
        extend_argv=['{prefix}.mop'],
        stdout_name='{prefix}.out')

    def __init__(self, restart=None,
                 ignore_bad_restart_file=FileIOCalculator._deprecated,
                 label='mopac', atoms=None, **kwargs):
        """Construct MOPAC-calculator object.

        Parameters:

        label: str
            Prefix for filenames (label.mop, label.out, ...)

        Examples:

        Use default values to do a single SCF calculation and print
        the forces (task='1SCF GRADIENTS'):

        >>> from ase.build import molecule
        >>> from ase.calculators.mopac import MOPAC
        >>>
        >>> atoms = molecule('O2')
        >>> atoms.calc = MOPAC(label='O2')
        >>> atoms.get_potential_energy()
        >>> eigs = atoms.calc.get_eigenvalues()
        >>> somos = atoms.calc.get_somo_levels()
        >>> homo, lumo = atoms.calc.get_homo_lumo_levels()

        Use the internal geometry optimization of Mopac:

        >>> atoms = molecule('H2')
        >>> atoms.calc = MOPAC(label='H2', task='GRADIENTS')
        >>> atoms.get_potential_energy()

        Read in and start from output file:

        >>> atoms = MOPAC.read_atoms('H2')
        >>> atoms.calc.get_homo_lumo_levels()

        """
        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                  label, atoms, **kwargs)

    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, atoms, properties, system_changes)
        p = Parameters(self.parameters.copy())

        # Ensure DISP so total energy is available
        if 'DISP' not in p.task.split():
            p.task = p.task + ' DISP'

        # Build string to hold .mop input file:
        s = f'{p.method} {p.task} '

        if p.relscf:
            s += f'RELSCF={p.relscf} '

        # Write charge:
        if p.charge is None:
            charge = atoms.get_initial_charges().sum()
        else:
            charge = p.charge

        if charge != 0:
            s += f'CHARGE={int(round(charge))} '

        magmom = int(round(abs(atoms.get_initial_magnetic_moments().sum())))
        if magmom:
            s += (['DOUBLET', 'TRIPLET', 'QUARTET', 'QUINTET'][magmom - 1] +
                  ' UHF ')

        s += '\nTitle: ASE calculation\n\n'

        # Write coordinates:
        for xyz, symbol in zip(atoms.positions, atoms.get_chemical_symbols()):
            s += ' {:2} {} 1 {} 1 {} 1\n'.format(symbol, *xyz)

        for v, pbc in zip(atoms.cell, atoms.pbc):
            if pbc:
                s += 'Tv {} {} {}\n'.format(*v)

        with open(self.label + '.mop', 'w') as fd:
            fd.write(s)

    def get_spin_polarized(self):
        return self.nspins == 2

    def get_index(self, lines, pattern):
        for i, line in enumerate(lines):
            if line.find(pattern) != -1:
                return i

    def read(self, label):
        FileIOCalculator.read(self, label)
        if not os.path.isfile(self.label + '.out'):
            raise ReadError

        with open(self.label + '.out') as fd:
            lines = fd.readlines()

        self.parameters = Parameters(task='', method='')
        p = self.parameters
        parm_line = self.read_parameters_from_file(lines)
        for keyword in parm_line.split():
            if 'RELSCF' in keyword:
                p.relscf = float(keyword.split('=')[-1])
            elif keyword in self.methods:
                p.method = keyword
            else:
                p.task += keyword + ' '

        p.task = p.task.rstrip()
        if 'charge' not in p:
            p.charge = None

        self.atoms = self.read_atoms_from_file(lines)
        self.read_results()

    def read_atoms_from_file(self, lines):
        """Read the Atoms from the output file stored as list of str in lines.
        Parameters:

            lines: list of str
        """
        # first try to read from final point (last image)
        i = self.get_index(lines, 'FINAL  POINT  AND  DERIVATIVES')
        if i is None:  # XXX should we read it from the input file?
            assert 0, 'Not implemented'

        lines1 = lines[i:]
        i = self.get_index(lines1, 'CARTESIAN COORDINATES')
        j = i + 2
        symbols = []
        positions = []
        while not lines1[j].isspace():  # continue until we hit a blank line
            l = lines1[j].split()
            symbols.append(l[1])
            positions.append([float(c) for c in l[2: 2 + 3]])
            j += 1

        return Atoms(symbols=symbols, positions=positions)

    def read_parameters_from_file(self, lines):
        """Find and return the line that defines a Mopac calculation

        Parameters:

            lines: list of str
        """
        for i, line in enumerate(lines):
            if line.find('CALCULATION DONE:') != -1:
                break

        lines1 = lines[i:]
        for i, line in enumerate(lines1):
            if line.find('****') != -1:
                return lines1[i + 1]

    def read_results(self):
        """Read the results, such as energy, forces, eigenvalues, etc.
        """
        FileIOCalculator.read(self, self.label)
        if not os.path.isfile(self.label + '.out'):
            raise ReadError

        with open(self.label + '.out') as fd:
            lines = fd.readlines()

        self.results['version'] = get_version_number(lines)

        total_energy_regex = re.compile(
            r'^\s+TOTAL ENERGY\s+\=\s+(-?\d+\.\d+) EV')
        final_heat_regex = re.compile(
            r'^\s+FINAL HEAT OF FORMATION\s+\=\s+(-?\d+\.\d+) KCAL/MOL')

        for i, line in enumerate(lines):
            if total_energy_regex.match(line):
                self.results['total_energy'] = float(
                    total_energy_regex.match(line).groups()[0])
            elif final_heat_regex.match(line):
                self.results['final_hof'] = float(final_heat_regex.match(line)
                                                  .groups()[0]) * kcal / mol
            elif line.find('NO. OF FILLED LEVELS') != -1:
                self.nspins = 1
                self.no_occ_levels = int(line.split()[-1])
            elif line.find('NO. OF ALPHA ELECTRON') != -1:
                self.nspins = 2
                self.no_alpha_electrons = int(line.split()[-1])
                self.no_beta_electrons = int(lines[i + 1].split()[-1])
                self.results['magmom'] = abs(self.no_alpha_electrons -
                                             self.no_beta_electrons)
            elif line.find('FINAL  POINT  AND  DERIVATIVES') != -1:
                forces = [-float(line.split()[6])
                          for line in lines[i + 3:i + 3 + 3 * len(self.atoms)]]
                self.results['forces'] = np.array(
                    forces).reshape((-1, 3)) * kcal / mol
            elif line.find('EIGENVALUES') != -1:
                if line.find('ALPHA') != -1:
                    j = i + 1
                    eigs_alpha = []
                    while not lines[j].isspace():
                        eigs_alpha += [float(eps) for eps in lines[j].split()]
                        j += 1
                elif line.find('BETA') != -1:
                    j = i + 1
                    eigs_beta = []
                    while not lines[j].isspace():
                        eigs_beta += [float(eps) for eps in lines[j].split()]
                        j += 1
                    eigs = np.array([eigs_alpha, eigs_beta]).reshape(2, 1, -1)
                    self.eigenvalues = eigs
                else:
                    eigs = []
                    j = i + 1
                    while not lines[j].isspace():
                        eigs += [float(e) for e in lines[j].split()]
                        j += 1
                    self.eigenvalues = np.array(eigs).reshape(1, 1, -1)
            elif line.find('DIPOLE   ') != -1:
                self.results['dipole'] = np.array(
                    lines[i + 3].split()[1:1 + 3], float) * Debye

        # Developers recommend using final HOF as it includes dispersion etc.
        # For backward compatibility we won't change the results of old MOPAC
        # calculations... yet
        if version.parse(self.results['version']) >= version.parse('22'):
            self.results['energy'] = self.results['final_hof']
        else:
            warn("Using a version of MOPAC lower than v22: ASE will use "
                 "TOTAL ENERGY as the potential energy. In future, "
                 "FINAL HEAT OF FORMATION will be preferred for all versions.")
            self.results['energy'] = self.results['total_energy']
        self.results['free_energy'] = self.results['energy']

    def get_eigenvalues(self, kpt=0, spin=0):
        return self.eigenvalues[spin, kpt]

    def get_homo_lumo_levels(self):
        eigs = self.eigenvalues
        if self.nspins == 1:
            nocc = self.no_occ_levels
            return np.array([eigs[0, 0, nocc - 1], eigs[0, 0, nocc]])
        else:
            na = self.no_alpha_electrons
            nb = self.no_beta_electrons
            if na == 0:
                return None, self.eigenvalues[1, 0, nb - 1]
            elif nb == 0:
                return self.eigenvalues[0, 0, na - 1], None
            else:
                eah, eal = eigs[0, 0, na - 1: na + 1]
                ebh, ebl = eigs[1, 0, nb - 1: nb + 1]
                return np.array([max(eah, ebh), min(eal, ebl)])

    def get_somo_levels(self):
        assert self.nspins == 2
        na, nb = self.no_alpha_electrons, self.no_beta_electrons
        if na == 0:
            return None, self.eigenvalues[1, 0, nb - 1]
        elif nb == 0:
            return self.eigenvalues[0, 0, na - 1], None
        else:
            return np.array([self.eigenvalues[0, 0, na - 1],
                             self.eigenvalues[1, 0, nb - 1]])

    def get_final_heat_of_formation(self):
        """Final heat of formation as reported in the Mopac output file"""
        warn("This method is deprecated, please use "
             "MOPAC.results['final_hof']", DeprecationWarning)
        return self.results['final_hof']

    @property
    def final_hof(self):
        warn("This property is deprecated, please use "
             "MOPAC.results['final_hof']", DeprecationWarning)
        return self.results['final_hof']

    @final_hof.setter
    def final_hof(self, new_hof):
        warn("This property is deprecated, please use "
             "MOPAC.results['final_hof']", DeprecationWarning)
        self.results['final_hof'] = new_hof
