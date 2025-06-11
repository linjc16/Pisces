# fmt: off

# flake8: noqa
"""This module defines an ASE interface to deMon-nano.

Link to the open-source DFTB code deMon-nano:
http://demon-nano.ups-tlse.fr/

export ASE_DEMONNANO_COMMAND="/path/to/bin/deMon.username.x"
export DEMONNANO_BASIS_PATH="/path/to/basis/"

The file 'deMon.inp' contains the input geometry and parameters
The file 'deMon.out' contains the results

"""
import os
import os.path as op
# import subprocess
import pathlib as pl

import numpy as np

import ase.data
import ase.io
from ase.calculators.calculator import FileIOCalculator, Parameters, ReadError
from ase.units import Bohr, Hartree


class DemonNanoParameters(Parameters):
    """Parameters class for the calculator.

    The options here are the most important ones that the user needs to be
    aware of. Further options accepted by deMon can be set in the dictionary
    input_arguments.

    """

    def __init__(
            self,
            label='.',
            atoms=None,
            command=None,
            basis_path=None,
            restart_path='.',
            print_out='ASE',
            title='deMonNano input file',
            forces=False,
            input_arguments=None):
        kwargs = locals()
        kwargs.pop('self')
        Parameters.__init__(self, **kwargs)


class DemonNano(FileIOCalculator):
    """Calculator interface to the deMon-nano code. """

    implemented_properties = ['energy', 'forces']

    def __init__(self, **kwargs):
        """ASE interface to the deMon-nano code.

        The deMon-nano code can be obtained from http://demon-nano.ups-tlse.fr/

        The ASE_DEMONNANO_COMMAND environment variable must be set to run the executable, in bash it would be set along the lines of
        export ASE_DEMONNANO_COMMAND="pathway-to-deMon-binary/deMon.username.x"

        Parameters:

        label : str
            relative path to the run directory
        atoms  : Atoms object
            the atoms object
        command  : str
            Command to run deMon. If not present, the environment variable ASE_DEMONNANO_COMMAND is used
        basis_path  : str
            Relative path to the directory containing DFTB-SCC or DFTB-0 parameters
            If not present, the environment variable DEMONNANO_BASIS_PATH is used
        restart_path  : str
            Relative path to the deMon restart dir
        title : str
            Title in the deMon input file.
        forces : bool
            If True a force calculation is enforced
        print_out : str | list
            Options for the printing in deMon
        input_arguments : dict
            Explicitly given input arguments. The key is the input keyword
            and the value is either a str, a list of str (will be written on the same line as the keyword),
            or a list of lists of str (first list is written on the first line, the others on following lines.)
        """

        parameters = DemonNanoParameters(**kwargs)

        # basis path
        basis_path = parameters['basis_path']
        if basis_path is None:
            basis_path = self.cfg.get('DEMONNANO_BASIS_PATH')

        if basis_path is None:
            mess = 'The "DEMONNANO_BASIS_PATH" environment is not defined.'
            raise ValueError(mess)
        else:
            parameters['basis_path'] = basis_path

        # Call the base class.
        FileIOCalculator.__init__(
            self,
            **parameters)

    def __getitem__(self, key):
        """Convenience method to retrieve a parameter as
        calculator[key] rather than calculator.parameters[key]

            Parameters:
                key       : str, the name of the parameters to get.
        """
        return self.parameters[key]

    def write_input(self, atoms, properties=None, system_changes=None):
        """Write input (in)-file.
        See calculator.py for further details.

        Parameters:
             atoms        : The Atoms object to write.
             properties   : The properties which should be calculated.
             system_changes : List of properties changed since last run.

        """
        # Call base calculator.
        FileIOCalculator.write_input(
            self,
            atoms=atoms,
            properties=properties,
            system_changes=system_changes)

        if system_changes is None and properties is None:
            return

        filename = self.label + '/deMon.inp'

        # Start writing the file.
        with open(filename, 'w') as fd:
            # write keyword argument keywords
            value = self.parameters['title']
            self._write_argument('TITLE', value, fd)
            fd.write('\n')

            # obtain forces through a single BOMD step
            # only if forces is in properties, or if keyword forces is True
            value = self.parameters['forces']
            if 'forces' in properties or value:
                self._write_argument('MDYNAMICS', 'ZERO', fd)
                self._write_argument('MDSTEP', 'MAX=1', fd)
                # default timestep is 0.25 fs if not enough - uncomment the line below
                # self._write_argument('TIMESTEP', '0.1', fd)

            # print argument, here other options could change this
            value = self.parameters['print_out']
            assert (isinstance(value, str))

            if len(value) != 0:
                self._write_argument('PRINT', value, fd)
                fd.write('\n')

            # write general input arguments
            self._write_input_arguments(fd)

            if 'BASISPATH' not in self.parameters['input_arguments']:
                value = self.parameters['basis_path']
                fd.write(value)
                fd.write('\n')

            # write geometry
            self._write_atomic_coordinates(fd, atoms)

            # write xyz file for good measure.
            ase.io.write(self.label + '/deMon_atoms.xyz', self.atoms)

    def read(self, restart_path):
        """Read parameters from directory restart_path."""

        self.set_label(restart_path)
        rpath = pl.Path(restart_path)

        if not (rpath / 'deMon.inp').exists():
            raise ReadError('The restart_path file {} does not exist'
                            .format(rpath))

        self.atoms = self.deMon_inp_to_atoms(rpath / 'deMon.inp')

        self.read_results()

    def _write_input_arguments(self, fd):
        """Write directly given input-arguments."""
        input_arguments = self.parameters['input_arguments']

        # Early return
        if input_arguments is None:
            return

        for key, value in input_arguments.items():
            self._write_argument(key, value, fd)

    def _write_argument(self, key, value, fd):
        """Write an argument to file.
        key :  a string coresponding to the input keyword
        value : the arguments, can be a string, a number or a list
        fd  :  and open file
        """
        if key == 'BASISPATH':
            # Write a basis path to file.
            # Has to be in lowercase for deMon-nano to work
            line = value.lower()
            fd.write(line)
            fd.write('\n')
        elif not isinstance(value, (tuple, list)):
            # for only one argument, write on same line
            line = key.upper()
            line += ' ' + str(value).upper()
            fd.write(line)
            fd.write('\n')

        # for a list, write first argument on the first line,
        # then the rest on new lines
        else:
            line = key
            if not isinstance(value[0], (tuple, list)):
                for i in range(len(value)):
                    line += ' ' + str(value[i].upper())
                fd.write(line)
                fd.write('\n')
            else:
                for i in range(len(value)):
                    for j in range(len(value[i])):
                        line += ' ' + str(value[i][j]).upper()
                    fd.write(line)
                    fd.write('\n')
                    line = ''

    def _write_atomic_coordinates(self, fd, atoms):
        """Write atomic coordinates.
        Parameters:
        - fd:     An open file object.
        - atoms: An atoms object.
        """
        # fd.write('#\n')
        # fd.write('# Atomic coordinates\n')
        # fd.write('#\n')
        fd.write('GEOMETRY CARTESIAN ANGSTROM\n')

        for sym, pos in zip(atoms.symbols, atoms.positions):
            fd.write('{:9s} {:10.5f} {:10.5f} {:10.5f}\n'.format(sym, *pos))

        fd.write('\n')

# Analysis routines
    def read_results(self):
        """Read the results from output files."""
        self.read_energy()
        self.read_forces(self.atoms)
        # self.read_eigenvalues()

    def read_energy(self):
        """Read energy from deMon.ase output file."""

        epath = pl.Path(self.label)

        if not (epath / 'deMon.ase').exists():
            raise ReadError('The deMonNano output file for ASE {} does not exist'
                            .format(epath))

        filename = self.label + '/deMon.ase'

        if op.isfile(filename):
            with open(filename) as fd:
                lines = fd.readlines()

        for i in range(len(lines)):
            if lines[i].startswith(' DFTB total energy [Hartree]'):
                self.results['energy'] = float(lines[i + 1]) * Hartree
                break

    def read_forces(self, atoms):
        """Read forces from the deMon.ase file."""

        natoms = len(atoms)
        epath = pl.Path(self.label)

        if not (epath / 'deMon.ase').exists():
            raise ReadError('The deMonNano output file for ASE {} does not exist'
                            .format(epath))

        filename = self.label + '/deMon.ase'

        with open(filename) as fd:
            lines = fd.readlines()

            # find line where the forces start
            flag_found = False
            for i in range(len(lines)):
                if 'DFTB gradients at 0 time step in a.u.' in lines[i]:
                    start = i + 1
                    flag_found = True
                    break

            if flag_found:
                self.results['forces'] = np.zeros((natoms, 3), float)
                for i in range(natoms):
                    line = [s for s in lines[i + start].strip().split(' ')
                            if len(s) > 0]
                    f = -np.array([float(x) for x in line[1:4]])
                    # output forces in a.u.
                    # self.results['forces'][i, :] = f
                    # output forces with real dimension
                    self.results['forces'][i, :] = f * (Hartree / Bohr)

    def deMon_inp_to_atoms(self, filename):
        """Routine to read deMon.inp and convert it to an atoms object."""

        read_flag = False
        chem_symbols = []
        xyz = []

        with open(filename) as fd:
            for line in fd:
                if 'GEOMETRY' in line:
                    read_flag = True
                    if 'ANGSTROM' in line:
                        coord_units = 'Ang'
                    elif 'BOHR' in line:
                        coord_units = 'Bohr'

                if read_flag:
                    tokens = line.split()
                    symbol = tokens[0]
                    xyz_loc = np.array(tokens[1:4]).astype(float)
                if read_flag and tokens:
                    chem_symbols.append(symbol)
                    xyz.append(xyz_loc)

        if coord_units == 'Bohr':
            xyz *= Bohr

        # set atoms object
        atoms = ase.Atoms(symbols=chem_symbols, positions=xyz)

        return atoms
