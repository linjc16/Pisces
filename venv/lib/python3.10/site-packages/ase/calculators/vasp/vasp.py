# fmt: off

# Copyright (C) 2008 CSC - Scientific Computing Ltd.
"""This module defines an ASE interface to VASP.

Developed on the basis of modules by Jussi Enkovaara and John
Kitchin.  The path of the directory containing the pseudopotential
directories (potpaw,potpaw_GGA, potpaw_PBE, ...) should be set
by the environmental flag $VASP_PP_PATH.

The user should also set the environmental flag $VASP_SCRIPT pointing
to a python script looking something like::

   import os
   exitcode = os.system('vasp')

Alternatively, user can set the environmental flag $VASP_COMMAND pointing
to the command use the launch vasp e.g. 'vasp' or 'mpirun -n 16 vasp'

http://cms.mpi.univie.ac.at/vasp/
"""

import os
import re
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Tuple
from warnings import warn
from xml.etree import ElementTree

import numpy as np

import ase
from ase.calculators import calculator
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointDFTCalculator
from ase.calculators.vasp.create_input import GenerateVaspInput
from ase.config import cfg
from ase.io import jsonio, read
from ase.utils import PurePath, deprecated
from ase.vibrations.data import VibrationsData


def _prohibit_directory_in_label(args: List, kwargs: Dict[str, Any]) -> bool:
    if len(args) >= 5 and "/" in args[4]:
        return True
    return "/" in kwargs.get("label", "")


class Vasp(GenerateVaspInput, Calculator):  # type: ignore[misc]
    """ASE interface for the Vienna Ab initio Simulation Package (VASP),
    with the Calculator interface.

    Parameters:

    atoms:  object
        Attach an atoms object to the calculator.

    label: str
        Prefix for the output file, and sets the working directory.
        Default is 'vasp'.

    directory: str
        Set the working directory. Is prepended to ``label``.

    restart: str or bool
        Sets a label for the directory to load files from.
        if :code:`restart=True`, the working directory from
        ``directory`` is used.

    txt: bool, None, str or writable object
        - If txt is None, output stream will be supressed

        - If txt is '-' the output will be sent through stdout

        - If txt is a string a file will be opened,\
            and the output will be sent to that file.

        - Finally, txt can also be a an output stream,\
            which has a 'write' attribute.

        Default is 'vasp.out'

        - Examples:
            >>> from ase.calculators.vasp import Vasp
            >>> calc = Vasp(label='mylabel', txt='vasp.out') # Redirect stdout
            >>> calc = Vasp(txt='myfile.txt') # Redirect stdout
            >>> calc = Vasp(txt='-') # Print vasp output to stdout
            >>> calc = Vasp(txt=None)  # Suppress txt output

    command: str
        Custom instructions on how to execute VASP. Has priority over
        environment variables.
    """
    name = 'vasp'
    ase_objtype = 'vasp_calculator'  # For JSON storage

    # Environment commands
    env_commands = ('ASE_VASP_COMMAND', 'VASP_COMMAND', 'VASP_SCRIPT')

    implemented_properties = [
        'energy', 'free_energy', 'forces', 'dipole', 'fermi', 'stress',
        'magmom', 'magmoms'
    ]

    # Can be used later to set some ASE defaults
    default_parameters: Dict[str, Any] = {}

    @deprecated(
        'Specifying directory in "label" is deprecated, '
        'use "directory" instead.',
        category=FutureWarning,
        callback=_prohibit_directory_in_label,
    )
    def __init__(self,
                 atoms=None,
                 restart=None,
                 directory='.',
                 label='vasp',
                 ignore_bad_restart_file=Calculator._deprecated,
                 command=None,
                 txt='vasp.out',
                 **kwargs):
        """
        .. deprecated:: 3.19.2
            Specifying directory in ``label`` is deprecated,
            use ``directory`` instead.
        """

        self._atoms = None
        self.results = {}

        # Initialize parameter dictionaries
        GenerateVaspInput.__init__(self)
        self._store_param_state()  # Initialize an empty parameter state

        # Store calculator from vasprun.xml here - None => uninitialized
        self._xml_calc = None

        # Set directory and label
        self.directory = directory
        if "/" in label:
            if self.directory != ".":
                msg = (
                    'Directory redundantly specified though directory='
                    f'"{self.directory}" and label="{label}".  Please omit '
                    '"/" in label.'
                )
                raise ValueError(msg)
            self.label = label
        else:
            self.prefix = label  # The label should only contain the prefix

        if isinstance(restart, bool):
            restart = self.label if restart is True else None

        Calculator.__init__(
            self,
            restart=restart,
            ignore_bad_restart_file=ignore_bad_restart_file,
            # We already, manually, created the label
            label=self.label,
            atoms=atoms,
            **kwargs)

        self.command = command

        self._txt = None
        self.txt = txt  # Set the output txt stream
        self.version = None

        # XXX: This seems to break restarting, unless we return first.
        # Do we really still need to enfore this?

        #  # If no XC combination, GGA functional or POTCAR type is specified,
        #  # default to PW91. This is mostly chosen for backwards compatibility.
        # if kwargs.get('xc', None):
        #     pass
        # elif not (kwargs.get('gga', None) or kwargs.get('pp', None)):
        #     self.input_params.update({'xc': 'PW91'})
        # # A null value of xc is permitted; custom recipes can be
        # # used by explicitly setting the pseudopotential set and
        # # INCAR keys
        # else:
        #     self.input_params.update({'xc': None})

    def make_command(self, command=None):
        """Return command if one is passed, otherwise try to find
        ASE_VASP_COMMAND, VASP_COMMAND or VASP_SCRIPT.
        If none are set, a CalculatorSetupError is raised"""
        if command:
            cmd = command
        else:
            # Search for the environment commands
            for env in self.env_commands:
                if env in cfg:
                    cmd = cfg[env].replace('PREFIX', self.prefix)
                    if env == 'VASP_SCRIPT':
                        # Make the system python exe run $VASP_SCRIPT
                        exe = sys.executable
                        cmd = ' '.join([exe, cmd])
                    break
            else:
                msg = ('Please set either command in calculator'
                       ' or one of the following environment '
                       'variables (prioritized as follows): {}').format(
                           ', '.join(self.env_commands))
                raise calculator.CalculatorSetupError(msg)
        return cmd

    def set(self, **kwargs):
        """Override the set function, to test for changes in the
        Vasp Calculator, then call the create_input.set()
        on remaining inputs for VASP specific keys.

        Allows for setting ``label``, ``directory`` and ``txt``
        without resetting the results in the calculator.
        """
        changed_parameters = {}

        if 'label' in kwargs:
            self.label = kwargs.pop('label')

        if 'directory' in kwargs:
            # str() call to deal with pathlib objects
            self.directory = str(kwargs.pop('directory'))

        if 'txt' in kwargs:
            self.txt = kwargs.pop('txt')

        if 'atoms' in kwargs:
            atoms = kwargs.pop('atoms')
            self.atoms = atoms  # Resets results

        if 'command' in kwargs:
            self.command = kwargs.pop('command')

        changed_parameters.update(Calculator.set(self, **kwargs))

        # We might at some point add more to changed parameters, or use it
        if changed_parameters:
            self.clear_results()  # We don't want to clear atoms
        if kwargs:
            # If we make any changes to Vasp input, we always reset
            GenerateVaspInput.set(self, **kwargs)
            self.results.clear()

    def reset(self):
        self.atoms = None
        self.clear_results()

    def clear_results(self):
        self.results.clear()
        self._xml_calc = None

    @contextmanager
    def _txt_outstream(self):
        """Custom function for opening a text output stream. Uses self.txt
        to determine the output stream, and accepts a string or an open
        writable object.
        If a string is used, a new stream is opened, and automatically closes
        the new stream again when exiting.

        Examples:
        # Pass a string
        calc.txt = 'vasp.out'
        with calc.txt_outstream() as out:
            calc.run(out=out)   # Redirects the stdout to 'vasp.out'

        # Use an existing stream
        mystream = open('vasp.out', 'w')
        calc.txt = mystream
        with calc.txt_outstream() as out:
            calc.run(out=out)
        mystream.close()

        # Print to stdout
        calc.txt = '-'
        with calc.txt_outstream() as out:
            calc.run(out=out)   # output is written to stdout
        """

        txt = self.txt
        open_and_close = False  # Do we open the file?

        if txt is None:
            # Suppress stdout
            out = subprocess.DEVNULL
        else:
            if isinstance(txt, str):
                if txt == '-':
                    # subprocess.call redirects this to stdout
                    out = None
                else:
                    # Open the file in the work directory
                    txt = self._indir(txt)
                    # We wait with opening the file, until we are inside the
                    # try/finally
                    open_and_close = True
            elif hasattr(txt, 'write'):
                out = txt
            else:
                raise RuntimeError('txt should either be a string'
                                   'or an I/O stream, got {}'.format(txt))

        try:
            if open_and_close:
                out = open(txt, 'w')
            yield out
        finally:
            if open_and_close:
                out.close()

    def calculate(self,
                  atoms=None,
                  properties=('energy', ),
                  system_changes=tuple(calculator.all_changes)):
        """Do a VASP calculation in the specified directory.

        This will generate the necessary VASP input files, and then
        execute VASP. After execution, the energy, forces. etc. are read
        from the VASP output files.
        """
        Calculator.calculate(self, atoms, properties, system_changes)
        # Check for zero-length lattice vectors and PBC
        # and that we actually have an Atoms object.
        check_atoms(self.atoms)

        self.clear_results()

        command = self.make_command(self.command)
        self.write_input(self.atoms, properties, system_changes)

        with self._txt_outstream() as out:
            errorcode, stderr = self._run(command=command,
                                          out=out,
                                          directory=self.directory)

        if errorcode:
            raise calculator.CalculationFailed(
                '{} in {} returned an error: {:d} stderr {}'.format(
                    self.name, Path(self.directory).resolve(), errorcode,
                    stderr))

        # Read results from calculation
        self.update_atoms(atoms)
        self.read_results()

    def _run(self, command=None, out=None, directory=None):
        """Method to explicitly execute VASP"""
        if command is None:
            command = self.command
        if directory is None:
            directory = self.directory
        result = subprocess.run(command,
                                shell=True,
                                cwd=directory,
                                capture_output=True,
                                text=True)
        if out is not None:
            out.write(result.stdout)
        return result.returncode, result.stderr

    def check_state(self, atoms, tol=1e-15):
        """Check for system changes since last calculation."""
        def compare_dict(d1, d2):
            """Helper function to compare dictionaries"""
            # Use symmetric difference to find keys which aren't shared
            # for python 2.7 compatibility
            if set(d1.keys()) ^ set(d2.keys()):
                return False

            # Check for differences in values
            for key, value in d1.items():
                if np.any(value != d2[key]):
                    return False
            return True

        # First we check for default changes
        system_changes = Calculator.check_state(self, atoms, tol=tol)

        # We now check if we have made any changes to the input parameters
        # XXX: Should we add these parameters to all_changes?
        for param_string, old_dict in self.param_state.items():
            param_dict = getattr(self, param_string)  # Get current param dict
            if not compare_dict(param_dict, old_dict):
                system_changes.append(param_string)

        return system_changes

    def _store_param_state(self):
        """Store current parameter state"""
        self.param_state = dict(
            float_params=self.float_params.copy(),
            exp_params=self.exp_params.copy(),
            string_params=self.string_params.copy(),
            int_params=self.int_params.copy(),
            input_params=self.input_params.copy(),
            bool_params=self.bool_params.copy(),
            list_int_params=self.list_int_params.copy(),
            list_bool_params=self.list_bool_params.copy(),
            list_float_params=self.list_float_params.copy(),
            dict_params=self.dict_params.copy(),
            special_params=self.special_params.copy())

    def asdict(self):
        """Return a dictionary representation of the calculator state.
        Does NOT contain information on the ``command``, ``txt`` or
        ``directory`` keywords.
        Contains the following keys:

            - ``ase_version``
            - ``vasp_version``
            - ``inputs``
            - ``results``
            - ``atoms`` (Only if the calculator has an ``Atoms`` object)
        """
        # Get versions
        asevers = ase.__version__
        vaspvers = self.get_version()

        self._store_param_state()  # Update param state
        # Store input parameters which have been set
        inputs = {
            key: value
            for param_dct in self.param_state.values()
            for key, value in param_dct.items() if value is not None
        }

        dct = {
            'ase_version': asevers,
            'vasp_version': vaspvers,
            # '__ase_objtype__': self.ase_objtype,
            'inputs': inputs,
            'results': self.results.copy()
        }

        if self.atoms:
            # Encode atoms as dict
            from ase.db.row import atoms2dict
            dct['atoms'] = atoms2dict(self.atoms)

        return dct

    def fromdict(self, dct):
        """Restore calculator from a :func:`~ase.calculators.vasp.Vasp.asdict`
        dictionary.

        Parameters:

        dct: Dictionary
            The dictionary which is used to restore the calculator state.
        """
        if 'vasp_version' in dct:
            self.version = dct['vasp_version']
        if 'inputs' in dct:
            self.set(**dct['inputs'])
            self._store_param_state()
        if 'atoms' in dct:
            from ase.db.row import AtomsRow
            atoms = AtomsRow(dct['atoms']).toatoms()
            self.atoms = atoms
        if 'results' in dct:
            self.results.update(dct['results'])

    def write_json(self, filename):
        """Dump calculator state to JSON file.

        Parameters:

        filename: string
            The filename which the JSON file will be stored to.
            Prepends the ``directory`` path to the filename.
        """
        filename = self._indir(filename)
        dct = self.asdict()
        jsonio.write_json(filename, dct)

    def read_json(self, filename):
        """Load Calculator state from an exported JSON Vasp file."""
        dct = jsonio.read_json(filename)
        self.fromdict(dct)

    def write_input(self, atoms, properties=None, system_changes=None):
        """Write VASP inputfiles, INCAR, KPOINTS and POTCAR"""
        self.initialize(atoms)
        GenerateVaspInput.write_input(self, atoms, directory=self.directory)

    def read(self, label=None):
        """Read results from VASP output files.
        Files which are read: OUTCAR, CONTCAR and vasprun.xml
        Raises ReadError if they are not found"""
        if label is None:
            label = self.label
        Calculator.read(self, label)

        # If we restart, self.parameters isn't initialized
        if self.parameters is None:
            self.parameters = self.get_default_parameters()

        # Check for existence of the necessary output files
        for f in ['OUTCAR', 'CONTCAR', 'vasprun.xml']:
            file = self._indir(f)
            if not file.is_file():
                raise calculator.ReadError(
                    f'VASP outputfile {file} was not found')

        # Build sorting and resorting lists
        self.read_sort()

        # Read atoms
        self.atoms = self.read_atoms(filename=self._indir('CONTCAR'))

        # Read parameters
        self.read_incar(filename=self._indir('INCAR'))
        self.read_kpoints(filename=self._indir('KPOINTS'))
        self.read_potcar(filename=self._indir('POTCAR'))

        # Read the results from the calculation
        self.read_results()

    def _indir(self, filename):
        """Prepend current directory to filename"""
        return Path(self.directory) / filename

    def read_sort(self):
        """Create the sorting and resorting list from ase-sort.dat.
        If the ase-sort.dat file does not exist, the sorting is redone.
        """
        sortfile = self._indir('ase-sort.dat')
        if os.path.isfile(sortfile):
            self.sort = []
            self.resort = []
            with open(sortfile) as fd:
                for line in fd:
                    sort, resort = line.split()
                    self.sort.append(int(sort))
                    self.resort.append(int(resort))
        else:
            # Redo the sorting
            atoms = read(self._indir('CONTCAR'))
            self.initialize(atoms)

    def read_atoms(self, filename):
        """Read the atoms from file located in the VASP
        working directory. Normally called CONTCAR."""
        return read(filename)[self.resort]

    def update_atoms(self, atoms):
        """Update the atoms object with new positions and cell"""
        if (self.int_params['ibrion'] is not None
                and self.int_params['nsw'] is not None):
            if self.int_params['ibrion'] > -1 and self.int_params['nsw'] > 0:
                # Update atomic positions and unit cell with the ones read
                # from CONTCAR.
                atoms_sorted = read(self._indir('CONTCAR'))
                atoms.positions = atoms_sorted[self.resort].positions
                atoms.cell = atoms_sorted.cell

        self.atoms = atoms  # Creates a copy

    def read_results(self):
        """Read the results from VASP output files"""
        # Temporarily load OUTCAR into memory
        outcar = self.load_file('OUTCAR')

        # Read the data we can from vasprun.xml
        calc_xml = self._read_xml()
        xml_results = calc_xml.results

        # Fix sorting
        xml_results['forces'] = xml_results['forces'][self.resort]

        self.results.update(xml_results)

        # Parse the outcar, as some properties are not loaded in vasprun.xml
        # We want to limit this as much as possible, as reading large OUTCAR's
        # is relatively slow
        # Removed for now
        # self.read_outcar(lines=outcar)

        # Update results dict with results from OUTCAR
        # which aren't written to the atoms object we read from
        # the vasprun.xml file.

        self.converged = self.read_convergence(lines=outcar)
        self.version = self.read_version()
        magmom, magmoms = self.read_mag(lines=outcar)
        dipole = self.read_dipole(lines=outcar)
        nbands = self.read_nbands(lines=outcar)
        self.results.update(
            dict(magmom=magmom, magmoms=magmoms, dipole=dipole, nbands=nbands))

        # Stress is not always present.
        # Prevent calculation from going into a loop
        if 'stress' not in self.results:
            self.results.update(dict(stress=None))

        self._set_old_keywords()

        # Store the parameters used for this calculation
        self._store_param_state()

    def _set_old_keywords(self):
        """Store keywords for backwards compatibility wd VASP calculator"""
        self.spinpol = self.get_spin_polarized()
        self.energy_free = self.get_potential_energy(force_consistent=True)
        self.energy_zero = self.get_potential_energy(force_consistent=False)
        self.forces = self.get_forces()
        self.fermi = self.get_fermi_level()
        self.dipole = self.get_dipole_moment()
        # Prevent calculation from going into a loop
        self.stress = self.get_property('stress', allow_calculation=False)
        self.nbands = self.get_number_of_bands()

    # Below defines some functions for faster access to certain common keywords
    @property
    def kpts(self):
        """Access the kpts from input_params dict"""
        return self.input_params['kpts']

    @kpts.setter
    def kpts(self, kpts):
        """Set kpts in input_params dict"""
        self.input_params['kpts'] = kpts

    @property
    def encut(self):
        """Direct access to the encut parameter"""
        return self.float_params['encut']

    @encut.setter
    def encut(self, encut):
        """Direct access for setting the encut parameter"""
        self.set(encut=encut)

    @property
    def xc(self):
        """Direct access to the xc parameter"""
        return self.get_xc_functional()

    @xc.setter
    def xc(self, xc):
        """Direct access for setting the xc parameter"""
        self.set(xc=xc)

    @property
    def atoms(self):
        return self._atoms

    @atoms.setter
    def atoms(self, atoms):
        if atoms is None:
            self._atoms = None
            self.clear_results()
        else:
            if self.check_state(atoms):
                self.clear_results()
            self._atoms = atoms.copy()

    def load_file(self, filename):
        """Reads a file in the directory, and returns the lines

        Example:
        >>> from ase.calculators.vasp import Vasp
        >>> calc = Vasp()
        >>> outcar = calc.load_file('OUTCAR')  # doctest: +SKIP
        """
        filename = self._indir(filename)
        with open(filename) as fd:
            return fd.readlines()

    @contextmanager
    def load_file_iter(self, filename):
        """Return a file iterator"""

        filename = self._indir(filename)
        with open(filename) as fd:
            yield fd

    def read_outcar(self, lines=None):
        """Read results from the OUTCAR file.
        Deprecated, see read_results()"""
        if not lines:
            lines = self.load_file('OUTCAR')
        # Spin polarized calculation?
        self.spinpol = self.get_spin_polarized()

        self.version = self.get_version()

        # XXX: Do we want to read all of this again?
        self.energy_free, self.energy_zero = self.read_energy(lines=lines)
        self.forces = self.read_forces(lines=lines)
        self.fermi = self.read_fermi(lines=lines)

        self.dipole = self.read_dipole(lines=lines)

        self.stress = self.read_stress(lines=lines)
        self.nbands = self.read_nbands(lines=lines)

        self.read_ldau()
        self.magnetic_moment, self.magnetic_moments = self.read_mag(
            lines=lines)

    def _read_xml(self) -> SinglePointDFTCalculator:
        """Read vasprun.xml, and return the last calculator object.
        Returns calculator from the xml file.
        Raises a ReadError if the reader is not able to construct a calculator.
        """
        file = self._indir('vasprun.xml')
        incomplete_msg = (
            f'The file "{file}" is incomplete, and no DFT data was available. '
            'This is likely due to an incomplete calculation.')
        try:
            _xml_atoms = read(file, index=-1, format='vasp-xml')
            # Silence mypy, we should only ever get a single atoms object
            assert isinstance(_xml_atoms, ase.Atoms)
        except ElementTree.ParseError as exc:
            raise calculator.ReadError(incomplete_msg) from exc

        if _xml_atoms is None or _xml_atoms.calc is None:
            raise calculator.ReadError(incomplete_msg)

        self._xml_calc = _xml_atoms.calc
        return self._xml_calc

    @property
    def _xml_calc(self) -> SinglePointDFTCalculator:
        if self.__xml_calc is None:
            raise RuntimeError('vasprun.xml data has not yet been loaded. '
                               'Run read_results() first.')
        return self.__xml_calc

    @_xml_calc.setter
    def _xml_calc(self, value):
        self.__xml_calc = value

    def get_ibz_k_points(self):
        calc = self._xml_calc
        return calc.get_ibz_k_points()

    def get_kpt(self, kpt=0, spin=0):
        calc = self._xml_calc
        return calc.get_kpt(kpt=kpt, spin=spin)

    def get_eigenvalues(self, kpt=0, spin=0):
        calc = self._xml_calc
        return calc.get_eigenvalues(kpt=kpt, spin=spin)

    def get_fermi_level(self):
        calc = self._xml_calc
        return calc.get_fermi_level()

    def get_homo_lumo(self):
        calc = self._xml_calc
        return calc.get_homo_lumo()

    def get_homo_lumo_by_spin(self, spin=0):
        calc = self._xml_calc
        return calc.get_homo_lumo_by_spin(spin=spin)

    def get_occupation_numbers(self, kpt=0, spin=0):
        calc = self._xml_calc
        return calc.get_occupation_numbers(kpt, spin)

    def get_spin_polarized(self):
        calc = self._xml_calc
        return calc.get_spin_polarized()

    def get_number_of_spins(self):
        calc = self._xml_calc
        return calc.get_number_of_spins()

    def get_number_of_bands(self):
        return self.results.get('nbands', None)

    def get_number_of_electrons(self, lines=None):
        if not lines:
            lines = self.load_file('OUTCAR')

        nelect = None
        for line in lines:
            if 'total number of electrons' in line:
                nelect = float(line.split('=')[1].split()[0].strip())
                break
        return nelect

    def get_k_point_weights(self):
        filename = 'IBZKPT'
        return self.read_k_point_weights(filename)

    def get_dos(self, spin=None, **kwargs):
        """
        The total DOS.

        Uses the ASE DOS module, and returns a tuple with
        (energies, dos).
        """
        from ase.dft.dos import DOS
        dos = DOS(self, **kwargs)
        e = dos.get_energies()
        d = dos.get_dos(spin=spin)
        return e, d

    def get_version(self):
        if self.version is None:
            # Try if we can read the version number
            self.version = self.read_version()
        return self.version

    def read_version(self):
        """Get the VASP version number"""
        # The version number is the first occurrence, so we can just
        # load the OUTCAR, as we will return soon anyway
        if not os.path.isfile(self._indir('OUTCAR')):
            return None
        with self.load_file_iter('OUTCAR') as lines:
            for line in lines:
                if ' vasp.' in line:
                    return line[len(' vasp.'):].split()[0]
        # We didn't find the version in VASP
        return None

    def get_number_of_iterations(self):
        return self.read_number_of_iterations()

    def read_number_of_iterations(self):
        niter = None
        with self.load_file_iter('OUTCAR') as lines:
            for line in lines:
                # find the last iteration number
                if '- Iteration' in line:
                    niter = list(map(int, re.findall(r'\d+', line)))[1]
        return niter

    def read_number_of_ionic_steps(self):
        niter = None
        with self.load_file_iter('OUTCAR') as lines:
            for line in lines:
                if '- Iteration' in line:
                    niter = list(map(int, re.findall(r'\d+', line)))[0]
        return niter

    def read_stress(self, lines=None):
        """Read stress from OUTCAR.

        Depreciated: Use get_stress() instead.
        """
        # We don't really need this, as we read this from vasprun.xml
        # keeping it around "just in case" for now
        if not lines:
            lines = self.load_file('OUTCAR')

        stress = None
        for line in lines:
            if ' in kB  ' in line:
                stress = -np.array([float(a) for a in line.split()[2:]])
                stress = stress[[0, 1, 2, 4, 5, 3]] * 1e-1 * ase.units.GPa
        return stress

    def read_ldau(self, lines=None):
        """Read the LDA+U values from OUTCAR"""
        if not lines:
            lines = self.load_file('OUTCAR')

        ldau_luj = None
        ldauprint = None
        ldau = None
        ldautype = None
        atomtypes = []
        # read ldau parameters from outcar
        for line in lines:
            if line.find('TITEL') != -1:  # What atoms are present
                atomtypes.append(line.split()[3].split('_')[0].split('.')[0])
            if line.find('LDAUTYPE') != -1:  # Is this a DFT+U calculation
                ldautype = int(line.split('=')[-1])
                ldau = True
                ldau_luj = {}
            if line.find('LDAUL') != -1:
                L = line.split('=')[-1].split()
            if line.find('LDAUU') != -1:
                U = line.split('=')[-1].split()
            if line.find('LDAUJ') != -1:
                J = line.split('=')[-1].split()
        # create dictionary
        if ldau:
            for i, symbol in enumerate(atomtypes):
                ldau_luj[symbol] = {
                    'L': int(L[i]),
                    'U': float(U[i]),
                    'J': float(J[i])
                }
            self.dict_params['ldau_luj'] = ldau_luj

        self.ldau = ldau
        self.ldauprint = ldauprint
        self.ldautype = ldautype
        self.ldau_luj = ldau_luj
        return ldau, ldauprint, ldautype, ldau_luj

    def get_xc_functional(self):
        """Returns the XC functional or the pseudopotential type

        If a XC recipe is set explicitly with 'xc', this is returned.
        Otherwise, the XC functional associated with the
        pseudopotentials (LDA, PW91 or PBE) is returned.
        The string is always cast to uppercase for consistency
        in checks."""
        if self.input_params.get('xc', None):
            return self.input_params['xc'].upper()
        if self.input_params.get('pp', None):
            return self.input_params['pp'].upper()
        raise ValueError('No xc or pp found.')

    # Methods for reading information from OUTCAR files:
    def read_energy(self, all=None, lines=None):
        """Method to read energy from OUTCAR file.
        Depreciated: use get_potential_energy() instead"""
        if not lines:
            lines = self.load_file('OUTCAR')

        [energy_free, energy_zero] = [0, 0]
        if all:
            energy_free = []
            energy_zero = []
        for line in lines:
            # Free energy
            if line.lower().startswith('  free  energy   toten'):
                if all:
                    energy_free.append(float(line.split()[-2]))
                else:
                    energy_free = float(line.split()[-2])
            # Extrapolated zero point energy
            if line.startswith('  energy  without entropy'):
                if all:
                    energy_zero.append(float(line.split()[-1]))
                else:
                    energy_zero = float(line.split()[-1])
        return [energy_free, energy_zero]

    def read_forces(self, all=False, lines=None):
        """Method that reads forces from OUTCAR file.

        If 'all' is switched on, the forces for all ionic steps
        in the OUTCAR file be returned, in other case only the
        forces for the last ionic configuration is returned."""

        if not lines:
            lines = self.load_file('OUTCAR')

        if all:
            all_forces = []

        for n, line in enumerate(lines):
            if 'TOTAL-FORCE' in line:
                forces = []
                for i in range(len(self.atoms)):
                    forces.append(
                        np.array(
                            [float(f) for f in lines[n + 2 + i].split()[3:6]]))

                if all:
                    all_forces.append(np.array(forces)[self.resort])

        if all:
            return np.array(all_forces)
        return np.array(forces)[self.resort]

    def read_fermi(self, lines=None):
        """Method that reads Fermi energy from OUTCAR file"""
        if not lines:
            lines = self.load_file('OUTCAR')

        E_f = None
        for line in lines:
            if 'E-fermi' in line:
                E_f = float(line.split()[2])
        return E_f

    def read_dipole(self, lines=None):
        """Read dipole from OUTCAR"""
        if not lines:
            lines = self.load_file('OUTCAR')

        dipolemoment = np.zeros([1, 3])
        for line in lines:
            if 'dipolmoment' in line:
                dipolemoment = np.array([float(f) for f in line.split()[1:4]])
        return dipolemoment

    def read_mag(self, lines=None):
        if not lines:
            lines = self.load_file('OUTCAR')
        p = self.int_params
        q = self.list_float_params
        if self.spinpol:
            magnetic_moment = self._read_magnetic_moment(lines=lines)
            if ((p['lorbit'] is not None and p['lorbit'] >= 10)
                    or (p['lorbit'] is None and q['rwigs'])):
                magnetic_moments = self._read_magnetic_moments(lines=lines)
            else:
                warn('Magnetic moment data not written in OUTCAR (LORBIT<10),'
                     ' setting magnetic_moments to zero.\nSet LORBIT>=10'
                     ' to get information on magnetic moments')
                magnetic_moments = np.zeros(len(self.atoms))
        else:
            magnetic_moment = 0.0
            magnetic_moments = np.zeros(len(self.atoms))
        return magnetic_moment, magnetic_moments

    def _read_magnetic_moments(self, lines=None):
        """Read magnetic moments from OUTCAR.
        Only reads the last occurrence. """
        if not lines:
            lines = self.load_file('OUTCAR')

        magnetic_moments = np.zeros(len(self.atoms))
        magstr = 'magnetization (x)'

        # Search for the last occurrence
        nidx = -1
        for n, line in enumerate(lines):
            if magstr in line:
                nidx = n

        # Read that occurrence
        if nidx > -1:
            for m in range(len(self.atoms)):
                magnetic_moments[m] = float(lines[nidx + m + 4].split()[-1])
        return magnetic_moments[self.resort]

    def _read_magnetic_moment(self, lines=None):
        """Read magnetic moment from OUTCAR"""
        if not lines:
            lines = self.load_file('OUTCAR')

        for line in lines:
            if 'number of electron  ' in line:
                _ = line.split()[-1]
                magnetic_moment = 0.0 if _ == "magnetization" else float(_)
        return magnetic_moment

    def read_nbands(self, lines=None):
        """Read number of bands from OUTCAR"""
        if not lines:
            lines = self.load_file('OUTCAR')

        for line in lines:
            line = self.strip_warnings(line)
            if 'NBANDS' in line:
                return int(line.split()[-1])
        return None

    def read_convergence(self, lines=None):
        """Method that checks whether a calculation has converged."""
        if not lines:
            lines = self.load_file('OUTCAR')

        converged = None
        # First check electronic convergence
        for line in lines:
            # VASP 6 actually labels loop exit reason
            if 'aborting loop' in line:
                converged = 'because EDIFF is reached' in line
                # NOTE: 'EDIFF was not reached (unconverged)'
                #        and
                #       'because hard stop was set'
                # will result in unconverged
                break
            # determine convergence by attempting to reproduce VASP's
            # internal logic
            if 'EDIFF  ' in line:
                ediff = float(line.split()[2])
            if 'total energy-change' in line:
                # I saw this in an atomic oxygen calculation. it
                # breaks this code, so I am checking for it here.
                if 'MIXING' in line:
                    continue
                split = line.split(':')
                a = float(split[1].split('(')[0])
                b = split[1].split('(')[1][0:-2]
                # sometimes this line looks like (second number wrong format!):
                # energy-change (2. order) :-0.2141803E-08  ( 0.2737684-111)
                # we are checking still the first number so
                # let's "fix" the format for the second one
                if 'e' not in b.lower():
                    # replace last occurrence of - (assumed exponent) with -e
                    bsplit = b.split('-')
                    bsplit[-1] = 'e' + bsplit[-1]
                    b = '-'.join(bsplit).replace('-e', 'e-')
                b = float(b)
                if [abs(a), abs(b)] < [ediff, ediff]:
                    converged = True
                else:
                    converged = False
                    continue
        # Then if ibrion in [1,2,3] check whether ionic relaxation
        # condition been fulfilled
        if (self.int_params['ibrion'] in [1, 2, 3]
                and self.int_params['nsw'] not in [0]):
            if not self.read_relaxed():
                converged = False
            else:
                converged = True
        return converged

    def read_k_point_weights(self, filename):
        """Read k-point weighting. Normally named IBZKPT."""

        lines = self.load_file(filename)

        if 'Tetrahedra\n' in lines:
            N = lines.index('Tetrahedra\n')
        else:
            N = len(lines)
        kpt_weights = []
        for n in range(3, N):
            kpt_weights.append(float(lines[n].split()[3]))
        kpt_weights = np.array(kpt_weights)
        kpt_weights /= np.sum(kpt_weights)

        return kpt_weights

    def read_relaxed(self, lines=None):
        """Check if ionic relaxation completed"""
        if not lines:
            lines = self.load_file('OUTCAR')
        for line in lines:
            if 'reached required accuracy' in line:
                return True
        return False

    def read_spinpol(self, lines=None):
        """Method which reads if a calculation from spinpolarized using OUTCAR.

        Depreciated: Use get_spin_polarized() instead.
        """
        if not lines:
            lines = self.load_file('OUTCAR')

        for line in lines:
            if 'ISPIN' in line:
                if int(line.split()[2]) == 2:
                    self.spinpol = True
                else:
                    self.spinpol = False
        return self.spinpol

    def strip_warnings(self, line):
        """Returns empty string instead of line from warnings in OUTCAR."""
        if line[0] == "|":
            return ""
        return line

    @property
    def txt(self):
        return self._txt

    @txt.setter
    def txt(self, txt):
        if isinstance(txt, PurePath):
            txt = str(txt)
        self._txt = txt

    def get_number_of_grid_points(self):
        raise NotImplementedError

    def get_pseudo_density(self):
        raise NotImplementedError

    def get_pseudo_wavefunction(self, n=0, k=0, s=0, pad=True):
        raise NotImplementedError

    def get_bz_k_points(self):
        raise NotImplementedError

    def read_vib_freq(self, lines=None) -> Tuple[List[float], List[float]]:
        """Read vibrational frequencies.

        Returns:
            List of real and list of imaginary frequencies
            (imaginary number as real number).
        """
        freq = []
        i_freq = []

        if not lines:
            lines = self.load_file('OUTCAR')

        for line in lines:
            data = line.split()
            if 'THz' in data:
                if 'f/i=' not in data:
                    freq.append(float(data[-2]))
                else:
                    i_freq.append(float(data[-2]))
        return freq, i_freq

    def _read_massweighted_hessian_xml(self) -> np.ndarray:
        """Read the Mass Weighted Hessian from vasprun.xml.

        Returns:
            The Mass Weighted Hessian as np.ndarray from the xml file.

            Raises a ReadError if the reader is not able to read the Hessian.

            Converts to ASE units for VASP version 6.
        """

        file = self._indir('vasprun.xml')
        try:
            tree = ElementTree.iterparse(file)
            hessian = None
            for event, elem in tree:
                if elem.tag == 'dynmat':
                    for i, entry in enumerate(
                            elem.findall('varray[@name="hessian"]/v')):
                        text_split = entry.text.split()
                        if not text_split:
                            raise ElementTree.ParseError(
                                "Could not find varray hessian!")
                        if i == 0:
                            n_items = len(text_split)
                            hessian = np.zeros((n_items, n_items))
                        assert isinstance(hessian, np.ndarray)
                        hessian[i, :] = np.array(
                            [float(val) for val in text_split])
                    if i != n_items - 1:
                        raise ElementTree.ParseError(
                            "Hessian is not quadratic!")
                    # VASP6+ uses THz**2 as unit, not mEV**2 as before
                    for entry in elem.findall('i[@name="unit"]'):
                        if entry.text.strip() == 'THz^2':
                            conv = ase.units._amu / ase.units._e / \
                                1e-4 * (2 * np.pi)**2  # THz**2 to eV**2
                            # VASP6 uses factor 2pi
                            # 1e-4 = (angstrom to meter times Hz to THz) squared
                            # = (1e10 times 1e-12)**2
                            break
                        else:  # Catch changes in VASP
                            vasp_version_error_msg = (
                                f'The file "{file}" is from a '
                                'non-supported VASP version. '
                                'Not sure what unit the Hessian '
                                'is in, aborting.')
                            raise calculator.ReadError(vasp_version_error_msg)

                    else:
                        conv = 1.0  # VASP version <6 unit is meV**2
                    assert isinstance(hessian, np.ndarray)
                    hessian *= conv
            if hessian is None:
                raise ElementTree.ParseError("Hessian is None!")

        except ElementTree.ParseError as exc:
            incomplete_msg = (
                f'The file "{file}" is incomplete, '
                'and no DFT data was available. '
                'This is likely due to an incomplete calculation.')
            raise calculator.ReadError(incomplete_msg) from exc
        # VASP uses the negative definition of the hessian compared to ASE
        return -hessian

    def get_vibrations(self) -> VibrationsData:
        """Get a VibrationsData Object from a VASP Calculation.

        Returns:
            VibrationsData object.

            Note that the atoms in the VibrationsData object can be resorted.

            Uses the (mass weighted) Hessian from vasprun.xml, different masses
            in the POTCAR can therefore result in different results.

            Note the limitations concerning k-points and symmetry mentioned in
            the VASP-Wiki.
        """

        mass_weighted_hessian = self._read_massweighted_hessian_xml()
        # get indices of freely moving atoms, i.e. respect constraints.
        indices = VibrationsData.indices_from_constraints(self.atoms)
        # save the corresponding sorted atom numbers
        sort_indices = np.array(self.sort)[indices]
        # mass weights = 1/sqrt(mass)
        mass_weights = np.repeat(self.atoms.get_masses()[sort_indices]**-0.5, 3)
        # get the unweighted hessian = H_w / m_w / m_w^T
        # ugly and twice the work, but needed since vasprun.xml does
        # not have the unweighted ase.vibrations.vibration will do the
        # opposite in Vibrations.read
        hessian = mass_weighted_hessian / \
            mass_weights / mass_weights[:, np.newaxis]

        return VibrationsData.from_2d(self.atoms[self.sort], hessian, indices)

    def get_nonselfconsistent_energies(self, bee_type):
        """ Method that reads and returns BEE energy contributions
            written in OUTCAR file.
        """
        assert bee_type == 'beefvdw'
        cmd = 'grep -32 "BEEF xc energy contributions" OUTCAR | tail -32'
        p = os.popen(cmd, 'r')
        s = p.readlines()
        p.close()
        xc = np.array([])
        for line in s:
            l_ = float(line.split(":")[-1])
            xc = np.append(xc, l_)
        assert len(xc) == 32
        return xc


#####################################
# Below defines helper functions
# for the VASP calculator
#####################################


def check_atoms(atoms: ase.Atoms) -> None:
    """Perform checks on the atoms object, to verify that
    it can be run by VASP.
    A CalculatorSetupError error is raised if the atoms are not supported.
    """

    # Loop through all check functions
    for check in (check_atoms_type, check_cell, check_pbc):
        check(atoms)


def check_cell(atoms: ase.Atoms) -> None:
    """Check if there is a zero unit cell.
    Raises CalculatorSetupError if the cell is wrong.
    """
    if atoms.cell.rank < 3:
        raise calculator.CalculatorSetupError(
            "The lattice vectors are zero! "
            "This is the default value - please specify a "
            "unit cell.")


def check_pbc(atoms: ase.Atoms) -> None:
    """Check if any boundaries are not PBC, as VASP
    cannot handle non-PBC.
    Raises CalculatorSetupError.
    """
    if not atoms.pbc.all():
        raise calculator.CalculatorSetupError(
            "Vasp cannot handle non-periodic boundaries. "
            "Please enable all PBC, e.g. atoms.pbc=True")


def check_atoms_type(atoms: ase.Atoms) -> None:
    """Check that the passed atoms object is in fact an Atoms object.
    Raises CalculatorSetupError.
    """
    if not isinstance(atoms, ase.Atoms):
        raise calculator.CalculatorSetupError(
            'Expected an Atoms object, '
            'instead got object of type {}'.format(type(atoms)))
