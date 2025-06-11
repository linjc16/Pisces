# fmt: off

"""This module defines an ASE interface to Amber16.

Usage: (Tested only with Amber16, http://ambermd.org/)

Before usage, input files (infile, topologyfile, incoordfile)

"""

import subprocess

import numpy as np
from scipy.io import netcdf_file

import ase.units as units
from ase.calculators.calculator import Calculator, FileIOCalculator
from ase.io.amber import read_amber_coordinates, write_amber_coordinates


class Amber(FileIOCalculator):
    """Class for doing Amber classical MM calculations.

    Example:

    mm.in::

        Minimization with Cartesian restraints
        &cntrl
        imin=1, maxcyc=200, (invoke minimization)
        ntpr=5, (print frequency)
        &end
    """

    implemented_properties = ['energy', 'forces']
    discard_results_on_any_change = True

    def __init__(self, restart=None,
                 ignore_bad_restart_file=FileIOCalculator._deprecated,
                 label='amber', atoms=None, command=None,
                 amber_exe='sander -O ',
                 infile='mm.in', outfile='mm.out',
                 topologyfile='mm.top', incoordfile='mm.crd',
                 outcoordfile='mm_dummy.crd', mdcoordfile=None,
                 **kwargs):
        """Construct Amber-calculator object.

        Parameters
        ==========
        label: str
            Name used for all files.  May contain a directory.
        atoms: Atoms object
            Optional Atoms object to which the calculator will be
            attached.  When restarting, atoms will get its positions and
            unit-cell updated from file.
        label: str
            Prefix to use for filenames (label.in, label.txt, ...).
        amber_exe: str
            Name of the amber executable, one can add options like -O
            and other parameters here
        infile: str
            Input filename for amber, contains instuctions about the run
        outfile: str
            Logfilename for amber
        topologyfile: str
            Name of the amber topology file
        incoordfile: str
            Name of the file containing the input coordinates of atoms
        outcoordfile: str
            Name of the file containing the output coordinates of atoms
            this file is not used in case minisation/dynamics is done by ase.
            It is only relevant
            if you run MD/optimisation many steps with amber.

        """

        self.out = 'mm.log'

        self.positions = None
        self.atoms = None

        self.set(**kwargs)

        self.amber_exe = amber_exe
        self.infile = infile
        self.outfile = outfile
        self.topologyfile = topologyfile
        self.incoordfile = incoordfile
        self.outcoordfile = outcoordfile
        self.mdcoordfile = mdcoordfile

        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                  label, atoms, command=command,
                                  **kwargs)

    @property
    def _legacy_default_command(self):
        command = (self.amber_exe +
                   ' -i ' + self.infile +
                   ' -o ' + self.outfile +
                   ' -p ' + self.topologyfile +
                   ' -c ' + self.incoordfile +
                   ' -r ' + self.outcoordfile)
        if self.mdcoordfile is not None:
            command = command + ' -x ' + self.mdcoordfile
        return command

    def write_input(self, atoms=None, properties=None, system_changes=None):
        """Write updated coordinates to a file."""

        FileIOCalculator.write_input(self, atoms, properties, system_changes)
        self.write_coordinates(atoms, self.incoordfile)

    def read_results(self):
        """ read energy and forces """
        self.read_energy()
        self.read_forces()

    def write_coordinates(self, atoms, filename):
        """ write amber coordinates in netCDF format,
            only rectangular unit cells are allowed"""
        write_amber_coordinates(atoms, filename)

    def read_coordinates(self, atoms):
        """Import AMBER16 netCDF restart files.

        Reads atom positions and
        velocities (if available),
        and unit cell (if available)

        This may be useful if you have run amber many steps and
        want to read new positions and velocities
        """
        # For historical reasons we edit the input atoms rather than
        # returning new atoms.
        _atoms = read_amber_coordinates(self.outcoordfile)
        atoms.cell[:] = _atoms.cell
        atoms.pbc[:] = _atoms.pbc
        atoms.positions[:] = _atoms.positions
        atoms.set_momenta(_atoms.get_momenta())

    def read_energy(self, filename='mden'):
        """ read total energy from amber file """
        with open(filename, 'r') as fd:
            lines = fd.readlines()
            blocks = []
            while 'L0' in lines[0].split()[0]:
                blocks.append(lines[0:10])
                lines = lines[10:]
                if lines == []:
                    break
        self.results['energy'] = \
            float(blocks[-1][6].split()[2]) * units.kcal / units.mol

    def read_forces(self, filename='mdfrc'):
        """ read forces from amber file """
        fd = netcdf_file(filename, 'r', mmap=False)
        try:
            forces = fd.variables['forces']
            self.results['forces'] = forces[-1, :, :] \
                / units.Ang * units.kcal / units.mol
        finally:
            fd.close()

    def set_charges(self, selection, charges, parmed_filename=None):
        """ Modify amber topology charges to contain the updated
            QM charges, needed in QM/MM.
            Using amber's parmed program to change charges.
        """
        qm_list = list(selection)
        with open(parmed_filename, 'w') as fout:
            fout.write('# update the following QM charges \n')
            for i, charge in zip(qm_list, charges):
                fout.write('change charge @' + str(i + 1) + ' ' +
                           str(charge) + ' \n')
            fout.write('# Output the topology file \n')
            fout.write('outparm ' + self.topologyfile + ' \n')
        parmed_command = ('parmed -O -i ' + parmed_filename +
                          ' -p ' + self.topologyfile +
                          ' > ' + self.topologyfile + '.log 2>&1')
        subprocess.check_call(parmed_command, shell=True, cwd=self.directory)

    def get_virtual_charges(self, atoms):
        with open(self.topologyfile, 'r') as fd:
            topology = fd.readlines()
        for n, line in enumerate(topology):
            if '%FLAG CHARGE' in line:
                chargestart = n + 2
        lines1 = topology[chargestart:(chargestart
                                       + (len(atoms) - 1) // 5 + 1)]
        mm_charges = []
        for line in lines1:
            for el in line.split():
                mm_charges.append(float(el) / 18.2223)
        charges = np.array(mm_charges)
        return charges

    def add_virtual_sites(self, positions):
        return positions  # no virtual sites

    def redistribute_forces(self, forces):
        return forces


def map(atoms, top):
    p = np.zeros((2, len(atoms)), dtype="int")

    elements = atoms.get_chemical_symbols()
    unique_elements = np.unique(atoms.get_chemical_symbols())

    for i in range(len(unique_elements)):
        idx = 0
        for j in range(len(atoms)):
            if elements[j] == unique_elements[i]:
                idx += 1
                symbol = unique_elements[i] + np.str(idx)
                for k in range(len(atoms)):
                    if top.atoms[k].name == symbol:
                        p[0, k] = j
                        p[1, j] = k
                        break
    return p


try:
    import sander
    have_sander = True
except ImportError:
    have_sander = False


class SANDER(Calculator):
    """
    Interface to SANDER using Python interface

    Requires sander Python bindings from http://ambermd.org/
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, atoms=None, label=None, top=None, crd=None,
                 mm_options=None, qm_options=None, permutation=None, **kwargs):
        if not have_sander:
            raise RuntimeError("sander Python module could not be imported!")
        Calculator.__init__(self, label, atoms)
        self.permutation = permutation
        if qm_options is not None:
            sander.setup(top, crd.coordinates, crd.box, mm_options, qm_options)
        else:
            sander.setup(top, crd.coordinates, crd.box, mm_options)

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        if system_changes:
            if 'energy' in self.results:
                del self.results['energy']
            if 'forces' in self.results:
                del self.results['forces']
        if 'energy' not in self.results:
            if self.permutation is None:
                crd = np.reshape(atoms.get_positions(), (1, len(atoms), 3))
            else:
                crd = np.reshape(atoms.get_positions()
                                 [self.permutation[0, :]], (1, len(atoms), 3))
            sander.set_positions(crd)
            e, f = sander.energy_forces()
            self.results['energy'] = e.tot * units.kcal / units.mol
            if self.permutation is None:
                self.results['forces'] = (np.reshape(np.array(f),
                                                     (len(atoms), 3)) *
                                          units.kcal / units.mol)
            else:
                ff = np.reshape(np.array(f), (len(atoms), 3)) * \
                    units.kcal / units.mol
                self.results['forces'] = ff[self.permutation[1, :]]
        if 'forces' not in self.results:
            if self.permutation is None:
                crd = np.reshape(atoms.get_positions(), (1, len(atoms), 3))
            else:
                crd = np.reshape(atoms.get_positions()[self.permutation[0, :]],
                                 (1, len(atoms), 3))
            sander.set_positions(crd)
            e, f = sander.energy_forces()
            self.results['energy'] = e.tot * units.kcal / units.mol
            if self.permutation is None:
                self.results['forces'] = (np.reshape(np.array(f),
                                                     (len(atoms), 3)) *
                                          units.kcal / units.mol)
            else:
                ff = np.reshape(np.array(f), (len(atoms), 3)) * \
                    units.kcal / units.mol
                self.results['forces'] = ff[self.permutation[1, :]]
