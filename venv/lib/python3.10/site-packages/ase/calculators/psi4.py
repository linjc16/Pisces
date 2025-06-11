# fmt: off

"""
authors: Ben Comer (Georgia Tech), Xiangyun (Ray) Lei (Georgia Tech)

"""
import json
import multiprocessing
import os
import warnings
from io import StringIO

import numpy as np

from ase import io
from ase.calculators.calculator import (
    Calculator,
    CalculatorSetupError,
    InputError,
    ReadError,
    all_changes,
)
from ase.config import cfg
from ase.units import Bohr, Hartree


class Psi4(Calculator):
    """
    An ase calculator for the popular open source Q-chem code
    psi4.
    method is the generic input for whatever method you wish to use, thus
    and quantum chemistry method implemented in psi4 can be input
    (i.e. ccsd(t))

    also note that you can always use the in-built psi4 module through:
    calc.psi4
    """
    implemented_properties = ['energy', 'forces']
    discard_results_on_any_change = True

    default_parameters = {
        "basis": "aug-cc-pvtz",
        "method": "hf",
        'symmetry': 'c1'}

    def __init__(self, restart=None, ignore_bad_restart=False,
                 label='psi4-calc', atoms=None, command=None,
                 **kwargs):
        Calculator.__init__(self, restart=restart,
                            ignore_bad_restart=ignore_bad_restart, label=label,
                            atoms=atoms, command=command, **kwargs)
        import psi4
        self.psi4 = psi4
        # perform initial setup of psi4 python API
        self.set_psi4(atoms=atoms)

    def set_psi4(self, atoms=None):
        """
        This function sets the imported psi4 module to the settings the user
        defines
        """

        # Set the scrath directory for electronic structure files.
        # The default is /tmp
        if 'PSI_SCRATCH' in cfg:
            pass
        elif self.parameters.get('PSI_SCRATCH'):
            # XXX We should probably not be setting envvars except
            # if we are creating new processes.
            os.environ['PSI_SCRATCH'] = self.parameters['PSI_SCRATCH']

        # Input spin settings
        if self.parameters.get('reference') is not None:
            self.psi4.set_options({'reference':
                                   self.parameters['reference']})
        # Memory
        if self.parameters.get('memory') is not None:
            self.psi4.set_memory(self.parameters['memory'])

        # Threads
        nthreads = self.parameters.get('num_threads', 1)
        if nthreads == 'max':
            nthreads = multiprocessing.cpu_count()
        self.psi4.set_num_threads(nthreads)

        # deal with some ASE specific inputs
        if 'kpts' in self.parameters:
            raise InputError('psi4 is a non-periodic code, and thus does not'
                             ' require k-points. Please remove this '
                             'argument.')

        if self.parameters['method'] == 'LDA':
            # svwn is equivalent to LDA
            self.parameters['method'] = 'svwn'

        if 'nbands' in self.parameters:
            raise InputError('psi4 does not support the keyword "nbands"')

        if 'smearing' in self.parameters:
            raise InputError('Finite temperature DFT is not implemented in'
                             ' psi4 currently, thus a smearing argument '
                             'cannot be utilized. please remove this '
                             'argument')

        if 'xc' in self.parameters:
            raise InputError('psi4 does not accept the `xc` argument please'
                             ' use the `method` argument instead')

        if atoms is None:
            if self.atoms is None:
                return
            else:
                atoms = self.atoms
        if self.atoms is None:
            self.atoms = atoms
        # Generate the atomic input
        geomline = '{}\t{:.15f}\t{:.15f}\t{:.15f}'
        geom = [geomline.format(atom.symbol, *atom.position) for atom in atoms]
        geom.append('symmetry {}'.format(self.parameters['symmetry']))
        geom.append('units angstrom')

        charge = self.parameters.get('charge')
        mult = self.parameters.get('multiplicity')
        if mult is None:
            mult = 1
            if charge is not None:
                warnings.warn('A charge was provided without a spin '
                              'multiplicity. A multiplicity of 1 is assumed')
        if charge is None:
            charge = 0

        geom.append(f'{charge} {mult}')
        geom.append('no_reorient')

        if not os.path.isdir(self.directory):
            os.mkdir(self.directory)
        self.molecule = self.psi4.geometry('\n'.join(geom))

    def read(self, label):
        """Read psi4 outputs made from this ASE calculator
        """
        filename = label + '.dat'
        if not os.path.isfile(filename):
            raise ReadError('Could not find the psi4 output file: ' + filename)

        with open(filename) as fd:
            txt = fd.read()
        if '!ASE Information\n' not in txt:
            raise Exception('The output file {} could not be read because '
                            'the file does not contain the "!ASE Information"'
                            ' lines inserted by this calculator. This likely'
                            ' means the output file was not made using this '
                            'ASE calculator or has since been modified and '
                            'thus cannot be read.'.format(filename))
        info = txt.split('!ASE Information\n')[1]
        info = info.split('!')[0]
        saved_dict = json.loads(info)
        # use io read to recode atoms
        with StringIO(str(saved_dict['atoms'])) as g:
            self.atoms = io.read(g, format='json')
        self.parameters = saved_dict['parameters']
        self.results = saved_dict['results']
        # convert forces to numpy array
        if 'forces' in self.results:
            self.results['forces'] = np.array(self.results['forces'])

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes, symmetry='c1'):

        Calculator.calculate(self, atoms=atoms)
        if self.atoms is None:
            raise CalculatorSetupError('An Atoms object must be provided to '
                                       'perform a calculation')
        atoms = self.atoms

        if atoms.get_initial_magnetic_moments().any():
            self.parameters['reference'] = 'uhf'
            self.parameters['multiplicity'] = None
        # this inputs all the settings into psi4
        self.set_psi4(atoms=atoms)
        self.psi4.core.set_output_file(self.label + '.dat',
                                       False)

        # Set up the method
        method = self.parameters['method']
        basis = self.parameters['basis']

        # Do the calculations
        if 'forces' in properties:
            grad, wf = self.psi4.driver.gradient(f'{method}/{basis}',
                                                 return_wfn=True)
            # energy comes for free
            energy = wf.energy()
            self.results['energy'] = energy * Hartree
            # convert to eV/A
            # also note that the gradient is -1 * forces
            self.results['forces'] = -1 * np.array(grad) * Hartree / Bohr
        elif 'energy' in properties:
            energy = self.psi4.energy(f'{method}/{basis}',
                                      molecule=self.molecule)
            # convert to eV
            self.results['energy'] = energy * Hartree

        # dump the calculator info to the psi4 file
        save_atoms = self.atoms.copy()
        # use io.write to encode atoms
        with StringIO() as fd:
            io.write(fd, save_atoms, format='json')
            json_atoms = fd.getvalue()
        # convert forces to list for json storage
        save_results = self.results.copy()
        if 'forces' in save_results:
            save_results['forces'] = save_results['forces'].tolist()
        save_dict = {'parameters': self.parameters,
                     'results': save_results,
                     'atoms': json_atoms}
        self.psi4.core.print_out('!ASE Information\n')
        self.psi4.core.print_out(json.dumps(save_dict))
        self.psi4.core.print_out('!')
