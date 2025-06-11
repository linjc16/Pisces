# fmt: off

import os
import subprocess
from pathlib import Path
from warnings import warn

import numpy as np

from ase.calculators.calculator import (
    BaseCalculator,
    Calculator,
    FileIOCalculator,
)
from ase.io import write
from ase.io.vasp import write_vasp
from ase.parallel import world
from ase.units import Bohr, Hartree


def dftd3_defaults():
    default_parameters = {'xc': None,  # PBE if no custom damping parameters
                          'grad': True,  # calculate forces/stress
                          'abc': False,  # ATM 3-body contribution
                          'cutoff': 95 * Bohr,  # Cutoff for 2-body calcs
                          'cnthr': 40 * Bohr,  # Cutoff for 3-body and CN calcs
                          'old': False,  # use old DFT-D2 method instead
                          'damping': 'zero',  # Default to zero-damping
                          'tz': False,  # 'triple zeta' alt. parameters
                          's6': None,  # damping parameters start here
                          'sr6': None,
                          's8': None,
                          'sr8': None,
                          'alpha6': None,
                          'a1': None,
                          'a2': None,
                          'beta': None}
    return default_parameters


class DFTD3(BaseCalculator):
    """Grimme DFT-D3 calculator"""

    def __init__(self,
                 label='ase_dftd3',  # Label for dftd3 output files
                 command=None,  # Command for running dftd3
                 dft=None,  # DFT calculator
                 comm=world,
                 **kwargs):

        # Convert from 'func' keyword to 'xc'. Internally, we only store
        # 'xc', but 'func' is also allowed since it is consistent with the
        # CLI dftd3 interface.
        func = kwargs.pop('func', None)
        if func is not None:
            if kwargs.get('xc') is not None:
                raise RuntimeError('Both "func" and "xc" were provided! '
                                   'Please provide at most one of these '
                                   'two keywords. The preferred keyword '
                                   'is "xc"; "func" is allowed for '
                                   'consistency with the CLI dftd3 '
                                   'interface.')
            kwargs['xc'] = func

        # If the user did not supply an XC functional, but did attach a
        # DFT calculator that has XC set, then we will use that. Note that
        # DFTD3's spelling convention is different from most, so in general
        # you will have to explicitly set XC for both the DFT calculator and
        # for DFTD3 (and DFTD3's will likely be spelled differently...)
        if dft is not None and kwargs.get('xc') is None:
            dft_xc = dft.parameters.get('xc')
            if dft_xc is not None:
                kwargs['xc'] = dft_xc

        dftd3 = PureDFTD3(label=label, command=command, comm=comm, **kwargs)

        # dftd3 only implements energy, forces, and stresses (for periodic
        # systems). But, if a DFT calculator is attached, and that calculator
        # implements more properties, we expose those properties.
        # dftd3 contributions for those properties will be zero.
        if dft is None:
            self.implemented_properties = list(dftd3.dftd3_properties)
        else:
            self.implemented_properties = list(dft.implemented_properties)

        # Should our arguments be "parameters" (passed to superclass)
        # or are they not really "parameters"?
        #
        # That's not really well defined.  Let's not do anything then.
        super().__init__()

        self.dftd3 = dftd3
        self.dft = dft

    def todict(self):
        return {}

    def calculate(self, atoms, properties, system_changes):
        common_props = set(self.dftd3.dftd3_properties) & set(properties)
        dftd3_results = self._get_properties(atoms, common_props, self.dftd3)

        if self.dft is None:
            results = dftd3_results
        else:
            dft_results = self._get_properties(atoms, properties, self.dft)
            results = dict(dft_results)
            for name in set(results) & set(dftd3_results):
                assert np.shape(results[name]) == np.shape(dftd3_results[name])
                results[name] += dftd3_results[name]

            # Although DFTD3 may have calculated quantities not provided
            # by the calculator (e.g. stress), it would be wrong to
            # return those!  Return only what corresponds to the DFT calc.
            assert set(results) == set(dft_results)
        self.results = results

    def _get_properties(self, atoms, properties, calc):
        # We want any and all properties that the calculator
        # normally produces.  So we intend to rob the calc.results
        # dictionary instead of only getting the requested properties.

        import copy
        for name in properties:
            calc.get_property(name, atoms)
            assert name in calc.results

        # XXX maybe use get_properties() when that makes sense.
        results = copy.deepcopy(calc.results)
        assert set(properties) <= set(results)
        return results


class PureDFTD3(FileIOCalculator):
    """DFTD3 calculator without corresponding DFT contribution.

    This class is an implementation detail."""

    name = 'puredftd3'

    dftd3_properties = {'energy', 'free_energy', 'forces', 'stress'}
    implemented_properties = list(dftd3_properties)
    default_parameters = dftd3_defaults()
    damping_methods = {'zero', 'bj', 'zerom', 'bjm'}
    _legacy_default_command = 'dftd3'

    def __init__(self,
                 *,
                 label='ase_dftd3',  # Label for dftd3 output files
                 command=None,  # Command for running dftd3
                 comm=world,
                 **kwargs):

        # FileIOCalculator would default to self.name to get the envvar
        # which determines the command.
        # We'll have to overrule that if we want to keep scripts working:
        command = command or self.cfg.get('ASE_DFTD3_COMMAND')

        super().__init__(label=label,
                         command=command,
                         **kwargs)

        # TARP: This is done because the calculator does not call
        # FileIOCalculator.calculate, but Calculator.calculate and does not
        # use the profile defined in FileIOCalculator.__init__
        self.comm = comm

    def set(self, **kwargs):
        changed_parameters = {}

        # Check for unknown arguments. Don't raise an error, just let the
        # user know that we don't understand what they're asking for.
        unknown_kwargs = set(kwargs) - set(self.default_parameters)
        if unknown_kwargs:
            warn('WARNING: Ignoring the following unknown keywords: {}'
                 ''.format(', '.join(unknown_kwargs)))

        changed_parameters.update(FileIOCalculator.set(self, **kwargs))

        # Ensure damping method is valid (zero, bj, zerom, bjm).
        damping = self.parameters['damping']
        if damping is not None:
            damping = damping.lower()
        if damping not in self.damping_methods:
            raise ValueError(f'Unknown damping method {damping}!')

        # d2 only is valid with 'zero' damping
        elif self.parameters['old'] and damping != 'zero':
            raise ValueError('Only zero-damping can be used with the D2 '
                             'dispersion correction method!')

        # If cnthr (cutoff for three-body and CN calculations) is greater
        # than cutoff (cutoff for two-body calculations), then set the former
        # equal to the latter, since that doesn't make any sense.
        if self.parameters['cnthr'] > self.parameters['cutoff']:
            warn('WARNING: CN cutoff value of {cnthr} is larger than '
                 'regular cutoff value of {cutoff}! Reducing CN cutoff '
                 'to {cutoff}.'
                 ''.format(cnthr=self.parameters['cnthr'],
                           cutoff=self.parameters['cutoff']))
            self.parameters['cnthr'] = self.parameters['cutoff']

        # If you only care about the energy, gradient calculations (forces,
        # stresses) can be bypassed. This will greatly speed up calculations
        # in dense 3D-periodic systems with three-body corrections. But, we
        # can no longer say that we implement forces and stresses.
        # if not self.parameters['grad']:
        #    for val in ['forces', 'stress']:
        #        if val in self.implemented_properties:
        #            self.implemented_properties.remove(val)

        # Check to see if we're using custom damping parameters.
        zero_damppars = {'s6', 'sr6', 's8', 'sr8', 'alpha6'}
        bj_damppars = {'s6', 'a1', 's8', 'a2', 'alpha6'}
        zerom_damppars = {'s6', 'sr6', 's8', 'beta', 'alpha6'}
        all_damppars = zero_damppars | bj_damppars | zerom_damppars

        self.custom_damp = False

        damppars = set(kwargs) & all_damppars
        if damppars:
            self.custom_damp = True
            if damping == 'zero':
                valid_damppars = zero_damppars
            elif damping in ['bj', 'bjm']:
                valid_damppars = bj_damppars
            elif damping == 'zerom':
                valid_damppars = zerom_damppars

            # If some but not all damping parameters are provided for the
            # selected damping method, raise an error. We don't have "default"
            # values for damping parameters, since those are stored in the
            # dftd3 executable & depend on XC functional.
            missing_damppars = valid_damppars - damppars
            if missing_damppars and missing_damppars != valid_damppars:
                raise ValueError('An incomplete set of custom damping '
                                 'parameters for the {} damping method was '
                                 'provided! Expected: {}; got: {}'
                                 ''.format(damping,
                                           ', '.join(valid_damppars),
                                           ', '.join(damppars)))

            # If a user provides damping parameters that are not used in the
            # selected damping method, let them know that we're ignoring them.
            # If the user accidentally provided the *wrong* set of parameters,
            # (e.g., the BJ parameters when they are using zero damping), then
            # the previous check will raise an error, so we don't need to
            # worry about that here.
            if damppars - valid_damppars:
                warn('WARNING: The following damping parameters are not '
                     'valid for the {} damping method and will be ignored: {}'
                     ''.format(damping,
                               ', '.join(damppars)))

        # The default XC functional is PBE, but this is only set if the user
        # did not provide their own value for xc or any custom damping
        # parameters.
        if self.parameters['xc'] and self.custom_damp:
            warn('WARNING: Custom damping parameters will be used '
                 'instead of those parameterized for {}!'
                 ''.format(self.parameters['xc']))

        if changed_parameters:
            self.results.clear()
        return changed_parameters

    def calculate(self, atoms, properties, system_changes):
        # We don't call FileIOCalculator.calculate here, because that method
        # calls subprocess.call(..., shell=True), which we don't want to do.
        # So, we reproduce some content from that method here.
        Calculator.calculate(self, atoms, properties, system_changes)

        # If a parameter file exists in the working directory, delete it
        # first. If we need that file, we'll recreate it later.
        localparfile = os.path.join(self.directory, '.dftd3par.local')
        if self.comm.rank == 0 and os.path.isfile(localparfile):
            os.remove(localparfile)

        # Write XYZ or POSCAR file and .dftd3par.local file if we are using
        # custom damping parameters.
        self.write_input(self.atoms, properties, system_changes)
        # command = self._generate_command()

        inputs = DFTD3Inputs(command=self.command, prefix=self.label,
                             atoms=self.atoms, parameters=self.parameters)
        command = inputs.get_argv(custom_damp=self.custom_damp)

        # Finally, call dftd3 and parse results.
        # DFTD3 does not run in parallel
        # so we only need it to run on 1 core
        errorcode = 0
        if self.comm.rank == 0:
            with open(self.label + '.out', 'w') as fd:
                errorcode = subprocess.call(command,
                                            cwd=self.directory, stdout=fd)

        errorcode = self.comm.sum_scalar(errorcode)

        if errorcode:
            raise RuntimeError('%s returned an error: %d' %
                               (self.name, errorcode))

        self.read_results()

    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, atoms, properties=properties,
                                     system_changes=system_changes)
        # dftd3 can either do fully 3D periodic or non-periodic calculations.
        # It cannot do calculations that are only periodic in 1 or 2
        # dimensions. If the atoms object is periodic in only 1 or 2
        # dimensions, then treat it as a fully 3D periodic system, but warn
        # the user.

        if self.custom_damp:
            damppars = _get_damppars(self.parameters)
        else:
            damppars = None

        pbc = any(atoms.pbc)
        if pbc and not all(atoms.pbc):
            warn('WARNING! dftd3 can only calculate the dispersion energy '
                 'of non-periodic or 3D-periodic systems. We will treat '
                 'this system as 3D-periodic!')

        if self.comm.rank == 0:
            self._actually_write_input(
                directory=Path(self.directory), atoms=atoms,
                properties=properties, prefix=self.label,
                damppars=damppars, pbc=pbc)

    def _actually_write_input(self, directory, prefix, atoms, properties,
                              damppars, pbc):
        if pbc:
            fname = directory / f'{prefix}.POSCAR'
            # We sort the atoms so that the atomtypes list becomes as
            # short as possible.  The dftd3 program can only handle 10
            # atomtypes
            write_vasp(fname, atoms, sort=True)
        else:
            fname = directory / f'{prefix}.xyz'
            write(fname, atoms, format='xyz', parallel=False)

        # Generate custom damping parameters file. This is kind of ugly, but
        # I don't know of a better way of doing this.
        if damppars is not None:
            damp_fname = directory / '.dftd3par.local'
            with open(damp_fname, 'w') as fd:
                fd.write(' '.join(damppars))

    def _outname(self):
        return Path(self.directory) / f'{self.label}.out'

    def _read_and_broadcast_results(self):
        from ase.parallel import broadcast
        if self.comm.rank == 0:
            output = DFTD3Output(directory=self.directory,
                                 stdout_path=self._outname())
            dct = output.read(atoms=self.atoms,
                              read_forces=bool(self.parameters['grad']))
        else:
            dct = None

        dct = broadcast(dct, root=0, comm=self.comm)
        return dct

    def read_results(self):
        results = self._read_and_broadcast_results()
        self.results = results


class DFTD3Inputs:
    dftd3_flags = {'grad', 'pbc', 'abc', 'old', 'tz'}

    def __init__(self, command, prefix, atoms, parameters):
        self.command = command
        self.prefix = prefix
        self.atoms = atoms
        self.parameters = parameters

    @property
    def pbc(self):
        return any(self.atoms.pbc)

    @property
    def inputformat(self):
        if self.pbc:
            return 'POSCAR'
        else:
            return 'xyz'

    def get_argv(self, custom_damp):
        argv = self.command.split()

        argv.append(f'{self.prefix}.{self.inputformat}')

        if not custom_damp:
            xc = self.parameters.get('xc')
            if xc is None:
                xc = 'pbe'
            argv += ['-func', xc.lower()]

        for arg in self.dftd3_flags:
            if self.parameters.get(arg):
                argv.append('-' + arg)

        if self.pbc:
            argv.append('-pbc')

        argv += ['-cnthr', str(self.parameters['cnthr'] / Bohr)]
        argv += ['-cutoff', str(self.parameters['cutoff'] / Bohr)]

        if not self.parameters['old']:
            argv.append('-' + self.parameters['damping'])

        return argv


class DFTD3Output:
    def __init__(self, directory, stdout_path):
        self.directory = Path(directory)
        self.stdout_path = Path(stdout_path)

    def read(self, *, atoms, read_forces):
        results = {}

        energy = self.read_energy()
        results['energy'] = energy
        results['free_energy'] = energy

        if read_forces:
            results['forces'] = self.read_forces(atoms)

        if any(atoms.pbc):
            results['stress'] = self.read_stress(atoms.cell)

        return results

    def read_forces(self, atoms):
        forcename = self.directory / 'dftd3_gradient'
        with open(forcename) as fd:
            forces = self.parse_forces(fd)

        assert len(forces) == len(atoms)

        forces *= -Hartree / Bohr
        # XXXX ordering!
        if any(atoms.pbc):
            # This seems to be due to vasp file sorting.
            # If that sorting rule changes, we will get garbled
            # forces!
            ind = np.argsort(atoms.symbols)
            forces[ind] = forces.copy()
        return forces

    def read_stress(self, cell):
        volume = cell.volume
        assert volume > 0

        stress = self.read_cellgradient()
        stress *= Hartree / Bohr / volume
        stress = stress.T @ cell
        return stress.flat[[0, 4, 8, 5, 2, 1]]

    def read_cellgradient(self):
        with (self.directory / 'dftd3_cellgradient').open() as fd:
            return self.parse_cellgradient(fd)

    def read_energy(self) -> float:
        with self.stdout_path.open() as fd:
            return self.parse_energy(fd, self.stdout_path)

    def parse_energy(self, fd, outname):
        for line in fd:
            if line.startswith(' program stopped'):
                if 'functional name unknown' in line:
                    message = ('Unknown DFTD3 functional name. '
                               'Please check the dftd3.f source file '
                               'for the list of known functionals '
                               'and their spelling.')
                else:
                    message = ('dftd3 failed! Please check the {} '
                               'output file and report any errors '
                               'to the ASE developers.'
                               ''.format(outname))
                raise RuntimeError(message)

            if line.startswith(' Edisp'):
                # line looks something like this:
                #
                #     Edisp /kcal,au,ev: xxx xxx xxx
                #
                parts = line.split()
                assert parts[1][0] == '/'
                index = 2 + parts[1][1:-1].split(',').index('au')
                e_dftd3 = float(parts[index]) * Hartree
                return e_dftd3

        raise RuntimeError('Could not parse energy from dftd3 '
                           'output, see file {}'.format(outname))

    def parse_forces(self, fd):
        forces = []
        for i, line in enumerate(fd):
            forces.append(line.split())
        return np.array(forces, dtype=float)

    def parse_cellgradient(self, fd):
        stress = np.zeros((3, 3))
        for i, line in enumerate(fd):
            for j, x in enumerate(line.split()):
                stress[i, j] = float(x)
        # Check if all stress elements are present?
        # Check if file is longer?
        return stress


def _get_damppars(par):
    damping = par['damping']

    damppars = []

    # s6 is always first
    damppars.append(str(float(par['s6'])))

    # sr6 is the second value for zero{,m} damping, a1 for bj{,m}
    if damping in ['zero', 'zerom']:
        damppars.append(str(float(par['sr6'])))
    elif damping in ['bj', 'bjm']:
        damppars.append(str(float(par['a1'])))

    # s8 is always third
    damppars.append(str(float(par['s8'])))

    # sr8 is fourth for zero, a2 for bj{,m}, beta for zerom
    if damping == 'zero':
        damppars.append(str(float(par['sr8'])))
    elif damping in ['bj', 'bjm']:
        damppars.append(str(float(par['a2'])))
    elif damping == 'zerom':
        damppars.append(str(float(par['beta'])))
    # alpha6 is always fifth
    damppars.append(str(int(par['alpha6'])))

    # last is the version number
    if par['old']:
        damppars.append('2')
    elif damping == 'zero':
        damppars.append('3')
    elif damping == 'bj':
        damppars.append('4')
    elif damping == 'zerom':
        damppars.append('5')
    elif damping == 'bjm':
        damppars.append('6')
    return damppars
