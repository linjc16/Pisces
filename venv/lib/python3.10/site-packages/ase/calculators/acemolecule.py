# fmt: off

import os
from copy import deepcopy

from ase.calculators.calculator import FileIOCalculator, ReadError
from ase.io import read


class ACE(FileIOCalculator):
    '''
    ACE-Molecule logfile reader
    It has default parameters of each input section
    And parameters' type = list of dictionaries
    '''
    name = 'ace'
    implemented_properties = ['energy', 'forces', 'excitation-energy']
    basic_list = [{
        'Type': 'Scaling', 'Scaling': '0.35', 'Basis': 'Sinc',
                  'Grid': 'Sphere',
                  'KineticMatrix': 'Finite_Difference', 'DerivativesOrder': '7',
                  'GeometryFilename': None, 'NumElectrons': None}
                  ]
    scf_list = [{
        'ExchangeCorrelation': {'XFunctional': 'GGA_X_PBE',
                                'CFunctional': 'GGA_C_PBE'},
        'NumberOfEigenvalues': None,
    }]

    force_list = [{'ForceDerivative': 'Potential'}]
    tddft_list = [{
        'SortOrbital': 'Order', 'MaximumOrder': '10',
        'ExchangeCorrelation': {'XFunctional': 'GGA_X_PBE',
                                'CFunctional': 'GGA_C_PBE'},
    }]

    order_list = ['BasicInformation', 'Guess', 'Scf']
    guess_list = [{}]  # type: ignore[var-annotated]
    default_parameters = {'BasicInformation': basic_list, 'Guess': guess_list,
                          'Scf': scf_list, 'Force': force_list,
                          'TDDFT': tddft_list, 'order': order_list}

    def __init__(
            self, restart=None,
            ignore_bad_restart_file=FileIOCalculator._deprecated,
            label='ace', atoms=None, command=None,
            basisfile=None, **kwargs):
        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                  label, atoms, command=command, **kwargs)

    def set(self, **kwargs):
        '''Update parameters self.parameter member variable.
        1. Add default values for repeated parameter sections with
           self.default_parameters using order.
        2. Also add empty dictionary as an indicator for section existence
           if no relevant default_parameters exist.
        3. Update parameters from arguments.

        Returns
        =======
        Updated parameter
        '''
        new_parameters = deepcopy(self.parameters)

        changed_parameters = FileIOCalculator.set(self, **kwargs)

        # Add default values for repeated parameter sections with
        # self.default_parameters using order.  Also add empty
        # dictionary as an indicator for section existence if no
        # relevant default_parameters exist.
        if 'order' in kwargs:
            new_parameters['order'] = kwargs['order']
            section_sets = set(kwargs['order'])
            for section_name in section_sets:
                repeat = kwargs['order'].count(section_name)
                if section_name in self.default_parameters.keys():
                    for _ in range(repeat - 1):
                        new_parameters[section_name] += deepcopy(
                            self.default_parameters[section_name])
                else:
                    new_parameters[section_name] = []
                    for _ in range(repeat):
                        new_parameters[section_name].append({})

        # Update parameters
        for section in new_parameters['order']:
            if section in kwargs:
                if isinstance(kwargs[section], dict):
                    kwargs[section] = [kwargs[section]]

                for i, section_param in enumerate(kwargs[section]):
                    new_parameters[section][i] = update_parameter(
                        new_parameters[section][i], section_param)
        self.parameters = new_parameters
        return changed_parameters

    def read(self, label):
        FileIOCalculator.read(self, label)
        filename = self.label + ".log"

        with open(filename) as fd:
            lines = fd.readlines()
        if 'WARNING' in lines:
            raise ReadError(
                f"Not convergy energy in log file {filename}.")
        if '! total energy' not in lines:
            raise ReadError(f"Wrong ACE-Molecule log file {filename}.")

        if not os.path.isfile(filename):
            raise ReadError(
                f"Wrong ACE-Molecule input file {filename}.")

        self.read_results()

    def write_input(self, atoms, properties=None, system_changes=None):
        '''Initializes input parameters and xyz files. If force calculation is
        requested, add Force section to parameters if not exists.

        Parameters
        ==========
        atoms: ASE atoms object.
        properties: List of properties to be calculated. Should be element
            of self.implemented_properties.
        system_chages: Ignored.

        '''
        FileIOCalculator.write_input(self, atoms, properties, system_changes)
        with open(self.label + '.inp', 'w') as inputfile:
            xyz_name = f"{self.label}.xyz"
            atoms.write(xyz_name)

            run_parameters = self.prepare_input(xyz_name, properties)
            self.write_acemolecule_input(inputfile, run_parameters)

    def prepare_input(self, geometry_filename, properties):
        '''Initialize parameters dictionary based on geometry filename and
        calculated properties.

        Parameters
        ==========
        geometry_filename: Geometry (XYZ format) file path.
        properties: Properties to be calculated.

        Returns
        =======
        Updated version of self.parameters; geometry file and
        optionally Force section are updated.

        '''
        copied_parameters = deepcopy(self.parameters)
        if (properties is not None and "forces" in properties
                and 'Force' not in copied_parameters['order']):
            copied_parameters['order'].append('Force')
        copied_parameters["BasicInformation"][0]["GeometryFilename"] = \
            f"{self.label}.xyz"
        copied_parameters["BasicInformation"][0]["GeometryFormat"] = "xyz"
        return copied_parameters

    def read_results(self):
        '''Read calculation results, speficied by 'quantities' variable, from
        the log file.

        quantities
        =======
        energy : obtaing single point energy(eV) from log file
        forces : obtaing force of each atom form log file
        excitation-energy : it able to calculate TDDFT.
        Return value is None. Result is not used.
        atoms : ASE atoms object

        '''
        filename = self.label + '.log'
        self.results = read(filename, format='acemolecule-out')

    def write_acemolecule_section(self, fpt, section, depth=0):
        '''Write parameters in each section of input

        Parameters
        ==========
        fpt: ACE-Moleucle input file object. Should be write mode.
        section: Dictionary of a parameter section.
        depth: Nested input depth.
        '''
        for section, section_param in section.items():
            if isinstance(section_param, (str, int, float)):
                fpt.write(
                    '    ' *
                    depth +
                    str(section) +
                    " " +
                    str(section_param) +
                    "\n")
            else:
                if isinstance(section_param, dict):
                    fpt.write('    ' * depth + "%% " + str(section) + "\n")
                    self.write_acemolecule_section(
                        fpt, section_param, depth + 1)
                    fpt.write('    ' * depth + "%% End\n")
                if isinstance(section_param, list):
                    for val in section_param:
                        fpt.write(
                            '    ' *
                            depth +
                            str(section) +
                            " " +
                            str(val) +
                            "\n")

    def write_acemolecule_input(self, fpt, param, depth=0):
        '''Write ACE-Molecule input

        ACE-Molecule input examples (not minimal)
        %% BasicInformation
            Type    Scaling
            Scaling 0.4
            Basis   Sinc
            Cell    10.0
            Grid    Sphere
            GeometryFormat      xyz
            SpinMultiplicity    3.0
            Polarize    1
            Centered    0
            %% Pseudopotential
                Pseudopotential 1
                UsingDoubleGrid 0
                FilterType      Sinc
                Format          upf
                PSFilePath      /PATH/TO/UPF
                PSFileSuffix    .pbe-theos.UPF
            %% End
            GeometryFilename    xyz/C.xyz
        %% End
        %% Guess
            InitialGuess        3
            InitialFilenames    001.cube
            InitialFilenames    002.cube
        %% End
        %% Scf
            IterateMaxCycle     150
            ConvergenceType     Energy
            ConvergenceTolerance    0.00001
            EnergyDecomposition     1
            ComputeInitialEnergy    1
            %% Diagonalize
                Tolerance           0.000001
            %% End
            %% ExchangeCorrelation
                XFunctional     GGA_X_PBE
                CFunctional     GGA_C_PBE
            %% End
            %% Mixing
                MixingMethod         1
                MixingType           Density
                MixingParameter      0.5
                PulayMixingParameter 0.1
            %% End
        %% End

        Parameters
        ==========
        fpt: File object, should be write mode.
        param: Dictionary of parameters. Also should contain
               special 'order' section_name for parameter section ordering.
        depth: Nested input depth.

        Notes
        =====
         - Order of parameter section
           (denoted using %% -- %% BasicInformation, %% Guess, etc.)
           is important, because it determines calculation order.
           For example, if Guess section comes after Scf section,
           calculation will not run because Scf will tries to run
            without initial Hamiltonian.
         - Order of each parameter section-section_name pair is
           not important unless their keys are the same.
         - Indentation unimportant and capital letters are important.
        '''
        prefix = "    " * depth

        for i in range(len(param['order'])):
            fpt.write(prefix + "%% " + param['order'][i] + "\n")
            section_list = param[param['order'][i]]
            if len(section_list) > 0:
                section = section_list.pop(0)
                self.write_acemolecule_section(fpt, section, 1)
            fpt.write("%% End\n")
        return


def update_parameter(oldpar, newpar):
    '''Update each section of parameter (oldpar) using newpar keys and values.
    If section of newpar exist in oldpar,
        - Replace the section_name with newpar's section_name if
          oldvar section_name type is not dict.
        - Append the section_name with newpar's section_name
           if oldvar section_name type is list.
        - If oldpar section_name type is dict, it is subsection.
          So call update_parameter again.
    otherwise, add the parameter section and section_name from newpar.

    Parameters
    ==========
    oldpar: dictionary of original parameters to be updated.
    newpar: dictionary containing parameter section and values to update.

    Return
    ======
    Updated parameter dictionary.
    '''
    for section, section_param in newpar.items():
        if section in oldpar:
            if isinstance(section_param, dict):
                oldpar[section] = update_parameter(
                    oldpar[section], section_param)
            else:
                oldpar[section] = section_param
        else:
            oldpar[section] = section_param
    return oldpar
