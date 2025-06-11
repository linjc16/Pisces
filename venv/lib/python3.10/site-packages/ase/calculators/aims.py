# fmt: off

"""This module defines an ASE interface to FHI-aims.

Felix Hanke hanke@liverpool.ac.uk
Jonas Bjork j.bjork@liverpool.ac.uk
Simon P. Rittmeyer simon.rittmeyer@tum.de

Edits on (24.11.2021) by Thomas A. R. Purcell purcell@fhi-berlin.mpg.de
"""

import os
import re

import numpy as np

from ase.calculators.genericfileio import (
    BaseProfile,
    CalculatorTemplate,
    GenericFileIOCalculator,
    read_stdout,
)
from ase.io.aims import write_aims, write_control


def get_aims_version(string):
    match = re.search(r'\s*FHI-aims version\s*:\s*(\S+)', string, re.M)
    return match.group(1)


class AimsProfile(BaseProfile):
    configvars = {'default_species_directory'}

    def __init__(self, command, default_species_directory=None, **kwargs):
        super().__init__(command, **kwargs)
        self.default_species_directory = default_species_directory

    def get_calculator_command(self, inputfile):
        return []

    def version(self):
        return get_aims_version(read_stdout(self._split_command))


class AimsTemplate(CalculatorTemplate):
    _label = 'aims'

    def __init__(self):
        super().__init__(
            'aims',
            [
                'energy',
                'free_energy',
                'forces',
                'stress',
                'stresses',
                'dipole',
                'magmom',
            ],
        )

        self.outputname = f'{self._label}.out'
        self.errorname = f'{self._label}.err'

    def update_parameters(self, properties, parameters):
        """Check and update the parameters to match the desired calculation

        Parameters
        ----------
        properties: list of str
            The list of properties to calculate
        parameters: dict
            The parameters used to perform the calculation.

        Returns
        -------
        dict
            The updated parameters object
        """
        parameters = dict(parameters)
        property_flags = {
            'forces': 'compute_forces',
            'stress': 'compute_analytical_stress',
            'stresses': 'compute_heat_flux',
        }
        # Ensure FHI-aims will calculate all desired properties
        for property in properties:
            aims_name = property_flags.get(property, None)
            if aims_name is not None:
                parameters[aims_name] = True

        if 'dipole' in properties:
            if 'output' in parameters and 'dipole' not in parameters['output']:
                parameters['output'] = list(parameters['output'])
                parameters['output'].append('dipole')
            elif 'output' not in parameters:
                parameters['output'] = ['dipole']

        return parameters

    def write_input(self, profile, directory, atoms, parameters, properties):
        """Write the geometry.in and control.in files for the calculation

        Parameters
        ----------
        directory : Path
            The working directory to store the input files.
        atoms : atoms.Atoms
            The atoms object to perform the calculation on.
        parameters: dict
            The parameters used to perform the calculation.
        properties: list of str
            The list of properties to calculate
        """
        parameters = self.update_parameters(properties, parameters)

        ghosts = parameters.pop('ghosts', None)
        geo_constrain = parameters.pop('geo_constrain', None)
        scaled = parameters.pop('scaled', None)
        write_velocities = parameters.pop('write_velocities', None)

        if scaled is None:
            scaled = np.all(atoms.pbc)
        if write_velocities is None:
            write_velocities = atoms.has('momenta')

        if geo_constrain is None:
            geo_constrain = scaled and 'relax_geometry' in parameters

        have_lattice_vectors = atoms.pbc.any()
        have_k_grid = (
            'k_grid' in parameters
            or 'kpts' in parameters
            or 'k_grid_density' in parameters
        )
        if have_lattice_vectors and not have_k_grid:
            raise RuntimeError('Found lattice vectors but no k-grid!')
        if not have_lattice_vectors and have_k_grid:
            raise RuntimeError('Found k-grid but no lattice vectors!')

        geometry_in = directory / 'geometry.in'

        write_aims(
            geometry_in,
            atoms,
            scaled,
            geo_constrain,
            write_velocities=write_velocities,
            ghosts=ghosts,
        )

        control = directory / 'control.in'

        if (
            'species_dir' not in parameters
            and profile.default_species_directory is not None
        ):
            parameters['species_dir'] = profile.default_species_directory

        write_control(control, atoms, parameters)

    def execute(self, directory, profile):
        profile.run(directory, None, self.outputname,
                    errorfile=self.errorname)

    def read_results(self, directory):
        from ase.io.aims import read_aims_results

        dst = directory / self.outputname
        return read_aims_results(dst, index=-1)

    def load_profile(self, cfg, **kwargs):
        return AimsProfile.from_config(cfg, self.name, **kwargs)

    def socketio_argv(self, profile, unixsocket, port):
        return [profile.command]

    def socketio_parameters(self, unixsocket, port):
        if port:
            use_pimd_wrapper = ('localhost', port)
        else:
            # (INET port number should be unused.)
            use_pimd_wrapper = (f'UNIX:{unixsocket}', 31415)

        return dict(use_pimd_wrapper=use_pimd_wrapper, compute_forces=True)


class Aims(GenericFileIOCalculator):
    def __init__(
        self,
        profile=None,
        directory='.',
        **kwargs,
    ):
        """Construct the FHI-aims calculator.

        The keyword arguments (kwargs) can be one of the ASE standard
        keywords: 'xc', 'kpts' and 'smearing' or any of FHI-aims'
        native keywords.


        Arguments:

        cubes: AimsCube object
            Cube file specification.

        tier: int or array of ints
            Set basis set tier for all atomic species.

        plus_u : dict
            For DFT+U. Adds a +U term to one specific shell of the species.

        kwargs : dict
            Any of the base class arguments.

        """

        super().__init__(
            template=AimsTemplate(),
            profile=profile,
            parameters=kwargs,
            directory=directory,
        )


class AimsCube:
    'Object to ensure the output of cube files, can be attached to Aims object'

    def __init__(
        self,
        origin=(0, 0, 0),
        edges=[(0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1)],
        points=(50, 50, 50),
        plots=(),
    ):
        """parameters:

        origin, edges, points:
            Same as in the FHI-aims output
        plots:
            what to print, same names as in FHI-aims"""

        self.name = 'AimsCube'
        self.origin = origin
        self.edges = edges
        self.points = points
        self.plots = plots

    def ncubes(self):
        """returns the number of cube files to output"""
        return len(self.plots)

    def move_to_base_name(self, basename):
        """when output tracking is on or the base namem is not standard,
        this routine will rename add the base to the cube file output for
        easier tracking"""
        for plot in self.plots:
            found = False
            cube = plot.split()
            if (
                cube[0] == 'total_density'
                or cube[0] == 'spin_density'
                or cube[0] == 'delta_density'
            ):
                found = True
                old_name = cube[0] + '.cube'
                new_name = basename + '.' + old_name
            if cube[0] == 'eigenstate' or cube[0] == 'eigenstate_density':
                found = True
                state = int(cube[1])
                s_state = cube[1]
                for i in [10, 100, 1000, 10000]:
                    if state < i:
                        s_state = '0' + s_state
                old_name = cube[0] + '_' + s_state + '_spin_1.cube'
                new_name = basename + '.' + old_name
            if found:
                # XXX Should not use platform dependent commands!
                os.system('mv ' + old_name + ' ' + new_name)

    def add_plot(self, name):
        """in case you forgot one ..."""
        self.plots += [name]

    def write(self, file):
        """write the necessary output to the already opened control.in"""
        file.write('output cube ' + self.plots[0] + '\n')
        file.write('   cube origin ')
        for ival in self.origin:
            file.write(str(ival) + ' ')
        file.write('\n')
        for i in range(3):
            file.write('   cube edge ' + str(self.points[i]) + ' ')
            for ival in self.edges[i]:
                file.write(str(ival) + ' ')
            file.write('\n')
        if self.ncubes() > 1:
            for i in range(self.ncubes() - 1):
                file.write('output cube ' + self.plots[i + 1] + '\n')
