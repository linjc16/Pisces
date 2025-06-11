# fmt: off

"""This module defines an ASE interface to ABINIT.

http://www.abinit.org/
"""

from pathlib import Path
from subprocess import check_output

import ase.io.abinit as io
from ase.calculators.genericfileio import (
    BaseProfile,
    CalculatorTemplate,
    GenericFileIOCalculator,
)


class AbinitProfile(BaseProfile):
    configvars = {'pp_paths'}

    def __init__(self, command, *, pp_paths=None, **kwargs):
        super().__init__(command, **kwargs)
        # XXX pp_paths is a raw configstring when it gets here.
        # All the config stuff should have been loaded somehow by now,
        # so this should be refactored.
        if isinstance(pp_paths, str):
            pp_paths = [path for path in pp_paths.splitlines() if path]
        if pp_paths is None:
            pp_paths = []
        self.pp_paths = pp_paths

    def version(self):
        argv = [*self._split_command, '--version']
        return check_output(argv, encoding='ascii').strip()

    def get_calculator_command(self, inputfile):
        return [str(inputfile)]

    def socketio_argv_unix(self, socket):
        # XXX clean up the passing of the inputfile
        inputfile = AbinitTemplate().input_file
        return [inputfile, '--ipi', f'{socket}:UNIX']


class AbinitTemplate(CalculatorTemplate):
    _label = 'abinit'  # Controls naming of files within calculation directory

    def __init__(self):
        super().__init__(
            name='abinit',
            implemented_properties=[
                'energy',
                'free_energy',
                'forces',
                'stress',
                'magmom',
            ],
        )

        # XXX superclass should require inputname and outputname

        self.inputname = f'{self._label}.in'
        self.outputname = f'{self._label}.log'
        self.errorname = f'{self._label}.err'

    def execute(self, directory, profile) -> None:
        profile.run(directory, self.inputname, self.outputname,
                    errorfile=self.errorname)

    def write_input(self, profile, directory, atoms, parameters, properties):
        directory = Path(directory)
        parameters = dict(parameters)
        pp_paths = parameters.pop('pp_paths', profile.pp_paths)
        assert pp_paths is not None

        kw = dict(xc='LDA', smearing=None, kpts=None, raw=None, pps='fhi')
        kw.update(parameters)

        io.prepare_abinit_input(
            directory=directory,
            atoms=atoms,
            properties=properties,
            parameters=kw,
            pp_paths=pp_paths,
        )

    def read_results(self, directory):
        return io.read_abinit_outputs(directory, self._label)

    def load_profile(self, cfg, **kwargs):
        return AbinitProfile.from_config(cfg, self.name, **kwargs)

    def socketio_argv(self, profile, unixsocket, port):
        # XXX This handling of --ipi argument is used by at least two
        # calculators, should refactor if needed yet again
        if unixsocket:
            ipi_arg = f'{unixsocket}:UNIX'
        else:
            ipi_arg = f'localhost:{port:d}'

        return profile.get_calculator_command(self.inputname) + [
            '--ipi',
            ipi_arg,
        ]

    def socketio_parameters(self, unixsocket, port):
        return dict(ionmov=28, expert_user=1, optcell=2)


class Abinit(GenericFileIOCalculator):
    """Class for doing ABINIT calculations.

    The default parameters are very close to those that the ABINIT
    Fortran code would use.  These are the exceptions::

      calc = Abinit(xc='LDA', ecut=400, toldfe=1e-5)
    """

    def __init__(
        self,
        *,
        profile=None,
        directory='.',
        **kwargs,
    ):
        """Construct ABINIT-calculator object.

        Examples
        ========
        Use default values:

        >>> h = Atoms('H', calculator=Abinit(ecut=200, toldfe=0.001))
        >>> h.center(vacuum=3.0)
        >>> e = h.get_potential_energy()

        """

        super().__init__(
            template=AbinitTemplate(),
            profile=profile,
            directory=directory,
            parameters=kwargs,
        )
