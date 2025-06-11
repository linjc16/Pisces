# fmt: off

import shlex
from abc import ABC, abstractmethod
from contextlib import ExitStack
from os import PathLike
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Set

from ase.calculators.abc import GetOutputsMixin
from ase.calculators.calculator import (
    BadConfiguration,
    BaseCalculator,
    _validate_command,
)
from ase.config import cfg as _cfg

link_calculator_docs = (
    "https://wiki.fysik.dtu.dk/ase/ase/calculators/"
    "calculators.html#calculator-configuration"
)


class BaseProfile(ABC):
    configvars: Set[str] = set()

    def __init__(self, command):
        self.command = _validate_command(command)

    @property
    def _split_command(self):
        return shlex.split(self.command)

    def get_command(self, inputfile, calc_command=None) -> List[str]:
        """
        Get the command to run. This should be a list of strings.

        Parameters
        ----------
        inputfile : str
        calc_command: list[str]: calculator command (used for sockets)

        Returns
        -------
        list of str
            The command to run.
        """
        if calc_command is None:
            calc_command = self.get_calculator_command(inputfile)
        return [*self._split_command, *calc_command]

    @abstractmethod
    def get_calculator_command(self, inputfile):
        """
        The calculator specific command as a list of strings.

        Parameters
        ----------
        inputfile : str

        Returns
        -------
        list of str
            The command to run.
        """

    def run(
        self, directory: Path, inputfile: Optional[str],
        outputfile: str, errorfile: Optional[str] = None,
        append: bool = False
    ) -> None:
        """
        Run the command in the given directory.

        Parameters
        ----------
        directory : pathlib.Path
            The directory to run the command in.
        inputfile : Optional[str]
            The name of the input file.
        outputfile : str
            The name of the output file.
        errorfile: Optional[str]
            the stderror file
        append: bool
            if True then use append mode
        """

        import os
        from subprocess import check_call

        argv_command = self.get_command(inputfile)
        mode = 'wb' if not append else 'ab'

        with ExitStack() as stack:
            output_path = directory / outputfile
            fd_out = stack.enter_context(open(output_path, mode))
            if errorfile is not None:
                error_path = directory / errorfile
                fd_err = stack.enter_context(open(error_path, mode))
            else:
                fd_err = None
            check_call(
                argv_command,
                cwd=directory,
                stdout=fd_out,
                stderr=fd_err,
                env=os.environ,
            )

    @abstractmethod
    def version(self):
        """Get the version of the code.

        Returns
        -------
        str
            The version of the code.
        """

    @classmethod
    def from_config(cls, cfg, section_name):
        """Create a profile from a configuration file.

        Parameters
        ----------
        cfg : ase.config.Config
            The configuration object.
        section_name : str
            The name of the section in the configuration file. E.g. the name
            of the template that this profile is for.

        Returns
        -------
        BaseProfile
            The profile object.
        """
        section = cfg.parser[section_name]
        command = section['command']

        kwargs = {
            varname: section[varname]
            for varname in cls.configvars if varname in section
        }

        try:
            return cls(command=command, **kwargs)
        except TypeError as err:
            raise BadConfiguration(*err.args)


def read_stdout(args, createfile=None):
    """Run command in tempdir and return standard output.

    Helper function for getting version numbers of DFT codes.
    Most DFT codes don't implement a --version flag, so in order to
    determine the code version, we just run the code until it prints
    a version number."""
    import tempfile
    from subprocess import PIPE, Popen

    with tempfile.TemporaryDirectory() as directory:
        if createfile is not None:
            path = Path(directory) / createfile
            path.touch()
        proc = Popen(
            args,
            stdout=PIPE,
            stderr=PIPE,
            stdin=PIPE,
            cwd=directory,
            encoding='utf-8',  # Make this a parameter if any non-utf8/ascii
        )
        stdout, _ = proc.communicate()
        # Exit code will be != 0 because there isn't an input file
    return stdout


class CalculatorTemplate(ABC):
    def __init__(self, name: str, implemented_properties: Iterable[str]):
        self.name = name
        self.implemented_properties = frozenset(implemented_properties)

    @abstractmethod
    def write_input(self, profile, directory, atoms, parameters, properties):
        ...

    @abstractmethod
    def execute(self, directory, profile):
        ...

    @abstractmethod
    def read_results(self, directory: PathLike) -> Mapping[str, Any]:
        ...

    @abstractmethod
    def load_profile(self, cfg):
        ...

    def socketio_calculator(
        self,
        profile,
        parameters,
        directory,
        # We may need quite a few socket kwargs here
        # if we want to expose all the timeout etc. from
        # SocketIOCalculator.
        unixsocket=None,
        port=None,
    ):
        import os
        from subprocess import Popen

        from ase.calculators.socketio import SocketIOCalculator

        if port and unixsocket:
            raise TypeError(
                'For the socketio_calculator only a UNIX '
                '(unixsocket) or INET (port) socket can be used'
                ' not both.'
            )

        if not port and not unixsocket:
            raise TypeError(
                'For the socketio_calculator either a '
                'UNIX (unixsocket) or INET (port) socket '
                'must be used'
            )

        if not (
            hasattr(self, 'socketio_argv')
            and hasattr(self, 'socketio_parameters')
        ):
            raise TypeError(
                f'Template {self} does not implement mandatory '
                'socketio_argv() and socketio_parameters()'
            )

        # XXX need socketio ABC or something
        argv = profile.get_command(
            inputfile=None,
            calc_command=self.socketio_argv(profile, unixsocket, port)
        )
        parameters = {
            **self.socketio_parameters(unixsocket, port),
            **parameters,
        }

        # Not so elegant that socket args are passed to this function
        # via socketiocalculator when we could make a closure right here.
        def launch(atoms, properties, port, unixsocket):
            directory.mkdir(exist_ok=True, parents=True)

            self.write_input(
                atoms=atoms,
                profile=profile,
                parameters=parameters,
                properties=properties,
                directory=directory,
            )

            with open(directory / self.outputname, 'w') as out_fd:
                return Popen(argv, stdout=out_fd, cwd=directory, env=os.environ)

        return SocketIOCalculator(
            launch_client=launch, unixsocket=unixsocket, port=port
        )


class GenericFileIOCalculator(BaseCalculator, GetOutputsMixin):
    cfg = _cfg

    def __init__(
        self,
        *,
        template,
        profile,
        directory,
        parameters=None,
    ):
        self.template = template
        if profile is None:
            if template.name not in self.cfg.parser:
                raise BadConfiguration(
                    f"No configuration of '{template.name}'. "
                    f"See '{link_calculator_docs}'"
                )
            try:
                profile = template.load_profile(self.cfg)
            except Exception as err:
                configvars = self.cfg.as_dict()
                raise BadConfiguration(
                    f'Failed to load section [{template.name}] '
                    f'from configuration: {configvars}'
                ) from err

        self.profile = profile

        # Maybe we should allow directory to be a factory, so
        # calculators e.g. produce new directories on demand.
        self.directory = Path(directory)
        super().__init__(parameters)

    def set(self, *args, **kwargs):
        raise RuntimeError(
            'No setting parameters for now, please.  '
            'Just create new calculators.'
        )

    def __repr__(self):
        return f'{type(self).__name__}({self.template.name})'

    @property
    def implemented_properties(self):
        return self.template.implemented_properties

    @property
    def name(self):
        return self.template.name

    def write_inputfiles(self, atoms, properties):
        # SocketIOCalculators like to write inputfiles
        # without calculating.
        self.directory.mkdir(exist_ok=True, parents=True)
        self.template.write_input(
            profile=self.profile,
            atoms=atoms,
            parameters=self.parameters,
            properties=properties,
            directory=self.directory,
        )

    def calculate(self, atoms, properties, system_changes):
        self.write_inputfiles(atoms, properties)
        self.template.execute(self.directory, self.profile)
        self.results = self.template.read_results(self.directory)
        # XXX Return something useful?

    def _outputmixin_get_results(self):
        return self.results

    def socketio(self, **socketkwargs):
        return self.template.socketio_calculator(
            directory=self.directory,
            parameters=self.parameters,
            profile=self.profile,
            **socketkwargs,
        )
