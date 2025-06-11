# fmt: off

"""ONETEP interface for the Atomic Simulation Environment (ASE) package

T. Demeyere, T.Demeyere@soton.ac.uk (2023)

https://onetep.org"""

from copy import deepcopy

from ase.calculators.genericfileio import (
    BaseProfile,
    CalculatorTemplate,
    GenericFileIOCalculator,
    read_stdout,
)
from ase.io import read, write


class OnetepProfile(BaseProfile):
    """
    ONETEP profile class.
    """

    configvars = {'pseudo_path'}

    def __init__(self, command, pseudo_path, **kwargs):
        """
        Parameters
        ----------
        command: str
            The onetep command (not including inputfile).
        **kwargs: dict
            Additional kwargs are passed to the BaseProfile
            class.
        """
        super().__init__(command, **kwargs)
        self.pseudo_path = pseudo_path

    def version(self):
        lines = read_stdout(self._split_command)
        return self.parse_version(lines)

    def parse_version(lines):
        return '1.0.0'

    def get_calculator_command(self, inputfile):
        return [str(inputfile)]


class OnetepTemplate(CalculatorTemplate):
    _label = 'onetep'

    def __init__(self, append):
        super().__init__(
            'ONETEP',
            implemented_properties=[
                'energy',
                'free_energy',
                'forces',
                'stress'])
        self.inputname = f'{self._label}.dat'
        self.outputname = f'{self._label}.out'
        self.errorname = f'{self._label}.err'
        self.append = append

    def execute(self, directory, profile):
        profile.run(directory, self.inputname, self.outputname,
                    self.errorname, append=self.append)

    def read_results(self, directory):
        output_path = directory / self.outputname
        atoms = read(output_path, format='onetep-out')
        return dict(atoms.calc.properties())

    def write_input(self, profile, directory, atoms, parameters, properties):
        input_path = directory / self.inputname

        parameters = deepcopy(parameters)

        keywords = parameters.get('keywords', {})
        keywords.setdefault('pseudo_path', profile.pseudo_path)
        parameters['keywords'] = keywords

        write(input_path, atoms, format='onetep-in',
              properties=properties, **parameters)

    def load_profile(self, cfg, **kwargs):
        return OnetepProfile.from_config(cfg, self.name, **kwargs)


class Onetep(GenericFileIOCalculator):
    """
    Class for the ONETEP calculator, uses ase/io/onetep.py.
    Need the env variable "ASE_ONETEP_COMMAND" defined to
    properly work. All other options are passed in kwargs.

    Parameters
    ----------
    autorestart : Bool
        When activated, manages restart keywords automatically.
    append: Bool
        Append to output instead of overwriting.
    directory: str
        Directory where to run the calculation(s).
    keywords: dict
        Dictionary with ONETEP keywords to write,
        keywords with lists as values will be
        treated like blocks, with each element
        of list being a different line.
    xc: str
        DFT xc to use e.g (PBE, RPBE, ...).
    ngwfs_count: int|list|dict
        Behaviour depends on the type:
            int: every species will have this amount
            of ngwfs.
            list: list of int, will be attributed
            alphabetically to species:
            dict: keys are species name(s),
            value are their number:
    ngwfs_radius: int|list|dict
        Behaviour depends on the type:
            float: every species will have this radius.
            list: list of float, will be attributed
            alphabetically to species:
            [10.0, 9.0]
            dict: keys are species name(s),
            value are their radius:
            {'Na': 9.0, 'Cl': 10.0}
    pseudopotentials: list|dict
        Behaviour depends on the type:
            list: list of string(s), will be attributed
            alphabetically to specie(s):
            ['Cl.usp', 'Na.usp']
            dict: keys are species name(s) their
            value are the pseudopotential file to use:
            {'Na': 'Na.usp', 'Cl': 'Cl.usp'}
    pseudo_path: str
        Where to look for pseudopotential, correspond
        to the pseudo_path keyword of ONETEP.

        .. note::
           write_forces is always turned on by default
           when using this interface.

        .. note::
           Little to no check is performed on the keywords provided by the user
           via the keyword dictionary, it is the user responsibility that they
           are valid ONETEP keywords.
    """

    def __init__(
            self,
            *,
            profile=None,
            directory='.',
            **kwargs):

        self.keywords = kwargs.get('keywords', None)
        self.template = OnetepTemplate(
            append=kwargs.pop('append', False)
        )

        super().__init__(profile=profile, template=self.template,
                         directory=directory,
                         parameters=kwargs)
