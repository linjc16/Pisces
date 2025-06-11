# fmt: off

from pathlib import Path

from ase.calculators.abc import GetOutputsMixin
from ase.calculators.calculator import FileIOCalculator
from ase.io import write
from ase.io.elk import ElkReader


class ELK(FileIOCalculator, GetOutputsMixin):
    _legacy_default_command = 'elk > elk.out'
    implemented_properties = ['energy', 'forces']
    ignored_changes = {'pbc'}
    discard_results_on_any_change = True

    fileio_rules = FileIOCalculator.ruleset(
        stdout_name='elk.out')

    def __init__(self, **kwargs):
        """Construct ELK calculator.

        The keyword arguments (kwargs) can be one of the ASE standard
        keywords: 'xc', 'kpts' and 'smearing' or any of ELK'
        native keywords.
        """

        super().__init__(**kwargs)

    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, atoms, properties, system_changes)

        parameters = dict(self.parameters)
        if 'forces' in properties:
            parameters['tforce'] = True

        directory = Path(self.directory)
        write(directory / 'elk.in', atoms, parameters=parameters,
              format='elk-in')

    def read_results(self):
        from ase.outputs import Properties
        reader = ElkReader(self.directory)
        dct = dict(reader.read_everything())

        converged = dct.pop('converged')
        if not converged:
            raise RuntimeError('Did not converge')

        # (Filter results thorugh Properties for error detection)
        props = Properties(dct)
        self.results = dict(props)

    def _outputmixin_get_results(self):
        return self.results
