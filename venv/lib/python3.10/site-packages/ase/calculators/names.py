# fmt: off

import importlib
from collections.abc import Mapping

# Recognized names of calculators sorted alphabetically:
names = ['abinit', 'ace', 'aims', 'amber', 'asap', 'castep', 'cp2k',
         'crystal', 'demon', 'demonnano', 'dftb', 'dftd3', 'dmol', 'eam',
         'elk', 'emt', 'espresso', 'exciting', 'ff', 'gamess_us',
         'gaussian', 'gpaw', 'gromacs', 'gulp', 'hotbit', 'kim',
         'lammpslib', 'lammpsrun', 'lj', 'mopac', 'morse', 'nwchem',
         'octopus', 'onetep', 'openmx', 'orca',
         'plumed', 'psi4', 'qchem', 'siesta',
         'tip3p', 'tip4p', 'turbomole', 'vasp']


builtin = {'eam', 'emt', 'ff', 'lj', 'morse', 'tip3p', 'tip4p'}


class Templates(Mapping):
    def __init__(self, dct):
        self._dct = dct

    def __iter__(self):
        return iter(self._dct)

    def __getitem__(self, index):
        importpath, clsname = self._dct[index].rsplit('.', 1)
        module = importlib.import_module(importpath)
        cls = getattr(module, clsname)
        return cls()

    def __len__(self):
        return len(self._dct)


templates = Templates({
    'abinit': 'ase.calculators.abinit.AbinitTemplate',
    'aims': 'ase.calculators.aims.AimsTemplate',
    'espresso': 'ase.calculators.espresso.EspressoTemplate',
    'octopus': 'ase.calculators.octopus.OctopusTemplate',
})
