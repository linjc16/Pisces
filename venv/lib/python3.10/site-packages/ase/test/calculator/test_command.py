import os
import subprocess

import pytest

from ase import Atoms
from ase.calculators.calculator import CalculatorSetupError

"""
These tests monkeypatch Popen so as to abort execution and verify that
a particular command as executed.

They test several cases:

 * command specified by environment
 * command specified via keyword
 * command not specified, with two possible behaviours:
   - command defaults to particular value
   - calculator raises CalculatorSetupError

(We do not bother to check e.g. conflicting combinations.)
"""


class InterceptedCommand(BaseException):
    def __init__(self, command):
        self.command = command


def mock_popen(command, shell=False, cwd=None, **kwargs):
    assert isinstance(command, str) is shell
    raise InterceptedCommand(command)


# Other calculators:
#  * cp2k uses command but is not FileIOCalculator
#  * turbomole hardcodes multiple commands but does not use command keyword


# Parameters for each calculator -- whatever it takes trigger a calculation
# without crashing first.
calculators = {
    'ace': {},
    'amber': {},
    'castep': dict(keyword_tolerance=3),
    'cp2k': {},
    'crystal': {},
    'demon': dict(basis_path='hello'),
    'demonnano': dict(input_arguments={},
                      basis_path='hello'),
    'dftb': {},
    'dftd3': {},
    'dmol': {},
    'elk': {},
    'gamess_us': {},
    'gaussian': {},
    'gromacs': {},
    'gulp': {},
    'lammpsrun': {},
    'mopac': {},
    'nwchem': {},
    'onetep': {},
    'openmx': dict(data_path='.', dft_data_year='13'),
    'psi4': {},
    'qchem': {},
    'siesta': dict(pseudo_path='.'),
    'turbomole': {},
    'vasp': {},
}


@pytest.fixture(autouse=True)
def miscellaneous_hacks(monkeypatch, tmp_path):
    from ase.calculators.calculator import FileIOCalculator
    from ase.calculators.crystal import CRYSTAL
    from ase.calculators.demon import Demon
    from ase.calculators.dftb import Dftb
    from ase.calculators.gamess_us import GAMESSUS
    from ase.calculators.gulp import GULP
    from ase.calculators.openmx import OpenMX
    from ase.calculators.siesta.siesta import FDFWriter
    from ase.calculators.vasp import Vasp

    def do_nothing(returnval=None):
        def mock_function(*args, **kwargs):
            return returnval
        return mock_function

    # Monkeypatches can be pretty dangerous because someone might obtain
    # a reference to the monkeypatched value before the patch is undone.
    #
    # We should try to refactor so we can avoid all the monkeypatches.

    monkeypatch.setattr(Demon, 'link_file', do_nothing())
    monkeypatch.setattr(CRYSTAL, '_write_crystal_in', do_nothing())
    monkeypatch.setattr(Dftb, 'write_dftb_in', do_nothing())

    # It calls super, but we'd like to skip the userscr handling:
    monkeypatch.setattr(GAMESSUS, 'calculate', FileIOCalculator.calculate)
    monkeypatch.setattr(GULP, 'library_check', do_nothing())

    # Attempts to read too many files.
    monkeypatch.setattr(OpenMX, 'write_input', do_nothing())

    monkeypatch.setattr(FDFWriter, 'link_pseudos_into_directory', do_nothing())
    monkeypatch.setattr(Vasp, '_build_pp_list', do_nothing(returnval=[]))


def mkcalc(name, **kwargs):
    from ase.calculators.calculator import get_calculator_class
    cls = get_calculator_class(name)
    kwargs = {**calculators[name], **kwargs}
    return cls(**kwargs)


@pytest.fixture(autouse=True)
def mock_subprocess_popen(monkeypatch):
    monkeypatch.setattr(subprocess, 'Popen', mock_popen)


def intercept_command(name, **kwargs):
    atoms = Atoms('H', pbc=True)
    atoms.center(vacuum=3.0)
    try:
        # cp2k runs cp2k_shell already in the constructor, and it has a right
        # to choose to do so.  Maybe other calculators do this as well.
        # So we include both lines in "try".
        atoms.calc = mkcalc(name, **kwargs)
        atoms.get_potential_energy()
    except InterceptedCommand as err:
        print(err.command)
        return err.command


envvars = {
    'ace': 'ASE_ACE_COMMAND',
    'amber': 'ASE_AMBER_COMMAND',
    'castep': 'CASTEP_COMMAND',
    'cp2k': 'ASE_CP2K_COMMAND',
    'crystal': 'ASE_CRYSTAL_COMMAND',
    'demon': 'ASE_DEMON_COMMAND',
    'demonnano': 'ASE_DEMONNANO_COMMAND',
    'dftb': 'DFTB_COMMAND',
    'dftd3': 'ASE_DFTD3_COMMAND',
    'dmol': 'DMOL_COMMAND',  # XXX Crashes when it runs along other tests
    'elk': 'ASE_ELK_COMMAND',
    'gamess_us': 'ASE_GAMESSUS_COMMAND',
    'gaussian': 'ASE_GAUSSIAN_COMMAND',
    'gromacs': 'ASE_GROMACS_COMMAND',
    'gulp': 'ASE_GULP_COMMAND',
    'lammpsrun': 'ASE_LAMMPSRUN_COMMAND',
    'mopac': 'ASE_MOPAC_COMMAND',
    'nwchem': 'ASE_NWCHEM_COMMAND',
    # 'onetep': 'ASE_ONETEP_COMMAND',  do we need to be compatible here?
    'openmx': 'ASE_OPENMX_COMMAND',  # fails in get_dft_data_year
    # 'psi4', <-- has command but is Calculator
    # 'qchem': 'ASE_QCHEM_COMMAND',  # ignores environment
    'siesta': 'ASE_SIESTA_COMMAND',
    # 'turbomole': turbomole is not really a calculator
    'vasp': 'ASE_VASP_COMMAND',
}


dftd3_boilerplate = (
    'ase_dftd3.POSCAR -func pbe -grad -pbc -cnthr 40.0 -cutoff 95.0 -zero')


def get_expected_command(command, name, tmp_path, from_envvar):
    if name == 'castep':
        return f'{command} castep'  # crazy

    if name == 'dftb' and from_envvar:
        # dftb modifies DFTB_COMMAND from envvar but not if given as keyword
        return f'{command} > dftb.out'

    if name == 'dftd3':
        return f'{command} {dftd3_boilerplate}'.split()

    if name == 'dmol' and from_envvar:
        return f'{command} tmp > tmp.out'

    if name == 'gromacs':
        return (f'{command} mdrun -s gromacs.tpr -o gromacs.trr '
                '-e gromacs.edr -g gromacs.log -c gromacs.g96  > MM.log 2>&1')

    if name == 'lammpsrun':
        # lammpsrun does not use a shell command
        return [*command.split(), '-echo', 'log',
                '-screen', 'none', '-log', '/dev/stdout']

    if name == 'onetep':
        return [*command.split(), 'onetep.dat']

    if name == 'openmx':
        # openmx converts the stream target to an abspath, so the command
        # will vary depending on the tempdir we're running in.
        return f'{command} openmx.dat > {os.path.join(tmp_path, "openmx.log")}'

    return command


@pytest.mark.parametrize('name', list(envvars))
def test_envvar_command(monkeypatch, name, tmp_path):
    command = 'dummy shell command from environment'

    if name == 'cp2k':
        command += 'cp2k_shell'  # circumvent sanity check

    expected_command = get_expected_command(command, name, tmp_path,
                                            from_envvar=True)
    monkeypatch.setenv(envvars[name], command)
    actual_command = intercept_command(name)
    assert actual_command == expected_command


def keyword_calculator_list():
    skipped = {
        'turbomole',  # commands are hardcoded in turbomole
        'qchem',  # qchem does something entirely different.  wth
        'psi4',  # needs external package
        'onetep',  # (takes profile)
    }
    return sorted(set(calculators) - skipped)


# castep uses another keyword than normal
command_keywords = {'castep': 'castep_command'}


@pytest.mark.parametrize('name', keyword_calculator_list())
def test_keyword_command(name, tmp_path):
    command = 'dummy command via keyword'

    if name == 'cp2k':
        command += ' cp2k_shell'  # circumvent sanity check

    expected_command = get_expected_command(command, name, tmp_path,
                                            from_envvar=False)

    # normally {'command': command}
    commandkwarg = {command_keywords.get(name, 'command'): command}
    print(intercept_command(name, **commandkwarg))
    assert intercept_command(name, **commandkwarg) == expected_command


# Calculators that (somewhat unwisely) have a hardcoded default command
default_commands = {
    'amber': ('sander -O  -i mm.in -o mm.out -p mm.top -c mm.crd '
              '-r mm_dummy.crd'),
    'castep': 'castep castep',  # wth?
    'cp2k': 'cp2k_shell',
    'dftb': 'dftb+ > dftb.out',
    'dftd3': f'dftd3 {dftd3_boilerplate}'.split(),
    'elk': 'elk > elk.out',
    'gamess_us': 'rungms gamess_us.inp > gamess_us.log 2> gamess_us.err',
    'gaussian': 'g16 < Gaussian.com > Gaussian.log',
    'gulp': 'gulp < gulp.gin > gulp.got',
    'lammpsrun': ['lammps', '-echo', 'log', '-screen', 'none',
                  '-log', '/dev/stdout'],
    'mopac': 'mopac mopac.mop 2> /dev/null',
    'nwchem': 'nwchem nwchem.nwi > nwchem.nwo',
    # 'openmx': '',  # command contains full path which is variable
    'qchem': 'qchem qchem.inp qchem.out',
    'siesta': 'siesta < siesta.fdf > siesta.out',
}

# Calculators that raise error if command not set
calculators_which_raise = [
    'ace',
    'demonnano',
    'crystal',
    'demon',
    'dmol',
    'gromacs',
    # Fixme: onetep raises AttributError
    # Should be handled universally by GenericFileIO system
    'vasp',
]


@pytest.mark.parametrize('name', list(default_commands))
def test_nocommand_default(name, monkeypatch, override_config):
    if name in envvars:
        monkeypatch.delenv(envvars[name], raising=False)

    assert intercept_command(name) == default_commands[name]


@pytest.mark.parametrize('name', calculators_which_raise)
def test_nocommand_raise(name, monkeypatch, override_config):
    if name in envvars:
        monkeypatch.delenv(envvars[name], raising=False)

    with pytest.raises(CalculatorSetupError):
        intercept_command(name)
