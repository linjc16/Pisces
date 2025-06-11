import os
import shutil
import tempfile
import zlib
from pathlib import Path
from subprocess import PIPE, Popen, check_output

import numpy as np
import pytest

import ase
from ase.config import Config, cfg
from ase.dependencies import all_dependencies
from ase.test.factories import (
    CalculatorInputs,
    NoSuchCalculator,
    factory_classes,
    get_factories,
    legacy_factory_calculator_names,
    make_factory_fixture,
    parametrize_calculator_tests,
)
from ase.test.factories import factory as factory_deco
from ase.utils import get_python_package_path_description, seterr, workdir

helpful_message = """\
 * Use --calculators option to select calculators.

 * See "ase test --help-calculators" on how to configure calculators.

 * This listing only includes external calculators known by the test
   system.  Others are "configured" by setting an environment variable
   like "ASE_xxx_COMMAND" in order to allow tests to run.  Please see
   the documentation of that individual calculator.
"""


@pytest.fixture(scope='session')
def testconfig():
    from ase.test.factories import MachineInformation
    return MachineInformation().cfg


def pytest_report_header(config, start_path):
    yield from library_header()
    yield ''
    yield from calculators_header(config)


def library_header():
    yield ''
    yield 'Libraries'
    yield '========='
    yield ''
    for name, path in all_dependencies():
        yield f'{name:24} {path}'


def calculators_header(config):
    try:
        factories = get_factories(config)
    except NoSuchCalculator as err:
        pytest.exit(f'No such calculator: {err}')

    machine_info = factories.machine_info
    configpaths = machine_info.cfg.paths
    # XXX FIXME may not be installed
    module = machine_info.datafiles_module

    yield ''
    yield 'Calculators'
    yield '==========='

    if not configpaths:
        configtext = 'No configuration file specified'
    else:
        configtext = ', '.join(str(path) for path in configpaths)
    yield f'Config: {configtext}'

    if module is None:
        datafiles_text = 'ase-datafiles package not installed'
    else:
        datafiles_text = str(Path(module.__file__).parent)

    yield f'Datafiles: {datafiles_text}'
    yield ''

    for name in sorted(factory_classes):
        if name in factories.builtin_calculators:
            # Not interesting to test presence of builtin calculators.
            continue

        factory = factories.factories.get(name)

        if factory is None:
            why_not = factories.why_not[name]
            configinfo = f'not installed: {why_not}'
        else:
            # Some really ugly hacks here:
            if hasattr(factory, 'importname'):
                pass
                # We want an to report from where we import calculators
                # that are defined in Python, but that's currently disabled.
                #
                # import importlib
                # XXXX reenable me somehow
                # module = importlib.import_module(factory.importname)
                # configinfo = get_python_package_path_description(module)
            else:
                configtokens = []
                for varname, variable in vars(factory).items():
                    configtokens.append(f'{varname}={variable}')
                configinfo = ', '.join(configtokens)

        enabled = factories.enabled(name)
        if enabled:
            version = '<unknown version>'
            if hasattr(factory, 'version'):
                try:
                    version = factory.version()
                except Exception:
                    # XXX Add a test for the version numbers so that
                    # will fail without crashing the whole test suite.
                    pass
            name = f'{name}-{version}'

        run = '[x]' if enabled else '[ ]'
        line = f'  {run} {name:16} {configinfo}'
        yield line

    yield ''
    yield helpful_message
    yield ''

    # (Where should we do this check?)
    for name in factories.requested_calculators:
        if not factories.is_adhoc(name) and not factories.installed(name):
            pytest.exit(f'Calculator "{name}" is not installed.  '
                        'Please run "ase test --help-calculators" on how '
                        'to install calculators')


@pytest.fixture(scope='session', autouse=True)
def sessionlevel_testing_path():
    # We cd into a tempdir so tests and fixtures won't create files
    # elsewhere (e.g. in the unsuspecting user's directory).
    #
    # However we regard it as an error if the tests leave files there,
    # because they can access each others' files and hence are not
    # independent.  Therefore we want them to explicitly use the
    # "testdir" fixture which ensures that each has a clean directory.
    #
    # To prevent tests from writing files, we chmod the directory.
    # But if the tests are killed, we cannot clean it up and it will
    # disturb other pytest runs if we use the pytest tempdir factory.
    #
    # So we use the tempfile module for this temporary directory.
    with tempfile.TemporaryDirectory(prefix='ase-test-workdir-') as tempdir:
        path = Path(tempdir)
        path.chmod(0o555)
        with workdir(path):
            yield path
        path.chmod(0o755)


@pytest.fixture(autouse=False)
def testdir(tmp_path):
    # Pytest can on some systems provide a Path from pathlib2.  Normalize:
    path = Path(str(tmp_path))
    print(f'Testdir: {path}')
    with workdir(path, mkdir=True):
        yield tmp_path


@pytest.fixture()
def allraise():
    with seterr(all='raise'):
        yield


@pytest.fixture()
def KIM():
    pytest.importorskip('kimpy')
    from ase.calculators.kim import KIM as _KIM
    from ase.calculators.kim.exceptions import KIMModelNotFound

    def KIM(*args, **kwargs):
        try:
            return _KIM(*args, **kwargs)
        except KIMModelNotFound:
            pytest.skip('KIM tests require the example KIM models.  '
                        'These models are available if the KIM API is '
                        'built from source.  See https://openkim.org/kim-api/'
                        'for more information.')

    return KIM


@pytest.fixture(scope='session')
def tkinter():
    import tkinter
    try:
        tkinter.Tk()
    except tkinter.TclError as err:
        pytest.skip(f'no tkinter: {err}')


@pytest.fixture(autouse=True)
def _plt_close_figures():
    import matplotlib.pyplot as plt
    yield
    fignums = plt.get_fignums()
    for fignum in fignums:
        plt.close(fignum)


@pytest.fixture(scope='session', autouse=True)
def _plt_use_agg():
    import matplotlib
    matplotlib.use('Agg')


@pytest.fixture(scope='session')
def plt(_plt_use_agg):
    import matplotlib.pyplot as plt
    return plt


@pytest.fixture()
def figure(plt):
    fig = plt.figure()
    yield fig
    plt.close(fig)


@pytest.fixture(scope='session')
def psycopg2():
    return pytest.importorskip('psycopg2')


@pytest.fixture(scope='session')
def factories(pytestconfig):
    return get_factories(pytestconfig)


# XXX Maybe we should not have individual factory fixtures, we could use
# the decorator @pytest.mark.calculator(name) instead.
abinit_factory = make_factory_fixture('abinit')
cp2k_factory = make_factory_fixture('cp2k')
dftb_factory = make_factory_fixture('dftb')
espresso_factory = make_factory_fixture('espresso')
gpaw_factory = make_factory_fixture('gpaw')
mopac_factory = make_factory_fixture('mopac')
octopus_factory = make_factory_fixture('octopus')
siesta_factory = make_factory_fixture('siesta')
orca_factory = make_factory_fixture('orca')


def make_dummy_factory(name):
    @factory_deco(name)
    class Factory:
        def __init__(self, cfg):
            self.cfg = cfg

        def calc(self, **kwargs):
            from ase.calculators.calculator import get_calculator_class
            cls = get_calculator_class(name)
            return cls(**kwargs)

        @classmethod
        def fromconfig(cls, config):
            return cls()

    Factory.__name__ = f'{name.upper()}Factory'
    globalvars = globals()
    globalvars[f'{name}_factory'] = make_factory_fixture(name)


for name in legacy_factory_calculator_names:
    make_dummy_factory(name)


@pytest.fixture()
def factory(request, factories):
    name, kwargs = request.param
    if not factories.installed(name):
        pytest.skip(f'Not installed: {name}')
    if not factories.enabled(name):
        pytest.skip(f'Not enabled: {name}')
    # TODO: nice reporting of installedness and configuration
    # if name in factories.builtin_calculators & factories.datafile_calculators:
    #    if not factories.datafiles_module:
    #        pytest.skip('ase-datafiles package not installed')
    try:
        factory = factories[name]
    except KeyError:
        pytest.skip(f'Not configured: {name}')
    return CalculatorInputs(factory, kwargs)


def check_missing_init(module):
    # We don't like missing __init__.py because pytest imports those
    # as toplevel modules, which can cause clashes.
    if not module.__name__.startswith('ase.test.'):
        raise RuntimeError(
            f'Test module {module.__name__} at {module.__file__} does not '
            'start with "ase.test".  Maybe __init__.py is missing?')


def pytest_generate_tests(metafunc):
    check_missing_init(metafunc.module)

    parametrize_calculator_tests(metafunc)

    if 'seed' in metafunc.fixturenames:
        seeds = metafunc.config.getoption('seed')
        if len(seeds) == 0:
            seeds = [0]
        else:
            seeds = list(map(int, seeds))
        metafunc.parametrize('seed', seeds)


@pytest.fixture()
def override_config(monkeypatch):
    parser = Config().parser
    monkeypatch.setattr(cfg, 'parser', parser)
    return cfg


@pytest.fixture()
def config_file(tmp_path, monkeypatch, override_config):
    dummy_config = """\
[parallel]
runner = mpirun
nprocs = -np
stdout = --output-filename

[abinit]
command = /home/ase/calculators/abinit/bin/abinit
abipy_mrgddb = /home/ase/calculators/abinit/bin/mrgddb
abipy_anaddb = /home/ase/calculators/abinit/bin/anaddb

[cp2k]
cp2k_shell = cp2k_shell
cp2k_main = cp2k

[dftb]
command = /home/ase/calculators/dftbplus/bin/dftb+

[dftd3]
command = /home/ase/calculators/dftd3/bin/dftd3

[elk]
command = /usr/bin/elk-lapw

[espresso]
command = /home/ase/calculators/espresso/bin/pw.x
pseudo_dir = /home/ase/.local/lib/python3.10/site-packages/asetest/\
datafiles/espresso/gbrv-lda-espresso

[exciting]
command = /home/ase/calculators/exciting/bin/exciting

[gromacs]
command = gmx

[lammps]
command = /home/ase/calculators/lammps/bin/lammps

[mopac]
command = /home/ase/calculators/mopac/bin/mopac

[nwchem]
command = /home/ase/calculators/nwchem/bin/nwchem

[octopus]
command = /home/ase/calculators/octopus/bin/octopus

[openmx]
# command = /usr/bin/openmx

[siesta]
command = /home/ase/calculators/siesta/bin/siesta
"""

    override_config.parser.read_string(dummy_config)


class CLI:
    def __init__(self, calculators):
        self.calculators = calculators

    def ase(self, *args, expect_fail=False):
        environment = {}
        environment.update(os.environ)
        # Prevent failures due to Tkinter-related default backend
        # on systems without Tkinter.
        environment['MPLBACKEND'] = 'Agg'

        proc = Popen(['ase', '-T'] + list(args),
                     stdout=PIPE, stdin=PIPE,
                     env=environment)
        stdout, _ = proc.communicate(b'')
        status = proc.wait()
        assert (status != 0) == expect_fail
        return stdout.decode('utf-8')

    def shell(self, command, calculator_name=None):
        # Please avoid using shell comamnds including this method!
        if calculator_name is not None:
            self.calculators.require(calculator_name)

        actual_command = ' '.join(command.split('\n')).strip()
        output = check_output(actual_command, shell=True)
        return output.decode()


@pytest.fixture(scope='session')
def cli(factories):
    return CLI(factories)


@pytest.fixture(scope='session')
def datadir():
    test_basedir = Path(__file__).parent
    return test_basedir / 'testdata'


@pytest.fixture(scope='session')
def asap3():
    return pytest.importorskip('asap3')


@pytest.fixture(autouse=True)
def arbitrarily_seed_rng(request):
    # We want tests to not use global stuff such as np.random.seed().
    # But they do.
    #
    # So in lieu of (yet) fixing it, we reseed and unseed the random
    # state for every test.  That makes each test deterministic if it
    # uses random numbers without seeding, but also repairs the damage
    # done to global state if it did seed.
    #
    # In order not to generate all the same random numbers in every test,
    # we seed according to a kind of hash:
    ase_path = get_python_package_path_description(ase, default='abort!')
    if "abort!" in ase_path:
        raise RuntimeError("Bad ase.__path__: {:}".format(
            ase_path.replace('abort!', '')))
    abspath = Path(request.module.__file__)
    relpath = abspath.relative_to(ase_path)
    module_identifier = relpath.as_posix()  # Same on all platforms
    function_name = request.function.__name__
    hashable_string = f'{module_identifier}:{function_name}'
    # We use zlib.adler32() rather than hash() because Python randomizes
    # the string hashing at startup for security reasons.
    seed = zlib.adler32(hashable_string.encode('ascii')) % 12345
    # (We should really use the full qualified name of the test method.)
    state = np.random.get_state()
    np.random.seed(seed)
    yield
    print(f'Global seed for "{hashable_string}" was: {seed}')
    np.random.set_state(state)


@pytest.fixture(scope='session')
def povray_executable():
    exe = shutil.which('povray')
    if exe is None:
        pytest.skip('povray not installed')
    return exe


def pytest_addoption(parser):
    parser.addoption('--calculators', metavar='NAMES', default='',
                     help='comma-separated list of calculators to test or '
                     '"auto" for all configured calculators')
    parser.addoption('--seed', action='append', default=[],
                     help='add a seed for tests where random number generators'
                          ' are involved. This option can be applied more'
                          ' than once.')
