# type: ignore
import sys

import pytest

from ase.calculators.turbomole import Turbomole
from ase.calculators.turbomole.executor import execute


@pytest.fixture()
def default_params():
    return {'multiplicity': 1}


def test_turbomole_empty():
    with pytest.raises(AssertionError) as err:
        assert Turbomole()
    assert str(err.value) == 'multiplicity not defined'


def test_turbomole_default(default_params):
    calc = Turbomole(**default_params)
    assert calc['label'] is None
    assert calc['prefix'] is None
    assert calc['directory'] == '.'
    assert not calc['restart']
    assert calc['atoms'] is None


def test_execute_good():
    python = sys.executable
    code = ('import sys; print(\"ended normally\", file=sys.stderr);'
            'print(\"Hello world\")')
    stdout_file = execute([python, '-c', code])
    with open(stdout_file) as fd:
        assert fd.read() == 'Hello world' + '\n'


def test_execute_fail():
    python = sys.executable
    with pytest.raises(OSError) as err:
        execute([python, '-c', 'print(\"Hello world\")'])
    assert 'Turbomole error' in str(err.value)
