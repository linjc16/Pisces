import numpy as np
import pytest

from ase.calculators.calculator import equal


def arrays(a, dtype):
    return [list(map(dtype, a)), np.array(a, dtype=dtype)]


@pytest.mark.parametrize('a', [1, 1.])
@pytest.mark.parametrize('b', [1, 1.])
@pytest.mark.parametrize('rtol', [None, 0, 1e-8])
@pytest.mark.parametrize('atol', [None, 0, 1e-8])
def test_single_value(a, b, rtol, atol):
    assert equal(a, b, rtol=rtol, atol=atol)


@pytest.mark.parametrize('a', arrays([1, 1], int) + arrays([1, 1], float))
@pytest.mark.parametrize('b', arrays([1, 1], int) + arrays([1, 1], float))
@pytest.mark.parametrize('rtol', [None, 0, 1e-8])
@pytest.mark.parametrize('atol', [None, 0, 1e-8])
def test_array_equal(a, b, rtol, atol):
    assert a is not b
    assert equal(a, b, rtol=rtol, atol=atol)
    assert equal({'size': a, 'gamma': True},
                 {'size': b, 'gamma': True}, rtol=rtol, atol=atol)
    assert not equal({'size': a, 'gamma': True},
                     {'size': b, 'gamma': False}, rtol=rtol, atol=atol)
    assert not equal({'size': a}, b, rtol=rtol, atol=atol)
    assert not equal(a, {'size': b}, rtol=rtol, atol=atol)


@pytest.mark.parametrize('a', arrays([2, 2], float))
@pytest.mark.parametrize('b', arrays(np.array([2, 2]) + 1.9e-8, float))
@pytest.mark.parametrize('rtol,atol', [[None, 2e-8], [0, 2e-8],
                                       [1e-8, None], [1e-8, 0],
                                       [0.5e-8, 1e-8]])
def test_array_almost_equal(a, b, rtol, atol):
    assert a is not b
    assert equal(a, b, rtol=rtol, atol=atol)
    assert equal({'size': a, 'gamma': True},
                 {'size': b, 'gamma': True}, rtol=rtol, atol=atol)


@pytest.mark.parametrize('a', arrays([2, 2], float))
@pytest.mark.parametrize('b', arrays(np.array([2, 2]) + 3.1e-8, float))
@pytest.mark.parametrize('rtol', [None, 0, 1e-8])
@pytest.mark.parametrize('atol', [None, 0, 1e-8])
def test_array_not_equal(a, b, rtol, atol):
    assert a is not b
    assert not equal(a, b, rtol=rtol, atol=atol)
    assert not equal({'size': a, 'gamma': True},
                     {'size': b, 'gamma': True}, rtol=rtol, atol=atol)
