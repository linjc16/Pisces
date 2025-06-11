# type: ignore
import pytest

from ase.calculators.turbomole import TurbomoleParameters


@pytest.fixture()
def default_params():
    return TurbomoleParameters(multiplicity=1)


def test_tm_parameters_default():
    params = TurbomoleParameters()
    defaults = params.default_parameters
    pspec = params.parameter_spec
    assert len(defaults) > 0
    assert all(v['type'] is not None for v in params.parameter_spec.values())
    assert all(params[k] == v for k, v in defaults.items())
    assert all(v['default'] == defaults[k] for k, v in pspec.items())
    assert all(isinstance(params[k], (v['type'], type(None)))
               for k, v in pspec.items())


def test_tm_parameters_uhf():
    params = TurbomoleParameters(uhf=True)
    assert params['uhf']


def test_tm_parameters_update_wrong_type(default_params):
    params_update = {'uhf': 2}
    with pytest.raises(TypeError) as err:
        assert default_params.update(params_update)
    assert str(err.value) == 'uhf has wrong type: <class \'int\'>'


def test_tm_parameters_update_invalid_name(default_params):
    with pytest.raises(ValueError) as err:
        assert default_params.update({'blah': 3})
    assert str(err.value) == 'invalid parameter: blah'


def test_tm_parameters_restart_update(default_params):
    params_update = {'uhf': True}
    with pytest.raises(ValueError) as err:
        assert default_params.update_restart(params_update)
    assert str(err.value) == 'parameters [\'uhf\'] cannot be changed'


def test_tm_parameters_verify_empty_define_str():
    params = TurbomoleParameters()
    params.define_str = {}
    with pytest.raises(AssertionError) as err:
        assert params.verify()
    assert str(err.value) == 'define_str must be str'
    params.define_str = ''
    with pytest.raises(AssertionError) as err:
        assert params.verify()
    assert str(err.value) == 'define_str may not be empty'


def test_tm_parameters_verify_mult_ndefined():
    params = TurbomoleParameters()
    with pytest.raises(AssertionError) as err:
        assert params.verify()
    assert str(err.value) == 'multiplicity not defined'


def test_tm_parameters_verify_mult_wrong_value(default_params):
    default_params.update({'multiplicity': 0})
    with pytest.raises(AssertionError) as err:
        assert default_params.verify()
    assert str(err.value) == 'multiplicity has wrong value'


def test_tm_parameters_verify_initial_guess(default_params):
    default_params.update({'initial guess': {}})
    with pytest.raises(ValueError) as err:
        assert default_params.verify()
    assert str(err.value) == 'Wrong input for initial guess'
    default_params.update({'initial guess': 'blah'})
    with pytest.raises(ValueError) as err:
        assert default_params.verify()
    assert str(err.value) == 'Wrong input for initial guess'


def test_tm_parameters_verify_roh(default_params):
    default_params.update({'rohf': True})
    with pytest.raises(NotImplementedError):
        assert default_params.verify()


def test_tm_parameters_verify_bsl(default_params):
    default_params.update({'use basis set library': False})
    with pytest.raises(NotImplementedError):
        assert default_params.verify()


def test_tm_parameters_verify_c2v(default_params):
    default_params.update({'point group': 'c2v'})
    with pytest.raises(NotImplementedError):
        assert default_params.verify()


def test_tm_parameters_define_str(default_params):
    ref = ('\n\na coord\n*\nno\nbb all def-SV(P)\n*\neht\ny\n0\ny\ndft\non'
           '\n*\ndft\nfunc b-p\n*\ndft\ngrid m3\n*\nscf\niter\n60\n\nq\n')
    assert default_params.get_define_str(5) == ref
    default_params.define_str = 'invalid define string'
    assert default_params.get_define_str(5) == 'invalid define string'
