from unittest import mock

import numpy as np
import pytest

from ase.build import bulk
from ase.calculators.vasp.create_input import (
    GenerateVaspInput,
    _args_without_comment,
    _from_vasp_bool,
    _to_vasp_bool,
)


def dict_is_subset(d1, d2):
    """True if all the key-value pairs in dict 1 are in dict 2"""
    # Note, we are using direct comparison, so we should not compare
    # floats if any real computations are made, as that would be unsafe.
    # Cannot use pytest.approx here, because of of string comparison
    # not being available in python 3.6.
    return all(key in d2 and d1[key] == d2[key] for key in d1)


@pytest.fixture()
def rng():
    return np.random.RandomState(seed=42)


@pytest.fixture()
def nacl(rng):
    atoms = bulk('NaCl', crystalstructure='rocksalt', a=4.1,
                 cubic=True) * (3, 3, 3)
    rng.shuffle(atoms.symbols)  # Ensure symbols are mixed
    return atoms


@pytest.fixture()
def vaspinput_factory(nacl):
    """Factory for GenerateVaspInput class, which mocks the generation of
    pseudopotentials."""
    def _vaspinput_factory(atoms=None, **kwargs) -> GenerateVaspInput:
        if atoms is None:
            atoms = nacl
        mocker = mock.Mock()
        inputs = GenerateVaspInput()
        inputs.set(**kwargs)
        inputs._build_pp_list = mocker(  # type: ignore[method-assign]
            return_value=None
        )
        inputs.initialize(atoms)
        return inputs

    return _vaspinput_factory


def test_sorting(nacl, vaspinput_factory):
    """Test that the sorting/resorting scheme works"""
    vaspinput = vaspinput_factory(atoms=nacl)
    srt = vaspinput.sort
    resrt = vaspinput.resort
    atoms = nacl.copy()
    assert atoms[srt] != nacl
    assert atoms[resrt] != nacl
    assert atoms[srt][resrt] == nacl

    # Check the first and second half of the sorted atoms have the same symbols
    assert len(atoms) % 2 == 0  # We should have an even number of atoms
    atoms_sorted = atoms[srt]
    N = len(atoms) // 2
    seq1 = set(atoms_sorted.symbols[:N])
    seq2 = set(atoms_sorted.symbols[N:])
    assert len(seq1) == 1
    assert len(seq2) == 1
    # Check that we have two different symbols
    assert len(seq1.intersection(seq2)) == 0


@pytest.fixture(params=['random', 'ones', 'binaries'])
def magmoms_factory(rng, request):
    """Factory for generating various kinds of magnetic moments"""
    kind = request.param
    if kind == 'random':
        # Array of random
        func = rng.rand
    elif kind == 'ones':
        # Array of just 1's
        func = np.ones
    elif kind == 'binaries':
        # Array of 0's and 1's
        def rand_binary(x):
            return rng.randint(2, size=x)

        func = rand_binary
    else:
        raise ValueError(f'Unknown kind: {kind}')

    def _magmoms_factory(atoms):
        magmoms = func(len(atoms))
        assert len(magmoms) == len(atoms)
        return magmoms

    return _magmoms_factory


def read_magmom_from_file(filename) -> np.ndarray:
    """Helper function to parse the magnetic moments from an INCAR file"""
    found = False
    with open(filename) as file:
        for line in file:
            # format "MAGMOM = n1*val1 n2*val2 ..."
            if 'MAGMOM = ' in line:
                found = True
                parts = line.strip().split()[2:]
                new_magmom = []
                for part in parts:
                    n, val = part.split('*')
                    # Add "val" to magmom "n" times
                    new_magmom += int(n) * [float(val)]
                break
    assert found
    return np.array(new_magmom)


@pytest.fixture()
def assert_magmom_equal_to_incar_value():
    """Fixture to compare a pre-made magmom array to the value
    a GenerateVaspInput.write_incar object writes to a file"""
    def _assert_magmom_equal_to_incar_value(atoms, expected_magmom, vaspinput):
        assert len(atoms) == len(expected_magmom)
        vaspinput.write_incar(atoms)
        new_magmom = read_magmom_from_file('INCAR')
        assert len(new_magmom) == len(expected_magmom)
        srt = vaspinput.sort
        resort = vaspinput.resort
        # We round to 4 digits
        assert np.allclose(expected_magmom, new_magmom[resort], atol=1e-3)
        assert np.allclose(np.array(expected_magmom)[srt],
                           new_magmom,
                           atol=1e-3)

    return _assert_magmom_equal_to_incar_value


@pytest.mark.parametrize('list_func', [list, tuple, np.array])
def test_write_magmom(magmoms_factory, list_func, nacl, vaspinput_factory,
                      assert_magmom_equal_to_incar_value, testdir):
    """Test writing magnetic moments to INCAR, and ensure we can do it
    passing different types of sequences"""
    magmom = magmoms_factory(nacl)

    vaspinput = vaspinput_factory(atoms=nacl, magmom=magmom, ispin=2)
    assert vaspinput.spinpol
    assert_magmom_equal_to_incar_value(nacl, magmom, vaspinput)


def test_atoms_with_initial_magmoms(magmoms_factory, nacl, vaspinput_factory,
                                    assert_magmom_equal_to_incar_value,
                                    testdir):
    """Test passing atoms with initial magnetic moments"""
    magmom = magmoms_factory(nacl)
    assert len(magmom) == len(nacl)
    nacl.set_initial_magnetic_moments(magmom)
    vaspinput = vaspinput_factory(atoms=nacl)
    assert vaspinput.spinpol
    assert_magmom_equal_to_incar_value(nacl, magmom, vaspinput)


def test_vasp_from_bool():
    for s in ('T', '.true.'):
        assert _from_vasp_bool(s) is True
    for s in ('f', '.False.'):
        assert _from_vasp_bool(s) is False
    with pytest.raises(ValueError):
        _from_vasp_bool('yes')
    with pytest.raises(AssertionError):
        _from_vasp_bool(True)


def test_vasp_to_bool():
    for x in ('T', '.true.', True):
        assert _to_vasp_bool(x) == '.TRUE.'
    for x in ('f', '.FALSE.', False):
        assert _to_vasp_bool(x) == '.FALSE.'

    with pytest.raises(ValueError):
        _to_vasp_bool('yes')
    with pytest.raises(AssertionError):
        _to_vasp_bool(1)


@pytest.mark.parametrize('args, expected_len',
                         [(['a', 'b', '#', 'c'], 2),
                          (['a', 'b', '!', 'c', '#', 'd'], 2),
                          (['#', 'a', 'b', '!', 'c', '#', 'd'], 0)])
def test_vasp_args_without_comment(args, expected_len):
    """Test comment splitting logic"""
    clean_args = _args_without_comment(args)
    assert len(clean_args) == expected_len


def test_vasp_xc(vaspinput_factory):
    """
    Run some tests to ensure that the xc setting in the VASP calculator
    works.
    """

    calc_vdw = vaspinput_factory(xc='optb86b-vdw')

    assert dict_is_subset({
        'param1': 0.1234,
        'param2': 1.0
    }, calc_vdw.float_params)
    assert calc_vdw.bool_params['luse_vdw'] is True

    calc_hse = vaspinput_factory(xc='hse06',
                                 hfscreen=0.1,
                                 gga='RE',
                                 encut=400,
                                 sigma=0.5)

    assert dict_is_subset({
        'hfscreen': 0.1,
        'encut': 400,
        'sigma': 0.5
    }, calc_hse.float_params)
    assert calc_hse.bool_params['lhfcalc'] is True
    assert dict_is_subset({'gga': 'RE'}, calc_hse.string_params)

    with pytest.warns(FutureWarning):
        calc_pw91 = vaspinput_factory(xc='pw91',
                                      kpts=(2, 2, 2),
                                      gamma=True,
                                      lreal='Auto')
        assert dict_is_subset(
            {
                'pp': 'PW91',
                'kpts': (2, 2, 2),
                'gamma': True,
                'reciprocal': False
            }, calc_pw91.input_params)


def test_ichain(vaspinput_factory):

    with pytest.warns(UserWarning):
        calc_warn = vaspinput_factory(ichain=1, ediffg=-0.01)
        calc_warn.write_incar(nacl)
        calc_warn.read_incar('INCAR')
        assert calc_warn.int_params['iopt'] == 1
        assert calc_warn.exp_params['ediffg'] == -0.01
        assert calc_warn.int_params['ibrion'] == 3
        assert calc_warn.float_params['potim'] == 0.0

    with pytest.raises(RuntimeError):
        calc_wrong = vaspinput_factory(ichain=1, ediffg=0.0001, iopt=1)
        calc_wrong.write_incar(nacl)
        calc_wrong.read_incar('INCAR')
        assert calc_wrong.int_params['iopt'] == 1

    calc = vaspinput_factory(ichain=1,
                             ediffg=-0.01,
                             iopt=1,
                             potim=0.0,
                             ibrion=3)
    calc.write_incar(nacl)
    calc.read_incar('INCAR')
    assert calc.int_params['iopt'] == 1
    assert calc.exp_params['ediffg'] == -0.01
    assert calc.int_params['ibrion'] == 3
    assert calc.float_params['potim'] == 0.0


def test_non_registered_keys(vaspinput_factory) -> None:
    """Test if non-registered INCAR keys can be written and read.

    Here the SCAN meta-GGA functional via LIBXC is tested.

    https://www.vasp.at/wiki/index.php/LIBXC1#Examples_of_INCAR

    """
    calc = vaspinput_factory(metagga='LIBXC')

    # Be sure that `libxc1` and `libxc2` are not in the registered INCAR keys.
    assert 'libxc1' not in calc.string_params
    assert 'libxc2' not in calc.int_params

    calc.set(libxc1='MGGA_X_SCAN')  # or 263
    calc.set(libxc2=267)  # or "MGGA_C_SCAN"

    calc.write_incar(nacl)
    calc.read_incar('INCAR')

    assert calc.string_params['libxc1'] == 'MGGA_X_SCAN'
    assert calc.int_params['libxc2'] == 267


def test_bool(tmp_path, vaspinput_factory):
    """Test that INCAR parser behaves similarly to Vasp, which uses
    default fortran 'read' parsing
    """

    for bool_str in ['t', 'T', 'true', 'TRUE', 'TrUe', '.true.', '.T', 'tbob']:
        with open(tmp_path / 'INCAR', 'w') as fout:
            fout.write('ENCUT = 100\n')
            fout.write(f'LCHARG = {bool_str}\n')
        calc = vaspinput_factory(encut=100)
        calc.read_incar(tmp_path / 'INCAR')
        assert calc.bool_params['lcharg']

    for bool_str in ['f', 'F', 'false', 'FALSE', 'FaLSe', '.false.', '.F',
                     'fbob']:
        with open(tmp_path / 'INCAR', 'w') as fout:
            fout.write('ENCUT = 100\n')
            fout.write(f'LCHARG = {bool_str}\n')
        calc = vaspinput_factory(encut=100)
        calc.read_incar(tmp_path / 'INCAR')
        assert not calc.bool_params['lcharg']

    for bool_str in ['x', '..true.', '1']:
        with open(tmp_path / 'INCAR', 'w') as fout:
            fout.write('ENCUT = 100\n')
            fout.write(f'LCHARG = {bool_str}\n')
        calc = vaspinput_factory(encut=100)
        with pytest.raises(ValueError):
            calc.read_incar(tmp_path / 'INCAR')
