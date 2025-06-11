import os

import numpy as np
import pytest

import ase.lattice.cubic
from ase.build import bulk
from ase.calculators.castep import (
    Castep,
    CastepKeywords,
    _get_indices_to_sort_back,
    make_cell_dict,
    make_param_dict,
)
from ase.dft.kpoints import BandPath
from ase.io.castep.castep_input_file import (
    CastepCell,
    CastepOption,
    CastepParam,
)

calc = pytest.mark.calculator

# We use 'fake keywords' to test all the generic CastepOptions and
# whether they work
kw_types = ['Real', 'String', 'Defined', 'Integer Vector',
            'Boolean (Logical)', 'Integer', 'Real Vector',
            'Block', 'Physical']
kw_levels = ['Dummy', 'Intermediate', 'Expert', 'Basic']


@pytest.fixture()
def testing_keywords():

    kw_data = {}

    for kwt in kw_types:
        kwtlow = kwt.lower().replace(' ', '_')
        if 'Boolean' in kwt:
            kwtlow = 'boolean'
        kw = f'test_{kwtlow}_kw'

        kw_data[kw] = {
            'docstring': f'A fake {kwt} keyword',
            'option_type': kwt,
            'keyword': kw,
            'level': 'Dummy'
        }

    # Add the special ones for cell and param that have custom parsers

    # Special keywords for the CastepParam object
    param_kws = [('continuation', 'String'), ('reuse', 'String')]

    param_kw_data = {
        pkw: {
            'docstring': f'Dummy {pkw} keyword',
            'option_type': t,
            'keyword': pkw,
            'level': 'Dummy',
        }
        for pkw, t in param_kws
    }
    param_kw_data.update(kw_data)

    # Special keywords for the CastepCell object
    cell_kws = [('species_pot', 'Block'),
                ('symmetry_ops', 'Block'),
                ('positions_abs_intermediate', 'Block'),
                ('positions_abs_product', 'Block'),
                ('positions_frac_intermediate', 'Block'),
                ('positions_frac_product', 'Block'),
                ('kpoint_mp_grid', 'Integer Vector'),
                ('kpoint_mp_offset', 'Real Vector'),
                ('kpoint_list', 'Block'),
                ('bs_kpoint_list', 'Block')]

    cell_kw_data = {
        ckw: {
            'docstring': f'Dummy {ckw} keyword',
            'option_type': t,
            'keyword': ckw,
            'level': 'Dummy',
        }
        for ckw, t in cell_kws
    }
    cell_kw_data.update(kw_data)

    param_dict = make_param_dict(param_kw_data)
    cell_dict = make_cell_dict(cell_kw_data)

    return CastepKeywords(param_dict, cell_dict, kw_types, kw_levels,
                          'Castep v.Fake')


@pytest.fixture()
def pspot_tmp_path(tmp_path):

    path = os.path.join(tmp_path, 'ppots')
    os.mkdir(path)

    for el in ase.data.chemical_symbols:
        with open(os.path.join(path, f'{el}_test.usp'), 'w') as fd:
            fd.write('Fake PPOT')

    return path


@pytest.fixture()
def testing_calculator(testing_keywords, tmp_path, pspot_tmp_path):
    castep_path = os.path.join(tmp_path, 'CASTEP')
    os.mkdir(castep_path)

    return Castep(castep_keywords=testing_keywords, directory=castep_path,
                  castep_pp_path=pspot_tmp_path)


def test_fundamental_params():
    # Start by testing the fundamental parts of a CastepCell/CastepParam object
    boolOpt = CastepOption('test_bool', 'basic', 'defined')
    boolOpt.value = 'TRUE'
    assert boolOpt.raw_value is True

    float3Opt = CastepOption('test_float3', 'basic', 'real vector')
    float3Opt.value = '1.0 2.0 3.0'
    assert np.isclose(float3Opt.raw_value, [1, 2, 3]).all()

    # Generate a mock keywords object
    mock_castep_keywords = CastepKeywords(make_param_dict(), make_cell_dict(),
                                          [], [], 0)
    mock_cparam = CastepParam(mock_castep_keywords, keyword_tolerance=2)
    mock_ccell = CastepCell(mock_castep_keywords, keyword_tolerance=2)

    # Test special parsers
    mock_cparam.continuation = 'default'
    with pytest.warns(UserWarning):
        mock_cparam.reuse = 'default'
    assert mock_cparam.reuse.value is None

    mock_ccell.species_pot = ('Si', 'Si.usp')
    mock_ccell.species_pot = ('C', 'C.usp')
    assert 'Si Si.usp' in mock_ccell.species_pot.value
    assert 'C C.usp' in mock_ccell.species_pot.value
    symops = (np.eye(3)[None], np.zeros(3)[None])
    mock_ccell.symmetry_ops = symops
    assert """1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
0.0 0.0 0.0""" == mock_ccell.symmetry_ops.value.strip()


def test_castep_option(testing_keywords):

    # check if the CastepOption assignment and comparison mechanisms work
    p1 = CastepParam(testing_keywords)
    p2 = CastepParam(testing_keywords)

    assert p1._options == p2._options

    # Set some values
    p1.test_real_kw = 3.0
    p1.test_string_kw = 'PBE'
    p1.test_defined_kw = True
    p1.test_integer_kw = 10
    p1.test_integer_vector_kw = [3, 3, 3]
    p1.test_real_vector_kw = [3.0, 3.0, 3.0]
    p1.test_boolean_kw = False
    p1.test_physical_kw = '3.0 ang'

    assert p1.test_real_kw.value == '3.0'
    assert p1.test_string_kw.value == 'PBE'
    assert p1.test_defined_kw.value == 'TRUE'
    assert p1.test_integer_kw.value == '10'
    assert p1.test_integer_vector_kw.value == '3 3 3'
    assert p1.test_real_vector_kw.value == '3.0 3.0 3.0'
    assert p1.test_boolean_kw.value == 'FALSE'
    assert p1.test_physical_kw.value == '3.0 ang'

    assert p1._options != p2._options


def test_castep_cell(testing_keywords):

    ccell = CastepCell(testing_keywords, keyword_tolerance=2)

    # Here we test the special keywords exclusive to cell

    # 1. species_pot
    ccell.species_pot = ('H', 'H_test.usp')  # Setting with a single value
    assert ccell.species_pot.value == """
H H_test.usp"""

    ccell.species_pot = [('H', 'H_test.usp'), ('He', 'He_test.usp')]  # Two
    assert ccell.species_pot.value == """
H H_test.usp
He He_test.usp"""

    # 2. symmetry_ops
    # Create for example the P-1 spacegroup
    R = np.array([np.eye(3), -np.eye(3)])
    T = np.zeros((2, 3))
    ccell.symmetry_ops = (R, T)
    strblock = [line.strip() for line in ccell.symmetry_ops.value.split('\n')
                if line.strip() != '']
    fblock = np.array([list(map(float, line.split())) for line in strblock])

    assert np.isclose(fblock[:3], R[0]).all()
    assert np.isclose(fblock[3], T[0]).all()
    assert np.isclose(fblock[4:7], R[1]).all()
    assert np.isclose(fblock[7], T[1]).all()

    # 3. transition state blocks (postponed until fix is merged)
    a = ase.Atoms('H', positions=[[0, 0, 1]], cell=np.eye(3) * 2)

    ccell.positions_abs_product = a
    ccell.positions_abs_intermediate = a

    def parse_posblock(pblock, has_units=False):

        lines = pblock.split('\n')

        units = None
        if has_units:
            units = lines.pop(0).strip()

        pos_lines = []
        while len(lines) > 0:
            line = lines.pop(0).strip()
            if line == '':
                continue
            el, x, y, z = line.split()
            xyz = np.array(list(map(float, [x, y, z])))
            pos_lines.append((el, xyz))

        return units, pos_lines

    pap = parse_posblock(ccell.positions_abs_product.value, True)
    pai = parse_posblock(ccell.positions_abs_intermediate.value, True)

    assert pap[0] == 'ang'
    assert pap[1][0][0] == 'H'
    assert np.isclose(pap[1][0][1], a.get_positions()[0]).all()

    assert pai[0] == 'ang'
    assert pai[1][0][0] == 'H'
    assert np.isclose(pai[1][0][1], a.get_positions()[0]).all()

    ccell.positions_frac_product = a
    ccell.positions_frac_intermediate = a

    pfp = parse_posblock(ccell.positions_frac_product.value)
    pfi = parse_posblock(ccell.positions_frac_intermediate.value)

    assert pfp[1][0][0] == 'H'
    assert np.isclose(pfp[1][0][1], a.get_scaled_positions()[0]).all()

    assert pfi[1][0][0] == 'H'
    assert np.isclose(pfi[1][0][1], a.get_scaled_positions()[0]).all()

    # Test example conflict
    ccell.kpoint_mp_grid = '3 3 3'
    with pytest.warns(UserWarning):
        ccell.kpoint_mp_spacing = 10.0


def test_castep_param(testing_keywords):

    cparam = CastepParam(testing_keywords, keyword_tolerance=2)

    # Special keywords for param

    # 1. continuation and reuse
    cparam.continuation = True
    with pytest.warns(UserWarning):
        cparam.reuse = False   # This conflicts with the previous one
    cparam.continuation = None
    cparam.reuse = True
    with pytest.warns(UserWarning):
        cparam.continuation = True   # This conflicts with the previous one

    # Test conflict
    cparam.cut_off_energy = 500
    with pytest.warns(UserWarning):
        cparam.basis_precision = 'FINE'


@pytest.mark.skipif(os.name == "nt", reason="No symlink on Windows")
def test_workflow(testing_calculator):
    c = testing_calculator
    c._build_missing_pspots = False
    c._find_pspots = True
    c.set_label('test_label_pspots')

    atoms = bulk('Ag')
    atoms.calc = c

    # Should find them automatically!
    c._fetch_pspots()

    assert os.path.isfile(os.path.join(c._directory, 'Ag_test.usp'))

    # Try creating input files
    c.prepare_input_files()

    assert os.path.isfile(os.path.join(c._directory, c._label + '.cell'))
    assert os.path.isfile(os.path.join(c._directory, c._label + '.param'))


def test_set_kpoints(testing_calculator):

    c = testing_calculator

    c.set_kpts([(0.0, 0.0, 0.0, 1.0)])
    assert c.cell.kpoint_list.value == '0.0 0.0 0.0 1.0'
    c.set_kpts(((0.0, 0.0, 0.0, 0.25), (0.25, 0.25, 0.3, 0.75)))
    assert (c.cell.kpoint_list.value ==
            '0.0 0.0 0.0 0.25\n0.25 0.25 0.3 0.75')
    c.set_kpts(c.cell.kpoint_list.value.split('\n'))
    assert (c.cell.kpoint_list.value ==
            '0.0 0.0 0.0 0.25\n0.25 0.25 0.3 0.75')
    c.set_kpts([3, 3, 2])
    assert c.cell.kpoint_mp_grid.value == '3 3 2'
    c.set_kpts(None)
    assert c.cell.kpoints_list.value is None
    assert c.cell.kpoint_list.value is None
    assert c.cell.kpoint_mp_grid.value is None
    c.set_kpts('2 2 3')
    assert c.cell.kpoint_mp_grid.value == '2 2 3'
    c.set_kpts({'even': True, 'gamma': True})
    assert c.cell.kpoint_mp_grid.value == '2 2 2'
    assert c.cell.kpoint_mp_offset.value == '0.25 0.25 0.25'
    c.set_kpts({'size': (2, 2, 4), 'even': False})
    assert c.cell.kpoint_mp_grid.value == '3 3 5'
    assert c.cell.kpoint_mp_offset.value == '0.0 0.0 0.0'
    atoms = bulk('Ag')
    atoms.calc = c
    c.set_kpts({'density': 10, 'gamma': False, 'even': None})
    assert c.cell.kpoint_mp_grid.value == '27 27 27'
    assert c.cell.kpoint_mp_offset.value == '0.018519 0.018519 0.018519'
    c.set_kpts({'spacing': (1 / (np.pi * 10)),
                'gamma': False, 'even': True})
    assert c.cell.kpoint_mp_grid.value == '14 14 14'
    assert c.cell.kpoint_mp_offset.value == '0.0 0.0 0.0'


def test_band_structure_setup(testing_calculator):

    c = testing_calculator

    atoms = bulk('Ag')
    bp = BandPath(cell=atoms.cell,
                  path='GX',
                  special_points={'G': [0, 0, 0], 'X': [0.5, 0, 0.5]})
    bp = bp.interpolate(npoints=10)

    c.set_bandpath(bp)

    kpt_list = c.cell.bs_kpoint_list.value.split('\n')
    assert len(kpt_list) == 10
    assert list(map(float, kpt_list[0].split())) == [0., 0., 0.]
    assert list(map(float, kpt_list[-1].split())) == [0.5, 0.0, 0.5]


def test_get_indices_to_sort_back():
    """Test if spicies in .castep are sorted back to atoms.symbols."""
    symbols = ['Si', 'Al', 'P', 'Al', 'P', 'Al', 'P', 'C']
    species = ['C', 'Al', 'Al', 'Al', 'Si', 'P', 'P', 'P']
    indices_ref = [4, 1, 5, 2, 6, 3, 7, 0]
    assert [species[_] for _ in indices_ref] == symbols
    indices = _get_indices_to_sort_back(symbols, species)
    assert indices.tolist() == indices_ref
