import io
import warnings

import numpy as np
import pytest

from ase import Atoms
from ase.build import molecule
from ase.calculators.calculator import compare_atoms
from ase.io.bytes import parse_atoms, to_bytes
from ase.io.cif import (
    CIFLoop,
    NoStructureData,
    parse_cif,
    parse_loop,
    read_cif,
    write_cif,
)


def parse_string(string):
    buf = io.BytesIO(string.encode('latin1'))
    blocks = list(parse_cif(buf))
    return blocks


def check_fractional_occupancies(atoms):
    """ Checks fractional occupancy entries in atoms.info dict """
    assert atoms.info['occupancy']
    assert list(atoms.arrays['spacegroup_kinds'])

    occupancies = atoms.info['occupancy']
    for key in occupancies:
        assert isinstance(key, str)

    kinds = atoms.arrays['spacegroup_kinds']
    for a in atoms:
        a_index_str = str(kinds[a.index])
        if a.symbol == 'Na':

            assert len(occupancies[a_index_str]) == 2
            assert occupancies[a_index_str]['K'] == 0.25
            assert occupancies[a_index_str]['Na'] == 0.75
        else:
            assert len(occupancies[a_index_str]) == 1
        if a.symbol == 'Cl':
            assert occupancies[a_index_str]['Cl'] == 0.3


content = """
data_1


_chemical_name_common                  'Mysterious something'
_cell_length_a                         5.50000
_cell_length_b                         5.50000
_cell_length_c                         5.50000
_cell_angle_alpha                      90
_cell_angle_beta                       90
_cell_angle_gamma                      90
_space_group_name_H-M_alt              'F m -3 m'
_space_group_IT_number                 225

loop_
_space_group_symop_operation_xyz
   'x, y, z'
   '-x, -y, -z'
   '-x, -y, z'
   'x, y, -z'
   '-x, y, -z'  # some comment
   'x, -y, z'
   'x, -y, -z'
   '-x, y, z'
   'z, x, y'
   '-z, -x, -y'
   'z, -x, -y'
   '-z, x, y'
   '-z, -x, y'
   'z, x, -y'
   '-z, x, -y'
   'z, -x, y'
   'y, z, x'
   '-y, -z, -x'
   '-y, z, -x'
   'y, -z, x'
   'y, -z, -x'
   '-y, z, x'
   '-y, -z, x'
   'y, z, -x'
   'y, x, -z'
   '-y, -x, z'
   '-y, -x, -z'
   'y, x, z'
   'y, -x, z'
   '-y, x, -z'
   '-y, x, z'
   'y, -x, -z'
   'x, z, -y'
   '-x, -z, y'
   '-x, z, y'
   'x, -z, -y'
   '-x, -z, -y'
   'x, z, y'
   'x, -z, y'
   '-x, z, -y'
   'z, y, -x'
   '-z, -y, x'
   'z, -y, x'
   '-z, y, -x'
   '-z, y, x'
   'z, -y, -x'
   '-z, -y, -x'
   'z, y, x'
   'x, y+1/2, z+1/2'
   '-x, -y+1/2, -z+1/2'
   '-x, -y+1/2, z+1/2'
   'x, y+1/2, -z+1/2'
   '-x, y+1/2, -z+1/2'
   'x, -y+1/2, z+1/2'
   'x, -y+1/2, -z+1/2'
   '-x, y+1/2, z+1/2'
   'z, x+1/2, y+1/2'
   '-z, -x+1/2, -y+1/2'
   'z, -x+1/2, -y+1/2'
   '-z, x+1/2, y+1/2'
   '-z, -x+1/2, y+1/2'
   'z, x+1/2, -y+1/2'
   '-z, x+1/2, -y+1/2'
   'z, -x+1/2, y+1/2'
   'y, z+1/2, x+1/2'
   '-y, -z+1/2, -x+1/2'
   '-y, z+1/2, -x+1/2'
   'y, -z+1/2, x+1/2'
   'y, -z+1/2, -x+1/2'
   '-y, z+1/2, x+1/2'
   '-y, -z+1/2, x+1/2'
   'y, z+1/2, -x+1/2'
   'y, x+1/2, -z+1/2'
   '-y, -x+1/2, z+1/2'
   '-y, -x+1/2, -z+1/2'
   'y, x+1/2, z+1/2'
   'y, -x+1/2, z+1/2'
   '-y, x+1/2, -z+1/2'
   '-y, x+1/2, z+1/2'
   'y, -x+1/2, -z+1/2'
   'x, z+1/2, -y+1/2'
   '-x, -z+1/2, y+1/2'
   '-x, z+1/2, y+1/2'
   'x, -z+1/2, -y+1/2'
   '-x, -z+1/2, -y+1/2'
   'x, z+1/2, y+1/2'
   'x, -z+1/2, y+1/2'
   '-x, z+1/2, -y+1/2'
   'z, y+1/2, -x+1/2'
   '-z, -y+1/2, x+1/2'
   'z, -y+1/2, x+1/2'
   '-z, y+1/2, -x+1/2'
   '-z, y+1/2, x+1/2'
   'z, -y+1/2, -x+1/2'
   '-z, -y+1/2, -x+1/2'
   'z, y+1/2, x+1/2'
   'x+1/2, y, z+1/2'
   '-x+1/2, -y, -z+1/2'
   '-x+1/2, -y, z+1/2'
   'x+1/2, y, -z+1/2'
   '-x+1/2, y, -z+1/2'
   'x+1/2, -y, z+1/2'
   'x+1/2, -y, -z+1/2'
   '-x+1/2, y, z+1/2'
   'z+1/2, x, y+1/2'
   '-z+1/2, -x, -y+1/2'
   'z+1/2, -x, -y+1/2'
   '-z+1/2, x, y+1/2'
   '-z+1/2, -x, y+1/2'
   'z+1/2, x, -y+1/2'
   '-z+1/2, x, -y+1/2'
   'z+1/2, -x, y+1/2'
   'y+1/2, z, x+1/2'
   '-y+1/2, -z, -x+1/2'
   '-y+1/2, z, -x+1/2'
   'y+1/2, -z, x+1/2'
   'y+1/2, -z, -x+1/2'
   '-y+1/2, z, x+1/2'
   '-y+1/2, -z, x+1/2'
   'y+1/2, z, -x+1/2'
   'y+1/2, x, -z+1/2'
   '-y+1/2, -x, z+1/2'
   '-y+1/2, -x, -z+1/2'
   'y+1/2, x, z+1/2'
   'y+1/2, -x, z+1/2'
   '-y+1/2, x, -z+1/2'
   '-y+1/2, x, z+1/2'
   'y+1/2, -x, -z+1/2'
   'x+1/2, z, -y+1/2'
   '-x+1/2, -z, y+1/2'
   '-x+1/2, z, y+1/2'
   'x+1/2, -z, -y+1/2'
   '-x+1/2, -z, -y+1/2'
   'x+1/2, z, y+1/2'
   'x+1/2, -z, y+1/2'
   '-x+1/2, z, -y+1/2'
   'z+1/2, y, -x+1/2'
   '-z+1/2, -y, x+1/2'
   'z+1/2, -y, x+1/2'
   '-z+1/2, y, -x+1/2'
   '-z+1/2, y, x+1/2'
   'z+1/2, -y, -x+1/2'
   '-z+1/2, -y, -x+1/2'
   'z+1/2, y, x+1/2'
   'x+1/2, y+1/2, z'
   '-x+1/2, -y+1/2, -z'
   '-x+1/2, -y+1/2, z'
   'x+1/2, y+1/2, -z'
   '-x+1/2, y+1/2, -z'
   'x+1/2, -y+1/2, z'
   'x+1/2, -y+1/2, -z'
   '-x+1/2, y+1/2, z'
   'z+1/2, x+1/2, y'
   '-z+1/2, -x+1/2, -y'
   'z+1/2, -x+1/2, -y'
   '-z+1/2, x+1/2, y'
   '-z+1/2, -x+1/2, y'
   'z+1/2, x+1/2, -y'
   '-z+1/2, x+1/2, -y'
   'z+1/2, -x+1/2, y'
   'y+1/2, z+1/2, x'
   '-y+1/2, -z+1/2, -x'
   '-y+1/2, z+1/2, -x'
   'y+1/2, -z+1/2, x'
   'y+1/2, -z+1/2, -x'
   '-y+1/2, z+1/2, x'
   '-y+1/2, -z+1/2, x'
   'y+1/2, z+1/2, -x'
   'y+1/2, x+1/2, -z'
   '-y+1/2, -x+1/2, z'
   '-y+1/2, -x+1/2, -z'
   'y+1/2, x+1/2, z'
   'y+1/2, -x+1/2, z'
   '-y+1/2, x+1/2, -z'
   '-y+1/2, x+1/2, z'
   'y+1/2, -x+1/2, -z'
   'x+1/2, z+1/2, -y'
   '-x+1/2, -z+1/2, y'
   '-x+1/2, z+1/2, y'
   'x+1/2, -z+1/2, -y'
   '-x+1/2, -z+1/2, -y'
   'x+1/2, z+1/2, y'
   'x+1/2, -z+1/2, y'
   '-x+1/2, z+1/2, -y'
   'z+1/2, y+1/2, -x'
   '-z+1/2, -y+1/2, x'
   'z+1/2, -y+1/2, x'
   '-z+1/2, y+1/2, -x'
   '-z+1/2, y+1/2, x'
   'z+1/2, -y+1/2, -x'
   '-z+1/2, -y+1/2, -x'
   'z+1/2, y+1/2, x'

loop_
   _atom_site_label
   _atom_site_occupancy
   _atom_site_fract_x
   _atom_site_fract_y
   _atom_site_fract_z
   _atom_site_adp_type
   _atom_site_B_iso_or_equiv
   _atom_site_type_symbol
   Na         0.7500  0.000000      0.000000      0.000000     Biso  1.000000 Na
   K          0.2500  0.000000      0.000000      0.000000     Biso  1.000000 K
   Cl         0.3000  0.500000      0.500000      0.500000     Biso  1.000000 Cl
   I          0.5000  0.250000      0.250000      0.250000     Biso  1.000000 I
"""


def test_cif():
    cif_file = io.StringIO(content)

    # legacy behavior is to not read the K atoms
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        atoms_leg = read_cif(cif_file, fractional_occupancies=False)
    elements = np.unique(atoms_leg.get_atomic_numbers())
    for n in (11, 17, 53):
        assert n in elements

    assert 'occupancy' not in atoms_leg.info

    cif_file = io.StringIO(content)
    # new behavior is to still not read the K atoms, but build info
    atoms = read_cif(cif_file, fractional_occupancies=True)

    # yield the same old atoms for fractional_occupancies case
    assert len(atoms_leg) == len(atoms)
    assert np.all(atoms_leg.get_atomic_numbers() == atoms.get_atomic_numbers())
    assert atoms_leg == atoms

    elements = np.unique(atoms_leg.get_atomic_numbers())
    for n in (11, 17, 53):
        assert n in elements

    check_fractional_occupancies(atoms)

    # read/write
    fname = 'testfile.cif'
    with open(fname, 'wb') as fd:
        write_cif(fd, atoms)

    with open(fname) as fd:
        atoms = read_cif(fd, fractional_occupancies=True)

    check_fractional_occupancies(atoms)

    # check repeating atoms
    atoms = atoms.repeat([2, 1, 1])
    assert len(atoms.arrays['spacegroup_kinds']) == len(atoms.arrays['numbers'])


# ICSD-like file from issue #293
content2 = """
data_global
_cell_length_a 9.378(5)
_cell_length_b 7.488(5)
_cell_length_c 6.513(5)
_cell_angle_alpha 90.
_cell_angle_beta 91.15(5)
_cell_angle_gamma 90.
_cell_volume 457.27
_cell_formula_units_Z 2
_symmetry_space_group_name_H-M 'P 1 n 1'
_symmetry_Int_Tables_number 7
_refine_ls_R_factor_all 0.071
loop_
_symmetry_equiv_pos_site_id
_symmetry_equiv_pos_as_xyz
1 'x+1/2, -y, z+1/2'
2 'x, y, z'
loop_
_atom_type_symbol
_atom_type_oxidation_number
Sn2+ 2
As4+ 4
Se2- -2
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_symmetry_multiplicity
_atom_site_Wyckoff_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_B_iso_or_equiv
_atom_site_occupancy
_atom_site_attached_hydrogens
Sn1 Sn2+ 2 a 0.5270(2) 0.3856(2) 0.7224(3) 0.0266(4) 1. 0
Sn2 Sn2+ 2 a 0.0279(2) 0.1245(2) 0.7870(2) 0.0209(4) 1. 0
As1 As4+ 2 a 0.6836(4) 0.1608(5) 0.8108(6) 0.0067(7) 1. 0
As2 As4+ 2 a 0.8174(4) 0.6447(6) 0.1908(6) 0.0057(6) 1. 0
Se1 Se2- 2 a 0.4898(4) 0.7511(6) 0.8491(6) 0.0110(6) 1. 0
Se2 Se2- 2 a 0.7788(4) 0.6462(6) 0.2750(6) 0.0097(6) 1. 0
Se3 Se2- 2 a 0.6942(4) 0.0517(5) 0.5921(6) 0.2095(6) 1. 0
Se4 Se2- 2 a 0.0149(4) 0.3437(6) 0.5497(7) 0.1123(7) 1. 0
Se5 Se2- 2 a 0.1147(4) 0.5633(4) 0.3288(6) 0.1078(6) 1. 0
Se6 Se2- 2 a 0.0050(4) 0.4480(6) 0.9025(6) 0.9102(6) 1. 0
"""


def test_cif_icsd():
    cif_file = io.StringIO(content2)
    atoms = read_cif(cif_file)
    # test something random so atoms is not unused
    assert 'occupancy' in atoms.info


@pytest.fixture()
def cif_atoms():
    cif_file = io.StringIO(content)
    return read_cif(cif_file)


def test_cif_loop_keys(cif_atoms):
    data = {}
    # test case has 20 entries
    data['someKey'] = [[str(i) + "test" for i in range(20)]]
    # test case has 20 entries
    data['someIntKey'] = [[str(i) + "123" for i in range(20)]]
    cif_atoms.write('testfile.cif', loop_keys=data)

    atoms1 = read_cif('testfile.cif', store_tags=True)
    # keys are read lowercase only
    r_data = {'someKey': atoms1.info['_somekey'],
              'someIntKey': atoms1.info['_someintkey']}
    assert r_data['someKey'] == data['someKey'][0]
    # data reading auto converts strins
    assert r_data['someIntKey'] == [int(x) for x in data['someIntKey'][0]]


# test if automatic numbers written after elements are correct
def test_cif_writer_label_numbers(cif_atoms):
    cif_atoms.write('testfile.cif')
    atoms1 = read_cif('testfile.cif', store_tags=True)
    labels = atoms1.info['_atom_site_label']
    # cannot use atoms.symbols as K is missing there
    elements = atoms1.info['_atom_site_type_symbol']
    build_labels = [
        "{:}{:}".format(
            x, i) for x in set(elements) for i in range(
            1, elements.count(x) + 1)]
    assert build_labels.sort() == labels.sort()


def test_cif_labels(cif_atoms):
    data = [["label" + str(i) for i in range(20)]]  # test case has 20 entries
    cif_atoms.write('testfile.cif', labels=data)

    atoms1 = read_cif('testfile.cif', store_tags=True)
    print(atoms1.info)
    assert data[0] == atoms1.info['_atom_site_label']


def test_cifloop():
    dct = {'_eggs': range(4),
           '_potatoes': [1.3, 7.1, -1, 0]}

    loop = CIFLoop()
    loop.add('_eggs', dct['_eggs'], '{:<2d}')
    loop.add('_potatoes', dct['_potatoes'], '{:.4f}')

    string = loop.tostring() + '\n\n'
    lines = string.splitlines()[::-1]
    assert lines.pop() == 'loop_'

    newdct = parse_loop(lines)
    print(newdct)
    assert set(dct) == set(newdct)
    for name in dct:
        assert dct[name] == pytest.approx(newdct[name])


@pytest.mark.parametrize('data', [b'', b'data_dummy'])
def test_empty_or_atomless(data):
    ciffile = io.BytesIO(data)

    images = read_cif(ciffile, index=':')
    assert len(images) == 0


def test_empty_or_atomless_cifblock():
    blocks = parse_string('data_dummy')

    assert len(blocks) == 1
    assert not blocks[0].has_structure()
    with pytest.raises(NoStructureData):
        blocks[0].get_atoms()


def test_symbols_questionmark():
    blocks = parse_string("""\
data_dummy
loop_
_atom_site_label
?""")

    assert not blocks[0].has_structure()
    with pytest.raises(NoStructureData, match='undetermined'):
        blocks[0].get_atoms()


def test_bad_occupancies(cif_atoms):
    assert 'Au' not in cif_atoms.symbols
    cif_atoms.symbols[0] = 'Au'
    with pytest.warns(UserWarning, match='no occupancy info'):
        write_cif('tmp.cif', cif_atoms)


@pytest.mark.parametrize(
    'setting_name, ref_setting',
    [
        ('hexagonal', 1),
        ('trigonal', 2),
        ('rhombohedral', 2)
    ]
)
def test_spacegroup_named_setting(setting_name, ref_setting):
    """The rhombohedral crystal system signifies setting=2"""
    blocks = parse_string("""\
data_test
_space_group_crystal_system {}
_symmetry_space_group_name_H-M         'R-3m'
""".format(setting_name))

    assert len(blocks) == 1
    spg = blocks[0].get_spacegroup(False)
    assert int(spg) == 166
    assert spg.setting == ref_setting


@pytest.fixture()
def atoms():
    return Atoms('CO', cell=[2., 3., 4., 50., 60., 70.], pbc=True,
                 scaled_positions=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])


def roundtrip(atoms):
    buf = to_bytes(atoms, format='cif')
    return parse_atoms(buf, format='cif')


def test_cif_roundtrip_periodic(atoms):
    # Reading and writing the cell loses the rotation information,
    # but preserves cellpar and scaled positions.
    atoms1 = roundtrip(atoms)

    assert str(atoms1.symbols) == 'CO'
    assert all(atoms1.pbc)
    assert atoms.cell.cellpar() == pytest.approx(
        atoms1.cell.cellpar(), abs=1e-5)
    assert atoms.get_scaled_positions() == pytest.approx(
        atoms1.get_scaled_positions(), abs=1e-5)


def test_cif_roundtrip_nonperiodic():
    atoms = molecule('H2O')
    atoms1 = roundtrip(atoms)
    assert not compare_atoms(atoms, atoms1, tol=1e-5)


def test_cif_missingvector(atoms):
    # We don't know any way to represent only 2 cell vectors in CIF.
    # So we discard them and warn the user.
    atoms.cell[0] = 0.0
    atoms.pbc[0] = False

    assert atoms.cell.rank == 2

    with pytest.raises(ValueError, match='CIF format can only'):
        roundtrip(atoms)


def test_cif_roundtrip_mixed():
    atoms = Atoms('Au', cell=[1., 2., 3.], pbc=[1, 1, 0])
    atoms1 = roundtrip(atoms)

    # We cannot preserve PBC info for this case:
    assert all(atoms1.pbc)
    assert compare_atoms(atoms, atoms1, tol=1e-5) == ['pbc']
    assert atoms.get_scaled_positions() == pytest.approx(
        atoms1.get_scaled_positions(), abs=1e-5)


def test_loop_with_space():
    # Regression test for https://gitlab.com/ase/ase/-/issues/859 .
    # (We do the \n\ to avoid automatic trailing whitespace cleanup)
    blocks = parse_string("""\
data_image0
loop_
 _hello
 banana
 \n\
_potato 42
""")

    assert len(blocks) == 1
    assert blocks[0]['_potato'] == 42


# Test case for symmetry loop without any symops in it.
empty_symloop_cif = """
data_cif

_audit_creation_method             'generated by GULP'

_symmetry_space_group_name_H-M     '(unknown)       '
_symmetry_Int_Tables_number        1

loop_
_symmetry_equiv_pos_site_id
_symmetry_equiv_pos_as_xyz
_cell_length_a                        5.1053
_cell_length_b                        5.1053
_cell_length_c                        5.1053
_cell_angle_alpha                   60.0000
_cell_angle_beta                    60.0000
_cell_angle_gamma                   60.0000

loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
  Cu          0.00000      0.00000      0.00000  1.0000
  Cu          0.00000      0.00000      0.50000  1.0000
  Cu          0.00000      0.50000      0.00000  1.0000
  Cu          1.00000      0.50000      0.50000  1.0000
  Cu          0.50000      0.00000      0.00000  1.0000
  Cu          0.50000      1.00000      0.50000  1.0000
  Cu          0.50000      0.50000      0.00000  1.0000
  Cu          0.50000      0.50000      0.50000  1.0000
"""


def test_cif_with_empty_symmetries_loop():
    blocks = parse_string(empty_symloop_cif)
    assert len(blocks) == 1
    atoms = blocks[0].get_atoms()
    assert str(atoms.symbols) == 'Cu8'
    assert atoms.cell.lengths() == pytest.approx(5.1053)
    assert atoms.cell.angles() == pytest.approx(60)


def test_cif_with_strings_for_integers():
    # Test that even if spacegroup is given as '42' with ticks, we can
    # still convert it to integer and thus actual Spacegroup.

    buf = """\
data_cif
_cell_length_a               1
_cell_length_b               1
_cell_length_c               1
_cell_angle_alpha            90
_cell_angle_beta             90
_cell_angle_gamma            90
_symmetry_space_group_H-M        'P1'
_symmetry_Int_Tables_number      '42'
"""

    block = parse_string(buf)[0]
    assert block.get_spacegroup(subtrans_included=True).no == 42
