# additional tests of the extended XYZ file I/O
# (which is also included in oi.py test case)
# maintained by James Kermode <james.kermode@gmail.com>

import sys
from pathlib import Path

import numpy as np
import pytest

import ase.io
from ase.atoms import Atoms
from ase.build import bulk, molecule
from ase.calculators.calculator import compare_atoms
from ase.calculators.emt import EMT
from ase.calculators.mixing import LinearCombinationCalculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms, FixCartesian
from ase.io import extxyz
from ase.io.extxyz import escape, save_calc_results

# array data of shape (N, 1) squeezed down to shape (N, ) -- bug fixed
# in commit r4541


@pytest.fixture()
def atoms():
    return bulk('Si')


@pytest.fixture()
def images(atoms):
    images = [atoms, atoms * (2, 1, 1), atoms * (3, 1, 1)]
    images[1].set_pbc([True, True, False])
    images[2].set_pbc([True, False, False])
    return images


def test_array_shape(atoms):
    # Check that unashable data type in info does not break output
    atoms.info['bad-info'] = [[1, np.array([0, 1])], [2, np.array([0, 1])]]
    with pytest.warns(UserWarning):
        ase.io.write('to.xyz', atoms, format='extxyz')
    del atoms.info['bad-info']
    atoms.arrays['ns_extra_data'] = np.zeros((len(atoms), 1))
    assert atoms.arrays['ns_extra_data'].shape == (2, 1)

    ase.io.write('to_new.xyz', atoms, format='extxyz')
    at_new = ase.io.read('to_new.xyz')
    assert at_new.arrays['ns_extra_data'].shape == (2, )


# test comment read/write with vec_cell
def test_comment(atoms):
    atoms.info['comment'] = 'test comment'
    ase.io.write('comment.xyz', atoms, comment=atoms.info['comment'],
                 vec_cell=True)
    r = ase.io.read('comment.xyz')
    assert atoms == r


# write sequence of images with different numbers of atoms -- bug fixed
# in commit r4542
def test_sequence(images):
    ase.io.write('multi.xyz', images, format='extxyz')
    read_images = ase.io.read('multi.xyz', index=':')
    assert read_images == images


# test vec_cell writing and reading
def test_vec_cell(atoms, images):
    ase.io.write('multi.xyz', images, vec_cell=True)
    cell = images[1].get_cell()
    cell[-1] = [0.0, 0.0, 0.0]
    images[1].set_cell(cell)
    cell = images[2].get_cell()
    cell[-1] = [0.0, 0.0, 0.0]
    cell[-2] = [0.0, 0.0, 0.0]
    images[2].set_cell(cell)
    read_images = ase.io.read('multi.xyz', index=':')
    assert read_images == images
    # also test for vec_cell with whitespaces
    Path('structure.xyz').write_text("""1
    Coordinates
    C         -7.28250        4.71303       -3.82016
      VEC1 1.0 0.1 1.1
    1

    C         -7.28250        4.71303       -3.82016
    VEC1 1.0 0.1 1.1
    """)

    a = ase.io.read('structure.xyz', index=0)
    b = ase.io.read('structure.xyz', index=1)
    assert a == b

    # read xyz containing trailing blank line
    # also test for upper case elements
    Path('structure.xyz').write_text("""4
    Coordinates
    MG        -4.25650        3.79180       -2.54123
    C         -1.15405        2.86652       -1.26699
    C         -5.53758        3.70936        0.63504
    C         -7.28250        4.71303       -3.82016

    """)

    a = ase.io.read('structure.xyz')
    assert a[0].symbol == 'Mg'


# read xyz with / and @ signs in key value
def test_read_slash():
    Path('slash.xyz').write_text("""4
    key1=a key2=a/b key3=a@b key4="a@b"
    Mg        -4.25650        3.79180       -2.54123
    C         -1.15405        2.86652       -1.26699
    C         -5.53758        3.70936        0.63504
    C         -7.28250        4.71303       -3.82016
    """)

    a = ase.io.read('slash.xyz')
    assert a.info['key1'] == r'a'
    assert a.info['key2'] == r'a/b'
    assert a.info['key3'] == r'a@b'
    assert a.info['key4'] == r'a@b'


def test_read_struct():
    struct = Atoms(
        'H4', pbc=[True, True, True],
        cell=[[4.00759, 0.0, 0.0],
              [-2.003795, 3.47067475, 0.0],
              [3.06349683e-16, 5.30613216e-16, 5.00307]],
        positions=[[-2.003795e-05, 2.31379473, 0.875437189],
                   [2.00381504, 1.15688001, 4.12763281],
                   [2.00381504, 1.15688001, 3.37697219],
                   [-2.003795e-05, 2.31379473, 1.62609781]],
    )
    struct.info = {'dataset': 'deltatest', 'kpoints': np.array([28, 28, 20]),
                   'identifier': 'deltatest_H_1.00',
                   'unique_id': '4cf83e2f89c795fb7eaf9662e77542c1'}
    ase.io.write('tmp.xyz', struct)


# Complex properties line. Keys and values that break with a regex parser.
# see https://gitlab.com/ase/ase/issues/53 for more info
def test_complex_key_val():
    complex_xyz_string = (
        ' '  # start with a separator
        'str=astring '
        'quot="quoted value" '
        'quote_special="a_to_Z_$%%^&*" '
        r'escaped_quote="esc\"aped" '
        'true_value '
        'false_value = F '
        'integer=22 '
        'floating=1.1 '
        'int_array={1 2 3} '
        'float_array="3.3 4.4" '
        'virial="1 4 7 2 5 8 3 6 9" '  # special 3x3, fortran ordering
        'not_a_3x3_array="1 4 7 2 5 8 3 6 9" '  # should be left as a 9-vector
        'Lattice="  4.3  0.0 0.0 0.0  3.3 0.0 0.0 0.0  7.0 " '  # spaces in arr
        'scientific_float=1.2e7 '
        'scientific_float_2=5e-6 '
        'scientific_float_array="1.2 2.2e3 4e1 3.3e-1 2e-2" '
        'not_array="1.2 3.4 text" '
        'bool_array={T F T F} '
        'bool_array_2=" T, F, T " '  # leading spaces
        'not_bool_array=[T F S] '
        # read and write
        # '\xfcnicode_key=val\xfce '  # fails on AppVeyor
        'unquoted_special_value=a_to_Z_$%%^&* '
        '2body=33.3 '
        'hyphen-ated '
        # parse only
        'many_other_quotes="4 8 12" '
        'comma_separated="7, 4, -1" '
        'bool_array_commas=[T, T, F, T] '
        'Properties=species:S:1:pos:R:3 '
        'multiple_separators       '
        'double_equals=abc=xyz '
        'trailing '
        '"with space"="a value" '
        r'space\"="a value" '
        # tests of JSON functionality
        'f_str_looks_like_array="[[1, 2, 3], [4, 5, 6]]" '
        'f_float_array="_JSON [[1.5, 2, 3], [4, 5, 6]]" '
        'f_int_array="_JSON [[1, 2], [3, 4]]" '
        'f_bool_bare '
        'f_bool_value=F '
        'f_dict={_JSON {"a" : 1}} '
    )

    expected_dict = {
        'str': 'astring',
        'quot': "quoted value",
        'quote_special': "a_to_Z_$%%^&*",
        'escaped_quote': 'esc"aped',
        'true_value': True,
        'false_value': False,
        'integer': 22,
        'floating': 1.1,
        'int_array': np.array([1, 2, 3]),
        'float_array': np.array([3.3, 4.4]),
        'virial': np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        'not_a_3x3_array': np.array([1, 4, 7, 2, 5, 8, 3, 6, 9]),
        'Lattice': np.array([[4.3, 0.0, 0.0],
                             [0.0, 3.3, 0.0],
                             [0.0, 0.0, 7.0]]),
        'scientific_float': 1.2e7,
        'scientific_float_2': 5e-6,
        'scientific_float_array': np.array([1.2, 2200, 40, 0.33, 0.02]),
        'not_array': "1.2 3.4 text",
        'bool_array': np.array([True, False, True, False]),
        'bool_array_2': np.array([True, False, True]),
        'not_bool_array': 'T F S',
        # '\xfcnicode_key': 'val\xfce',  # fails on AppVeyor
        'unquoted_special_value': 'a_to_Z_$%%^&*',
        '2body': 33.3,
        'hyphen-ated': True,
        'many_other_quotes': np.array([4, 8, 12]),
        'comma_separated': np.array([7, 4, -1]),
        'bool_array_commas': np.array([True, True, False, True]),
        'Properties': 'species:S:1:pos:R:3',
        'multiple_separators': True,
        'double_equals': 'abc=xyz',
        'trailing': True,
        'with space': 'a value',
        'space"': 'a value',
        'f_str_looks_like_array': '[[1, 2, 3], [4, 5, 6]]',
        'f_float_array': np.array([[1.5, 2, 3], [4, 5, 6]]),
        'f_int_array': np.array([[1, 2], [3, 4]]),
        'f_bool_bare': True,
        'f_bool_value': False,
        'f_dict': {"a": 1}
    }

    parsed_dict = extxyz.key_val_str_to_dict(complex_xyz_string)
    np.testing.assert_equal(parsed_dict, expected_dict)

    key_val_str = extxyz.key_val_dict_to_str(expected_dict)
    parsed_dict = extxyz.key_val_str_to_dict(key_val_str)
    np.testing.assert_equal(parsed_dict, expected_dict)

    # Round trip through a file with complex line.
    # Create file with the complex line and re-read it afterwards.
    with open('complex.xyz', 'w', encoding='utf-8') as f_out:
        f_out.write(f'1\n{complex_xyz_string}\nH 1.0 1.0 1.0')
    complex_atoms = ase.io.read('complex.xyz')

    # test all keys end up in info, as expected
    for key, value in expected_dict.items():
        if key in ['Properties', 'Lattice']:
            continue  # goes elsewhere
        else:
            np.testing.assert_equal(complex_atoms.info[key], value)


def test_write_multiple(atoms, images):
    # write multiple atoms objects to one xyz
    for atoms in images:
        atoms.write('append.xyz', append=True)
        atoms.write('comp_append.xyz.gz', append=True)
        atoms.write('not_append.xyz', append=False)
    readFrames = ase.io.read('append.xyz', index=slice(0, None))
    assert readFrames == images
    readFrames = ase.io.read('comp_append.xyz.gz', index=slice(0, None))
    assert readFrames == images
    singleFrame = ase.io.read('not_append.xyz', index=slice(0, None))
    assert singleFrame[-1] == images[-1]


# read xyz with blank comment line
def test_blank_comment():
    Path('blankcomment.xyz').write_text("""4

    Mg        -4.25650        3.79180       -2.54123
    C         -1.15405        2.86652       -1.26699
    C         -5.53758        3.70936        0.63504
    C         -7.28250        4.71303       -3.82016
    """)

    a = ase.io.read('blankcomment.xyz')
    assert a.info == {}


def test_escape():
    assert escape('plain_string') == 'plain_string'
    assert escape('string_containing_"') == r'"string_containing_\""'
    assert escape('string with spaces') == '"string with spaces"'


def test_stress():
    # build a water dimer, which has 6 atoms
    water1 = molecule('H2O')
    water2 = molecule('H2O')
    water2.positions[:, 0] += 5.0
    atoms = water1 + water2
    atoms.cell = [10, 10, 10]
    atoms.pbc = True

    atoms.calc = EMT()
    a_stress = atoms.get_stress()
    atoms.write('tmp.xyz')
    b = ase.io.read('tmp.xyz')
    assert abs(b.get_stress() - a_stress).max() < 1e-6


def test_json_scalars():
    a = bulk('Si')
    a.info['val_1'] = 42.0
    a.info['val_2'] = 42.0  # was np.float but that's the same.  Can remove
    a.info['val_3'] = np.int64(42)
    a.write('tmp.xyz')
    with open('tmp.xyz') as fd:
        comment_line = fd.readlines()[1]
    assert ("val_1=42.0" in comment_line
            and "val_2=42.0" in comment_line
            and "val_3=42" in comment_line)
    b = ase.io.read('tmp.xyz')
    assert abs(b.info['val_1'] - 42.0) < 1e-6
    assert abs(b.info['val_2'] - 42.0) < 1e-6
    assert abs(b.info['val_3'] - 42) == 0


@pytest.mark.parametrize('constraint', [FixAtoms(indices=(0, 2)),
                                        FixCartesian(1, mask=(1, 0, 1)),
                                        [FixCartesian(0), FixCartesian(2)]])
def test_constraints(constraint):
    atoms = molecule('H2O')
    atoms.set_constraint(constraint)

    columns = ['symbols', 'positions', 'move_mask']
    ase.io.write('tmp.xyz', atoms, columns=columns)

    atoms2 = ase.io.read('tmp.xyz')
    assert not compare_atoms(atoms, atoms2)

    constraint2 = atoms2.constraints
    cls = type(constraint)
    if cls == FixAtoms:
        assert len(constraint2) == 1
        assert isinstance(constraint2[0], cls)
        assert np.all(constraint2[0].index == constraint.index)
    elif cls == FixCartesian:
        assert len(constraint2) == len(atoms)
        assert isinstance(constraint2[0], cls)
        assert np.all(constraint2[0].mask)
        assert np.all(constraint2[1].mask == constraint.mask)
        assert np.all(constraint2[2].mask)
    elif cls is list:
        assert len(constraint2) == len(atoms)
        assert np.all(constraint2[0].mask == constraint[0].mask)
        assert np.all(constraint2[1].mask)
        assert np.all(constraint2[2].mask == constraint[1].mask)


def test_constraints_int():
    # check for regressions of issue #1015
    Path('movemask.xyz').write_text("""3
Properties=species:S:1:pos:R:3:move_mask:I:1 pbc="F F F"
O        0.00000000       0.00000000       0.11926200  1
H        0.00000000       0.76323900      -0.47704700  0
H        0.00000000      -0.76323900      -0.47704700  0""")

    a = ase.io.read('movemask.xyz')
    assert isinstance(a.constraints[0], FixAtoms)
    assert np.all(a.constraints[0].index == [1, 2])


# test read/write with both initial_charges & charges
@pytest.mark.parametrize("enable_initial_charges", [True, False])
@pytest.mark.parametrize("enable_charges", [True, False])
def test_write_read_charges(atoms, tmpdir, enable_initial_charges,
                            enable_charges):
    initial_charges = [1.0, -1.0]
    charges = [-2.0, 2.0]
    if enable_initial_charges:
        atoms.set_initial_charges(initial_charges)
    if enable_charges:
        atoms.calc = SinglePointCalculator(atoms, charges=charges)
        atoms.get_charges()
    ase.io.write(str(tmpdir / 'charge.xyz'), atoms, format='extxyz')
    r = ase.io.read(str(tmpdir / 'charge.xyz'))
    assert atoms == r
    if enable_initial_charges:
        assert np.allclose(r.get_initial_charges(), initial_charges)
    if enable_charges:
        assert np.allclose(r.get_charges(), charges)


@pytest.mark.parametrize("pbc,atoms_pbc", (
    ("True True True", [True, True, True]),
    ("True True False", [True, True, False]),
    ("False false T", [False, False, True]),
    ("True true T", [True, True, True]),
    ("True false T", [True, False, True]),
    ("F F F", [False, False, False]),
    ("T T F", [True, True, False]),
    ("True", [True, True, True]),
    ("False", [False, False, False]),
))
def test_pbc_property(pbc, atoms_pbc):
    """Test various specifications of the ``pbc`` property."""
    Path('pbc-test.xyz').write_text(f"""2
Lattice="3.608 0.0 0.0 -1.804 3.125 0.0 0.0 0.0 21.3114930844" pbc="{pbc}"
As           1.8043384632       1.0417352974      11.3518747709
As          -0.0000000002       2.0834705948       9.9596183135""")
    atoms = ase.io.read('pbc-test.xyz')
    assert (atoms.pbc == atoms_pbc).all()


def test_conflicting_fields():
    atoms = Atoms('Cu', cell=[2] * 3, pbc=[True] * 3)
    atoms.calc = EMT()

    _ = atoms.get_potential_energy()
    atoms.info["energy"] = 100
    # info / per-config conflict
    with pytest.raises(KeyError):
        ase.io.write(sys.stdout, atoms, format="extxyz")

    atoms = Atoms('Cu', cell=[2] * 3, pbc=[True] * 3)
    atoms.calc = EMT()

    _ = atoms.get_forces()
    atoms.new_array("forces", np.ones(atoms.positions.shape))
    # arrays / per-atom conflict
    with pytest.raises(KeyError):
        ase.io.write(sys.stdout, atoms, format="extxyz")


def test_save_calc_results():
    # DEFAULT (class name)
    atoms = Atoms('Cu', cell=[2] * 3, pbc=[True] * 3)
    atoms.calc = EMT()
    _ = atoms.get_potential_energy()

    calc_prefix = atoms.calc.__class__.__name__ + '_'
    save_calc_results(atoms, remove_atoms_calc=True)
    # make sure calculator was removed
    assert atoms.calc is None

    # make sure info/arrays keys with right names exist
    assert calc_prefix + 'energy' in atoms.info
    assert calc_prefix + 'forces' in atoms.arrays

    # EXPLICIT STRING
    atoms = Atoms('Cu', cell=[2] * 3, pbc=[True] * 3)
    atoms.calc = EMT()
    _ = atoms.get_potential_energy()

    calc_prefix = 'REF_'
    save_calc_results(atoms, calc_prefix=calc_prefix)
    # make sure calculator was not removed
    assert atoms.calc is not None

    # make sure info/arrays keys with right names exist
    assert calc_prefix + 'energy' in atoms.info
    assert calc_prefix + 'forces' in atoms.arrays

    # make sure conflicting field names raise an error
    with pytest.raises(KeyError):
        save_calc_results(atoms, calc_prefix=calc_prefix)

    # make sure conflicting field names do not raise an error when force=True
    save_calc_results(atoms, calc_prefix=calc_prefix, force=True)


def test_basic_functionality(tmp_path):
    atoms = Atoms('Cu2', cell=[4, 2, 2], positions=[[0, 0, 0], [2.05, 0, 0]],
                  pbc=[True] * 3)
    atoms.calc = EMT()
    atoms.get_potential_energy()

    atoms.info["REF_energy"] = 5

    ase.io.write(tmp_path / 'test.xyz', atoms)
    with open(tmp_path / 'test.xyz') as fin:
        for line_i, line in enumerate(fin):
            if line_i == 0:
                assert line.strip() == str(len(atoms))
            elif line_i == 1:
                assert ('Properties=species:S:1:pos:R:3:'
                        'energies:R:1:forces:R:3') in line
                assert 'energy=' in line
                assert 'stress=' in line
                assert 'REF_energy=' in line
            else:
                assert len(line.strip().split()) == 1 + 3 + 1 + 3


def test_linear_combination_calculator():
    """Test if results from `LinearCombinationCalculator` can be written

    `LinearCombinationCalculator` has non-standard properties like
    `energy_contributions` in `results`. Here we check if this causes errors.
    """
    atoms = bulk('Cu')
    atoms.calc = LinearCombinationCalculator([EMT()], [1.0])
    atoms.get_potential_energy()
    atoms.write('tmp.xyz')


def test_outputs_not_properties(tmp_path):
    atoms = Atoms('Cu2', cell=[4, 2, 2], positions=[[0, 0, 0], [2.05, 0, 0]],
                  pbc=[True] * 3, info={'nbands': 1})
    ase.io.write(tmp_path / 'nbands.extxyz', atoms)
    _ = ase.io.read(tmp_path / 'nbands.extxyz')
