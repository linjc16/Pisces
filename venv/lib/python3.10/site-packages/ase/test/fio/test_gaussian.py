import copy
from io import StringIO

import numpy as np
import pytest

from ase.atoms import Atoms
from ase.calculators.calculator import InputError
from ase.calculators.gaussian import Gaussian
from ase.io import ParseError
from ase.io.gaussian import (
    _get_atoms_info,
    _get_cartesian_atom_coords,
    _get_zmatrix_line,
    _re_chgmult,
    _re_link0,
    _re_method_basis,
    _re_nuclear_props,
    _re_output_type,
    _validate_symbol_string,
    read_gaussian_in,
)


# Refactoring needed.
#
# Lots of calculators are instantiated here even though no calculations
# are taking place.  These tests should instead use the GaussianConfiguration
# class to ensure that all the formatting/parsing works well.
@pytest.fixture(autouse=True)
def hack_gaussian_command(monkeypatch):
    monkeypatch.setenv('ASE_GAUSSIAN_COMMAND', '_does_not_exist_')


@pytest.fixture()
def fd_cartesian():
    # make an example input string with cartesian coords:
    fd_cartesian = StringIO('''
    %chk=example.chk
    %Nprocshared=16
    # N B3LYP/6-31G(d',p') ! ASE formatted method and basis
    # POpt(Tight, MaxCyc=100)/Integral=Ultrafine

    Gaussian input prepared by ASE

    0 1
    8,  -0.464,   0.177,   0.0
    1(iso=0.1134289259, NMagM=-8.89, ZEff=-1), -0.464,   1.137,   0.0
    1(iso=2, spin=1, QMom=1, RadNuclear=1, ZNuc=2),   0.441,  -0.143,   0.0
    TV        10.0000000000        0.0000000000        0.0000000000
    TV         0.0000000000       10.0000000000        0.0000000000
    TV         0.0000000000        0.0000000000       10.0000000000

    ''')
    return fd_cartesian


_basis_set_text = '''H     0
S    2   1.00
    0.5447178000D+01       0.1562849787D+00
    0.8245472400D+00       0.9046908767D+00
S    1   1.00
    0.1831915800D+00       1.0000000
****
O     0
S    6   1.00
    0.5472270000D+04       0.1832168810D-02
    0.8178060000D+03       0.1410469084D-01
    0.1864460000D+03       0.6862615542D-01
    0.5302300000D+02       0.2293758510D+00
    0.1718000000D+02       0.4663986970D+00
    0.5911960000D+01       0.3641727634D+00
SP   2   1.00
    0.7402940000D+01      -0.4044535832D+00       0.2445861070D+00
    0.1576200000D+01       0.1221561761D+01       0.8539553735D+00
SP   1   1.00
    0.3736840000D+00       0.1000000000D+01       0.1000000000D+01
****'''


@pytest.fixture()
def fd_cartesian_basis_set():
    # make an example input string with cartesian coords and a basis set
    # definition:
    fd_cartesian_basis_set = StringIO('''
    %chk=example.chk
    %Nprocshared=16
    %Save
    # N g1/Gen/TZVPFit ! ASE formatted method and basis
    # Opt(Tight MaxCyc=100) Integral=Ultrafine
    Frequency=(ReadIsotopes, Anharmonic)

    Gaussian input prepared by ASE

    0 1
    O1  -0.464   0.177   0.0
    H1  -0.464   1.137   0.0
    H2   0.441  -0.143   0.0

    300 1.0 1.0

    0.1134289259 ! mass of first H
    ! test comment
    2 ! mass of 2nd hydrogen
    ! test comment


''' + _basis_set_text + '\n')

    return fd_cartesian_basis_set


_zmatrix_file_text = '''
    %chk=example.chk
    %Nprocshared=16
    # T B3LYP/Gen
    # opt=(Tight, MaxCyc=100) integral(Ultrafine) freq=ReadIso

    Gaussian input with Z matrix

    0 1
    B 0 0.00 0.00 0.00
    H 0 1.31 0.00 0.00
    H 1 r1 2 a1
    B 2 r1 1 a2 3 0
    ! test comment inside the z-matrix
    H 1 r2 4 a3 2 90
    H 1 r2 4 a3 2 -90
    H 4 r2 1 a3 2 90
    H 4 r2 1 a3 2 -90
    Variables:
    r1 1.31
    r2 1.19
    a1 97
    a2 83
    a3 120

    ! test comment after molecule spec.

    300 1.0

    0.1134289259 ! mass of first H

    @basis-set-filename.gbs

    '''


@pytest.fixture()
def fd_zmatrix():
    # make an example input string with a z-matrix:
    fd_zmatrix = StringIO(_zmatrix_file_text)
    return fd_zmatrix


@pytest.fixture()
def fd_incorrect_zmatrix_var():
    # Make an example input string with a z-matrix with
    # incorrect variable definitions
    incorrect_zmatrix_text = ""
    for i, line in enumerate(_zmatrix_file_text.split('\n')):
        if i == 10:
            # add in variable name that isn't defined:
            incorrect_zmatrix_text += 'H 1 test 2 a1 \n'
        elif i == 18:
            incorrect_zmatrix_text += 'Constants: \n'
        else:
            incorrect_zmatrix_text += line + '\n'

    return StringIO(incorrect_zmatrix_text)


def fd_incorrect_zmatrix_symbol():
    # Make an example input string with a z-matrix with
    # an unrecognised symbol
    incorrect_zmatrix_text = ""
    for i, line in enumerate(_zmatrix_file_text.split('\n')):
        if i == 10:
            # add in variable name that isn't defined:
            incorrect_zmatrix_text += 'UnknownSymbol 0 1.31 0.00 0.00\n'
        else:
            incorrect_zmatrix_text += line + '\n'

    return StringIO(incorrect_zmatrix_text)


def fd_unsupported_option():
    # Make an example string with an unsupported route option:
    unsupported_text = ""
    for i, line in enumerate(_zmatrix_file_text.split('\n')):
        if i == 4:
            # add in unsupported setting:
            unsupported_text += 'Geom=ModRedundant freq=ReadIso\n'
        else:
            unsupported_text += line + '\n'

    return StringIO(unsupported_text)


def fd_no_charge_mult():
    # Make an example input string without specifying charge and multiplicity:
    unsupported_text = ""
    for i, line in enumerate(_zmatrix_file_text.split('\n')):
        if i == 8:
            # add in unsupported setting:
            unsupported_text += ''
        else:
            unsupported_text += line + '\n'

    return StringIO(unsupported_text)


@pytest.fixture()
def fd_command_set():
    # Make an example input string where command is set in link0:
    unsupported_text = ""
    for i, line in enumerate(_zmatrix_file_text.split('\n')):
        if i == 1:
            # add in unsupported setting:
            unsupported_text += '%command = echo arbitrary_code_execution'
        else:
            unsupported_text += line + '\n'

    return StringIO(unsupported_text)


def _test_write_gaussian(atoms, params_expected, properties=None):
    '''Writes atoms to gaussian input file, reads this back in and
    checks that the resulting atoms object is equal to atoms and
    the calculator has parameters params_expected'''

    atoms.calc.label = 'gaussian_input_file'
    out_file = atoms.calc.label + '.com'

    atoms_expected = atoms.copy()

    atoms.calc.write_input(atoms, properties=properties)

    with open(out_file) as fd:
        atoms_written = read_gaussian_in(fd, True)
        if _get_iso_masses(atoms_written):
            atoms_written.set_masses(_get_iso_masses(atoms_written))
    _check_atom_properties(atoms_expected, atoms_written, params_expected)


def _check_atom_properties(atoms, atoms_new, params):
    ''' Checks that the properties of atoms is equal to the properties
    of atoms_new, and the parameters of atoms_new.calc is equal to params.'''
    assert np.all(atoms_new.numbers == atoms.numbers)
    assert np.allclose(atoms_new.get_masses(), atoms.get_masses())
    assert np.allclose(atoms_new.positions, atoms.positions, atol=1e-3)
    assert np.all(atoms_new.pbc == atoms.pbc)
    assert np.allclose(atoms_new.cell, atoms.cell)

    new_params = atoms_new.calc.parameters
    new_params_to_check = copy.deepcopy(new_params)
    params_to_check = copy.deepcopy(params)

    if 'basis_set' in params:
        # Makes sure both basis sets are formatted comparably for the test:
        params_to_check['basis_set'] = params_to_check['basis_set'].split(
            '\n')
        params_to_check['basis_set'] = [line.strip() for line in
                                        params_to_check['basis_set']]
        new_params_to_check['basis_set'] = new_params_to_check[
            'basis_set'].strip().split('\n')
    for key, value in new_params_to_check.items():
        params_equal = new_params_to_check.get(
            key) == params_to_check.get(key)
        if isinstance(params_equal, np.ndarray):
            assert (new_params_to_check.get(key)
                    == params_to_check.get(key)).all()
        else:
            assert (new_params_to_check.get(key)
                    == params_to_check.get(key))


def _get_iso_masses(atoms):
    if atoms.calc.parameters.get('isolist'):
        return list(atoms.calc.parameters['isolist'])


@pytest.fixture()
def cartesian_setup():
    positions = [[-0.464, 0.177, 0.0],
                 [-0.464, 1.137, 0.0],
                 [0.441, -0.143, 0.0]]
    cell = [[10., 0., 0.], [0., 10., 0.], [0., 0., 10.]]
    masses = [15.999, 0.1134289259, 2]

    atoms = Atoms('OH2', cell=cell, positions=positions,
                  masses=masses, pbc=True)

    params = {'chk': 'example.chk', 'nprocshared': '16',
              'output_type': 'n', 'method': 'b3lyp',
              'basis': "6-31g(d',p')", 'opt': 'tight, maxcyc=100',
              'integral': 'ultrafine', 'charge': 0, 'mult': 1,
              'isolist': np.array([None, 0.1134289259, 2])}

    return atoms, params


def test_read_write_gaussian_cartesian(fd_cartesian, cartesian_setup):
    '''Tests the read_gaussian_in and write_gaussian_in methods.
    For the input text given by each fixture we do the following:
    - Check reading in the text generates the Atoms object and Calculator that
      we would expect to get.
    - Check that writing out the resulting Atoms object and reading it back in
      generates the same Atoms object and parameters. '''

    # Tests reading a Gaussian input file with:
    # - Cartesian coordinates for the atom positions.
    # - ASE formatted method and basis
    # - PBCs
    # - All nuclei properties set
    # - Masses defined using nuclei properties
    atoms, params = cartesian_setup
    params['nmagmlist'] = np.array([None, -8.89, None])
    params['zefflist'] = np.array([None, -1, None])
    params['znuclist'] = np.array([None, None, 2])
    params['qmomlist'] = np.array([None, None, 1])
    params['radnuclearlist'] = np.array([None, None, 1])
    params['spinlist'] = np.array([None, None, 1])
    # Expect warning due to fragments not being supported:
    with pytest.warns(UserWarning):
        atoms_new = read_gaussian_in(fd_cartesian, True)
    atoms_new.set_masses(_get_iso_masses(atoms_new))
    _check_atom_properties(atoms, atoms_new, params)

    # Now we have tested reading the input, we can test writing it
    # and reading it back in.

    _test_write_gaussian(atoms_new, params)


def test_read_write_gaussian_cartesian_basis_set(fd_cartesian_basis_set,
                                                 cartesian_setup):
    # Tests reading a Gaussian input file with:
    # - Cartesian coordinates for the atom positions.
    # - ASE formatted method and basis
    # - Masses defined using readiso section
    atoms, params = cartesian_setup
    atoms.pbc = None
    atoms.cell = None
    iso_params = {'temperature': '300', 'pressure': '1.0', 'scale': '1.0'}
    params.update(iso_params)
    params['opt'] = 'tight maxcyc=100'
    params['frequency'] = 'anharmonic'
    params['basis'] = 'gen'
    params['method'] = 'g1'
    params['fitting_basis'] = 'tzvpfit'
    params['save'] = ''
    params['basis_set'] = _basis_set_text

    atoms_new = read_gaussian_in(fd_cartesian_basis_set, True)
    atoms_new.set_masses(_get_iso_masses(atoms_new))
    _check_atom_properties(atoms, atoms_new, params)

    # Now we have tested reading the input, we can test writing it
    # and reading it back in.
    # Expect warning due to g1 being a composite method:
    with pytest.warns(UserWarning):
        _test_write_gaussian(atoms_new, params)


def test_read_write_gaussian_zmatrix(fd_zmatrix):
    # Tests reading a Gaussian input file with:
    # - Z-matrix format for structure definition
    # - Variables in the Z-matrix
    # - Masses defined using 'ReadIso'
    # - Method and basis not formatted by ASE
    # - Basis file used instead of standard basis set.
    positions = np.array([
        [+0.000, +0.000, +0.000],
        [+1.310, +0.000, +0.000],
        [-0.160, +1.300, +0.000],
        [+1.150, +1.300, +0.000],
        [-0.394, -0.446, +1.031],
        [-0.394, -0.446, -1.031],
        [+1.545, +1.746, -1.031],
        [+1.545, +1.746, +1.031],
    ])
    masses = [None] * 8
    masses[1] = 0.1134289259
    atoms = Atoms('BH2BH4', positions=positions, masses=masses)

    params = {'chk': 'example.chk', 'nprocshared': '16', 'output_type': 't',
              'b3lyp': None, 'gen': None, 'opt': 'tight, maxcyc=100',
              'freq': None, 'integral': 'ultrafine', 'charge': 0, 'mult': 1,
              'temperature': '300', 'pressure': '1.0',
              'basisfile': '@basis-set-filename.gbs'}
    params['isolist'] = np.array(masses)

    # Note that although the freq is set to ReadIso in the input text,
    # here we have set it to None. This is because when reading in a file
    # this option does not get saved to the calculator, it instead saves
    # the temperature, pressure, scale and masses separately.

    # Test reading the gaussian input
    atoms_new = read_gaussian_in(fd_zmatrix, True)
    atoms_new.set_masses(_get_iso_masses(atoms_new))
    _check_atom_properties(atoms, atoms_new, params)

    # Now we have tested reading the input, we can test writing it
    # and reading it back in.

    # ASE does not support reading in output files in 'terse' format, so
    # it does not support writing input files with this setting. Therefore,
    # the output type automatically set to P in the case that T is chosen

    params['output_type'] = 'p'

    _test_write_gaussian(atoms_new, params)


def test_incorrect_mol_spec(fd_incorrect_zmatrix_var):
    ''' Tests that incorrect lines in the molecule
    specification fail to be read.'''
    # checks parse error raised when freezecode set:
    freeze_code_line = 'H 1 1.0 2.0 3.0'
    symbol, pos = _get_atoms_info(freeze_code_line)
    with pytest.raises(ParseError):
        _get_cartesian_atom_coords(symbol, pos)

    # checks parse error raised when 'alternate' z-matrix
    # definition is used
    incorrect_zmatrix = 'C4 O1 0.8 C2 121.4 O2 150.0 1'

    with pytest.raises(ParseError):
        _get_zmatrix_line(incorrect_zmatrix)

    incorrect_symbol = 'C1-7 0 1 2 3'
    # Checks parse error raised when
    # molecule specifications for molecular mechanics
    # calculations have been used.
    with pytest.raises(ParseError):
        _validate_symbol_string(incorrect_symbol)

    # Expect warning as constants aren't supported so they're
    # set as vars instead:
    with pytest.warns(UserWarning):
        # Expect error as undefined var appears in matrix:
        with pytest.raises(ParseError):
            read_gaussian_in(fd_incorrect_zmatrix_var, True)


@pytest.mark.parametrize("unsupported_file", [fd_incorrect_zmatrix_symbol(),
                                              fd_unsupported_option(),
                                              fd_no_charge_mult()])
def test_read_gaussian_in_errors(fd_command_set, unsupported_file):
    with pytest.raises(ParseError):
        read_gaussian_in(unsupported_file, True)


def test_read_gaussian_in_command(fd_command_set):
    # Expect error if 'command' is set in link0 section as this
    # would try to set the command for the calculator:
    with pytest.raises(TypeError):
        read_gaussian_in(fd_command_set, True)


def test_write_gaussian_calc():
    ''' Tests writing an input file for a Gaussian calculator. Reads this
    back in and checks that we get the parameters we expect.

    This allows testing of 'addsec', 'extra', 'ioplist',
    which we weren't able to test by reading and then writing files.'''

    # Generate an atoms object and calculator to test writing to a gaussian
    # input file:
    atoms = Atoms('H2', [[0, 0, 0], [0, 0, 0.74]])
    params = {'mem': '1GB', 'charge': 0, 'mult': 1, 'xc': 'PBE',
              'save': None, 'basis': 'EPR-III', 'scf': 'qc',
              'ioplist': ['1/2', '2/3'], 'freq': 'readiso',
              'addsec': '297 3 1', 'extra': 'Opt = Tight'}
    atoms.calc = Gaussian(**params)

    # Here we generate the params we expect to read back from the
    # input file:
    params_expected = {}
    for k, v in params.items():
        if v is None:
            params_expected[k] = ''
        elif type(v) in [list, int]:
            params_expected[k] = v
        else:
            params_expected[k] = v.lower()

    # We haven't specified the output type so it will be set to 'p'
    params_expected['output_type'] = 'p'

    # No method is set in the calculator, so the xc is used as the method.
    # The XC PBE should be converted to pbepbe automatically
    params_expected.pop('xc')
    params_expected['method'] = 'pbepbe'

    # The 'extra' text is added into the route section,
    # and will be read as a key-value pair.
    params_expected.pop('extra')
    params_expected['opt'] = 'tight'

    # Freq= ReadIso is always removed because ReadIso section
    # is converted into being saved in route section and nuclei
    # properties section:
    params_expected['freq'] = None

    # Addsec gets added to the end of the file.
    # We expect the addsec we have set to be read in
    # as the readiso section where temperature, pressure and scale are set.
    params_expected.pop('addsec')
    params_expected['temperature'] = '297'
    params_expected['pressure'] = '3'
    params_expected['scale'] = '1'

    # The IOPlist will be read in as a string:
    ioplist = params_expected.pop('ioplist')
    ioplist_txt = ''
    for iop in ioplist:
        ioplist_txt += iop + ', '
    ioplist_txt = ioplist_txt.strip(', ')
    params_expected['iop'] = ioplist_txt

    # We will request the forces property, so forces should be added to the
    # route section:
    params_expected['forces'] = None

    _test_write_gaussian(atoms, params_expected, properties='forces')

    calc = Gaussian(basis='gen')
    # Can't set basis to gen without defining basis set:
    with pytest.raises(InputError):
        calc.write_input(atoms)

    # Test case where we have the basis set in a separate file:
    basisfilename = 'basis.txt'
    with open(basisfilename, 'w+') as fd:
        fd.write(_basis_set_text)
    calc = Gaussian(basisfile=basisfilename, output_type='p',
                    mult=0, charge=1, basis='gen')
    atoms.calc = calc
    params_expected = calc.parameters
    params_expected['basis_set'] = _basis_set_text
    _test_write_gaussian(atoms, params_expected, properties='forces')


def test_read_gaussian_regex():
    ''' Test regex used in read_gaussian_in'''
    # Test link0 regex:
    link0_line = '%chk=example.chk'
    link0_match = _re_link0.match(link0_line)
    assert link0_match.group(1) == 'chk'
    assert link0_match.group(2) == 'example.chk'
    link0_line = '%chk'
    link0_match = _re_link0.match(link0_line)
    assert link0_match.group(1) == 'chk'
    assert link0_match.group(2) is None

    # Test output type regex:
    output_type_lines = ['#P B3LYP', ' #P', '# P']
    for line in output_type_lines:
        output_type_match = _re_output_type.match(line)
        assert output_type_match.group(1) == 'P'

    # Test method and basis regex:
    # On line with method/basis/fitting basis
    method_basis_line = 'g1/Gen/TZVPFit ! ASE formatted method and basis'
    method_basis_match = _re_method_basis.match(method_basis_line)
    assert method_basis_match.group(1) == 'g1'
    assert method_basis_match.group(2) == 'Gen'
    assert method_basis_match.group(4) == 'TZVPFit '
    assert method_basis_match.group(5) == '! ASE formatted method and basis'
    # On line with method/basis
    method_basis_line = 'g1/Gen ! ASE formatted method and basis'
    method_basis_match = _re_method_basis.match(method_basis_line)
    assert method_basis_match.group(1) == 'g1'
    assert method_basis_match.group(2) == 'Gen '
    assert method_basis_match.group(5) == '! ASE formatted method and basis'

    # Test charge and multiplicity regex - here we are just interested in
    #  whether we get a match
    chgmult_lines = ['0 1', ' 0 1', '0, 2']
    for line in chgmult_lines:
        assert _re_chgmult.match(line).group(0) == line

    # Test nuclear properties regex:
    nuclear_props = '(iso=0.1134289259, NMagM=-8.89, ZEff=-1)'
    nuclear_prop_line = f'1{nuclear_props}, -0.464,   1.137,   0.0'
    assert (_re_nuclear_props.search(nuclear_prop_line).group(0)
            == nuclear_props)
