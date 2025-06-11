# flake8: noqa
import numpy as np
import pytest

import ase.io.vasp_parsers.vasp_outcar_parsers as vop
from ase.io import ParseError


def compare_result_to_expected(result, exp):
    """Helper function for doing comparisons between
    and expected result, and the actual result"""
    if isinstance(exp, (np.ndarray, list, tuple)):
        assert len(result) == len(exp)
        for v1, v2 in zip(result, exp):
            if isinstance(v2, str):
                # Compare strings directly
                assert v1 == v2
            else:
                # Assume numerical stuff
                np.allclose(v1, v2)
    elif exp is None or isinstance(exp, bool):
        assert result == exp
    else:
        assert pytest.approx(result) == exp


@pytest.fixture
def do_test_parser():
    def _do_test_parser(header, cursor, lines, parser, expected):
        parser.header = header
        assert parser.has_property(cursor, lines)
        result = parser.parse(cursor, lines)
        for k, v in result.items():
            exp = expected[k]
            compare_result_to_expected(v, exp)

    return _do_test_parser


@pytest.fixture
def do_test_header_parser():
    def _do_test_parser(cursor, lines, parser, expected):
        assert parser.has_property(cursor, lines)
        result = parser.parse(cursor, lines)
        for k, v in result.items():
            exp = expected[k]
            compare_result_to_expected(v, exp)

    return _do_test_parser


@pytest.fixture
def do_test_stress(do_test_parser):
    def _do_test_stress(line, expected):
        parser = vop.Stress()
        cursor = 0
        lines = [line]
        header = {}
        expected = {'stress': expected}
        do_test_parser(header, cursor, lines, parser, expected)

    return _do_test_stress


@pytest.mark.parametrize('stress, expected', [([1, 2, 3, 4, 5, 6], [
    -0.00062415, -0.0012483, -0.00187245, -0.00312075, -0.00374491, -0.0024966
])])
def test_convert_stress(stress, expected):
    """Test the stress conversion function"""
    assert np.allclose(vop.convert_vasp_outcar_stress(stress), expected)


L1 = ("  in kB      -4.29429    -4.58894    -4.50342 "
      "    0.50047    -0.94049     0.36481")
L2 = ("  in kB     -47.95544   -39.91706   -34.79627  "
      "   9.20266   -15.74132    -1.85167")


@pytest.mark.parametrize(
    'line, expected',
    [
        (L1,
         [-4.29429, -4.58894, -4.50342, 0.50047, -0.94049, 0.36481]),
        (L2,
         [-47.95544, -39.91706, -34.79627, 9.20266, -15.74132, -1.85167])],
    ids=['stress1', 'stress2'])
def test_stress(line, expected, do_test_stress):
    """Test reading a particular line for parsing stress"""
    do_test_stress(line, vop.convert_vasp_outcar_stress(expected))


@pytest.mark.parametrize(
    'line, expected',
    [
        # Problematic line
        ("  in kB  358197.07841357849.97016357508.47884 19769.97820-30359.31165-19835.82336",
         None),
    ],
    ids=['stress1'])
def test_stress_problematic(line, expected, do_test_stress):
    with pytest.warns(UserWarning):
        do_test_stress(line, expected)


def test_cell_parser(do_test_parser):
    lines = """
    direct lattice vectors                 reciprocal lattice vectors
    17.934350000  0.000000000  0.000000000     0.055758921  0.000000000  0.000000000
     0.000000000 17.934350000  0.000000000     0.000000000  0.055758921  0.000000000
     0.000000000  0.000000000 17.934350000     0.000000000  0.000000000  0.055758921
    """
    lines = lines.splitlines()
    cursor = 1
    header = {}
    expected = {'cell': np.diag(3 * [17.934350000])}
    parser = vop.Cell()
    do_test_parser(header, cursor, lines, parser, expected)


def test_position_and_forces(do_test_parser):
    lines = """
    POSITION                                       TOTAL-FORCE (eV/Angst)
    -----------------------------------------------------------------------------------
        -1.48687      1.72231      1.61649       -57.777920    114.691339     68.153037
        -0.71946      0.83302      0.79517        57.777920   -114.691339    -68.153037
        0.01288      1.67248      1.67648        59.172191     73.797754    121.926341
        0.71933      0.79146      0.82334       -59.172191    -73.797754   -121.926341
        1.46928     -0.04570      0.04408         0.000000     -0.000000     -0.000000
        0.00000     -0.00000     -0.00000         0.000000     -0.000000      0.000000
    -----------------------------------------------------------------------------------
        total drift:                               -0.000000     -0.000000     -0.000000
    """
    lines = lines.splitlines()
    cursor = 1
    header = {'natoms': 6}

    expected_pos = [
        [-1.48687, 1.72231, 1.61649],
        [-0.71946, 0.83302, 0.79517],
        [0.01288, 1.67248, 1.67648],
        [0.71933, 0.79146, 0.82334],
        [1.46928, -0.04570, 0.04408],
        [0, 0, 0],
    ]
    expected_forces = [
        [-57.777920, 114.691339, 68.153037],
        [57.777920, -114.691339, -68.153037],
        [59.172191, 73.797754, 121.926341],
        [-59.172191, -73.797754, -121.926341],
        [0, 0, 0],
        [0, 0, 0],
    ]
    expected = {'positions': expected_pos, 'forces': expected_forces}

    parser = vop.PositionsAndForces()
    do_test_parser(header, cursor, lines, parser, expected)


def test_magmom(do_test_parser):
    lines = """
     number of electron     180.0000000 magnetization      18.0000000
     """
    lines = lines.splitlines()
    cursor = 1
    header = {}
    expected = {'magmom': 18}

    parser = vop.Magmom()
    do_test_parser(header, cursor, lines, parser, expected)


def test_non_collinear_magmom(do_test_parser):
    lines = """
     number of electron      31.9999969 magnetization       6.0504868     -0.0017494     -0.0024904
     """
    lines = lines.splitlines()
    cursor = 1
    header = {}
    expected = {'magmom': [6.0504868, -0.0017494, -0.0024904]}

    parser = vop.Magmom()
    do_test_parser(header, cursor, lines, parser, expected)


def test_magmom_wrong_line():
    """Test a line which we test should not be read as magmom"""
    lines = ['   NELECT =     180.0000    total number of electrons']
    cursor = 0
    parser = vop.Magmom()
    assert not parser.has_property(cursor, lines)


def test_magmoms(do_test_parser):
    lines = """
     magnetization (x)
 
    # of ion       s       p       d       tot
    ------------------------------------------
        1       -0.019  -0.019   0.000  -0.038
        2       -0.019  -0.019   0.000  -0.038
        3       -0.021  -0.016  -0.000  -0.038
        4       -0.021  -0.016  -0.000  -0.038
        5       -0.027  -0.030   0.004  -0.053
        6        0.019   0.023   3.188   3.231
    --------------------------------------------------
    tot         -0.087  -0.078   3.191   3.026
    """
    lines = lines.splitlines()
    cursor = 1
    header = {'natoms': 6}
    expected = {'magmoms': [-0.038, -0.038, -0.038, -0.038, -0.053, 3.231]}

    parser = vop.Magmoms()
    do_test_parser(header, cursor, lines, parser, expected)


def test_non_collinearmagmoms(do_test_parser):
    lines = """
     magnetization (x)

    # of ion       s       p       d       tot
    ------------------------------------------
        1        0.143   0.564   4.442   5.149
        2        0.143   0.564   4.441   5.148
    --------------------------------------------------
    tot          0.285   1.129   8.883  10.297



     magnetization (y)

    # of ion       s       p       d       tot
    ------------------------------------------
        1       -0.000  -0.000  -0.005  -0.005
        2        0.000   0.000   0.002   0.002
    --------------------------------------------------
    tot         -0.000   0.000  -0.003  -0.003



     magnetization (z)

    # of ion       s       p       d       tot
    ------------------------------------------
        1       -0.000   0.000  -0.004  -0.004
        2        0.000   0.000   0.001   0.001
    --------------------------------------------------
    tot         -0.000   0.001  -0.004  -0.003
    """
    lines = lines.splitlines()
    cursor = 1
    header = {'natoms': 2}
    expected = {'magmoms': [[5.149, -0.005, -0.004], [5.148, 0.002, 0.001]]}
    parser = vop.Magmoms()
    do_test_parser(header, cursor, lines, parser, expected)


def test_energy(do_test_parser):
    lines = """
  FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)
  ---------------------------------------------------
  free  energy   TOTEN  =       -68.22868532 eV

  energy  without entropy=      -68.23570214  energy(sigma->0) =      -68.23102426
    """
    lines = lines.splitlines()
    cursor = 1
    header = {}
    expected = {'free_energy': -68.22868532, 'energy': -68.23102426}

    parser = vop.Energy()
    do_test_parser(header, cursor, lines, parser, expected)


def test_efermi(do_test_parser):
    lines = """
     E-fermi :  -3.7404     XC(G=0):  -1.0024     alpha+bet : -0.5589
    """
    lines = lines.splitlines()
    cursor = 1
    header = {}
    expected = {'efermi': -3.7404}
    parser = vop.EFermi()
    do_test_parser(header, cursor, lines, parser, expected)


def test_kpoints():
    # Note: The following lines have been manually adjusted from an OUTCAR to
    # compress the test, as there is usually quite a lot of lines here
    lines = """
 spin component 1

 k-point   1 :       0.0000    0.0000    0.0000
  band No.  band energies     occupation
      1      -9.9948      0.80000
      2      -8.2511      0.50000

 k-point   2 :       0.5000    0.5000    0.5000
  band No.  band energies     occupation
      1      -9.9837      1.00000
      2      -1.2511      0.00000

 spin component 2

 k-point   1 :       0.0000    0.0000    0.0000
  band No.  band energies     occupation
      1      -9.9948      1.00000
      2      -8.2511      1.00000

 k-point   2 :       0.5000    0.5000    0.5000
  band No.  band energies     occupation
      1      -9.9948      1.00000
      2      -1.2511      1.00000
    """
    lines = lines.splitlines()
    cursor = 1
    header = {
        'nbands': 2,
        'spinpol': True,
        'nkpts': 2,
        'kpt_weights': [1, 0.75]
    }

    parser = vop.Kpoints(header=header)
    assert parser.has_property(cursor, lines)

    kpts = parser.parse(cursor, lines)['kpts']

    # Some expected values
    exp_s = [0, 0, 1, 1]  # spin
    exp_w = [1, 0.75, 1, 0.75]  # weights
    exp_f_n = [
        [0.8, 0.5],
        [1.0, 0],
        [1, 1],
        [1, 1],
    ]
    exp_eps_n = [
        [-9.9948, -8.2511],
        [-9.9837, -1.2511],
        [-9.9948, -8.2511],
        [-9.9948, -1.2511],
    ]
    # Test the first two kpoints
    for i, kpt in enumerate(kpts):
        assert kpt.s == exp_s[i]
        assert kpt.weight == pytest.approx(exp_w[i])
        assert np.allclose(kpt.eps_n, exp_eps_n[i])
        assert np.allclose(kpt.f_n, exp_f_n[i])


def test_kpoints_header_multiple_lines():
    # Correct line
    line1 = "k-points           NKPTS =      2   k-points in BZ     NKDIM =      2   number of bands    NBANDS=    336"
    # Wrong line (unclear when this is written.)
    line2 = "k-points           NKPTS =      1   k-points in BZ     NKDIM ="

    parser = vop.KpointHeader()

    assert parser.has_property(0, [line1])
    assert not parser.has_property(0, [line2])


def test_kpoints_header(do_test_header_parser):
    lines = """
   k-points           NKPTS =     63   k-points in BZ     NKDIM =     63   number of bands    NBANDS=     32

Here, a bunch of stuff follows, which should just be skipped automatically.
<snipped>
<More stuff is snipped>

k-points in reciprocal lattice and weights: KPOINTS created by Atomic Simulation Env
   0.00000000  0.00000000  0.00000000       0.008
   0.20000000  0.00000000  0.00000000       0.016
   0.40000000  0.00000000  0.00000000       0.016
   0.00000000  0.20000000  0.00000000       0.016
   0.20000000  0.20000000  0.00000000       0.016
   0.40000000  0.20000000  0.00000000       0.016
  -0.40000000  0.20000000  0.00000000       0.016
  -0.20000000  0.20000000  0.00000000       0.016
   0.00000000  0.40000000  0.00000000       0.016
   0.20000000  0.40000000  0.00000000       0.016
   0.40000000  0.40000000  0.00000000       0.016
  -0.40000000  0.40000000  0.00000000       0.016
  -0.20000000  0.40000000  0.00000000       0.016
   0.00000000  0.00000000  0.20000000       0.016
   0.20000000  0.00000000  0.20000000       0.016
   0.40000000  0.00000000  0.20000000       0.016
  -0.40000000  0.00000000  0.20000000       0.016
  -0.20000000  0.00000000  0.20000000       0.016
   0.00000000  0.20000000  0.20000000       0.016
   0.20000000  0.20000000  0.20000000       0.016
   0.40000000  0.20000000  0.20000000       0.016
  -0.40000000  0.20000000  0.20000000       0.016
  -0.20000000  0.20000000  0.20000000       0.016
   0.00000000  0.40000000  0.20000000       0.016
   0.20000000  0.40000000  0.20000000       0.016
   0.40000000  0.40000000  0.20000000       0.016
  -0.40000000  0.40000000  0.20000000       0.016
  -0.20000000  0.40000000  0.20000000       0.016
   0.00000000 -0.40000000  0.20000000       0.016
   0.20000000 -0.40000000  0.20000000       0.016
   0.40000000 -0.40000000  0.20000000       0.016
  -0.40000000 -0.40000000  0.20000000       0.016
  -0.20000000 -0.40000000  0.20000000       0.016
   0.00000000 -0.20000000  0.20000000       0.016
   0.20000000 -0.20000000  0.20000000       0.016
   0.40000000 -0.20000000  0.20000000       0.016
  -0.40000000 -0.20000000  0.20000000       0.016
  -0.20000000 -0.20000000  0.20000000       0.016
   0.00000000  0.00000000  0.40000000       0.016
   0.20000000  0.00000000  0.40000000       0.016
   0.40000000  0.00000000  0.40000000       0.016
  -0.40000000  0.00000000  0.40000000       0.016
  -0.20000000  0.00000000  0.40000000       0.016
   0.00000000  0.20000000  0.40000000       0.016
   0.20000000  0.20000000  0.40000000       0.016
   0.40000000  0.20000000  0.40000000       0.016
  -0.40000000  0.20000000  0.40000000       0.016
  -0.20000000  0.20000000  0.40000000       0.016
   0.00000000  0.40000000  0.40000000       0.016
   0.20000000  0.40000000  0.40000000       0.016
   0.40000000  0.40000000  0.40000000       0.016
  -0.40000000  0.40000000  0.40000000       0.016
  -0.20000000  0.40000000  0.40000000       0.016
   0.00000000 -0.40000000  0.40000000       0.016
   0.20000000 -0.40000000  0.40000000       0.016
   0.40000000 -0.40000000  0.40000000       0.016
  -0.40000000 -0.40000000  0.40000000       0.016
  -0.20000000 -0.40000000  0.40000000       0.016
   0.00000000 -0.20000000  0.40000000       0.016
   0.20000000 -0.20000000  0.40000000       0.016
   0.40000000 -0.20000000  0.40000000       0.016
  -0.40000000 -0.20000000  0.40000000       0.016
  -0.20000000 -0.20000000  0.40000000       0.016

  """
    lines = lines.splitlines()
    cursor = 1
    expected = {
        'nkpts':
        63,
        'nbands':
        32,
        'ibzkpts':
        np.array([[0., 0., 0.], [0.2, 0., 0.], [0.4, 0., 0.], [0., 0.2, 0.],
                  [0.2, 0.2, 0.], [0.4, 0.2, 0.], [-0.4, 0.2, 0.],
                  [-0.2, 0.2, 0.], [0., 0.4, 0.], [0.2, 0.4,
                                                   0.], [0.4, 0.4, 0.],
                  [-0.4, 0.4, 0.], [-0.2, 0.4, 0.], [0., 0., 0.2],
                  [0.2, 0., 0.2], [0.4, 0., 0.2], [-0.4, 0., 0.2],
                  [-0.2, 0., 0.2], [0., 0.2, 0.2], [0.2, 0.2, 0.2],
                  [0.4, 0.2, 0.2], [-0.4, 0.2, 0.2], [-0.2, 0.2, 0.2],
                  [0., 0.4, 0.2], [0.2, 0.4, 0.2], [0.4, 0.4, 0.2],
                  [-0.4, 0.4, 0.2], [-0.2, 0.4, 0.2], [0., -0.4, 0.2],
                  [0.2, -0.4, 0.2], [0.4, -0.4, 0.2], [-0.4, -0.4, 0.2],
                  [-0.2, -0.4, 0.2], [0., -0.2, 0.2], [0.2, -0.2, 0.2],
                  [0.4, -0.2, 0.2], [-0.4, -0.2, 0.2], [-0.2, -0.2, 0.2],
                  [0., 0., 0.4], [0.2, 0., 0.4], [0.4, 0., 0.4],
                  [-0.4, 0., 0.4], [-0.2, 0., 0.4], [0., 0.2, 0.4],
                  [0.2, 0.2, 0.4], [0.4, 0.2, 0.4], [-0.4, 0.2, 0.4],
                  [-0.2, 0.2, 0.4], [0., 0.4, 0.4], [0.2, 0.4, 0.4],
                  [0.4, 0.4, 0.4], [-0.4, 0.4, 0.4], [-0.2, 0.4, 0.4],
                  [0., -0.4, 0.4], [0.2, -0.4, 0.4], [0.4, -0.4, 0.4],
                  [-0.4, -0.4, 0.4], [-0.2, -0.4, 0.4], [0., -0.2, 0.4],
                  [0.2, -0.2, 0.4], [0.4, -0.2, 0.4], [-0.4, -0.2, 0.4],
                  [-0.2, -0.2, 0.4]]),
        'kpt_weights':
        np.array([
            0.008, 0.016, 0.016, 0.016, 0.016, 0.016, 0.016, 0.016, 0.016,
            0.016, 0.016, 0.016, 0.016, 0.016, 0.016, 0.016, 0.016, 0.016,
            0.016, 0.016, 0.016, 0.016, 0.016, 0.016, 0.016, 0.016, 0.016,
            0.016, 0.016, 0.016, 0.016, 0.016, 0.016, 0.016, 0.016, 0.016,
            0.016, 0.016, 0.016, 0.016, 0.016, 0.016, 0.016, 0.016, 0.016,
            0.016, 0.016, 0.016, 0.016, 0.016, 0.016, 0.016, 0.016, 0.016,
            0.016, 0.016, 0.016, 0.016, 0.016, 0.016, 0.016, 0.016, 0.016
        ])
    }

    parser = vop.KpointHeader()

    # Test we got some expected results
    do_test_header_parser(cursor, lines, parser, expected)


@pytest.mark.parametrize(
    'line, expected',
    [
        (' POTCAR:    PAW_PBE Ni 02Aug2007', ['Ni']),
        (' POTCAR:    PAW_PBE Fe_pv 02Aug2007', ['Fe']),
        (' POTCAR:    H  1/r potential', ['H']),  # The H_AE POTCAR
        (' POTCAR:    PAW_PBE H1.25 07Sep2000', ['H']),
        # Non-PBE potential
        (' POTCAR:    PAW Ca_sv_GW 31Mar2010', ['Ca']),
    ])
def test_parse_potcar_in_outcar(line, expected, do_test_header_parser):
    cursor = 0
    lines = [line]
    parser = vop.SpeciesTypes()
    expected = {'species': expected}
    do_test_header_parser(cursor, lines, parser, expected)


@pytest.mark.parametrize(
    'line',
    [
        ' POTCAR:    PAW_PBE Nis 02Aug2007',  # Purely made-up typo in the element
        ' POTCAR:    PAW_PBE M 02Aug2007',  # Purely made-up typo in the element
    ])
def test_parse_potcar_parse_error(line):
    """Test that we raise a ParseError for a corrupted POTCAR line.
    Note, that this line is purely made-up, just to test a crash"""
    cursor = 0
    lines = [line]
    parser = vop.SpeciesTypes()
    with pytest.raises(ParseError):
        parser.parse(cursor, lines)


@pytest.mark.parametrize(
    'line, expected',
    [
        ('   ions per type =              32  31   2', (32, 31, 2)),
        ('   ions per type =               2   4', (2, 4)),
    ],
    # Add ID, as the line is a little long, looks quite verbose
    ids=['ions1', 'ions2'])
def test_ions_per_species(line, expected, do_test_header_parser):
    cursor = 0  # single line, cursor always starts at 0
    lines = [line]
    parser = vop.IonsPerSpecies()
    expected = {'ion_types': expected}
    do_test_header_parser(cursor, lines, parser, expected)


def test_potcar_repeated_entry():
    """Test reading an OUTCAR where we have repeated "POTCAR:" entries.
    We should only expect to insert every second entry.
    """

    lines = """
    POTCAR:    PAW_PBE Ni 02Aug2007
    POTCAR:    PAW_PBE H1.25 02Aug2007
    POTCAR:    PAW_PBE Au_GW 02Aug2007
    POTCAR:    PAW_PBE Ni 02Aug2007
    POTCAR:    PAW_PBE H1.25 02Aug2007 
    POTCAR:    PAW_PBE Au_GW 02Aug2007
    """
    # Prepare input as list of strings
    lines = lines.splitlines()[1:]

    # Emulate the parser, reading the lines 1-by-1
    parser = vop.SpeciesTypes()
    for line in lines:
        if not line.strip():
            # Blank line, just skip
            continue
        line = [line]

        assert parser.has_property(0, line)
        parser.parse(0, line)
    assert len(parser.species) == 6
    assert parser.species == ['Ni', 'H', 'Au', 'Ni', 'H', 'Au']
    assert len(parser.get_species()) == 3

    assert parser.get_species() == ['Ni', 'H', 'Au']


def test_default_header_parser_make_parsers():
    """Test we can make two sets of identical parsers,
    but that we do not actually return the same parser
    instances
    """
    parsers1 = vop.default_header_parsers.make_parsers()
    parsers2 = vop.default_header_parsers.make_parsers()

    assert len(parsers1) > 0
    assert len(parsers1) == len(parsers2)
    # Test we made all of the parsers
    assert len(parsers1) == len(vop.default_header_parsers.parsers_dct)

    # Compare parsers
    for p1, p2 in zip(parsers1, parsers2):
        # We should've made instances of the same type
        assert type(p1) == type(p2)
        assert p1.get_name() == p2.get_name()
        if isinstance(p1, vop.SimpleProperty):
            assert p1.LINE_DELIMITER == p2.LINE_DELIMITER
            assert p1.LINE_DELIMITER is not None
        # However, they should not actually BE the same parser
        # but separate instances, i.e. two separate memory addresses
        assert p1 is not p2


def test_vasp6_kpoints_reading():
    """Vasp6 v6.2 introduced a new line in the kpoints lines.
    Verify we can read them.
    """

    lines = """
     spin component 1

    k-point     1 :       0.0000    0.0000   0.0000
    band No.  band energies     occupation
        1      -10.000      1.00000
        2       0.0000      1.00000

    k-point     2 :       0.1250    0.0417    0.0417
    band No.  band energies     occupation
        1      -10.000      1.00000
        2       -5.000      1.00000
     Fermi energy:         -6.789

     spin component 2

    k-point     1 :       0.0000    0.0000   0.0000
    band No.  band energies     occupation
        1      -10.000      1.00000
        2       0.0000      1.00000

    k-point     2 :       0.1250    0.0417    0.0417
    band No.  band energies     occupation
        1      -10.000      1.00000
        2       -5.000      1.00000
     Fermi energy:         -8.123

    """
    lines = lines.splitlines()
    cursor = 1
    header = {
        'nbands': 2,
        'spinpol': True,
        'nkpts': 2,
        'kpt_weights': [1, 0.75]
    }

    parser = vop.Kpoints(header=header)
    assert parser.has_property(cursor, lines)

    kpts = parser.parse(cursor, lines)['kpts']

    # Some expected values
    exp_s = [0, 0, 1, 1]  # spin
    exp_w = 2 * [1, 0.75]  # weights
    exp_eps_n = [
        [-10, 0.],
        [-10, -5],
        [-10, 0],
        [-10, -5],
    ]
    exp_f_n = 4 * [[1.0, 1.0]]
    # Test the first two kpoints
    for i, kpt in enumerate(kpts):
        assert kpt.s == exp_s[i]
        assert kpt.weight == pytest.approx(exp_w[i])
        assert np.allclose(kpt.eps_n, exp_eps_n[i])
        assert np.allclose(kpt.f_n, exp_f_n[i])
