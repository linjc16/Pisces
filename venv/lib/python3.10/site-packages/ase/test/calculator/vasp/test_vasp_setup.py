import pytest

from ase.atoms import Atoms

calc = pytest.mark.calculator


def check_potcar(setups, filename='POTCAR'):
    """Return true if labels in setups are found in POTCAR"""

    pp = []
    with open(filename) as fd:
        for line in fd:
            if 'TITEL' in line.split():
                pp.append(line.split()[3])
    for setup in setups:
        assert setup in pp


@pytest.fixture()
def atoms_1():
    return Atoms('CaGdCs',
                 positions=[[0, 0, 1], [0, 0, 2], [0, 0, 3]],
                 cell=[5, 5, 5])


@pytest.fixture()
def atoms_2():
    return Atoms('CaInI',
                 positions=[[0, 0, 1], [0, 0, 2], [0, 0, 3]],
                 cell=[5, 5, 5])


@pytest.fixture()
def do_check():
    def _do_check(factory, atoms, expected, settings, should_raise=False):
        calc = factory.calc(**settings)
        calc.initialize(atoms)
        calc.write_potcar()
        if should_raise:
            # We passed in bad potentials on purpose
            with pytest.raises(AssertionError):
                check_potcar(expected, filename='POTCAR')
        else:
            check_potcar(expected, filename='POTCAR')

    return _do_check


@calc('vasp')
@pytest.mark.parametrize('settings, expected', [
    (dict(xc='pbe'), ('Ca_pv', 'Gd', 'Cs_sv')),
    (dict(xc='pbe', setups='recommended'), ('Ca_sv', 'Gd_3', 'Cs_sv')),
])
def test_vasp_setup_atoms_1(factory, do_check, atoms_1, settings, expected):
    """Run some tests to ensure that VASP calculator constructs correct
    POTCAR files"""
    do_check(factory, atoms_1, expected, settings)


@calc('vasp')
@pytest.mark.parametrize('settings, expected', [
    (dict(xc='pbe', setups={'base': 'gw'}), ('Ca_sv_GW', 'In_d_GW', 'I_GW')),
    (dict(xc='pbe', setups={
        'base': 'gw',
        'I': ''
    }), ('Ca_sv_GW', 'In_d_GW', 'I')),
    (dict(xc='pbe', setups={
        'base': 'gw',
        'Ca': '_sv',
        2: 'I'
    }), ('Ca_sv', 'In_d_GW', 'I')),
])
def test_vasp_setup_atoms_2(factory, do_check, atoms_2, settings, expected):
    do_check(factory, atoms_2, expected, settings)


@calc('vasp')
@pytest.mark.parametrize('settings, expected', [
    (dict(xc='pbe'), ('Ca_sv', 'Gd', 'Cs_sv')),
    (dict(xc='pbe', setups='recommended'), ('Ca_sv', 'Gd_31', 'Cs_sv')),
])
def test_setup_error(factory, do_check, atoms_1, settings, expected):
    """Do a test, where we purposely make mistakes"""

    do_check(factory, atoms_1, expected, settings, should_raise=True)
