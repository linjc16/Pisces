import pytest

from ase.build import bulk

calc = pytest.mark.calculator


@pytest.fixture()
def system():
    return bulk('Al', 'fcc', a=4.5, cubic=True)


@pytest.fixture()
def expected_nelect_from_vasp():
    # Expected number of electrons from the specified system
    # with no charge
    return 12


@calc('vasp')
def test_vasp_charge(factory, system, expected_nelect_from_vasp):
    """
    Run VASP tests to ensure that determining number of electrons from
    user-supplied charge works correctly.

    Test that the number of charge found matches the expected.
    """

    # Dummy calculation to let VASP determine default number of electrons
    calc = factory.calc(xc='LDA',
                        nsw=-1,
                        ibrion=-1,
                        nelm=1,
                        lwave=False,
                        lcharg=False)
    system.calc = calc
    system.get_potential_energy()

    default_nelect_from_vasp = calc.get_number_of_electrons()
    assert default_nelect_from_vasp == expected_nelect_from_vasp


@calc('vasp')
def test_vasp_no_inputs(system, factory):
    # Make sure that no nelect was written into INCAR yet (as it wasn't
    # necessary)
    calc = factory.calc()
    system.calc = calc
    system.get_potential_energy()
    calc.read_incar('INCAR')
    assert calc.float_params['nelect'] is None


@calc('vasp')
def test_vasp_minus_charge(factory, system, expected_nelect_from_vasp):
    # Compare VASP's output nelect from before minus charge to default nelect
    # determined by us minus charge
    charge = -2
    calc = factory.calc(xc='LDA',
                        nsw=-1,
                        ibrion=-1,
                        nelm=1,
                        lwave=False,
                        lcharg=False,
                        charge=charge)
    calc.initialize(system)
    calc.write_input(system)
    calc.read_incar('INCAR')
    assert calc.float_params['nelect'] == expected_nelect_from_vasp - charge


@calc('vasp')
def test_vasp_nelect_charge_conflict(factory, system,
                                     expected_nelect_from_vasp):
    # Test that conflicts between explicitly given nelect and charge are
    # detected
    charge = -2
    calc = factory.calc(xc='LDA',
                        nsw=-1,
                        ibrion=-1,
                        nelm=1,
                        lwave=False,
                        lcharg=False,
                        nelect=expected_nelect_from_vasp - charge + 1,
                        charge=charge)
    system.calc = calc
    with pytest.raises(ValueError):
        system.get_potential_energy()


@calc('vasp')
def test_vasp_nelect_no_write(factory, system):
    # Test that nothing is written if charge is 0 and nelect not given
    calc = factory.calc(xc='LDA',
                        nsw=-1,
                        ibrion=-1,
                        nelm=1,
                        lwave=False,
                        lcharg=False,
                        charge=0)
    calc.initialize(system)
    calc.write_input(system)
    calc.read_incar('INCAR')
    assert calc.float_params['nelect'] is None


@calc('vasp')
def test_vasp_nelect(factory, system):
    # Test that explicitly given nelect still works as expected
    calc = factory.calc(xc='LDA',
                        nsw=-1,
                        ibrion=-1,
                        nelm=1,
                        lwave=False,
                        lcharg=False,
                        nelect=15)
    calc.calculate(system)
    assert calc.get_number_of_electrons() == 15
