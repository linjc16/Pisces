import pytest

from ase.build import bulk

calc = pytest.mark.calculator


@calc('vasp')
def test_vasp_Si_get_dos(factory):
    """
    Run VASP tests to ensure that the get_dos function works properly.
    This test is corresponding to the tutorial:
    https://wiki.fysik.dtu.dk/ase/ase/calculators/vasp.html#density-of-states
    This is conditional on the existence of the VASP_COMMAND or VASP_SCRIPT
    environment variables.

    """
    Si = bulk('Si')
    test_dir = 'test_dos'
    test_npts = 401  # Default number of points for ASE DOS module
    calc = factory.calc(kpts=(4, 4, 4), directory=test_dir)
    Si.calc = calc
    Si.get_potential_energy()  # Execute
    test_energies, test_dos = calc.get_dos()  # Obtain energies and DOS
    assert len(test_energies) == test_npts
    assert len(test_dos) == test_npts

    # Clean up
    Si.calc.clean()
