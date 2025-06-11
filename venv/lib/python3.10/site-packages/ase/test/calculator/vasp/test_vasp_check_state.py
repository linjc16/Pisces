import os

import pytest

calc = pytest.mark.calculator


@calc('vasp')
def test_vasp_check_state(factory, atoms_2co):
    """Run tests to ensure that the VASP check_state() function call
    works correctly, i.e. correctly sets the working directories and
    works in that directory.

    This is conditional on the existence of the VASP_COMMAND or
    VASP_SCRIPT environment variables"""

    atoms = atoms_2co  # aliasing

    # Test
    settings = dict(xc='LDA',
                    prec='Low',
                    algo='Fast',
                    ismear=0,
                    sigma=1.,
                    istart=0,
                    lwave=False,
                    lcharg=False)

    s1 = atoms.get_chemical_symbols()

    calc = factory.calc(**settings)

    atoms.calc = calc

    en1 = atoms.get_potential_energy()

    # Test JSON dumping and restarting works
    fi = 'json_test.json'
    calc.write_json(filename=fi)

    assert os.path.isfile(fi)

    calc2 = factory.calc()
    calc2.read_json(fi)
    assert not calc2.calculation_required(atoms, ['energy', 'forces'])
    en2 = calc2.get_potential_energy()
    assert abs(en1 - en2) < 1e-8
    os.remove(fi)  # Clean up the JSON file

    # Check that the symbols remain in order (non-sorted)
    s2 = calc.atoms.get_chemical_symbols()
    assert s1 == s2
    s3 = sorted(s2)
    assert s2 != s3

    # Check that get_atoms() doesn't reset results
    r1 = dict(calc.results)  # Force a copy
    calc.get_atoms()
    r2 = dict(calc.results)
    assert r1 == r2

    # Make a parameter change to the calculator
    calc.set(sigma=0.5)

    # Check that we capture a change for float params
    assert calc.check_state(atoms) == ['float_params']
    assert calc.calculation_required(atoms, ['energy', 'forces'])

    en2 = atoms.get_potential_energy()

    # The change in sigma should result in a small change in energy
    assert (en1 - en2) > 1e-7

    # Now we make a change in input_params instead
    calc.kpts = 2

    # Check that this requires a new calculation
    assert calc.check_state(atoms) == ['input_params']
    assert calc.calculation_required(atoms, ['energy', 'forces'])

    # Clean up
    calc.clean()
