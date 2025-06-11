from pathlib import Path

import pytest

from ase.calculators.calculator import Calculator


def test_directory_and_label():
    def normalize(path):
        """Helper function to normalize path"""
        return str(Path(path))

    calc = Calculator()

    assert calc.directory == '.'
    assert calc.label is None

    calc.directory = 'somedir'

    assert calc.directory == 'somedir'
    assert calc.label == 'somedir/'

    # We cannot redundantly specify directory
    with pytest.raises(ValueError):
        calc = Calculator(directory='somedir',
                          label='anotherdir/label')

    # Test only directory in directory
    calc = Calculator(directory='somedir',
                      label='label')

    assert calc.directory == 'somedir'
    assert calc.label == 'somedir/label'

    wdir = '/home/somedir'
    calc = Calculator(directory=wdir,
                      label='label')

    assert calc.directory == normalize(wdir)
    assert calc.label == normalize(wdir) + '/label'

    # Test we can handle pathlib directories
    wdir = Path('/home/somedir')
    calc = Calculator(directory=wdir,
                      label='label')
    assert calc.directory == normalize(wdir)
    assert calc.label == normalize(wdir) + '/label'

    with pytest.raises(ValueError):
        calc = Calculator(directory=wdir,
                          label='somedir/label')

    # Passing in empty directories with directories in label should be OK
    for wdir in ['somedir', '/home/directory']:
        label = wdir + '/label'
        expected_label = normalize(wdir) + '/label'
        calc = Calculator(directory='', label=label)
        assert calc.label == expected_label
        assert calc.directory == normalize(wdir)

        calc = Calculator(directory='.', label=label)
        assert calc.label == expected_label
        assert calc.directory == normalize(wdir)


def test_deprecated_get_spin_polarized():
    calc = Calculator()
    with pytest.warns(FutureWarning):
        spinpol = calc.get_spin_polarized()
    assert spinpol is False
