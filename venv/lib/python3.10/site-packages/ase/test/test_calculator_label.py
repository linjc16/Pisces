from ase.calculators.calculator import Calculator


def test_calculator_label():

    calc = Calculator()
    assert calc.directory == '.'
    assert calc.prefix is None
    assert calc.label is None

    calc.label = 'dir/pref'
    assert calc.directory == 'dir'
    assert calc.prefix == 'pref'
    assert calc.label == 'dir/pref'

    calc.label = 'dir2/'
    assert calc.directory == 'dir2'
    assert calc.prefix is None
    assert calc.label == 'dir2/'

    calc.label = 'hello'
    assert calc.directory == '.'
    assert calc.prefix == 'hello'
    assert calc.label == 'hello'

    calc.label = None
    assert calc.label is None
    assert calc.prefix is None
    assert calc.directory == '.'
