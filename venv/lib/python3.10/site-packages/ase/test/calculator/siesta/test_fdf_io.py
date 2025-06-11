import pytest

from ase.build import bulk
from ase.io.siesta import _read_fdf_lines


@pytest.mark.calculator_lite()
@pytest.mark.calculator('siesta')
def test_fdf_io(factory):

    atoms = bulk('Ti')
    calc = factory.calc()
    atoms.calc = calc
    calc.write_input(atoms, properties=['energy'])
    # Should produce siesta.fdf but really we should be more explicit

    fname = 'siesta.fdf'

    with open(fname) as fd:
        thing = _read_fdf_lines(fd)
    print(thing)

    assert thing[0].split() == ['SystemName', 'siesta']
