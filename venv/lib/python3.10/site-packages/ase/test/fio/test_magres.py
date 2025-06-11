import numpy as np

from ase.build import bulk
from ase.calculators.singlepoint import SinglePointDFTCalculator
from ase.io import read, write


def test_magres():

    # Test with fictional data
    si2 = bulk('Si')

    ms = np.ones((2, 3, 3))
    si2.set_array('ms', ms)
    efg = np.repeat([[[1, 0, 0], [0, 1, 0], [0, 0, -2]]], 2, axis=0)
    si2.set_array('efg', efg)

    calc = SinglePointDFTCalculator(si2)
    calc.results['sus'] = np.eye(3) * 2
    si2.calc = calc

    si2.info['magres_units'] = {'ms': 'ppm',
                                'efg': 'au',
                                'sus': '10^-6.cm^3.mol^-1'}

    write('si2_test.magres', si2)
    si2 = read('si2_test.magres')

    assert (np.trace(si2.get_array('ms')[0]) == 3)
    assert (np.all(np.isclose(si2.get_array('efg')[:, 2, 2], -2)))
    assert (np.all(np.isclose(si2.calc.results['sus'], np.eye(3) * 2)))


def test_magres_large(datadir):

    # Test with big structure
    assert len(read(datadir / "large_atoms.magres")) == 240
