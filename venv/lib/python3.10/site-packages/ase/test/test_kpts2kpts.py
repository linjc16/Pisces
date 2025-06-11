import pytest

from ase import Atoms
from ase.calculators.calculator import kpts2kpts
from ase.lattice import all_variants


# This function tests whether giving a bandpath
# and an atoms object with a completed cell to
# kpts2kpts actually produces a band path with
# the same special points in the end. If this
# isn't fulfilled then it means that something
# has gone wrong (most likely that the bravais
# lattice wasn't correctly identified).
@pytest.mark.parametrize('lat', all_variants())
def test_kpts2kpts(lat):
    print()
    print(lat)
    bandpath = lat.bandpath()
    a = Atoms()
    a.cell = lat.tocell().complete()
    a.pbc[:lat.ndim] = True
    path = {'path': bandpath.path}
    bandpath2 = kpts2kpts(path, atoms=a)
    print('cell', a.cell)
    print('Original', bandpath)
    print('path', path)
    print('Produced by kpts2kpts', bandpath2)
    sp = set(bandpath.special_points)
    sp2 = set(bandpath2.special_points)
    msg = ('Input and output bandpath from kpts2kpts dont agree!\n'
           'Input: {}\n Output: {}'.format(bandpath, bandpath2))
    assert sp == sp2, msg
