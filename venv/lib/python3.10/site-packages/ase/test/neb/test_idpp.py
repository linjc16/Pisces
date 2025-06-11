import pytest

from ase.build import molecule
from ase.mep import NEB, idpp_interpolate


# I think idpp uses an optimizer;
# at the very least how an optimizer is called during this test
@pytest.mark.optimize()
def test_idpp(testdir):
    initial = molecule('C2H6')
    final = initial.copy()
    final.positions[2:5] = initial.positions[[3, 4, 2]]

    images = [initial]
    for _ in range(5):
        images.append(initial.copy())
    images.append(final)

    neb = NEB(images)
    d0 = images[3].get_distance(2, 3)
    neb.interpolate()
    d1 = images[3].get_distance(2, 3)
    idpp_interpolate(neb, fmax=0.005)
    d2 = images[3].get_distance(2, 3)
    print(d0, d1, d2)
    assert abs(d2 - 1.74) < 0.01
