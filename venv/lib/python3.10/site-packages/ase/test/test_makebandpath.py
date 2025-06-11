from ase.build import bulk
from ase.dft.kpoints import bandpath


def test_makebandpath():
    atoms = bulk('Au')
    cell = atoms.cell

    path0 = bandpath('GXL', cell)
    print(path0)
    path1 = bandpath([[0., 0., 0.], [.5, .5, .5]], cell, npoints=50)
    print(path1)
    path2 = bandpath([[0., 0., 0.], [.5, .5, .5], [.1, .2, .3]], cell,
                     npoints=50,
                     special_points={'G': [0., 0., 0.]})
    print(path2)
