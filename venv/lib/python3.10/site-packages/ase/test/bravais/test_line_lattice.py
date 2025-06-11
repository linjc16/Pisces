from ase.cell import Cell


def test_line_lattice():
    kx = Cell.new([5, 0, 0]).bandpath(path='GX', npoints=2).kpts
    kz = Cell.new([0, 0, 5]).bandpath(path='GX', npoints=2).kpts
    print(kx)
    print(kz)
    kx[1, 0] -= 0.5
    kz[1, 2] -= 0.5
    assert abs(kx).max() == 0.0
    assert abs(kz).max() == 0.0
