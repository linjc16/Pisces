import numpy as np
import pytest

from ase.build import bulk, fcc111
from ase.cell import Cell
from ase.lattice import bravais_lattices

bravais = {name.lower(): bravais_lattices[name] for name in bravais_lattices}


def check_single(name, cell, pbc=(True, True, True)):
    c = Cell(cell)

    pbc = np.array(pbc)
    lattice = c.get_bravais_lattice(pbc=pbc)
    name1 = lattice.name.lower()
    latname = name.split('@')[0]
    ok = latname == name1
    print(name, '-->', name1, 'OK' if ok else 'ERR', c.cellpar())
    assert ok, f'Expected {latname} but found {name1}'


def check(name, cell, pbc=None):
    if pbc is None:
        pbc = cell.any(1)
    pbc = np.asarray(pbc)
    cell = Cell(cell)

    # Check all three positive permutations:
    check_single(name + '@012', cell[[0, 1, 2]], pbc=pbc[[0, 1, 2]])
    # 2D lattice determination only supports pbc=(1,1,0) and hence we
    # check the permutations only for 3D lattices:
    if cell.rank == 3 and pbc.sum() != 1:
        check_single(name + '@201', cell[[2, 0, 1]], pbc=pbc[[2, 0, 1]])
        check_single(name + '@120', cell[[1, 2, 0]], pbc=pbc[[1, 2, 0]])


@pytest.mark.parametrize(
    'name, cell', [
        ('cub', bravais['cub'](3.3).tocell()),
        ('fcc', bravais['fcc'](3.4).tocell()),
        ('fcc', bulk('Au').cell),
        ('bcc', bravais['bcc'](3.5).tocell()),
        ('bcc', bulk('Fe').cell),
        ('tet', bravais['tet'](4., 5.).tocell()),
        ('tet', np.diag([4., 5., 5.])),
        ('tet', np.diag([5., 4., 5.])),
        ('tet', np.diag([5., 5., 4.])),
        ('bct', bravais['bct'](3., 4.).tocell()),
        ('orc', bravais['orc'](3., 4., 5.).tocell()),
        ('orcf', bravais['orcf'](4., 5., 7.).tocell()),
        ('orci', bravais['orci'](2., 5., 6.).tocell()),
        ('orcc', bravais['orcc'](3., 4., 5.).tocell()),
        ('hex', fcc111('Au', size=(1, 1, 3), periodic=True).cell),
        ('hex', bravais['hex'](5., 6.).tocell()),
        ('rhl', bravais['rhl'](4., 54.).tocell()),
        ('mcl', bravais['mcl'](2., 3., 4., 62.).tocell()),
        ('mclc', bravais['mclc'](3., 4., 5., 75.).tocell()),
        ('tri', bravais['tri'](7., 6., 5., 65., 70., 80.).tocell())])
def test_bravais_check(name, cell):
    check(name, cell)


# For 2D materials we have to check both the tocell() method
# but also for realistic cell nonzero nonperiodic axis.
def test_2d():
    check('sqr', bravais['sqr'](3.).tocell())
    check('sqr', Cell(np.diag([3., 3., 10.])),
          pbc=np.array([True, True, False]))

    check('crect', bravais['crect'](3., 40).tocell())
    check('rect', bravais['rect'](3., 4.).tocell())
    check('rect', Cell.new([3, 4, 10]), pbc=[1, 1, 0])
    check('hex2d', bravais['hex2d'](3.).tocell())
    check('obl', bravais['obl'](3., 4., 40).tocell())


def test_crect():
    a = 3
    alpha = 40 / 360 * 2 * np.pi
    x = np.cos(alpha)
    y = np.sin(alpha)
    crectcell = np.array([[a, 0, 0],
                          [a * x, a * y, 0],
                          [0, 0, 10]])
    check('crect', Cell(crectcell), pbc=[1, 1, 0])


def test_hex2d():
    a = 3
    x = 0.5 * np.sqrt(3)
    hexcell = np.array([[a, 0, 0],
                        [-0.5 * a, x * a, 0],
                        [0., 0., 0.]])
    check('hex2d', Cell(hexcell))


def test_obl():
    a = 1.234
    alpha = 40 / 360 * 2 * np.pi
    b = 4
    x = np.cos(alpha)
    y = np.sin(alpha)
    oblcell = np.array([[a, 0, 0],
                        [b * x, b * y, 0],
                        [0, 0, 10]])
    check('obl', Cell(oblcell), pbc=np.array([True, True, False]))


def test_1d():
    a = 1.234
    check('line', Cell(np.diag([a, 0, 0.0])))
    check('line', Cell(np.diag([a, 1, 1.0])), pbc=np.array([1, 0, 0]))
    check('line', Cell(np.diag([0.0, 0, a])))
    check('line', Cell(np.diag([1.0, 1, a])), pbc=np.array([0, 0, 1]))
