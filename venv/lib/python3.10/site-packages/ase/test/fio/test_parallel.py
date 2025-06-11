from ase import Atoms
from ase.io import read, write
from ase.parallel import world


def test_parallel():

    n = world.rank + 1
    a = Atoms('H' * n)
    name = f'H{n}.xyz'
    write(name, a, parallel=False)
    b = read(name, parallel=False)
    assert n == len(b)
