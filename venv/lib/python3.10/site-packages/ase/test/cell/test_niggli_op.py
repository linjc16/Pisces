import numpy as np

from ase.cell import Cell


def test_niggli_op():

    rng = np.random.RandomState(3)

    for _ in range(5):
        cell = Cell(rng.random((3, 3)))
        print(cell.cellpar())
        rcell, op = cell.niggli_reduce()
        print(op)
        rcell1 = Cell(op.T @ cell)

        rcellpar = rcell.cellpar()
        rcellpar1 = rcell1.cellpar()
        err = np.abs(rcellpar - rcellpar1).max()
        print(err)
        assert err < 1e-10, err
