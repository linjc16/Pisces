import numpy as np

from ase.cell import Cell
from ase.db.core import bytes_to_object, object_to_bytes


def test_o2b2o():
    for o1 in [1.0,
               {'a': np.zeros((2, 2), np.float32),
                'b': np.zeros((0, 2), int)},
               ['a', 42, True, None, np.nan, np.inf, 1j],
               Cell(np.eye(3)),
               {'a': {'b': {'c': np.ones(3)}}}]:
        b1 = object_to_bytes(o1)
        o2 = bytes_to_object(b1)
        print(o2)
        assert repr(o1) == repr(o2)
