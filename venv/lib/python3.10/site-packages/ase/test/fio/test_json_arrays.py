import numpy as np

from ase.io.jsonio import decode, encode


def test_json_arrays():

    def check(obj):
        txt = encode(obj)
        newobj = decode(txt, always_array=False)
        print(obj, '-->', newobj)
        assert isinstance(obj, type(newobj)), '{} vs {}'.format(type(obj),
                                                                type(newobj))
        assert np.shape(obj) == np.shape(newobj)
        assert np.array_equal(obj, newobj)

    check([1, 2, 3])
    check([1.0, 2.0, 3.0])
    check([])

    check(np.arange(3))
    check(np.arange(3).astype(float))
    check(np.empty((3, 0, 7)))
    check(np.empty((0, 3, 7), dtype=int))
    check(np.ones(2, complex))
    check(np.ones(2, np.complex64))
