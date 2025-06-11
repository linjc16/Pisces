import io
from datetime import datetime

import numpy as np

from ase.io.jsonio import decode, encode, read_json, write_json


def test_jsonio():
    """Test serialization of ndarrays and other stuff."""
    assert decode(encode(np.int64(42))) == 42

    c = np.array([0.1j])
    assert (decode(encode(c)) == c).all()

    c = np.array(['a'])
    assert (decode(encode(c)) == c).all()

    fd = io.StringIO()

    obj1 = {'hello': 'world'}
    write_json(fd, obj1)
    fd.seek(0)
    obj2 = read_json(fd)

    print(obj1)
    print(obj2)

    for obj in [0.5 + 1.5j,
                datetime.now()]:
        s = encode(obj)
        o = decode(s)
        print(obj)
        print(s)
        print(obj)
        assert obj == o, (obj, o, s)


def test_dict_with_int_key():
    assert decode(encode({1: 2}), False)[1] == 2
