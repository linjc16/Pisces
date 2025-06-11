"""Test ase.io.ulm file stuff."""
import numpy as np
import pytest

import ase.io.ulm as ulm


class A:
    def write(self, writer):
        writer.write(x=np.ones((2, 3)))

    @staticmethod
    def read(reader):
        a = A()
        a.x = reader.x
        return a


@pytest.fixture()
def ulmfile(tmp_path):
    path = tmp_path / 'a.ulm'

    with ulm.open(path, 'w') as w:
        w.write(a=A(), y=9)
        w.write(s='abc')
        w.sync()
        w.write(s='abc2')
        w.sync()
        w.write(s='abc3', z=np.ones(7, int))

    return path


class MyFile:
    def __fspath__(self):
        return 'hello'


def test_open_anypathlike():
    # File does not exist, but we still want to test that it is correctly
    # converted into a (nonexistent) path
    with pytest.raises(FileNotFoundError):
        ulm.open(MyFile())


def test_ulm(ulmfile):
    with ulm.open(ulmfile) as r:
        assert r.y == 9
        assert r.s == 'abc'
        assert (A.read(r.a).x == np.ones((2, 3))).all()
        assert (r.a.x == np.ones((2, 3))).all()
        assert r[1].s == 'abc2'
        assert r[2].s == 'abc3'
        assert (r[2].z == np.ones(7)).all()


def test_append(ulmfile):
    path = ulmfile.with_name('b.ulm')
    path.write_bytes(ulmfile.read_bytes())
    with ulm.open(path, 'a') as w:
        assert w.nitems == 3
        w.write(d={'h': [1, 'asdf']})
        w.add_array('psi', (4, 2))
        w.fill(np.ones((1, 2)))
        w.fill(np.ones((1, 2)) * 2)
        w.fill(np.ones((2, 2)) * 3)

    with ulm.open(path, 'r', 3) as r:
        assert r.d['h'] == [1, 'asdf']
    with ulm.open(path) as r:
        assert (r[2].z == np.ones(7)).all()
    with ulm.open(path, index=3) as r:
        psi = r.proxy('psi')[0:3]
        assert (psi == [[1, 1], [2, 2], [3, 3]]).all()


def test_ulm_copy(ulmfile):
    path = ulmfile.with_name('c.ulm')
    ulm.copy(ulmfile, path, exclude={'.a'})

    with ulm.open(path) as r:
        assert 'a' not in r
        assert 'y' in r
