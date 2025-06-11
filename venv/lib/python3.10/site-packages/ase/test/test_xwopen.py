import pytest

from ase.utils import xwopen

pytestmark = pytest.mark.usefixtures('testdir')


poem = 'Wer reitet so spät durch Nacht und Wind\n'.encode()
filename = 'poem.txt'


def test_xwopen():
    with xwopen(filename) as fd:
        fd.write(poem)

    assert fd.closed

    with open(filename, 'rb') as fd:
        assert fd.read() == poem


def test_xwopen_locked():
    with xwopen(filename) as fd:
        assert fd is not None
        with xwopen(filename) as fd2:
            assert fd2 is None


def test_xwopen_fail(tmp_path):
    with pytest.raises(OSError):
        with xwopen(tmp_path / 'does_not_exist/file'):
            pass
