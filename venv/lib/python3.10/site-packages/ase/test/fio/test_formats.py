from pathlib import Path

import pytest

from ase.io import read
from ase.io.formats import UnknownFileTypeError


def mkfile(path, text):
    path = Path(path)
    path.write_text(text)
    return path


def test_no_such_file():
    fname = 'nosuchfile.traj'
    with pytest.raises(FileNotFoundError, match=fname):
        read(fname)


def test_empty_file():
    path = mkfile('empty.xyz', '')
    with pytest.raises(UnknownFileTypeError, match='Empty file'):
        read(path)


def test_bad_format():
    path = mkfile('strangefile._no_such_format', 'strange file contents')
    with pytest.raises(UnknownFileTypeError, match='_no_such_format'):
        read(path)
