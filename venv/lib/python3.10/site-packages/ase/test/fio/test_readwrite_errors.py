import warnings
from io import StringIO

import pytest

from ase.build import bulk
from ase.io import read, write
from ase.io.formats import UnknownFileTypeError, parse_filename


def test_readwrite_errors():

    atoms = bulk('Au')
    fd = StringIO()

    with pytest.raises(UnknownFileTypeError):
        write(fd, atoms, format='hello')

    with pytest.raises(UnknownFileTypeError):
        read(fd, format='hello')


def test_parse_filename_with_at_in_ext():
    # parse filename with '@' in extension
    filename, index = parse_filename('file_name.traj@1:4:2')
    assert filename == 'file_name.traj'
    assert index == slice(1, 4, 2)


def test_parse_filename_with_at_in_path():
    # parse filename with '@' in path, but not in name
    filename, index = parse_filename('user@local/filename.xyz')
    assert filename == 'user@local/filename.xyz'
    assert index is None


# this will not work if we allow flexible endings for databases
# and simultaneously allow files without extensions.
# i.e. `core@shell.xyz` can be both a filename and
# a query to database `core` with key `shell.xyz`
#
# def test_parse_filename_with_at_in_name():
#    # parse filename with '@' in name
#    from ase.io.formats import parse_filename
#    filename, index = parse_filename('userlocal/file@name.xyz')
#    assert filename == 'userlocal/file@name.xyz'
#    assert index is None


def test_parse_filename_no_ext():
    # parse filename with no extension
    filename, index = parse_filename('path.to/filename')
    assert filename == 'path.to/filename'
    assert index is None


def test_parse_filename_with_at_no_ext():
    # parse filename with no extension but with @-slice
    filename, index = parse_filename('path.to/filename@1:4')
    assert filename == 'path.to/filename'
    assert index == slice(1, 4, None)


def test_parse_filename_bad_slice():
    # parse filename with malformed @-slice
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        filename, _index = parse_filename('path.to/filename@s:4')
        assert filename == 'path.to/filename'
        assert len(w) == 1
        assert 'Can not parse index' in str(w[-1].message)


def test_parse_filename_db_entry():
    # parse filename targetting database entry
    filename, index = parse_filename('path.to/filename.db@anything')
    assert filename == 'path.to/filename.db'
    assert index == 'anything'


def test_parse_filename_do_not_split():
    # check if do_not_split_by_at_sign flag works
    filename, index = parse_filename('user@local/file@name',
                                     do_not_split_by_at_sign=True)
    assert filename == 'user@local/file@name'
    assert index is None
