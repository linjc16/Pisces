"""
Tests of the plugin functionality for defining IO formats
outside of the ase package
"""
import copy
import io
from importlib.metadata import EntryPoint

import pytest

from ase.build import bulk
from ase.io import formats, read, write
from ase.io.formats import define_external_io_format
from ase.utils.plugins import ExternalIOFormat


@pytest.fixture(autouse=True)
def reset_ioformats_after_test():
    ioformats_before = copy.deepcopy(formats.ioformats)
    try:
        yield
    finally:
        formats.ioformats = ioformats_before
        formats.all_formats = ioformats_before


VALID_IO_FORMAT = ExternalIOFormat(
    desc='Test IO format',
    code='1F',
    module='ase.test.fio.test_external_io_formats'
)


# These are dummy functions for reading and writing the dummy io format
def read_dummy(file):
    return "Atoms dummy"


def write_dummy(file, atoms):
    file.write("dummy output")


def test_external_ioformat_valid(tmp_path):
    """
    Test of the external io format utility correctly
    registering a valid external entry format
    """

    test_entry_point = EntryPoint(
        name='dummy',
        value='ase.test.fio.test_external_io_formats:VALID_IO_FORMAT',
        group='ase.ioformats')

    define_external_io_format(test_entry_point)

    assert 'dummy' in formats.ioformats

    TEST_FILE = """
        THIS IS A DUMMY
    """
    assert read(io.StringIO(TEST_FILE), format='dummy') == 'Atoms dummy'

    atom = bulk('Ti')
    write(tmp_path / 'dummy_output', atom, format='dummy')
    with open(tmp_path / 'dummy_output') as file:
        assert file.read() == 'dummy output'


def test_external_ioformat_already_existing():
    """
    Test of the external io format utility correctly
    refusing to register an IOformat that is already present
    """

    test_entry_point = EntryPoint(
        name='xyz',
        value='ase.test.fio.test_external_io_formats:VALID_IO_FORMAT',
        group='ase.ioformats')

    with pytest.raises(ValueError, match='Format xyz already defined'):
        define_external_io_format(test_entry_point)

    assert 'xyz' in formats.ioformats
    assert formats.ioformats['xyz'].description != 'Test IO format'


# Io format not specified with the required namedtuple
INVALID_IO_FORMAT = {
    'desc': 'Test IO format',
    'code': '1F',
    'module': 'ase.test.fio.test_external_io_formats'
}


def test_external_ioformat_wrong_type():
    """
    Test of the external io format utility correctly
    refusing to register an IOformat that is not specified using the
    namedtuple
    """

    test_entry_point = EntryPoint(
        name='dummy',
        value='ase.test.fio.test_external_io_formats:INVALID_IO_FORMAT',
        group='ase.ioformats')

    with pytest.raises(TypeError,
                       match='Wrong type for registering external IO formats'):
        define_external_io_format(test_entry_point)

    assert 'dummy' not in formats.ioformats


def test_external_ioformat_import_error():
    """
    Test of the external io format utility correctly
    refusing to register an IOformat that is already present
    """

    test_entry_point = EntryPoint(
        name='dummy',
        value='ase.test.fio.test_external_io_formats:NOT_EXISTING',
        group='ase.ioformats')

    with pytest.raises(AttributeError):
        define_external_io_format(test_entry_point)

    assert 'dummy' not in formats.ioformats
