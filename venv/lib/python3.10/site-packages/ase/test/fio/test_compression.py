"""
Read and write on compressed files.
"""

import os
import os.path

import numpy as np
import pytest

from ase import io
from ase.build import bulk
from ase.io import formats

single = bulk('Au')
multiple = [bulk('Fe'), bulk('Zn'), bulk('Li')]
compressions = ['gz', 'bz2', 'xz']


def test_get_compression():
    """Identification of supported compression from filename."""
    assert formats.get_compression('H2O.pdb.gz') == ('H2O.pdb', 'gz')
    assert formats.get_compression('CH4.pdb.bz2') == ('CH4.pdb', 'bz2')
    assert formats.get_compression('Alanine.pdb.xz') == ('Alanine.pdb', 'xz')
    # zip not implemented ;)
    assert formats.get_compression('DNA.pdb.zip') == ('DNA.pdb.zip', None)
    assert formats.get_compression('crystal.cif') == ('crystal.cif', None)


@pytest.mark.parametrize('ext', compressions)
def test_compression_write_single(ext):
    """Writing compressed file."""
    filename = f'single.xsf.{ext}'
    io.write(filename, single)
    assert os.path.exists(filename)


@pytest.mark.parametrize('fmt', ('xsf', 'traj'))
@pytest.mark.parametrize('ext', compressions)
def test_compression_read_write_single(ext, fmt):
    """Re-reading a compressed file."""
    # Use xsf filetype as it needs to check the 'magic'
    # filetype guessing when reading
    filename = f'single.{fmt}.{ext}'
    io.write(filename, single)
    assert os.path.exists(filename)
    reread = io.read(filename)
    assert reread.get_chemical_symbols() == single.get_chemical_symbols()
    assert np.allclose(reread.positions, single.positions)


@pytest.mark.parametrize('ext', compressions)
def test_compression_write_multiple(ext):
    """Writing compressed file, with multiple configurations."""
    filename = f'multiple.xyz.{ext}'
    io.write(filename, multiple)
    assert os.path.exists(filename)


@pytest.mark.parametrize('fmt', ('xyz', 'traj'))
@pytest.mark.parametrize('ext', compressions)
def test_compression_read_write_multiple(ext, fmt):
    """Re-reading a compressed file with multiple configurations."""
    filename = f'multiple.{fmt}.{ext}'
    io.write(filename, multiple)
    assert os.path.exists(filename)
    reread = io.read(filename, ':')
    assert len(reread) == len(multiple)
    assert np.allclose(reread[-1].positions, multiple[-1].positions)


@pytest.mark.parametrize('ext', compressions)
def test_modes(ext):
    """Test the different read/write modes for a compression format."""
    filename = f'testrw.{ext}'
    for mode in ['w', 'wb', 'wt']:
        with formats.open_with_compression(filename, mode) as tmp:
            if 'b' in mode:
                tmp.write(b'some text')
            else:
                tmp.write('some text')

    for mode in ['r', 'rb', 'rt']:
        with formats.open_with_compression(filename, mode) as tmp:
            if 'b' in mode:
                assert tmp.read() == b'some text'
            else:
                assert tmp.read() == 'some text'
