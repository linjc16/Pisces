# fmt: off

"""
Reader for CP2Ks DCD_ALIGNED_CELL format and restart files.

Based on [pwtools](https://github.com/elcorto/pwtools).
All information about the dcd format is taken from there.
The way of reading it is also copied from pwtools.
Thanks to Steve for agreeing to this.

Some information also comes directly from the CP2K source,
so if they decide to change anything this here might break.

Some parts are adapted from the extxyz reader.

Contributed by Patrick Melix <chemistry@melix.me>
"""

import os
from itertools import islice

import numpy as np

from ase.atoms import Atom, Atoms
from ase.cell import Cell
from ase.data import atomic_numbers
from ase.io.formats import index2range

__all__ = ['read_cp2k_dcd', 'iread_cp2k_dcd', 'read_cp2k_restart']

# DCD file header
#   (name, dtype, shape)
# numpy dtypes:
#   i4  = int32
#   f4  = float32 (single precision)
#   f8  = float64 (double precision)
#   S80 = string of length 80 (80 chars)
_HEADER_TYPES = [
    ('blk0-0', 'i4'),       # 84 (start of first block, size=84 bytes)
    ('hdr', 'S4'),          # 'CORD'
    ('9int', ('i4', 9)),    # 9 ints, mostly 0
    ('timestep', 'f4'),     # timestep (float32)
    ('10int', ('i4', 10)),  # 10 ints, mostly 0, last is 24
    ('blk0-1', 'i4'),       # 84
    ('blk1-0', 'i4'),       # 164
    ('ntitle', 'i4'),       # 2
    ('remark1', 'S80'),     # remark1
    ('remark2', 'S80'),     # remark2
    ('blk1-1', 'i4'),       # 164
    ('blk2-0', 'i4'),       # 4 (4 bytes = int32)
    ('natoms', 'i4'),       # natoms (int32)
    ('blk2-1', 'i4'),       # 4
]

_HEADER_DTYPE = np.dtype(_HEADER_TYPES)


def _bytes_per_timestep(natoms):
    return (4 + 6 * 8 + 7 * 4 + 3 * 4 * natoms)


def _read_metainfo(fileobj):
    if not hasattr(fileobj, 'seek'):
        raise TypeError("You should have passed a fileobject opened in binary "
                        "mode, it seems you did not.")
    fileobj.seek(0)
    header = np.fromfile(fileobj, _HEADER_DTYPE, 1)[0]
    natoms = header['natoms']
    remark1 = str(header['remark1'])  # taken from CP2Ks source/motion_utils.F
    if 'CP2K' not in remark1:
        raise ValueError("Header should contain mention of CP2K, are you sure "
                         "this file was created by CP2K?")

    # dtype for fromfile: nStep times dtype of a timestep data block
    dtype = np.dtype([('x0', 'i4'),
                      ('x1', 'f8', (6,)),
                      ('x2', 'i4', (2,)),
                      ('x3', 'f4', (natoms,)),
                      ('x4', 'i4', (2,)),
                      ('x5', 'f4', (natoms,)),
                      ('x6', 'i4', (2,)),
                      ('x7', 'f4', (natoms,)),
                      ('x8', 'i4')])

    fd_pos = fileobj.tell()
    header_end = fd_pos
    # seek to end
    fileobj.seek(0, os.SEEK_END)
    # number of bytes between fd_pos and end
    fd_rest = fileobj.tell() - fd_pos
    # reset to pos after header
    fileobj.seek(fd_pos)
    # calculate nstep: fd_rest / bytes_per_timestep
    # 4 - initial 48
    # 6*8 - cryst_const_dcd
    # 7*4 - markers between x,y,z and at the end of the block
    # 3*4*natoms - float32 cartesian coords
    nsteps = fd_rest / _bytes_per_timestep(natoms)
    if fd_rest % _bytes_per_timestep(natoms) != 0:
        raise ValueError("Calculated number of steps is not int, "
                         "cannot read file")
    nsteps = int(nsteps)
    return dtype, natoms, nsteps, header_end


class DCDChunk:
    def __init__(self, chunk, dtype, natoms, symbols, aligned):
        self.chunk = chunk
        self.dtype = dtype
        self.natoms = natoms
        self.symbols = symbols
        self.aligned = aligned

    def build(self):
        """Convert unprocessed chunk into Atoms."""
        return _read_cp2k_dcd_frame(self.chunk, self.dtype, self.natoms,
                                    self.symbols, self.aligned)


def idcdchunks(fd, ref_atoms, aligned):
    """Yield unprocessed chunks for each image."""
    if ref_atoms:
        symbols = ref_atoms.get_chemical_symbols()
    else:
        symbols = None
    dtype, natoms, nsteps, header_end = _read_metainfo(fd)
    bytes_per_step = _bytes_per_timestep(natoms)
    fd.seek(header_end)
    for i in range(nsteps):
        fd.seek(bytes_per_step * i + header_end)
        yield DCDChunk(fd, dtype, natoms, symbols, aligned)


class DCDImageIterator:
    """"""

    def __init__(self, ichunks):
        self.ichunks = ichunks

    def __call__(self, fd, indices=-1, ref_atoms=None, aligned=False):
        self.ref_atoms = ref_atoms
        self.aligned = aligned
        if not hasattr(indices, 'start'):
            if indices < 0:
                indices = slice(indices - 1, indices)
            else:
                indices = slice(indices, indices + 1)

        for chunk in self._getslice(fd, indices):
            yield chunk.build()

    def _getslice(self, fd, indices):
        try:
            iterator = islice(self.ichunks(fd, self.ref_atoms, self.aligned),
                              indices.start, indices.stop, indices.step)
        except ValueError:
            # Negative indices. Adjust slice to positive numbers.
            _dtype, _natoms, nsteps, _header_end = _read_metainfo(fd)
            indices_tuple = indices.indices(nsteps + 1)
            iterator = islice(self.ichunks(fd, self.ref_atoms, self.aligned),
                              *indices_tuple)
        return iterator


iread_cp2k_dcd = DCDImageIterator(idcdchunks)


def read_cp2k_dcd(fileobj, index=-1, ref_atoms=None, aligned=False):
    """ Read a DCD file created by CP2K.

    To yield one Atoms object at a time use ``iread_cp2k_dcd``.

    Note: Other programs use other formats, they are probably not compatible.

    If ref_atoms is not given, all atoms will have symbol 'X'.

    To make sure that this is a dcd file generated with the *DCD_ALIGNED_CELL* key
    in the CP2K input, you need to set ``aligned`` to *True* to get cell information.
    Make sure you do not set it otherwise, the cell will not match the atomic coordinates!
    See the following url for details:
    https://groups.google.com/forum/#!searchin/cp2k/Outputting$20cell$20information$20and$20fractional$20coordinates%7Csort:relevance/cp2k/72MhYykrSrQ/5c9Jaw7S9yQJ.
    """  # noqa: E501
    dtype, natoms, nsteps, header_end = _read_metainfo(fileobj)
    bytes_per_timestep = _bytes_per_timestep(natoms)
    if ref_atoms:
        symbols = ref_atoms.get_chemical_symbols()
    else:
        symbols = ['X' for _ in range(natoms)]
    if natoms != len(symbols):
        raise ValueError("Length of ref_atoms does not match natoms "
                         "from dcd file")
    trbl = index2range(index, nsteps)

    for index in trbl:
        frame_pos = int(header_end + index * bytes_per_timestep)
        fileobj.seek(frame_pos)
        yield _read_cp2k_dcd_frame(fileobj, dtype, natoms, symbols,
                                   aligned=aligned)


def _read_cp2k_dcd_frame(fileobj, dtype, natoms, symbols, aligned=False):
    arr = np.fromfile(fileobj, dtype, 1)
    cryst_const = np.empty(6, dtype=np.float64)
    cryst_const[0] = arr['x1'][0, 0]
    cryst_const[1] = arr['x1'][0, 2]
    cryst_const[2] = arr['x1'][0, 5]
    cryst_const[3] = arr['x1'][0, 4]
    cryst_const[4] = arr['x1'][0, 3]
    cryst_const[5] = arr['x1'][0, 1]
    coords = np.empty((natoms, 3), dtype=np.float32)
    coords[..., 0] = arr['x3'][0, ...]
    coords[..., 1] = arr['x5'][0, ...]
    coords[..., 2] = arr['x7'][0, ...]
    assert len(coords) == len(symbols)
    if aligned:
        # convention of the cell is (see cp2ks src/particle_methods.F):
        # A is in x
        # B lies in xy plane
        # luckily this is also the ASE convention for Atoms.set_cell()
        atoms = Atoms(symbols, coords, cell=cryst_const, pbc=True)
    else:
        atoms = Atoms(symbols, coords)
    return atoms


def read_cp2k_restart(fileobj):
    """Read Atoms and Cell from Restart File.

    Reads the elements, coordinates and cell information from the
    '&SUBSYS' section of a CP2K restart file.

    Tries to convert atom names to elements, if this fails element is set to X.

    Returns an Atoms object.
    """

    def _parse_section(inp):
        """Helper to parse structure to nested dict"""
        ret = {'content': []}
        while inp:
            line = inp.readline().strip()
            if line.startswith('&END'):
                return ret
            elif line.startswith('&'):
                key = line.replace('&', '')
                ret[key] = _parse_section(inp)
            else:
                ret['content'].append(line)
        return ret

    def _fast_forward_to(fileobj, section_header):
        """Helper to forward to a section"""
        found = False
        while fileobj:
            line = fileobj.readline()
            if section_header in line:
                found = True
                break
        if not found:
            raise RuntimeError(f"No {section_header} section found!")

    def _read_cell(data):
        """Helper to read cell data, returns cell and pbc"""
        cell = None
        pbc = [False, False, False]
        if 'CELL' in data:
            content = data['CELL']['content']
            cell = Cell([[0, 0, 0] for i in range(3)])
            char2idx = {'A ': 0, 'B ': 1, 'C ': 2}
            for line in content:
                # lines starting with A/B/C<whitespace> have cell
                if line[:2] in char2idx:
                    idx = char2idx[line[:2]]
                    cell[idx] = [float(x) for x in line.split()[1:]]
                    pbc[idx] = True
            if {len(v) for v in cell} != {3}:
                raise RuntimeError("Bad Cell Definition found.")
        return cell, pbc

    def _read_geometry(content):
        """Helper to read geometry, returns a list of Atoms"""
        atom_list = []
        for entry in content:
            entry = entry.split()
            # Get letters for element symbol
            el = [char.lower() for char in entry[0] if char.isalpha()]
            el = "".join(el).capitalize()
            # Get positions
            pos = [float(x) for x in entry[1:4]]
            if el in atomic_numbers.keys():
                atom_list.append(Atom(el, pos))
            else:
                atom_list.append(Atom('X', pos))
        return atom_list

    # fast forward to &SUBSYS section
    _fast_forward_to(fileobj, '&SUBSYS')
    # parse sections into dictionary
    data = _parse_section(fileobj)
    # look for &CELL
    cell, pbc = _read_cell(data)
    # get the atoms
    atom_list = _read_geometry(data['COORD']['content'])
    return Atoms(atom_list, cell=cell, pbc=pbc)
