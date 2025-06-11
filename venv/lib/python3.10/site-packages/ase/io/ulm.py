# fmt: off

"""
ULM files
=========

*Simple and efficient pythonic file-format*

Stores ndarrays as binary data and Python's built-in datatypes
(bool, int, float, complex, str, dict, list, tuple, None) as json.

.. autofunction:: open
.. autoexception:: InvalidULMFileError


File layout
-----------

When there is only a single item::

    0: "- of Ulm" (magic prefix, ascii)
    8: "                " (tag, ascii)
    24: version (int64)
    32: nitems (int64)
    40: 48 (position of offsets, int64)
    48: p0 (offset to json data, int64)
    56: array1, array2, ... (8-byte aligned ndarrays)
    p0: n (length of json data, int64)
    p0+8: json data
    p0+8+n: EOF


Examples
--------

Writing:

>>> import numpy as np
>>> import ase.io.ulm as ulm
>>> with ulm.open('x.ulm', 'w') as w:
...     w.write(a=np.ones(7), b=42, c='abc')
...     w.write(d=3.14)


Reading:

>>> r = ulm.open('x.ulm')
>>> print(r.c)
abc
>>> r.close()

To see what's inside 'x.ulm' do this::

    $ ase ulm x.ulm
    x.ulm  (tag: "", 1 item)
    item #0:
    {
        a: <ndarray shape=(7,) dtype=float64>,
        b: 42,
        c: abc,
        d: 3.14}


.. autoclass:: Writer
    :members:

.. autoclass:: Reader
    :members:


More examples
-------------

In the following we append to the ulm-file from above and demonstrae
how to write a big array in chunks:

>>> w = ulm.open('x.ulm', 'a')
>>> w.add_array('bigarray', (10, 1000), float)
>>> for i in range(10):
...     w.fill(np.ones(1000))
...
>>> w.close()

Now read first and second items:

>>> with ulm.open('x.ulm') as r:
...     print(r.keys())
dict_keys(['a', 'b', 'c', 'd'])
>>> with ulm.open('x.ulm', index=1) as r:
...     print(r.keys())
dict_keys(['bigarray'])

To get all the data, it is possible to iterate over the items in the file.

>>> for i, r in enumerate(ulm.Reader('x.ulm')):
...     for k in r.keys():
...         print(i, k)
0 a
0 b
0 c
0 d
1 bigarray
>>> r.close()

The different parts (items) of the file are numbered by the index
argument:

>>> r = ulm.Reader('x.ulm')
>>> r[1].bigarray.shape
(10, 1000)
>>> r.close()


Versions
--------

1) Initial version.

2) Added support for big endian machines.  Json data may now have
   _little_endian=False item.

3) Changed magic string from "AFFormat" to "- of Ulm".
"""

import numbers
from pathlib import Path
from typing import Set, Union

import numpy as np

from ase.io.formats import is_compressed
from ase.io.jsonio import decode, encode
from ase.utils import plural

VERSION = 3
N1 = 42  # block size - max number of items: 1, N1, N1*N1, N1*N1*N1, ...


def open(filename, mode='r', index=None, tag=None):
    """Open ulm-file.

    filename: str
        Filename.
    mode: str
        Mode.  Must be 'r' for reading, 'w' for writing to a new file
        (overwriting an existing one) or 'a' for appending to an existing file.
    index: int
        Index of item to read.  Defaults to 0.
    tag: str
        Magic ID string.

    Returns a :class:`Reader` or a :class:`Writer` object.  May raise
    :class:`InvalidULMFileError`.
    """
    if mode == 'r':
        assert tag is None
        return Reader(filename, index or 0)
    if mode not in 'wa':
        2 / 0
    assert index is None
    return Writer(filename, mode, tag or '')


ulmopen = open


def align(fd):
    """Advance file descriptor to 8 byte alignment and return position."""
    pos = fd.tell()
    r = pos % 8
    if r == 0:
        return pos
    fd.write(b'#' * (8 - r))
    return pos + 8 - r


def writeint(fd, n, pos=None):
    """Write 64 bit integer n at pos or current position."""
    if pos is not None:
        fd.seek(pos)
    a = np.array(n, np.int64)
    if not np.little_endian:
        a.byteswap(True)
    fd.write(a.tobytes())


def readints(fd, n):
    a = np.frombuffer(fd.read(int(n * 8)), dtype=np.int64, count=n)
    if not np.little_endian:
        # Cannot use in-place byteswap because frombuffer()
        # returns readonly view
        a = a.byteswap()
    return a


def file_has_fileno(fd):
    """Tell whether file implements fileio() or not.

    array.tofile(fd) works only on files with fileno().
    numpy may write faster to physical files using fileno().

    For files without fileno() we use instead fd.write(array.tobytes()).
    Either way we need to distinguish."""

    try:
        fno = fd.fileno  # AttributeError?
        fno()  # IOError/OSError?  (Newer python: OSError is IOError)
    except (AttributeError, OSError):
        return False
    return True


class Writer:
    def __init__(self, fd, mode='w', tag='', data=None):
        """Create writer object.

        fd: str
            Filename.
        mode: str
            Mode.  Must be 'w' for writing to a new file (overwriting an
            existing one) and 'a' for appending to an existing file.
        tag: str
            Magic ID string.
        """

        assert mode in 'aw'

        # Header to be written later:
        self.header = b''

        if data is None:
            if np.little_endian:
                data = {}
            else:
                data = {'_little_endian': False}

            if isinstance(fd, str):
                fd = Path(fd)

            if mode == 'w' or (isinstance(fd, Path) and
                               not (fd.is_file() and
                                    fd.stat().st_size > 0)):
                self.nitems = 0
                self.pos0 = 48
                self.offsets = np.array([-1], np.int64)

                if isinstance(fd, Path):
                    fd = fd.open('wb')

                # File format identifier and other stuff:
                a = np.array([VERSION, self.nitems, self.pos0], np.int64)
                if not np.little_endian:
                    a.byteswap(True)
                self.header = (f'- of Ulm{tag:16}'.encode('ascii') +
                               a.tobytes() +
                               self.offsets.tobytes())
            else:
                if isinstance(fd, Path):
                    fd = fd.open('r+b')

                version, self.nitems, self.pos0, offsets = read_header(fd)[1:]
                assert version == VERSION
                n = 1
                while self.nitems > n:
                    n *= N1
                padding = np.zeros(n - self.nitems, np.int64)
                self.offsets = np.concatenate((offsets, padding))
                fd.seek(0, 2)

        self.fd = fd
        self.hasfileno = file_has_fileno(fd)

        self.data = data

        # date for array being filled:
        self.nmissing = 0  # number of missing numbers
        self.shape = None
        self.dtype = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.close()

    def add_array(self, name, shape, dtype=float):
        """Add ndarray object.

        Set name, shape and dtype for array and fill in the data in chunks
        later with the fill() method.
        """

        self._write_header()

        if isinstance(shape, int):
            shape = (shape,)

        shape = tuple(int(s) for s in shape)  # Convert np.int64 to int

        i = align(self.fd)

        self.data[name + '.'] = {
            'ndarray': (shape, np.dtype(dtype).name, i)}

        assert self.nmissing == 0, 'last array not done'

        self.dtype = dtype
        self.shape = shape
        self.nmissing = np.prod(shape)

    def _write_header(self):
        # We want to delay writing until there is any real data written.
        # Some people rely on zero file size.
        if self.header:
            self.fd.write(self.header)
            self.header = b''

    def fill(self, a):
        """Fill in ndarray chunks for array currently being written."""
        assert a.dtype == self.dtype
        assert a.shape[1:] == self.shape[len(self.shape) - a.ndim + 1:]
        self.nmissing -= a.size
        assert self.nmissing >= 0

        if self.hasfileno:
            a.tofile(self.fd)
        else:
            self.fd.write(a.tobytes())

    def sync(self):
        """Write data dictionary.

        Write bool, int, float, complex and str data, shapes and
        dtypes for ndarrays."""

        self._write_header()

        assert self.nmissing == 0
        i = self.fd.tell()
        s = encode(self.data).encode()
        writeint(self.fd, len(s))
        self.fd.write(s)

        n = len(self.offsets)
        if self.nitems >= n:
            offsets = np.zeros(n * N1, np.int64)
            offsets[:n] = self.offsets
            self.pos0 = align(self.fd)

            buf = offsets if np.little_endian else offsets.byteswap()

            if self.hasfileno:
                buf.tofile(self.fd)
            else:
                self.fd.write(buf.tobytes())
            writeint(self.fd, self.pos0, 40)
            self.offsets = offsets

        self.offsets[self.nitems] = i
        writeint(self.fd, i, self.pos0 + self.nitems * 8)
        self.nitems += 1
        writeint(self.fd, self.nitems, 32)
        self.fd.flush()
        self.fd.seek(0, 2)  # end of file
        if np.little_endian:
            self.data = {}
        else:
            self.data = {'_little_endian': False}

    def write(self, *args, **kwargs):
        """Write data.

        Examples::

            writer.write('n', 7)
            writer.write(n=7)
            writer.write(n=7, s='abc', a=np.zeros(3), abc=obj)

        If obj is not one of the supported data types (bool, int, float,
        complex, tupl, list, dict, None or ndarray) then it must have a
        obj.write(childwriter) method.
        """

        if args:
            name, value = args
            kwargs[name] = value

        self._write_header()

        for name, value in kwargs.items():
            if isinstance(value, (bool, int, float, complex,
                                  dict, list, tuple, str,
                                  type(None))):
                self.data[name] = value
            elif hasattr(value, '__array__'):
                value = np.asarray(value)
                if value.ndim == 0:
                    self.data[name] = value.item()
                else:
                    self.add_array(name, value.shape, value.dtype)
                    self.fill(value)
            else:
                value.write(self.child(name))

    def child(self, name):
        """Create child-writer object."""
        self._write_header()
        dct = self.data[name + '.'] = {}
        return Writer(self.fd, data=dct)

    def close(self):
        """Close file."""
        n = int('_little_endian' in self.data)
        if len(self.data) > n:
            # There is more than the "_little_endian" key.
            # Write that stuff before closing:
            self.sync()
        else:
            # Make sure header has been written (empty ulm-file):
            self._write_header()
        self.fd.close()

    def __len__(self):
        return int(self.nitems)


class DummyWriter:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.close()

    def add_array(self, name, shape, dtype=float):
        pass

    def fill(self, a):
        pass

    def sync(self):
        pass

    def write(self, *args, **kwargs):
        pass

    def child(self, name):
        return self

    def close(self):
        pass

    def __len__(self):
        return 0


def read_header(fd):
    fd.seek(0)
    if fd.read(8) not in [b'- of Ulm', b'AFFormat']:
        raise InvalidULMFileError('This is not an ULM formatted file.')
    tag = fd.read(16).decode('ascii').rstrip()
    version, nitems, pos0 = readints(fd, 3)
    fd.seek(pos0)
    offsets = readints(fd, nitems)
    return tag, version, nitems, pos0, offsets


class InvalidULMFileError(IOError):
    pass


class Reader:
    def __init__(self, fd, index=0, data=None, _little_endian=None):
        """Create reader."""

        self._little_endian = _little_endian

        self.must_close_fd = False
        if not hasattr(fd, 'read'):
            self.must_close_fd = True
            fd = Path(fd).open('rb')

        self._fd = fd
        self._index = index

        if data is None:
            try:
                (self._tag, self._version, self._nitems, self._pos0,
                 self._offsets) = read_header(fd)
            except BaseException:
                if self.must_close_fd:
                    fd.close()
                raise
            if self._nitems > 0:
                data = self._read_data(index)
            else:
                data = {}

        self._parse_data(data)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.close()

    def _parse_data(self, data):
        self._data = {}
        for name, value in data.items():
            if name.endswith('.'):
                if 'ndarray' in value:
                    shape, dtype, offset = value['ndarray']
                    dtype = dtype.encode()  # compatibility with Numpy 1.4
                    value = NDArrayReader(self._fd,
                                          shape,
                                          np.dtype(dtype),
                                          offset,
                                          self._little_endian)
                else:
                    value = Reader(self._fd, data=value,
                                   _little_endian=self._little_endian)
                name = name[:-1]

            self._data[name] = value

    def get_tag(self):
        """Return special tag string."""
        return self._tag

    def keys(self):
        """Return list of keys."""
        return self._data.keys()

    def asdict(self):
        """Read everything now and convert to dict."""
        dct = {}
        for key, value in self._data.items():
            if isinstance(value, NDArrayReader):
                value = value.read()
            elif isinstance(value, Reader):
                value = value.asdict()
            dct[key] = value
        return dct

    __dir__ = keys  # needed for tab-completion

    def __getattr__(self, attr):
        try:
            value = self._data[attr]
        except KeyError:
            raise AttributeError(attr)
        if isinstance(value, NDArrayReader):
            return value.read()
        return value

    def __contains__(self, key):
        return key in self._data

    def __iter__(self):
        yield self
        for i in range(self._index + 1, self._nitems):
            self._index = i
            data = self._read_data(i)
            self._parse_data(data)
            yield self

    def get(self, attr, value=None):
        """Get attr or value if no such attr."""
        try:
            return self.__getattr__(attr)
        except AttributeError:
            return value

    def proxy(self, name, *indices):
        value = self._data[name]
        assert isinstance(value, NDArrayReader)
        if indices:
            return value.proxy(*indices)
        return value

    def __len__(self):
        return int(self._nitems)

    def _read_data(self, index):
        self._fd.seek(self._offsets[index])
        size = int(readints(self._fd, 1)[0])
        data = decode(self._fd.read(size).decode(), False)
        self._little_endian = data.pop('_little_endian', True)
        return data

    def __getitem__(self, index):
        """Return Reader for item *index*."""
        data = self._read_data(index)
        return Reader(self._fd, index, data, self._little_endian)

    def tostr(self, verbose=False, indent='    '):
        keys = sorted(self._data)
        strings = []
        for key in keys:
            value = self._data[key]
            if verbose and isinstance(value, NDArrayReader):
                value = value.read()
            if isinstance(value, NDArrayReader):
                s = '<ndarray shape={} dtype={}>'.format(value.shape,
                                                         value.dtype)
            elif isinstance(value, Reader):
                s = value.tostr(verbose, indent + '    ')
            else:
                s = str(value).replace('\n', '\n  ' + ' ' * len(key) + indent)
            strings.append(f'{indent}{key}: {s}')
        return '{\n' + ',\n'.join(strings) + '}'

    def __str__(self):
        return self.tostr(False, '').replace('\n', ' ')

    def close(self):
        if self.must_close_fd:
            self._fd.close()


class NDArrayReader:
    def __init__(self, fd, shape, dtype, offset, little_endian):
        self.fd = fd
        self.hasfileno = file_has_fileno(fd)
        self.shape = tuple(shape)
        self.dtype = dtype
        self.offset = offset
        self.little_endian = little_endian

        self.ndim = len(self.shape)
        self.itemsize = dtype.itemsize
        self.size = np.prod(self.shape)
        self.nbytes = self.size * self.itemsize

        self.scale = 1.0
        self.length_of_last_dimension = None

    def __len__(self):
        return int(self.shape[0])  # Python-2.6 needs int

    def read(self):
        return self[:]

    def __getitem__(self, i):
        if isinstance(i, numbers.Integral):
            if i < 0:
                i += len(self)
            return self[i:i + 1][0]
        start, stop, step = i.indices(len(self))
        stride = np.prod(self.shape[1:], dtype=int)
        offset = self.offset + start * self.itemsize * stride
        self.fd.seek(offset)
        count = (stop - start) * stride
        if not is_compressed(self.fd) and self.hasfileno:
            a = np.fromfile(self.fd, self.dtype, count)
        else:
            # Not as fast, but works for reading from tar-files:
            a = np.frombuffer(self.fd.read(int(count * self.itemsize)),
                              self.dtype)
        a.shape = (stop - start,) + self.shape[1:]
        if step != 1:
            a = a[::step].copy()
        if self.little_endian != np.little_endian:
            # frombuffer() returns readonly array
            a = a.byteswap(inplace=a.flags.writeable)
        if self.length_of_last_dimension is not None:
            a = a[..., :self.length_of_last_dimension]
        if self.scale != 1.0:
            a *= self.scale
        return a

    def proxy(self, *indices):
        stride = self.size // len(self)
        start = 0
        for i, index in enumerate(indices):
            start += stride * index
            stride //= self.shape[i + 1]
        offset = self.offset + start * self.itemsize
        p = NDArrayReader(self.fd, self.shape[i + 1:], self.dtype,
                          offset, self.little_endian)
        p.scale = self.scale
        return p


def print_ulm_info(filename, index=None, verbose=False):
    b = ulmopen(filename, 'r')
    if index is None:
        indices = range(len(b))
    else:
        indices = [index]
    print('{}  (tag: "{}", {})'.format(filename, b.get_tag(),
                                       plural(len(b), 'item')))
    for i in indices:
        print(f'item #{i}:')
        print(b[i].tostr(verbose))


def copy(reader: Union[str, Path, Reader],
         writer: Union[str, Path, Writer],
         exclude: Set[str] = set(),
         name: str = '') -> None:
    """Copy from reader to writer except for keys in exclude."""
    close_reader = False
    close_writer = False
    if not isinstance(reader, Reader):
        reader = Reader(reader)
        close_reader = True
    if not isinstance(writer, Writer):
        writer = Writer(writer)
        close_writer = True
    for key, value in reader._data.items():
        if name + '.' + key in exclude:
            continue
        if isinstance(value, NDArrayReader):
            value = value.read()
        if isinstance(value, Reader):
            copy(value, writer.child(key), exclude, name + '.' + key)
        else:
            writer.write(key, value)
    if close_reader:
        reader.close()
    if close_writer:
        writer.close()
