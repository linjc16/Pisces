# fmt: off

import json
from collections.abc import Mapping, MutableMapping
from contextlib import contextmanager
from pathlib import Path

from ase.io.jsonio import encode as encode_json
from ase.io.jsonio import read_json, write_json
from ase.io.ulm import InvalidULMFileError, NDArrayReader, Writer, ulmopen
from ase.parallel import world
from ase.utils import opencew


def missing(key):
    raise KeyError(key)


class Locked(Exception):
    pass


# Note:
#
# The communicator handling is a complete hack.
# We should entirely remove communicators from these objects.
# (Actually: opencew() should not know about communicators.)
# Then the caller is responsible for handling parallelism,
# which makes life simpler for both the caller and us!
#
# Also, things like clean()/__del__ are not correctly implemented
# in parallel.  The reason why it currently "works" is that
# we don't call those functions from Vibrations etc., or they do so
# only for rank==0.


class JSONBackend:
    extension = '.json'
    DecodeError = json.decoder.JSONDecodeError

    @staticmethod
    def open_for_writing(path, comm):
        return opencew(path, world=comm)

    @staticmethod
    def read(fname):
        return read_json(fname, always_array=False)

    @staticmethod
    def open_and_write(target, data, comm):
        if comm.rank == 0:
            write_json(target, data)

    @staticmethod
    def write(fd, value):
        fd.write(encode_json(value).encode('utf-8'))

    @classmethod
    def dump_cache(cls, path, dct, comm):
        return CombinedJSONCache.dump_cache(path, dct, comm)

    @classmethod
    def create_multifile_cache(cls, directory, comm):
        return MultiFileJSONCache(directory, comm=comm)


class ULMBackend:
    extension = '.ulm'
    DecodeError = InvalidULMFileError

    @staticmethod
    def open_for_writing(path, comm):
        fd = opencew(path, world=comm)
        if fd is not None:
            return Writer(fd, 'w', '')

    @staticmethod
    def read(fname):
        with ulmopen(fname, 'r') as r:
            data = r._data['cache']
            if isinstance(data, NDArrayReader):
                return data.read()
        return data

    @staticmethod
    def open_and_write(target, data, comm):
        if comm.rank == 0:
            with ulmopen(target, 'w') as w:
                w.write('cache', data)

    @staticmethod
    def write(fd, value):
        fd.write('cache', value)

    @classmethod
    def dump_cache(cls, path, dct, comm):
        return CombinedULMCache.dump_cache(path, dct, comm)

    @classmethod
    def create_multifile_cache(cls, directory, comm):
        return MultiFileULMCache(directory, comm=comm)


class CacheLock:
    def __init__(self, fd, key, backend):
        self.fd = fd
        self.key = key
        self.backend = backend

    def save(self, value):
        try:
            self.backend.write(self.fd, value)
        except Exception as ex:
            raise RuntimeError(f'Failed to save {value} to cache') from ex
        finally:
            self.fd.close()


class _MultiFileCacheTemplate(MutableMapping):
    writable = True

    def __init__(self, directory, comm=world):
        self.directory = Path(directory)
        self.comm = comm

    def _filename(self, key):
        return self.directory / (f'cache.{key}' + self.backend.extension)

    def _glob(self):
        return self.directory.glob('cache.*' + self.backend.extension)

    def __iter__(self):
        for path in self._glob():
            cache, key = path.stem.split('.', 1)
            if cache != 'cache':
                continue
            yield key

    def __len__(self):
        # Very inefficient this, but not a big usecase.
        return len(list(self._glob()))

    @contextmanager
    def lock(self, key):
        if self.comm.rank == 0:
            self.directory.mkdir(exist_ok=True, parents=True)
        path = self._filename(key)
        fd = self.backend.open_for_writing(path, self.comm)
        try:
            if fd is None:
                yield None
            else:
                yield CacheLock(fd, key, self.backend)
        finally:
            if fd is not None:
                fd.close()

    def __setitem__(self, key, value):
        with self.lock(key) as handle:
            if handle is None:
                raise Locked(key)
            handle.save(value)

    def __getitem__(self, key):
        path = self._filename(key)
        try:
            return self.backend.read(path)
        except FileNotFoundError:
            missing(key)
        except self.backend.DecodeError:
            # May be partially written, which typically means empty
            # because the file was locked with exclusive-write-open.
            #
            # Since we decide what keys we have based on which files exist,
            # we are obligated to return a value for this case too.
            # So we return None.
            return None

    def __delitem__(self, key):
        try:
            self._filename(key).unlink()
        except FileNotFoundError:
            missing(key)

    def combine(self):
        cache = self.backend.dump_cache(self.directory, dict(self),
                                        comm=self.comm)
        assert set(cache) == set(self)
        self.clear()
        assert len(self) == 0
        return cache

    def split(self):
        return self

    def filecount(self):
        return len(self)

    def strip_empties(self):
        empties = [key for key, value in self.items() if value is None]
        for key in empties:
            del self[key]
        return len(empties)


class _CombinedCacheTemplate(Mapping):
    writable = False

    def __init__(self, directory, dct, comm=world):
        self.directory = Path(directory)
        self._dct = dict(dct)
        self.comm = comm

    def filecount(self):
        return int(self._filename.is_file())

    @property
    def _filename(self):
        return self.directory / ('combined' + self.backend.extension)

    def __len__(self):
        return len(self._dct)

    def __iter__(self):
        return iter(self._dct)

    def __getitem__(self, index):
        return self._dct[index]

    def _dump(self):
        target = self._filename
        if target.exists():
            raise RuntimeError(f'Already exists: {target}')
        self.directory.mkdir(exist_ok=True, parents=True)
        self.backend.open_and_write(target, self._dct, comm=self.comm)

    @classmethod
    def dump_cache(cls, path, dct, comm=world):
        cache = cls(path, dct, comm=comm)
        cache._dump()
        return cache

    @classmethod
    def load(cls, path, comm):
        # XXX Very hacky this one
        cache = cls(path, {}, comm=comm)
        dct = cls.backend.read(cache._filename)
        cache._dct.update(dct)
        return cache

    def clear(self):
        self._filename.unlink()
        self._dct.clear()

    def combine(self):
        return self

    def split(self):
        cache = self.backend.create_multifile_cache(self.directory,
                                                    comm=self.comm)
        assert len(cache) == 0
        cache.update(self)
        assert set(cache) == set(self)
        self.clear()
        return cache


class MultiFileJSONCache(_MultiFileCacheTemplate):
    backend = JSONBackend()


class MultiFileULMCache(_MultiFileCacheTemplate):
    backend = ULMBackend()


class CombinedJSONCache(_CombinedCacheTemplate):
    backend = JSONBackend()


class CombinedULMCache(_CombinedCacheTemplate):
    backend = ULMBackend()


def get_json_cache(directory, comm=world):
    try:
        return CombinedJSONCache.load(directory, comm=comm)
    except FileNotFoundError:
        return MultiFileJSONCache(directory, comm=comm)
