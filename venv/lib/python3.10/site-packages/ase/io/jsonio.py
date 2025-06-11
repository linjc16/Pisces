# fmt: off

import datetime
import json

import numpy as np

from ase.utils import reader, writer

# Note: We are converting JSON classes to the recommended mechanisms
# by the json module.  That means instead of classes, we will use the
# functions default() and object_hook().
#
# The encoder classes are to be deprecated (but maybe not removed, if
# widely used).


def default(obj):
    if hasattr(obj, 'todict'):
        dct = obj.todict()

        if not isinstance(dct, dict):
            raise RuntimeError('todict() of {} returned object of type {} '
                               'but should have returned dict'
                               .format(obj, type(dct)))
        if hasattr(obj, 'ase_objtype'):
            # We modify the dictionary, so it is wise to take a copy.
            dct = dct.copy()
            dct['__ase_objtype__'] = obj.ase_objtype

        return dct
    if isinstance(obj, np.ndarray):
        flatobj = obj.ravel()
        if np.iscomplexobj(obj):
            flatobj.dtype = obj.real.dtype
        # We use str(obj.dtype) here instead of obj.dtype.name, because
        # they are not always the same (e.g. for numpy arrays of strings).
        # Using obj.dtype.name can break the ability to recursively decode/
        # encode such arrays.
        return {'__ndarray__': (obj.shape,
                                str(obj.dtype),
                                flatobj.tolist())}
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, datetime.datetime):
        return {'__datetime__': obj.isoformat()}
    if isinstance(obj, complex):
        return {'__complex__': (obj.real, obj.imag)}

    raise TypeError(f'Cannot convert object of type {type(obj)} to '
                    'dictionary for JSON')


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        # (Note the name "default" comes from the outer namespace, so
        # not actually recursive)
        return default(obj)


encode = MyEncoder().encode


def object_hook(dct):
    if '__datetime__' in dct:
        return datetime.datetime.strptime(dct['__datetime__'],
                                          '%Y-%m-%dT%H:%M:%S.%f')

    if '__complex__' in dct:
        return complex(*dct['__complex__'])

    if '__ndarray__' in dct:
        return create_ndarray(*dct['__ndarray__'])

    # No longer used (only here for backwards compatibility):
    if '__complex_ndarray__' in dct:
        r, i = (np.array(x) for x in dct['__complex_ndarray__'])
        return r + i * 1j

    if '__ase_objtype__' in dct:
        objtype = dct.pop('__ase_objtype__')
        dct = numpyfy(dct)
        return create_ase_object(objtype, dct)

    return dct


def create_ndarray(shape, dtype, data):
    """Create ndarray from shape, dtype and flattened data."""
    array = np.empty(shape, dtype=dtype)
    flatbuf = array.ravel()
    if np.iscomplexobj(array):
        flatbuf.dtype = array.real.dtype
    flatbuf[:] = data
    return array


def create_ase_object(objtype, dct):
    # We just try each object type one after another and instantiate
    # them manually, depending on which kind it is.
    # We can formalize this later if it ever becomes necessary.
    if objtype == 'cell':
        from ase.cell import Cell
        dct.pop('pbc', None)  # compatibility; we once had pbc
        obj = Cell(**dct)
    elif objtype == 'bandstructure':
        from ase.spectrum.band_structure import BandStructure
        obj = BandStructure(**dct)
    elif objtype == 'bandpath':
        from ase.dft.kpoints import BandPath
        obj = BandPath(path=dct.pop('labelseq'), **dct)
    elif objtype == 'atoms':
        from ase import Atoms
        obj = Atoms.fromdict(dct)
    elif objtype == 'vibrationsdata':
        from ase.vibrations import VibrationsData
        obj = VibrationsData.fromdict(dct)
    else:
        raise ValueError('Do not know how to decode object type {} '
                         'into an actual object'.format(objtype))
    assert obj.ase_objtype == objtype
    return obj


mydecode = json.JSONDecoder(object_hook=object_hook).decode


def intkey(key):
    """Convert str to int if possible."""
    try:
        return int(key)
    except ValueError:
        return key


def fix_int_keys_in_dicts(obj):
    """Convert "int" keys: "1" -> 1.

    The json.dump() function will convert int keys in dicts to str keys.
    This function goes the other way.
    """
    if isinstance(obj, dict):
        return {intkey(key): fix_int_keys_in_dicts(value)
                for key, value in obj.items()}
    return obj


def numpyfy(obj):
    if isinstance(obj, dict):
        if '__complex_ndarray__' in obj:
            r, i = (np.array(x) for x in obj['__complex_ndarray__'])
            return r + i * 1j
    if isinstance(obj, list) and len(obj) > 0:
        try:
            a = np.array(obj)
        except ValueError:
            pass
        else:
            if a.dtype in [bool, int, float]:
                return a
        obj = [numpyfy(value) for value in obj]
    return obj


def decode(txt, always_array=True):
    obj = mydecode(txt)
    obj = fix_int_keys_in_dicts(obj)
    if always_array:
        obj = numpyfy(obj)
    return obj


@reader
def read_json(fd, always_array=True):
    dct = decode(fd.read(), always_array=always_array)
    return dct


@writer
def write_json(fd, obj):
    fd.write(encode(obj))
