import pickle
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from ase.utils.filecache import MultiFileJSONCache

description = """
Convert legacy pickle files from ASE vibrations calculations to JSON.

Pickles are no longer supported for file storage.

WARNING: Only run this command on trusted pickles since unpickling
a maliciously crafted pickle allows arbitrary code execution.
Indeed that is why pickles are no longer used.
"""


def port(picklefile):
    picklefile = Path(picklefile)

    name = picklefile.name

    vibname, key, pckl = name.rsplit('.', 3)
    assert pckl == 'pckl'

    cache = MultiFileJSONCache(picklefile.parent / vibname)

    obj = pickle.loads(picklefile.read_bytes())
    if isinstance(obj, np.ndarray):  # vibrations
        dct = {'forces': obj}
    else:  # Infrared
        forces, dipole = obj
        assert isinstance(forces, np.ndarray), f'not supported: {type(forces)}'
        assert isinstance(dipole, np.ndarray), f'not supported: {type(dipole)}'
        dct = {'forces': forces, 'dipole': dipole}

    outfilename = cache._filename(key)

    if key in cache:
        del cache[key]

    cache[key] = dct
    print(f'wrote {picklefile} ==> {outfilename}')


def main(argv=None):
    parser = ArgumentParser(description=description)
    parser.add_argument('picklefile', nargs='+')
    args = parser.parse_args(argv)

    for fname in args.picklefile:
        port(fname)


if __name__ == '__main__':
    main()
