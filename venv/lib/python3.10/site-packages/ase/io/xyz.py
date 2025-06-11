# fmt: off

"""Reference implementation of reader and writer for standard XYZ files.

See https://en.wikipedia.org/wiki/XYZ_file_format

Note that the .xyz files are handled by the extxyz module by default.
"""
from ase.atoms import Atoms


def read_xyz(fileobj, index):
    # This function reads first all atoms and then yields based on the index.
    # Perfomance could be improved, but this serves as a simple reference.
    # It'd require more code to estimate the total number of images
    # without reading through the whole file (note: the number of atoms
    # can differ for every image).
    lines = fileobj.readlines()
    images = []
    while len(lines) > 0:
        symbols = []
        positions = []
        natoms = int(lines.pop(0))
        lines.pop(0)  # Comment line; ignored
        for _ in range(natoms):
            line = lines.pop(0)
            symbol, x, y, z = line.split()[:4]
            symbol = symbol.lower().capitalize()
            symbols.append(symbol)
            positions.append([float(x), float(y), float(z)])
        images.append(Atoms(symbols=symbols, positions=positions))
    yield from images[index]


def write_xyz(fileobj, images, comment='', fmt='%22.15f'):
    comment = comment.rstrip()
    if '\n' in comment:
        raise ValueError('Comment line should not have line breaks.')
    for atoms in images:
        natoms = len(atoms)
        fileobj.write('%d\n%s\n' % (natoms, comment))
        for s, (x, y, z) in zip(atoms.symbols, atoms.positions):
            fileobj.write('%-2s %s %s %s\n' % (s, fmt % x, fmt % y, fmt % z))


# Compatibility with older releases
simple_read_xyz = read_xyz
simple_write_xyz = write_xyz
