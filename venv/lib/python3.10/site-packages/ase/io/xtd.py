# fmt: off

import xml.etree.ElementTree as ET
from xml.dom import minidom

import numpy as np

from ase import Atoms
from ase.io.xsd import SetChild, _write_xsd_html

_image_header = ' ' * 74 + '0.0000\n!DATE     Jan 01 00:00:00 2000\n'
_image_footer = 'end\nend\n'


def _get_atom_str(an, xyz):
    s = f'{an:<5}'
    s += f'{xyz[0]:>15.9f}{xyz[1]:>15.9f}{xyz[2]:>15.9f}'
    s += ' XXXX 1      xx      '
    s += f'{an:<2}'
    s += '  0.000\n'
    return s


def write_xtd(filename, images, connectivity=None, moviespeed=10):
    """Takes Atoms object, and write materials studio file
    atoms: Atoms object
    filename: path of the output file
    moviespeed: speed of animation. between 0 and 10

    note: material studio file cannot use a partial periodic system. If partial
    perodic system was inputted, full periodicity was assumed.
    """
    if moviespeed < 0 or moviespeed > 10:
        raise ValueError('moviespeed only between 0 and 10 allowed')

    if hasattr(images, 'get_positions'):
        images = [images]

    XSD, ATR = _write_xsd_html(images, connectivity)
    ATR.attrib['NumChildren'] = '2'
    natoms = len(images[0])

    bonds = []
    if connectivity is not None:
        for i in range(connectivity.shape[0]):
            for j in range(i + 1, connectivity.shape[0]):
                if connectivity[i, j]:
                    bonds.append([i, j])

    # non-periodic system
    s = '!BIOSYM archive 3\n'
    if not images[0].pbc.all():
        # Write trajectory
        SetChild(ATR, 'Trajectory', dict(
            ID=str(natoms + 3 + len(bonds)),
            Increment='-1',
            End=str(len(images)),
            Type='arc',
            Speed=str(moviespeed),
            FrameClassType='Atom',
        ))

        # write frame information file
        s += 'PBC=OFF\n'
        for image in images:
            s += _image_header
            s += '\n'
            an = image.get_chemical_symbols()
            xyz = image.get_positions()
            for i in range(natoms):
                s += _get_atom_str(an[i], xyz[i, :])
            s += _image_footer

    # periodic system
    else:
        SetChild(ATR, 'Trajectory', dict(
            ID=str(natoms + 9 + len(bonds)),
            Increment='-1',
            End=str(len(images)),
            Type='arc',
            Speed=str(moviespeed),
            FrameClassType='Atom',
        ))

        # write frame information file
        s += 'PBC=ON\n'
        for image in images:
            s += _image_header
            s += 'PBC'
            vec = image.cell.lengths()
            s += f'{vec[0]:>10.4f}{vec[1]:>10.4f}{vec[2]:>10.4f}'
            angles = image.cell.angles()
            s += '{:>10.4f}{:>10.4f}{:>10.4f}'.format(*angles)
            s += '\n'
            an = image.get_chemical_symbols()

            angrad = np.deg2rad(angles)
            cell = np.zeros((3, 3))
            cell[0, :] = [vec[0], 0, 0]
            cell[1, :] = (np.array([np.cos(angrad[2]), np.sin(angrad[2]), 0])
                          * vec[1])
            cell[2, 0] = vec[2] * np.cos(angrad[1])
            cell[2, 1] = ((vec[1] * vec[2] * np.cos(angrad[0])
                           - cell[1, 0] * cell[2, 0]) / cell[1, 1])
            cell[2, 2] = np.sqrt(vec[2]**2 - cell[2, 0]**2 - cell[2, 1]**2)
            xyz = np.dot(image.get_scaled_positions(), cell)
            for i in range(natoms):
                s += _get_atom_str(an[i], xyz[i, :])
            s += _image_footer

    # print arc file
    if isinstance(filename, str):
        farcname = filename[:-3] + 'arc'
    else:
        farcname = filename.name[:-3] + 'arc'

    with open(farcname, 'w') as farc:
        farc.write(s)

    # check if file is an object or not.
    openandclose = False
    try:
        if isinstance(filename, str):
            fd = open(filename, 'w')
            openandclose = True
        else:  # Assume it's a 'file-like object'
            fd = filename

        # Return a pretty-printed XML string for the Element.
        rough_string = ET.tostring(XSD, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        Document = reparsed.toprettyxml(indent='\t')

        # write
        fd.write(Document)
    finally:
        if openandclose:
            fd.close()


def read_xtd(filename, index=-1):
    """Import xtd file (Materials Studio)

    Xtd files always come with arc file, and arc file
    contains all the relevant information to make atoms
    so only Arc file needs to be read
    """
    if isinstance(filename, str):
        arcfilename = filename[:-3] + 'arc'
    else:
        arcfilename = filename.name[:-3] + 'arc'

    # This trick with opening a totally different file is a gross violation of
    # common sense.
    with open(arcfilename) as fd:
        return read_arcfile(fd, index)


def read_arcfile(fd, index):
    images = []

    # the first line is comment
    fd.readline()
    pbc = 'ON' in fd.readline()
    L = fd.readline()
    while L != '':
        if '!' not in L:  # flag for the start of an image
            L = fd.readline()
            continue
        if pbc:
            L = fd.readline()
            cell = [float(d) for d in L.split()[1:]]
        else:
            fd.readline()
        symbols = []
        coords = []
        while True:
            line = fd.readline()
            L = line.split()
            if not line or 'end' in L:
                break
            symbols.append(L[0])
            coords.append([float(x) for x in L[1:4]])
        if pbc:
            image = Atoms(symbols, positions=coords, cell=cell, pbc=pbc)
        else:
            image = Atoms(symbols, positions=coords, pbc=pbc)
        images.append(image)
        L = fd.readline()

    if not index:
        return images
    else:
        return images[index]
