# fmt: off

import gzip
import struct
from collections import deque
from os.path import splitext

import numpy as np

from ase.atoms import Atoms
from ase.calculators.lammps import convert
from ase.calculators.singlepoint import SinglePointCalculator
from ase.data import atomic_masses, chemical_symbols
from ase.parallel import paropen
from ase.quaternions import Quaternions


def read_lammps_dump(infileobj, **kwargs):
    """Method which reads a LAMMPS dump file.

       LAMMPS chooses output method depending on the given suffix:
        - .bin  : binary file
        - .gz   : output piped through gzip
        - .mpiio: using mpiio (should be like cleartext,
                  with different ordering)
        - else  : normal clear-text format

    :param infileobj: string to file, opened file or file-like stream

    """
    # !TODO: add support for lammps-regex naming schemes (output per
    # processor and timestep wildcards)

    opened = False
    if isinstance(infileobj, str):
        opened = True
        suffix = splitext(infileobj)[-1]
        if suffix == ".bin":
            fileobj = paropen(infileobj, "rb")
        elif suffix == ".gz":
            # !TODO: save for parallel execution?
            fileobj = gzip.open(infileobj, "rb")
        else:
            fileobj = paropen(infileobj)
    else:
        suffix = splitext(infileobj.name)[-1]
        fileobj = infileobj

    if suffix == ".bin":
        out = read_lammps_dump_binary(fileobj, **kwargs)
        if opened:
            fileobj.close()
        return out

    out = read_lammps_dump_text(fileobj, **kwargs)

    if opened:
        fileobj.close()

    return out


def lammps_data_to_ase_atoms(
    data,
    colnames,
    cell,
    celldisp,
    pbc=False,
    atomsobj=Atoms,
    order=True,
    specorder=None,
    prismobj=None,
    units="metal",
):
    """Extract positions and other per-atom parameters and create Atoms

    :param data: per atom data
    :param colnames: index for data
    :param cell: cell dimensions
    :param celldisp: origin shift
    :param pbc: periodic boundaries
    :param atomsobj: function to create ase-Atoms object
    :param order: sort atoms by id. Might be faster to turn off.
    Disregarded in case `id` column is not given in file.
    :param specorder: list of species to map lammps types to ase-species
    (usually .dump files to not contain type to species mapping)
    :param prismobj: Coordinate transformation between lammps and ase
    :type prismobj: Prism
    :param units: lammps units for unit transformation between lammps and ase
    :returns: Atoms object
    :rtype: Atoms

    """
    if len(data.shape) == 1:
        data = data[np.newaxis, :]

    # read IDs if given and order if needed
    if "id" in colnames:
        ids = data[:, colnames.index("id")].astype(int)
        if order:
            sort_order = np.argsort(ids)
            data = data[sort_order, :]

    # determine the elements
    if "element" in colnames:
        # priority to elements written in file
        elements = data[:, colnames.index("element")]
    elif "mass" in colnames:
        # try to determine elements from masses
        elements = [
            _mass2element(m)
            for m in data[:, colnames.index("mass")].astype(float)
        ]
    elif "type" in colnames:
        # fall back to `types` otherwise
        elements = data[:, colnames.index("type")].astype(int)

        # reconstruct types from given specorder
        if specorder:
            elements = [specorder[t - 1] for t in elements]
    else:
        # todo: what if specorder give but no types?
        # in principle the masses could work for atoms, but that needs
        # lots of cases and new code I guess
        raise ValueError("Cannot determine atom types form LAMMPS dump file")

    def get_quantity(labels, quantity=None):
        try:
            cols = [colnames.index(label) for label in labels]
            if quantity:
                return convert(data[:, cols].astype(float), quantity,
                               units, "ASE")

            return data[:, cols].astype(float)
        except ValueError:
            return None

    # Positions
    positions = None
    scaled_positions = None
    if "x" in colnames:
        # doc: x, y, z = unscaled atom coordinates
        positions = get_quantity(["x", "y", "z"], "distance")
    elif "xs" in colnames:
        # doc: xs,ys,zs = scaled atom coordinates
        scaled_positions = get_quantity(["xs", "ys", "zs"])
    elif "xu" in colnames:
        # doc: xu,yu,zu = unwrapped atom coordinates
        positions = get_quantity(["xu", "yu", "zu"], "distance")
    elif "xsu" in colnames:
        # xsu,ysu,zsu = scaled unwrapped atom coordinates
        scaled_positions = get_quantity(["xsu", "ysu", "zsu"])
    else:
        raise ValueError("No atomic positions found in LAMMPS output")

    velocities = get_quantity(["vx", "vy", "vz"], "velocity")
    charges = get_quantity(["q"], "charge")
    forces = get_quantity(["fx", "fy", "fz"], "force")
    # !TODO: how need quaternions be converted?
    quaternions = get_quantity(["c_q[1]", "c_q[2]", "c_q[3]", "c_q[4]"])

    # convert cell
    cell = convert(cell, "distance", units, "ASE")
    celldisp = convert(celldisp, "distance", units, "ASE")
    if prismobj:
        celldisp = prismobj.vector_to_ase(celldisp)
        cell = prismobj.update_cell(cell)

    if quaternions is not None:
        out_atoms = Quaternions(
            symbols=elements,
            positions=positions,
            cell=cell,
            celldisp=celldisp,
            pbc=pbc,
            quaternions=quaternions,
        )
    elif positions is not None:
        # reverse coordinations transform to lammps system
        # (for all vectors = pos, vel, force)
        if prismobj:
            positions = prismobj.vector_to_ase(positions, wrap=True)

        out_atoms = atomsobj(
            symbols=elements,
            positions=positions,
            pbc=pbc,
            celldisp=celldisp,
            cell=cell
        )
    elif scaled_positions is not None:
        out_atoms = atomsobj(
            symbols=elements,
            scaled_positions=scaled_positions,
            pbc=pbc,
            celldisp=celldisp,
            cell=cell,
        )

    if velocities is not None:
        if prismobj:
            velocities = prismobj.vector_to_ase(velocities)
        out_atoms.set_velocities(velocities)
    if charges is not None:
        out_atoms.set_initial_charges([charge[0] for charge in charges])
    if forces is not None:
        if prismobj:
            forces = prismobj.vector_to_ase(forces)
        # !TODO: use another calculator if available (or move forces
        #        to atoms.property) (other problem: synchronizing
        #        parallel runs)
        calculator = SinglePointCalculator(out_atoms, energy=0.0,
                                           forces=forces)
        out_atoms.calc = calculator

    # process the extra columns of fixes, variables and computes
    #    that can be dumped, add as additional arrays to atoms object
    for colname in colnames:
        # determine if it is a compute, fix or
        # custom property/atom (but not the quaternian)
        if (colname.startswith('f_') or colname.startswith('v_') or
            colname.startswith('d_') or colname.startswith('d2_') or
            (colname.startswith('c_') and not colname.startswith('c_q['))):
            out_atoms.new_array(colname, get_quantity([colname]),
                                dtype='float')

        elif colname.startswith('i_') or colname.startswith('i2_'):
            out_atoms.new_array(colname, get_quantity([colname]),
                                dtype='int')

    return out_atoms


def construct_cell(diagdisp, offdiag):
    """Help function to create an ASE-cell with displacement vector from
    the lammps coordination system parameters.

    :param diagdisp: cell dimension convoluted with the displacement vector
    :param offdiag: off-diagonal cell elements
    :returns: cell and cell displacement vector
    :rtype: tuple
    """
    xlo, xhi, ylo, yhi, zlo, zhi = diagdisp
    xy, xz, yz = offdiag

    # create ase-cell from lammps-box
    xhilo = (xhi - xlo) - abs(xy) - abs(xz)
    yhilo = (yhi - ylo) - abs(yz)
    zhilo = zhi - zlo
    celldispx = xlo - min(0, xy) - min(0, xz)
    celldispy = ylo - min(0, yz)
    celldispz = zlo
    cell = np.array([[xhilo, 0, 0], [xy, yhilo, 0], [xz, yz, zhilo]])
    celldisp = np.array([celldispx, celldispy, celldispz])

    return cell, celldisp


def get_max_index(index):
    if np.isscalar(index):
        return index
    elif isinstance(index, slice):
        return index.stop if (index.stop is not None) else float("inf")


def read_lammps_dump_text(fileobj, index=-1, **kwargs):
    """Process cleartext lammps dumpfiles

    :param fileobj: filestream providing the trajectory data
    :param index: integer or slice object (default: get the last timestep)
    :returns: list of Atoms objects
    :rtype: list
    """
    # Load all dumped timesteps into memory simultaneously
    lines = deque(fileobj.readlines())
    index_end = get_max_index(index)

    n_atoms = 0
    images = []

    # avoid references before assignment in case of incorrect file structure
    cell, celldisp, pbc, info = None, None, False, {}

    while len(lines) > n_atoms:
        line = lines.popleft()

        if "ITEM: TIMESTEP" in line:
            line = lines.popleft()
            # !TODO: pyflakes complains about this line -> do something
            ntimestep = int(line.split()[0])  # NOQA
            info["timestep"] = ntimestep

        if "ITEM: NUMBER OF ATOMS" in line:
            line = lines.popleft()
            n_atoms = int(line.split()[0])

        if "ITEM: BOX BOUNDS" in line:
            # save labels behind "ITEM: BOX BOUNDS" in triclinic case
            # (>=lammps-7Jul09)
            tilt_items = line.split()[3:]
            celldatarows = [lines.popleft() for _ in range(3)]
            celldata = np.loadtxt(celldatarows)
            diagdisp = celldata[:, :2].reshape(6, 1).flatten()

            # determine cell tilt (triclinic case!)
            if len(celldata[0]) > 2:
                # for >=lammps-7Jul09 use labels behind "ITEM: BOX BOUNDS"
                # to assign tilt (vector) elements ...
                offdiag = celldata[:, 2]
                # ... otherwise assume default order in 3rd column
                # (if the latter was present)
                if len(tilt_items) >= 3:
                    sort_index = [tilt_items.index(i)
                                  for i in ["xy", "xz", "yz"]]
                    offdiag = offdiag[sort_index]
            else:
                offdiag = (0.0,) * 3

            cell, celldisp = construct_cell(diagdisp, offdiag)

            # Handle pbc conditions
            if len(tilt_items) == 3:
                pbc_items = tilt_items
            elif len(tilt_items) > 3:
                pbc_items = tilt_items[3:6]
            else:
                pbc_items = ["f", "f", "f"]
            pbc = ["p" in d.lower() for d in pbc_items]

        if "ITEM: ATOMS" in line:
            colnames = line.split()[2:]
            datarows = [lines.popleft() for _ in range(n_atoms)]
            data = np.loadtxt(datarows, dtype=str, ndmin=2)
            out_atoms = lammps_data_to_ase_atoms(
                data=data,
                colnames=colnames,
                cell=cell,
                celldisp=celldisp,
                atomsobj=Atoms,
                pbc=pbc,
                **kwargs,
            )
            out_atoms.info.update(info)
            images.append(out_atoms)

        if len(images) > index_end >= 0:
            break

    return images[index]


def read_lammps_dump_binary(
    fileobj, index=-1, colnames=None, intformat="SMALLBIG", **kwargs
):
    """Read binary dump-files (after binary2txt.cpp from lammps/tools)

    :param fileobj: file-stream containing the binary lammps data
    :param index: integer or slice object (default: get the last timestep)
    :param colnames: data is columns and identified by a header
    :param intformat: lammps support different integer size.  Parameter set \
    at compile-time and can unfortunately not derived from data file
    :returns: list of Atoms-objects
    :rtype: list
    """
    # depending on the chosen compilation flag lammps uses either normal
    # integers or long long for its id or timestep numbering
    # !TODO: tags are cast to double -> missing/double ids (add check?)
    _tagformat, bigformat = dict(
        SMALLSMALL=("i", "i"), SMALLBIG=("i", "q"), BIGBIG=("q", "q")
    )[intformat]

    index_end = get_max_index(index)

    # Standard columns layout from lammpsrun
    if not colnames:
        colnames = ["id", "type", "x", "y", "z",
                    "vx", "vy", "vz", "fx", "fy", "fz"]

    images = []

    # wrap struct.unpack to raise EOFError
    def read_variables(string):
        obj_len = struct.calcsize(string)
        data_obj = fileobj.read(obj_len)
        if obj_len != len(data_obj):
            raise EOFError
        return struct.unpack(string, data_obj)

    while True:
        try:
            # Assume that the binary dump file is in the old (pre-29Oct2020)
            # format
            magic_string = None

            # read header
            ntimestep, = read_variables("=" + bigformat)

            # In the new LAMMPS binary dump format (version 29Oct2020 and
            # onward), a negative timestep is used to indicate that the next
            # few bytes will contain certain metadata
            if ntimestep < 0:
                # First bigint was actually encoding the negative of the format
                # name string length (we call this 'magic_string' to
                magic_string_len = -ntimestep

                # The next `magic_string_len` bytes will hold a string
                # indicating the format of the dump file
                magic_string = b''.join(read_variables(
                    "=" + str(magic_string_len) + "c"))

                # Read endianness (integer). For now, we'll disregard the value
                # and simply use the host machine's endianness (via '='
                # character used with struct.calcsize).
                #
                # TODO: Use the endianness of the dump file in subsequent
                #       read_variables rather than just assuming it will match
                #       that of the host
                read_variables("=i")

                # Read revision number (integer)
                revision, = read_variables("=i")

                # Finally, read the actual timestep (bigint)
                ntimestep, = read_variables("=" + bigformat)

            _n_atoms, triclinic = read_variables("=" + bigformat + "i")
            boundary = read_variables("=6i")
            diagdisp = read_variables("=6d")
            if triclinic != 0:
                offdiag = read_variables("=3d")
            else:
                offdiag = (0.0,) * 3
            size_one, = read_variables("=i")

            if len(colnames) != size_one:
                raise ValueError("Provided columns do not match binary file")

            if magic_string and revision > 1:
                # New binary dump format includes units string,
                # columns string, and time
                units_str_len, = read_variables("=i")

                if units_str_len > 0:
                    # Read lammps units style
                    _ = b''.join(
                        read_variables("=" + str(units_str_len) + "c"))

                flag, = read_variables("=c")
                if flag != b'\x00':
                    # Flag was non-empty string
                    read_variables("=d")

                # Length of column string
                columns_str_len, = read_variables("=i")

                # Read column string (e.g., "id type x y z vx vy vz fx fy fz")
                _ = b''.join(read_variables("=" + str(columns_str_len) + "c"))

            nchunk, = read_variables("=i")

            # lammps cells/boxes can have different boundary conditions on each
            # sides (makes mainly sense for different non-periodic conditions
            # (e.g. [f]ixed and [s]hrink for a irradiation simulation))
            # periodic case: b 0 = 'p'
            # non-peridic cases 1: 'f', 2 : 's', 3: 'm'
            pbc = np.sum(np.array(boundary).reshape((3, 2)), axis=1) == 0

            cell, celldisp = construct_cell(diagdisp, offdiag)

            data = []
            for _ in range(nchunk):
                # number-of-data-entries
                n_data, = read_variables("=i")
                # retrieve per atom data
                data += read_variables("=" + str(n_data) + "d")
            data = np.array(data).reshape((-1, size_one))

            # map data-chunk to ase atoms
            out_atoms = lammps_data_to_ase_atoms(
                data=data,
                colnames=colnames,
                cell=cell,
                celldisp=celldisp,
                pbc=pbc,
                **kwargs
            )

            images.append(out_atoms)

            # stop if requested index has been found
            if len(images) > index_end >= 0:
                break

        except EOFError:
            break

    return images[index]


def _mass2element(mass):
    """
    Guess the element corresponding to a given atomic mass.

    :param mass: Atomic mass for searching.
    :return: Element symbol as a string.
    """
    min_idx = np.argmin(np.abs(atomic_masses - mass))
    element = chemical_symbols[min_idx]
    return element
