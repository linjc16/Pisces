# fmt: off

""" Read/Write DL_POLY_4 CONFIG files """
import itertools
import re
from typing import IO, Generator, Iterable, List, Optional, Tuple, Union

import numpy as np

from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.data import chemical_symbols
from ase.units import _amu, _auf, _auv
from ase.utils import reader, writer

__all__ = ["read_dlp4", "write_dlp4"]

# dlp4 labels will be registered in atoms.arrays[DLP4_LABELS_KEY]
DLP4_LABELS_KEY = "dlp4_labels"
DLP4_DISP_KEY = "dlp4_displacements"
DLP_F_SI = 1.0e-10 * _amu / (1.0e-12 * 1.0e-12)  # Å*Da/ps^2
DLP_F_ASE = DLP_F_SI / _auf
DLP_V_SI = 1.0e-10 / 1.0e-12  # Å/ps
DLP_V_ASE = DLP_V_SI / _auv


def _get_frame_positions(fd: IO) -> Tuple[int, int, int, List[int]]:
    """Get positions of frames in HISTORY file."""
    init_pos = fd.tell()

    fd.seek(0)

    record_len = len(fd.readline())  # system name, and record size

    items = fd.readline().strip().split()

    if len(items) not in (3, 5):
        raise RuntimeError("Cannot determine version of HISTORY file format.")

    levcfg, imcon, natoms = map(int, items[0:3])

    if len(items) == 3:   # DLPoly classic
        # we have to iterate over the entire file
        pos = [fd.tell() for line in fd if "timestep" in line]
    else:
        nframes = int(items[3])
        pos = [((natoms * (levcfg + 2) + 4) * i + 3) * record_len
               for i in range(nframes)]

    fd.seek(init_pos)
    return levcfg, imcon, natoms, pos


@reader
def read_dlp_history(fd: IO,
                     index: Optional[Union[int, slice]] = None,
                     symbols: Optional[List[str]] = None) -> List[Atoms]:
    """Read a HISTORY file.

    Compatible with DLP4 and DLP_CLASSIC.

    *Index* can be integer or a slice.

    Provide a list of element strings to symbols to ignore naming
    from the HISTORY file.
    """
    return list(iread_dlp_history(fd, index, symbols))


@reader
def iread_dlp_history(fd: IO,
                      index: Optional[Union[int, slice]] = None,
                      symbols: Optional[List[str]] = None
                      ) -> Generator[Atoms, None, None]:
    """Generator version of read_dlp_history

    Compatible with DLP4 and DLP_CLASSIC.

    *Index* can be integer or a slice.

    Provide a list of element strings to symbols to ignore naming
    from the HISTORY file.
    """
    levcfg, imcon, natoms, pos = _get_frame_positions(fd)

    positions: Iterable[int] = ()
    if index is None:
        positions = pos
    elif isinstance(index, int):
        positions = (pos[index],)
    elif isinstance(index, slice):
        positions = itertools.islice(pos, index.start, index.stop, index.step)

    for pos_i in positions:
        fd.seek(pos_i + 1)
        yield read_single_image(fd, levcfg, imcon, natoms, is_trajectory=True,
                                symbols=symbols)


@reader
def read_dlp4(fd: IO,
              symbols: Optional[List[str]] = None) -> Atoms:
    """Read a DL_POLY_4 config/revcon file.

    Typically used indirectly through read("filename", atoms, format="dlp4").

    Can be unforgiving with custom chemical element names.
    Please complain to alin@elena.space for bugs."""

    fd.readline()
    levcfg, imcon = map(int, fd.readline().split()[0:2])

    position = fd.tell()
    is_trajectory = fd.readline().split()[0] == "timestep"

    if not is_trajectory:
        # Difficult to distinguish between trajectory and non-trajectory
        # formats without reading one line too many.
        fd.seek(position)

    natoms = (int(fd.readline().split()[2]) if is_trajectory else None)
    return read_single_image(fd, levcfg, imcon, natoms, is_trajectory,
                             symbols)


def read_single_image(fd: IO, levcfg: int, imcon: int,
                      natoms: Optional[int], is_trajectory: bool,
                      symbols: Optional[List[str]] = None) -> Atoms:
    """ Read a DLP frame """
    sym = symbols if symbols else []
    positions = []
    velocities = []
    forces = []
    charges = []
    masses = []
    disps = []
    labels = []

    is_pbc = imcon > 0

    cell = np.zeros((3, 3), dtype=np.float64)
    if is_pbc or is_trajectory:
        for j, line in enumerate(itertools.islice(fd, 3)):
            cell[j, :] = np.array(line.split(), dtype=np.float64)

    i = 0
    for i, line in enumerate(itertools.islice(fd, natoms), 1):
        match = re.match(r"\s*([A-Za-z][a-z]?)(\S*)", line)
        if not match:
            raise OSError(f"Line {line} does not match valid format.")

        symbol, label = match.group(1, 2)
        symbol = symbol.capitalize()

        if is_trajectory:
            mass, charge, *disp = map(float, line.split()[2:])
            charges.append(charge)
            masses.append(mass)
            disps.extend(disp if disp else [None])  # type: ignore[list-item]

        if not symbols:
            if symbol not in chemical_symbols:
                raise ValueError(
                    f"Symbol '{symbol}' not found in chemical symbols table"
                )
            sym.append(symbol)

        # make sure label is not empty
        labels.append(label if label else "")

        positions.append([float(x) for x in next(fd).split()[:3]])
        if levcfg > 0:
            velocities.append([float(x) * DLP_V_ASE
                               for x in next(fd).split()[:3]])
        if levcfg > 1:
            forces.append([float(x) * DLP_F_ASE
                           for x in next(fd).split()[:3]])

    if symbols and (i != len(symbols)):
        raise ValueError(
            f"Counter is at {i} but {len(symbols)} symbols provided."
        )

    if imcon == 0:
        pbc = (False, False, False)
    elif imcon in (1, 2, 3):
        pbc = (True, True, True)
    elif imcon == 6:
        pbc = (True, True, False)
    elif imcon in (4, 5):
        raise ValueError(f"Unsupported imcon: {imcon}")
    else:
        raise ValueError(f"Invalid imcon: {imcon}")

    atoms = Atoms(positions=positions,
                  symbols=sym,
                  cell=cell,
                  pbc=pbc,
                  # Cell is centered around (0, 0, 0) in dlp4:
                  celldisp=-cell.sum(axis=0) / 2)

    if is_trajectory:
        atoms.set_masses(masses)
        atoms.set_array(DLP4_DISP_KEY, disps, float)
        atoms.set_initial_charges(charges)

    atoms.set_array(DLP4_LABELS_KEY, labels, str)
    if levcfg > 0:
        atoms.set_velocities(velocities)
    if levcfg > 1:
        atoms.calc = SinglePointCalculator(atoms, forces=forces)

    return atoms


@writer
def write_dlp4(fd: IO, atoms: Atoms,
               levcfg: int = 0,
               title: str = "CONFIG generated by ASE"):
    """Write a DL_POLY_4 config file.

    Typically used indirectly through write("filename", atoms, format="dlp4").

    Can be unforgiven with custom chemical element names.
    Please complain to alin@elena.space in case of bugs"""

    def float_format(flt):
        return format(flt, "20.10f")

    natoms = len(atoms)

    if tuple(atoms.pbc) == (False, False, False):
        imcon = 0
    elif tuple(atoms.pbc) == (True, True, True):
        imcon = 3
    elif tuple(atoms.pbc) == (True, True, False):
        imcon = 6
    else:
        raise ValueError(f"Unsupported pbc: {atoms.pbc}. "
                         "Supported pbc are 000, 110, and 111.")

    print(f"{title:72s}", file=fd)
    print(f"{levcfg:10d}{imcon:10d} {natoms}", file=fd)

    if imcon > 0:
        for row in atoms.get_cell():
            print("".join(map(float_format, row)), file=fd)

    vels = atoms.get_velocities() / DLP_V_ASE if levcfg > 0 else None
    forces = atoms.get_forces() / DLP_F_ASE if levcfg > 1 else None

    labels = atoms.arrays.get(DLP4_LABELS_KEY)

    for i, atom in enumerate(atoms):

        sym = atom.symbol
        if labels is not None:
            sym += labels[i]

        print(f"{sym:8s}{atom.index + 1:10d}", file=fd)

        to_write = (atom.x, atom.y, atom.z)
        print("".join(map(float_format, to_write)), file=fd)

        if levcfg > 0:
            to_write = (vels[atom.index, :]
                        if vels is not None else (0.0, 0.0, 0.0))
            print("".join(map(float_format, to_write)), file=fd)

        if levcfg > 1:
            to_write = (forces[atom.index, :]
                        if forces is not None else (0.0, 0.0, 0.0))
            print("".join(map(float_format, to_write)), file=fd)
