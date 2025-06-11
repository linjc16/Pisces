# fmt: off

"""Module to read and write atoms in xtl file format for the muSTEM software.

See http://tcmp.ph.unimelb.edu.au/mustem/muSTEM.html for a few examples of
this format and the documentation of muSTEM.

See https://github.com/HamishGBrown/MuSTEM for the source code of muSTEM.
"""

import numpy as np

from ase.atoms import Atoms, symbols2numbers
from ase.data import chemical_symbols
from ase.utils import reader, writer

from .utils import verify_cell_for_export, verify_dictionary


@reader
def read_mustem(fd):
    r"""Import muSTEM input file.

    Reads cell, atom positions, etc. from muSTEM xtl file.
    The mustem xtl save the root mean square (RMS) displacement, which is
    convert to Debye-Waller (in Å²) factor by:

    .. math::

        B = RMS * 8\pi^2

    """
    from ase.geometry import cellpar_to_cell

    # Read comment:
    fd.readline()

    # Parse unit cell parameter
    cellpar = [float(i) for i in fd.readline().split()[:3]]
    cell = cellpar_to_cell(cellpar)

    # beam energy
    fd.readline()

    # Number of different type of atoms
    element_number = int(fd.readline().strip())

    # List of numpy arrays:
    # length of the list = number of different atom type (element_number)
    # length of each array = number of atoms for each atom type
    atomic_numbers = []
    positions = []
    debye_waller_factors = []
    occupancies = []

    for _ in range(element_number):
        # Read the element
        _ = fd.readline()
        line = fd.readline().split()
        atoms_number = int(line[0])
        atomic_number = int(line[1])
        occupancy = float(line[2])
        DW = float(line[3]) * 8 * np.pi**2
        # read all the position for each element
        positions.append(np.genfromtxt(fname=fd, max_rows=atoms_number))
        atomic_numbers.append(np.ones(atoms_number, dtype=int) * atomic_number)
        occupancies.append(np.ones(atoms_number) * occupancy)
        debye_waller_factors.append(np.ones(atoms_number) * DW)

    positions = np.vstack(positions)

    atoms = Atoms(cell=cell, scaled_positions=positions)
    atoms.set_atomic_numbers(np.hstack(atomic_numbers))
    atoms.set_array('occupancies', np.hstack(occupancies))
    atoms.set_array('debye_waller_factors', np.hstack(debye_waller_factors))

    return atoms


class XtlmuSTEMWriter:
    """See the docstring of the `write_mustem` function.
    """

    def __init__(self, atoms, keV, debye_waller_factors=None,
                 comment=None, occupancies=None, fit_cell_to_atoms=False):
        verify_cell_for_export(atoms.get_cell())

        self.atoms = atoms.copy()
        self.atom_types = sorted(set(atoms.symbols))
        self.keV = keV
        self.comment = comment
        self.occupancies = self._get_occupancies(occupancies)
        self.RMS = self._get_RMS(debye_waller_factors)

        self.numbers = symbols2numbers(self.atom_types)
        if fit_cell_to_atoms:
            self.atoms.translate(-self.atoms.positions.min(axis=0))
            self.atoms.set_cell(self.atoms.positions.max(axis=0))

    def _get_occupancies(self, occupancies):
        if occupancies is None:
            if 'occupancies' in self.atoms.arrays:
                occupancies = {element:
                               self._parse_array_from_atoms(
                                   'occupancies', element, True)
                               for element in self.atom_types}
            else:
                occupancies = 1.0
        if np.isscalar(occupancies):
            occupancies = {atom: occupancies for atom in self.atom_types}
        elif isinstance(occupancies, dict):
            verify_dictionary(self.atoms, occupancies, 'occupancies')

        return occupancies

    def _get_RMS(self, DW):
        if DW is None:
            if 'debye_waller_factors' in self.atoms.arrays:
                DW = {element: self._parse_array_from_atoms(
                    'debye_waller_factors', element, True) / (8 * np.pi**2)
                    for element in self.atom_types}
        elif np.isscalar(DW):
            if len(self.atom_types) > 1:
                raise ValueError('This cell contains more then one type of '
                                 'atoms and the Debye-Waller factor needs to '
                                 'be provided for each atom using a '
                                 'dictionary.')
            DW = {self.atom_types[0]: DW / (8 * np.pi**2)}
        elif isinstance(DW, dict):
            verify_dictionary(self.atoms, DW, 'debye_waller_factors')
            for key, value in DW.items():
                DW[key] = value / (8 * np.pi**2)

        if DW is None:
            raise ValueError('Missing Debye-Waller factors. It can be '
                             'provided as a dictionary with symbols as key or '
                             'if the cell contains only a single type of '
                             'element, the Debye-Waller factor can also be '
                             'provided as float.')

        return DW

    def _parse_array_from_atoms(self, name, element, check_same_value):
        """
        Return the array "name" for the given element.

        Parameters
        ----------
        name : str
            The name of the arrays. Can be any key of `atoms.arrays`
        element : str, int
            The element to be considered.
        check_same_value : bool
            Check if all values are the same in the array. Necessary for
            'occupancies' and 'debye_waller_factors' arrays.

        Returns
        -------
        array containing the values corresponding defined by "name" for the
        given element. If check_same_value, return a single element.

        """
        if isinstance(element, str):
            element = symbols2numbers(element)[0]
        sliced_array = self.atoms.arrays[name][self.atoms.numbers == element]

        if check_same_value:
            # to write the occupancies and the Debye Waller factor of xtl file
            # all the value must be equal
            if np.unique(sliced_array).size > 1:
                raise ValueError(
                    "All the '{}' values for element '{}' must be "
                    "equal.".format(name, chemical_symbols[element])
                )
            sliced_array = sliced_array[0]

        return sliced_array

    def _get_position_array_single_atom_type(self, number):
        # Get the scaled (reduced) position for a single atom type
        return self.atoms.get_scaled_positions()[
            self.atoms.numbers == number]

    def _get_file_header(self):
        # 1st line: comment line
        if self.comment is None:
            s = "{} atoms with chemical formula: {}\n".format(
                len(self.atoms),
                self.atoms.get_chemical_formula())
        else:
            s = self.comment
        # 2nd line: lattice parameter
        s += "{} {} {} {} {} {}\n".format(
            *self.atoms.cell.cellpar().tolist())
        # 3td line: acceleration voltage
        s += f"{self.keV}\n"
        # 4th line: number of different atom
        s += f"{len(self.atom_types)}\n"
        return s

    def _get_element_header(self, atom_type, number, atom_type_number,
                            occupancy, RMS):
        return "{}\n{} {} {} {:.3g}\n".format(atom_type,
                                              number,
                                              atom_type_number,
                                              occupancy,
                                              RMS)

    def _get_file_end(self):
        return "Orientation\n   1 0 0\n   0 1 0\n   0 0 1\n"

    def write_to_file(self, fd):
        if isinstance(fd, str):
            fd = open(fd, 'w')

        fd.write(self._get_file_header())
        for atom_type, number, occupancy in zip(self.atom_types,
                                                self.numbers,
                                                self.occupancies):
            positions = self._get_position_array_single_atom_type(number)
            atom_type_number = positions.shape[0]
            fd.write(self._get_element_header(atom_type, atom_type_number,
                                              number,
                                              self.occupancies[atom_type],
                                              self.RMS[atom_type]))
            np.savetxt(fname=fd, X=positions, fmt='%.6g', newline='\n')

        fd.write(self._get_file_end())


@writer
def write_mustem(fd, *args, **kwargs):
    r"""Write muSTEM input file.

    Parameters:

    atoms: Atoms object

    keV: float
        Energy of the electron beam in keV required for the image simulation.

    debye_waller_factors: float or dictionary of float with atom type as key
        Debye-Waller factor of each atoms. Since the prismatic/computem
        software use root means square RMS) displacements, the Debye-Waller
        factors (B) needs to be provided in Å² and these values are converted
        to RMS displacement by:

        .. math::

            RMS = \frac{B}{8\pi^2}

    occupancies: float or dictionary of float with atom type as key (optional)
        Occupancy of each atoms. Default value is `1.0`.

    comment: str (optional)
        Comments to be written in the first line of the file. If not
        provided, write the total number of atoms and the chemical formula.

    fit_cell_to_atoms: bool (optional)
        If `True`, fit the cell to the atoms positions. If negative coordinates
        are present in the cell, the atoms are translated, so that all
        positions are positive. If `False` (default), the atoms positions and
        the cell are unchanged.
    """
    writer = XtlmuSTEMWriter(*args, **kwargs)
    writer.write_to_file(fd)
