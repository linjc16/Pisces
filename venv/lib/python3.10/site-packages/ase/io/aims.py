# fmt: off

"""Defines class/functions to write input and parse output for FHI-aims."""
import os
import re
import time
import warnings
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np

from ase import Atom, Atoms
from ase.calculators.calculator import kpts2mp
from ase.calculators.singlepoint import SinglePointDFTCalculator
from ase.constraints import FixAtoms, FixCartesian
from ase.data import atomic_numbers
from ase.io import ParseError
from ase.units import Ang, fs
from ase.utils import deprecated, reader, writer

v_unit = Ang / (1000.0 * fs)

LINE_NOT_FOUND = object()


class AimsParseError(Exception):
    """Exception raised if an error occurs when parsing an Aims output file"""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


# Read aims geometry files
@reader
def read_aims(fd, apply_constraints=True):
    """Import FHI-aims geometry type files.

    Reads unitcell, atom positions and constraints from
    a geometry.in file.

    If geometric constraint (symmetry parameters) are in the file
    include that information in atoms.info["symmetry_block"]
    """

    lines = fd.readlines()
    return parse_geometry_lines(lines, apply_constraints=apply_constraints)


def parse_geometry_lines(lines, apply_constraints=True):

    from ase import Atoms
    from ase.constraints import (
        FixAtoms,
        FixCartesian,
        FixCartesianParametricRelations,
        FixScaledParametricRelations,
    )

    atoms = Atoms()

    positions = []
    cell = []
    symbols = []
    velocities = []
    magmoms = []
    symmetry_block = []
    charges = []
    fix = []
    fix_cart = []
    xyz = np.array([0, 0, 0])
    i = -1
    n_periodic = -1
    periodic = np.array([False, False, False])
    cart_positions, scaled_positions = False, False
    for line in lines:
        inp = line.split()
        if inp == []:
            continue
        if inp[0] in ["atom", "atom_frac"]:

            if inp[0] == "atom":
                cart_positions = True
            else:
                scaled_positions = True

            if xyz.all():
                fix.append(i)
            elif xyz.any():
                fix_cart.append(FixCartesian(i, xyz))
            floatvect = float(inp[1]), float(inp[2]), float(inp[3])
            positions.append(floatvect)
            symbols.append(inp[4])
            magmoms.append(0.0)
            charges.append(0.0)
            xyz = np.array([0, 0, 0])
            i += 1

        elif inp[0] == "lattice_vector":
            floatvect = float(inp[1]), float(inp[2]), float(inp[3])
            cell.append(floatvect)
            n_periodic = n_periodic + 1
            periodic[n_periodic] = True

        elif inp[0] == "initial_moment":
            magmoms[-1] = float(inp[1])

        elif inp[0] == "initial_charge":
            charges[-1] = float(inp[1])

        elif inp[0] == "constrain_relaxation":
            if inp[1] == ".true.":
                fix.append(i)
            elif inp[1] == "x":
                xyz[0] = 1
            elif inp[1] == "y":
                xyz[1] = 1
            elif inp[1] == "z":
                xyz[2] = 1

        elif inp[0] == "velocity":
            floatvect = [v_unit * float(line) for line in inp[1:4]]
            velocities.append(floatvect)

        elif inp[0] in [
            "symmetry_n_params",
            "symmetry_params",
            "symmetry_lv",
            "symmetry_frac",
        ]:
            symmetry_block.append(" ".join(inp))

    if xyz.all():
        fix.append(i)
    elif xyz.any():
        fix_cart.append(FixCartesian(i, xyz))

    if cart_positions and scaled_positions:
        raise Exception(
            "Can't specify atom positions with mixture of "
            "Cartesian and fractional coordinates"
        )
    elif scaled_positions and periodic.any():
        atoms = Atoms(
            symbols,
            scaled_positions=positions,
            cell=cell,
            pbc=periodic)
    else:
        atoms = Atoms(symbols, positions)

    if len(velocities) > 0:
        if len(velocities) != len(positions):
            raise Exception(
                "Number of positions and velocities have to coincide.")
        atoms.set_velocities(velocities)

    fix_params = []

    if len(symmetry_block) > 5:
        params = symmetry_block[1].split()[1:]

        lattice_expressions = []
        lattice_params = []

        atomic_expressions = []
        atomic_params = []

        n_lat_param = int(symmetry_block[0].split(" ")[2])

        lattice_params = params[:n_lat_param]
        atomic_params = params[n_lat_param:]

        for ll, line in enumerate(symmetry_block[2:]):
            expression = " ".join(line.split(" ")[1:])
            if ll < 3:
                lattice_expressions += expression.split(",")
            else:
                atomic_expressions += expression.split(",")

        fix_params.append(
            FixCartesianParametricRelations.from_expressions(
                list(range(3)),
                lattice_params,
                lattice_expressions,
                use_cell=True,
            )
        )

        fix_params.append(
            FixScaledParametricRelations.from_expressions(
                list(range(len(atoms))), atomic_params, atomic_expressions
            )
        )

    if any(magmoms):
        atoms.set_initial_magnetic_moments(magmoms)
    if any(charges):
        atoms.set_initial_charges(charges)

    if periodic.any():
        atoms.set_cell(cell)
        atoms.set_pbc(periodic)
    if len(fix):
        atoms.set_constraint([FixAtoms(indices=fix)] + fix_cart + fix_params)
    else:
        atoms.set_constraint(fix_cart + fix_params)

    if fix_params and apply_constraints:
        atoms.set_positions(atoms.get_positions())
    return atoms


def get_aims_header():
    """Returns the header for aims input files"""
    lines = ["#" + "=" * 79]
    for line in [
        "Created using the Atomic Simulation Environment (ASE)",
        time.asctime(),
    ]:
        lines.append("# " + line + "\n")
    return lines


def _write_velocities_alias(args: List, kwargs: Dict[str, Any]) -> bool:
    arg_position = 5
    if len(args) > arg_position and args[arg_position]:
        args[arg_position - 1] = True
    elif kwargs.get("velocities", False):
        if len(args) < arg_position:
            kwargs["write_velocities"] = True
        else:
            args[arg_position - 1] = True
    else:
        return False
    return True


# Write aims geometry files
@deprecated(
    "Use of `velocities` is deprecated, please use `write_velocities`",
    category=FutureWarning,
    callback=_write_velocities_alias,
)
@writer
def write_aims(
    fd,
    atoms,
    scaled=False,
    geo_constrain=False,
    write_velocities=False,
    velocities=False,
    ghosts=None,
    info_str=None,
    wrap=False,
):
    """Method to write FHI-aims geometry files.

    Writes the atoms positions and constraints (only FixAtoms is
    supported at the moment).

    Args:
        fd: file object
            File to output structure to
        atoms: ase.atoms.Atoms
            structure to output to the file
        scaled: bool
            If True use fractional coordinates instead of Cartesian coordinates
        symmetry_block: list of str
            List of geometric constraints as defined in:
            :arxiv:`1908.01610`
        write_velocities: bool
            If True add the atomic velocity vectors to the file
        velocities: bool
            NOT AN ARRAY OF VELOCITIES, but the legacy version of
            `write_velocities`
        ghosts: list of Atoms
            A list of ghost atoms for the system
        info_str: str
            A string to be added to the header of the file
        wrap: bool
            Wrap atom positions to cell before writing

    .. deprecated:: 3.23.0
        Use of ``velocities`` is deprecated, please use ``write_velocities``.
    """

    if scaled and not np.all(atoms.pbc):
        raise ValueError(
            "Requesting scaled for a calculation where scaled=True, but "
            "the system is not periodic")

    if geo_constrain:
        if not scaled and np.all(atoms.pbc):
            warnings.warn(
                "Setting scaled to True because a symmetry_block is detected."
            )
            scaled = True
        elif not np.all(atoms.pbc):
            warnings.warn(
                "Parameteric constraints can only be used in periodic systems."
            )
            geo_constrain = False

    for line in get_aims_header():
        fd.write(line + "\n")

    # If writing additional information is requested via info_str:
    if info_str is not None:
        fd.write("\n# Additional information:\n")
        if isinstance(info_str, list):
            fd.write("\n".join([f"#  {s}" for s in info_str]))
        else:
            fd.write(f"# {info_str}")
        fd.write("\n")

    fd.write("#=======================================================\n")

    i = 0
    if atoms.get_pbc().any():
        for n, vector in enumerate(atoms.get_cell()):
            fd.write("lattice_vector ")
            for i in range(3):
                fd.write(f"{vector[i]:16.16f} ")
            fd.write("\n")

    fix_cart = np.zeros((len(atoms), 3), dtype=bool)
    for constr in atoms.constraints:
        if isinstance(constr, FixAtoms):
            fix_cart[constr.index] = (True, True, True)
        elif isinstance(constr, FixCartesian):
            fix_cart[constr.index] = constr.mask

    if ghosts is None:
        ghosts = np.zeros(len(atoms))
    else:
        assert len(ghosts) == len(atoms)

    wrap = wrap and not geo_constrain
    scaled_positions = atoms.get_scaled_positions(wrap=wrap)

    for i, atom in enumerate(atoms):
        if ghosts[i] == 1:
            atomstring = "empty "
        elif scaled:
            atomstring = "atom_frac "
        else:
            atomstring = "atom "
        fd.write(atomstring)
        if scaled:
            for pos in scaled_positions[i]:
                fd.write(f"{pos:16.16f} ")
        else:
            for pos in atom.position:
                fd.write(f"{pos:16.16f} ")
        fd.write(atom.symbol)
        fd.write("\n")
        # (1) all coords are constrained:
        if fix_cart[i].all():
            fd.write("    constrain_relaxation .true.\n")
        # (2) some coords are constrained:
        elif fix_cart[i].any():
            xyz = fix_cart[i]
            for n in range(3):
                if xyz[n]:
                    fd.write(f"    constrain_relaxation {'xyz'[n]}\n")
        if atom.charge:
            fd.write(f"    initial_charge {atom.charge:16.6f}\n")
        if atom.magmom:
            fd.write(f"    initial_moment {atom.magmom:16.6f}\n")

        if write_velocities and atoms.get_velocities() is not None:
            v = atoms.get_velocities()[i] / v_unit
            fd.write(f"    velocity {v[0]:.16f} {v[1]:.16f} {v[2]:.16f}\n")

    if geo_constrain:
        for line in get_sym_block(atoms):
            fd.write(line)


def get_sym_block(atoms):
    """Get symmetry block for Parametric constraints in atoms.constraints"""
    from ase.constraints import (
        FixCartesianParametricRelations,
        FixScaledParametricRelations,
    )

    # Initialize param/expressions lists
    atomic_sym_params = []
    lv_sym_params = []
    atomic_param_constr = np.zeros((len(atoms),), dtype="<U100")
    lv_param_constr = np.zeros((3,), dtype="<U100")

    # Populate param/expressions list
    for constr in atoms.constraints:
        if isinstance(constr, FixScaledParametricRelations):
            atomic_sym_params += constr.params

            if np.any(atomic_param_constr[constr.indices] != ""):
                warnings.warn(
                    "multiple parametric constraints defined for the same "
                    "atom, using the last one defined"
                )

            atomic_param_constr[constr.indices] = [
                ", ".join(expression) for expression in constr.expressions
            ]
        elif isinstance(constr, FixCartesianParametricRelations):
            lv_sym_params += constr.params

            if np.any(lv_param_constr[constr.indices] != ""):
                warnings.warn(
                    "multiple parametric constraints defined for the same "
                    "lattice vector, using the last one defined"
                )

            lv_param_constr[constr.indices] = [
                ", ".join(expression) for expression in constr.expressions
            ]

    if np.all(atomic_param_constr == "") and np.all(lv_param_constr == ""):
        return []

    # Check Constraint Parameters
    if len(atomic_sym_params) != len(np.unique(atomic_sym_params)):
        warnings.warn(
            "Some parameters were used across constraints, they will be "
            "combined in the aims calculations"
        )
        atomic_sym_params = np.unique(atomic_sym_params)

    if len(lv_sym_params) != len(np.unique(lv_sym_params)):
        warnings.warn(
            "Some parameters were used across constraints, they will be "
            "combined in the aims calculations"
        )
        lv_sym_params = np.unique(lv_sym_params)

    if np.any(atomic_param_constr == ""):
        raise OSError(
            "FHI-aims input files require all atoms have defined parametric "
            "constraints"
        )

    cell_inds = np.where(lv_param_constr == "")[0]
    for ind in cell_inds:
        lv_param_constr[ind] = "{:.16f}, {:.16f}, {:.16f}".format(
            *atoms.cell[ind])

    n_atomic_params = len(atomic_sym_params)
    n_lv_params = len(lv_sym_params)
    n_total_params = n_atomic_params + n_lv_params

    sym_block = []
    if n_total_params > 0:
        sym_block.append("#" + "=" * 55 + "\n")
        sym_block.append("# Parametric constraints\n")
        sym_block.append("#" + "=" * 55 + "\n")
        sym_block.append(
            "symmetry_n_params {:d} {:d} {:d}\n".format(
                n_total_params, n_lv_params, n_atomic_params
            )
        )
        sym_block.append(
            "symmetry_params %s\n" % " ".join(lv_sym_params + atomic_sym_params)
        )

        for constr in lv_param_constr:
            sym_block.append(f"symmetry_lv {constr:s}\n")

        for constr in atomic_param_constr:
            sym_block.append(f"symmetry_frac {constr:s}\n")
    return sym_block


def format_aims_control_parameter(key, value, format="%s"):
    """Format a line for the aims control.in

    Parameter
    ---------
    key: str
        Name of the paramteter to format
    value: Object
        The value to pass to the parameter
    format: str
        string to format the the text as

    Returns
    -------
    str
        The properly formatted line for the aims control.in
    """
    return f"{key:35s}" + (format % value) + "\n"


# Write aims control.in files
@writer
def write_control(fd, atoms, parameters, verbose_header=False):
    """Write the control.in file for FHI-aims
    Parameters
    ----------
    fd: str
        The file object to write to
    atoms: atoms.Atoms
        The Atoms object for the requested calculation
    parameters: dict
        The dictionary of all paramters for the calculation
    verbose_header: bool
        If True then explcitly list the paramters used to generate the
         control.in file inside the header
    """

    parameters = dict(parameters)
    lim = "#" + "=" * 79

    if parameters["xc"] == "LDA":
        parameters["xc"] = "pw-lda"

    cubes = parameters.pop("cubes", None)

    for line in get_aims_header():
        fd.write(line + "\n")

    if verbose_header:
        fd.write("# \n# List of parameters used to initialize the calculator:")
        for p, v in parameters.items():
            s = f"#     {p}:{v}\n"
            fd.write(s)
    fd.write(lim + "\n")

    assert "kpts" not in parameters or "k_grid" not in parameters
    assert "smearing" not in parameters or "occupation_type" not in parameters

    for key, value in parameters.items():
        if key == "kpts":
            mp = kpts2mp(atoms, parameters["kpts"])
            dk = 0.5 - 0.5 / np.array(mp)
            fd.write(
                format_aims_control_parameter(
                    "k_grid",
                    tuple(mp),
                    "%d %d %d"))
            fd.write(
                format_aims_control_parameter(
                    "k_offset",
                    tuple(dk),
                    "%f %f %f"))
        elif key in ("species_dir", "tier"):
            continue
        elif key == "aims_command":
            continue
        elif key == "plus_u":
            continue
        elif key == "smearing":
            name = parameters["smearing"][0].lower()
            if name == "fermi-dirac":
                name = "fermi"
            width = parameters["smearing"][1]
            if name == "methfessel-paxton":
                order = parameters["smearing"][2]
                order = " %d" % order
            else:
                order = ""

            fd.write(
                format_aims_control_parameter(
                    "occupation_type", (name, width, order), "%s %f%s"
                )
            )
        elif key == "output":
            for output_type in value:
                fd.write(format_aims_control_parameter(key, output_type, "%s"))
        elif key == "vdw_correction_hirshfeld" and value:
            fd.write(format_aims_control_parameter(key, "", "%s"))
        elif isinstance(value, bool):
            fd.write(
                format_aims_control_parameter(
                    key, str(value).lower(), ".%s."))
        elif isinstance(value, (tuple, list)):
            fd.write(
                format_aims_control_parameter(
                    key, " ".join([str(x) for x in value]), "%s"
                )
            )
        elif isinstance(value, str):
            fd.write(format_aims_control_parameter(key, value, "%s"))
        else:
            fd.write(format_aims_control_parameter(key, value, "%r"))

    if cubes:
        cubes.write(fd)

    fd.write(lim + "\n\n")

    # Get the species directory
    species_dir = get_species_directory
    # dicts are ordered as of python 3.7
    species_array = np.array(list(dict.fromkeys(atoms.symbols)))
    # Grab the tier specification from the parameters. THis may either
    # be None, meaning the default should be used for all species, or a
    # list of integers/None values giving a specific basis set size
    # for each species in the calculation.
    tier = parameters.pop("tier", None)
    tier_array = np.full(len(species_array), tier)
    # Path to species files for FHI-aims. In this files are specifications
    # for the basis set sizes depending on which basis set tier is used.
    species_dir = get_species_directory(parameters.get("species_dir"))
    # Parse the species files for each species present in the calculation
    # according to the tier of each species.
    species_basis_dict = parse_species_path(
        species_array=species_array, tier_array=tier_array,
        species_dir=species_dir)
    # Write the basis functions to be included for each species in the
    # calculation into the control.in file (fd).
    write_species(fd, species_basis_dict, parameters)


def get_species_directory(species_dir=None):
    """Get the directory where the basis set information is stored

    If the requested directory does not exist then raise an Error

    Parameters
    ----------
    species_dir: str
        Requested directory to find the basis set info from. E.g.
        `~/aims2022/FHIaims/species_defaults/defaults_2020/light`.

    Returns
    -------
    Path
        The Path to the requested or default species directory.

    Raises
    ------
    RuntimeError
        If both the requested directory and the default one is not defined
        or does not exit.
    """
    if species_dir is None:
        species_dir = os.environ.get("AIMS_SPECIES_DIR")

    if species_dir is None:
        raise RuntimeError(
            "Missing species directory!  Use species_dir "
            + "parameter or set $AIMS_SPECIES_DIR environment variable."
        )

    species_path = Path(species_dir)
    if not species_path.exists():
        raise RuntimeError(
            f"The requested species_dir {species_dir} does not exist")

    return species_path


def write_species(control_file_descriptor, species_basis_dict, parameters):
    """Write species for the calculation depending on basis set size.

    The calculation should include certain basis set size function depending
    on the numerical settings (light, tight, really tight) and the basis set
    size (minimal, tier1, tier2, tier3, tier4). If the basis set size is not
    given then a 'standard' basis set size is used for each numerical setting.
    The species files are defined according to these standard basis set sizes
    for the numerical settings in the FHI-aims repository.

    Note, for FHI-aims in ASE, we don't explicitly give the numerical setting.
    Instead we include the numerical setting in the species path: e.g.
    `~/aims2022/FHIaims/species_defaults/defaults_2020/light` this path has
    `light`, the numerical setting, as the last folder in the path.

    Example - a basis function might be commented in the standard basis set size
        such as "#     hydro 4 f 7.4" and this basis function should be
        uncommented for another basis set size such as tier4.

    Args:
        control_file_descriptor: File descriptor for the control.in file into
            which we need to write relevant basis functions to be included for
            the calculation.
        species_basis_dict: Dictionary where keys as the species symbols and
            each value is a single string containing all the basis functions
            to be included in the caclculation.
        parameters: Calculation parameters as a dict.
    """
    # Now for every species (key) in the species_basis_dict, save the
    # relevant basis functions (values) from the species_basis_dict, by
    # writing to the file handle (species_file_descriptor) given to this
    # function.
    for species_symbol, basis_set_text in species_basis_dict.items():
        control_file_descriptor.write(basis_set_text)
        if parameters.get("plus_u") is not None:
            if species_symbol in parameters.plus_u:
                control_file_descriptor.write(
                    f"plus_u {parameters.plus_u[species_symbol]} \n")


def parse_species_path(species_array, tier_array, species_dir):
    """Parse the species files for each species according to the tier given.

    Args:
        species_array: An array of species/element symbols present in the unit
            cell (e.g. ['C', 'H'].)
        tier_array: An array of None/integer values which define which basis
            set size to use for each species/element in the calcualtion.
        species_dir: Directory containing FHI-aims species files.

    Returns:
        Dictionary containing species as keys and the basis set specification
            for each species as text as the value for the key.
    """
    if len(species_array) != len(tier_array):
        raise ValueError(
            f"The species array length: {len(species_array)}, "
            f"is not the same as the tier_array length: {len(tier_array)}")

    species_basis_dict = {}

    for symbol, tier in zip(species_array, tier_array):
        path = species_dir / f"{atomic_numbers[symbol]:02}_{symbol}_default"
        # Open the species file:
        with open(path, encoding="utf8") as species_file_handle:
            # Read the species file into a string.
            species_file_str = species_file_handle.read()
            species_basis_dict[symbol] = manipulate_tiers(
                species_file_str, tier)
    return species_basis_dict


def manipulate_tiers(species_string: str, tier: Union[None, int] = 1):
    """Adds basis set functions based on the tier value.

    This function takes in the species file as a string, it then searches
    for relevant basis functions based on the tier value to include in a new
    string that is returned.

    Args:
        species_string: species file (default) for a given numerical setting
            (light, tight, really tight) given as a string.
        tier: The basis set size. This will dictate which basis set functions
            are included in the returned string.

    Returns:
        Basis set functions defined by the tier as a string.
    """
    if tier is None:  # Then we use the default species file untouched.
        return species_string
    tier_pattern = r"(#  \".* tier\" .*|# +Further.*)"
    top, *tiers = re.split(tier_pattern, species_string)
    tier_comments = tiers[::2]
    tier_basis = tiers[1::2]
    assert len(
        tier_comments) == len(tier_basis), "Something wrong with splitting"
    n_tiers = len(tier_comments)
    assert tier <= n_tiers, f"Only {n_tiers} tiers available, you choose {tier}"
    string_new = top
    for i, (c, b) in enumerate(zip(tier_comments, tier_basis)):
        b = re.sub(r"\n( *for_aux| *hydro| *ionic| *confined)", r"\n#\g<1>", b)
        if i < tier:
            b = re.sub(
                r"\n#( *for_aux| *hydro| *ionic| *confined)", r"\n\g<1>", b)
        string_new += c + b
    return string_new


# Read aims.out files
scalar_property_to_line_key = {
    "free_energy": ["| Electronic free energy"],
    "number_of_iterations": ["| Number of self-consistency cycles"],
    "magnetic_moment": ["N_up - N_down"],
    "n_atoms": ["| Number of atoms"],
    "n_bands": [
        "Number of Kohn-Sham states",
        "Reducing total number of  Kohn-Sham states",
        "Reducing total number of Kohn-Sham states",
    ],
    "n_electrons": ["The structure contains"],
    "n_kpts": ["| Number of k-points"],
    "n_spins": ["| Number of spin channels"],
    "electronic_temp": ["Occupation type:"],
    "fermi_energy": ["| Chemical potential (Fermi level)"],
}


class AimsOutChunk:
    """Base class for AimsOutChunks"""

    def __init__(self, lines):
        """Constructor

        Parameters
        ----------
        lines: list of str
            The set of lines from the output file the encompasses either
            a single structure within a trajectory or
            general information about the calculation (header)
        """
        self.lines = lines

    def reverse_search_for(self, keys, line_start=0):
        """Find the last time one of the keys appears in self.lines

        Parameters
        ----------
        keys: list of str
            The key strings to search for in self.lines
        line_start: int
            The lowest index to search for in self.lines

        Returns
        -------
        int
            The last time one of the keys appears in self.lines
        """
        for ll, line in enumerate(self.lines[line_start:][::-1]):
            if any(key in line for key in keys):
                return len(self.lines) - ll - 1

        return LINE_NOT_FOUND

    def search_for_all(self, key, line_start=0, line_end=-1):
        """Find the all times the key appears in self.lines

        Parameters
        ----------
        key: str
            The key string to search for in self.lines
        line_start: int
            The first line to start the search from
        line_end: int
            The last line to end the search at

        Returns
        -------
        list of ints
            All times the key appears in the lines
        """
        line_index = []
        for ll, line in enumerate(self.lines[line_start:line_end]):
            if key in line:
                line_index.append(ll + line_start)
        return line_index

    def parse_scalar(self, property):
        """Parse a scalar property from the chunk

        Parameters
        ----------
        property: str
            The property key to parse

        Returns
        -------
        float
            The scalar value of the property
        """
        line_start = self.reverse_search_for(
            scalar_property_to_line_key[property])

        if line_start == LINE_NOT_FOUND:
            return None

        line = self.lines[line_start]
        return float(line.split(":")[-1].strip().split()[0])


class AimsOutHeaderChunk(AimsOutChunk):
    """The header of the aims.out file containint general information"""

    def __init__(self, lines):
        """Constructor

        Parameters
        ----------
        lines: list of str
            The lines inside the aims.out header
        """
        super().__init__(lines)

    @cached_property
    def constraints(self):
        """Parse the constraints from the aims.out file

        Constraints for the lattice vectors are not supported.
        """

        line_inds = self.search_for_all("Found relaxation constraint for atom")
        if len(line_inds) == 0:
            return []

        fix = []
        fix_cart = []
        for ll in line_inds:
            line = self.lines[ll]
            xyz = [0, 0, 0]
            ind = int(line.split()[5][:-1]) - 1
            if "All coordinates fixed" in line:
                if ind not in fix:
                    fix.append(ind)
            if "coordinate fixed" in line:
                coord = line.split()[6]
                if coord == "x":
                    xyz[0] = 1
                elif coord == "y":
                    xyz[1] = 1
                elif coord == "z":
                    xyz[2] = 1
                keep = True
                for n, c in enumerate(fix_cart):
                    if ind == c.index:
                        keep = False
                        break
                if keep:
                    fix_cart.append(FixCartesian(ind, xyz))
                else:
                    fix_cart[n].mask[xyz.index(1)] = 1
        if len(fix) > 0:
            fix_cart.append(FixAtoms(indices=fix))

        return fix_cart

    @cached_property
    def initial_cell(self):
        """Parse the initial cell from the aims.out file"""
        line_start = self.reverse_search_for(["| Unit cell:"])
        if line_start == LINE_NOT_FOUND:
            return None

        return [
            [float(inp) for inp in line.split()[-3:]]
            for line in self.lines[line_start + 1:line_start + 4]
        ]

    @cached_property
    def initial_atoms(self):
        """Create an atoms object for the initial geometry.in structure
        from the aims.out file"""
        line_start = self.reverse_search_for(["Atomic structure:"])
        if line_start == LINE_NOT_FOUND:
            raise AimsParseError(
                "No information about the structure in the chunk.")

        line_start += 2

        cell = self.initial_cell
        positions = np.zeros((self.n_atoms, 3))
        symbols = [""] * self.n_atoms
        for ll, line in enumerate(
                self.lines[line_start:line_start + self.n_atoms]):
            inp = line.split()
            positions[ll, :] = [float(pos) for pos in inp[4:7]]
            symbols[ll] = inp[3]

        atoms = Atoms(symbols=symbols, positions=positions)

        if cell:
            atoms.set_cell(cell)
            atoms.set_pbc([True, True, True])
        atoms.set_constraint(self.constraints)

        return atoms

    @cached_property
    def is_md(self):
        """Determine if calculation is a molecular dynamics calculation"""
        return LINE_NOT_FOUND != self.reverse_search_for(
            ["Complete information for previous time-step:"]
        )

    @cached_property
    def is_relaxation(self):
        """Determine if the calculation is a geometry optimization or not"""
        return LINE_NOT_FOUND != self.reverse_search_for(
            ["Geometry relaxation:"])

    @cached_property
    def _k_points(self):
        """Get the list of k-points used in the calculation"""
        n_kpts = self.parse_scalar("n_kpts")
        if n_kpts is None:
            return {
                "k_points": None,
                "k_point_weights": None,
            }
        n_kpts = int(n_kpts)

        line_start = self.reverse_search_for(["| K-points in task"])
        line_end = self.reverse_search_for(["| k-point:"])
        if (
            (line_start == LINE_NOT_FOUND)
            or (line_end == LINE_NOT_FOUND)
            or (line_end - line_start != n_kpts)
        ):
            return {
                "k_points": None,
                "k_point_weights": None,
            }

        k_points = np.zeros((n_kpts, 3))
        k_point_weights = np.zeros(n_kpts)
        for kk, line in enumerate(self.lines[line_start + 1:line_end + 1]):
            k_points[kk] = [float(inp) for inp in line.split()[4:7]]
            k_point_weights[kk] = float(line.split()[-1])

        return {
            "k_points": k_points,
            "k_point_weights": k_point_weights,
        }

    @cached_property
    def n_atoms(self):
        """The number of atoms for the material"""
        n_atoms = self.parse_scalar("n_atoms")
        if n_atoms is None:
            raise AimsParseError(
                "No information about the number of atoms in the header."
            )
        return int(n_atoms)

    @cached_property
    def n_bands(self):
        """The number of Kohn-Sham states for the chunk"""
        line_start = self.reverse_search_for(
            scalar_property_to_line_key["n_bands"])

        if line_start == LINE_NOT_FOUND:
            raise AimsParseError(
                "No information about the number of Kohn-Sham states "
                "in the header.")

        line = self.lines[line_start]
        if "| Number of Kohn-Sham states" in line:
            return int(line.split(":")[-1].strip().split()[0])

        return int(line.split()[-1].strip()[:-1])

    @cached_property
    def n_electrons(self):
        """The number of electrons for the chunk"""
        line_start = self.reverse_search_for(
            scalar_property_to_line_key["n_electrons"])

        if line_start == LINE_NOT_FOUND:
            raise AimsParseError(
                "No information about the number of electrons in the header."
            )

        line = self.lines[line_start]
        return int(float(line.split()[-2]))

    @cached_property
    def n_k_points(self):
        """The number of k_ppoints for the calculation"""
        n_kpts = self.parse_scalar("n_kpts")
        if n_kpts is None:
            return None

        return int(n_kpts)

    @cached_property
    def n_spins(self):
        """The number of spin channels for the chunk"""
        n_spins = self.parse_scalar("n_spins")
        if n_spins is None:
            raise AimsParseError(
                "No information about the number of spin "
                "channels in the header.")
        return int(n_spins)

    @cached_property
    def electronic_temperature(self):
        """The electronic temperature for the chunk"""
        line_start = self.reverse_search_for(
            scalar_property_to_line_key["electronic_temp"]
        )
        if line_start == LINE_NOT_FOUND:
            return 0.10

        line = self.lines[line_start]
        return float(line.split("=")[-1].strip().split()[0])

    @property
    def k_points(self):
        """All k-points listed in the calculation"""
        return self._k_points["k_points"]

    @property
    def k_point_weights(self):
        """The k-point weights for the calculation"""
        return self._k_points["k_point_weights"]

    @cached_property
    def header_summary(self):
        """Dictionary summarizing the information inside the header"""
        return {
            "initial_atoms": self.initial_atoms,
            "initial_cell": self.initial_cell,
            "constraints": self.constraints,
            "is_relaxation": self.is_relaxation,
            "is_md": self.is_md,
            "n_atoms": self.n_atoms,
            "n_bands": self.n_bands,
            "n_electrons": self.n_electrons,
            "n_spins": self.n_spins,
            "electronic_temperature": self.electronic_temperature,
            "n_k_points": self.n_k_points,
            "k_points": self.k_points,
            "k_point_weights": self.k_point_weights,
        }


class AimsOutCalcChunk(AimsOutChunk):
    """A part of the aims.out file correponding to a single structure"""

    def __init__(self, lines, header):
        """Constructor

        Parameters
        ----------
        lines: list of str
            The lines used for the structure
        header: dict
            A summary of the relevant information from the aims.out header
        """
        super().__init__(lines)
        self._header = header.header_summary

    @cached_property
    def _atoms(self):
        """Create an atoms object for the subsequent structures
        calculated in the aims.out file"""
        start_keys = [
            "Atomic structure (and velocities) as used in the preceding "
            "time step",
            "Updated atomic structure",
            "Atomic structure that was used in the preceding time step of "
            "the wrapper",
        ]
        line_start = self.reverse_search_for(start_keys)
        if line_start == LINE_NOT_FOUND:
            return self.initial_atoms

        line_start += 1

        line_end = self.reverse_search_for(
            [
                'Next atomic structure:',
                'Writing the current geometry to file "geometry.in.next_step"'
            ],
            line_start
        )
        if line_end == LINE_NOT_FOUND:
            line_end = len(self.lines)

        cell = []
        velocities = []
        atoms = Atoms()
        for line in self.lines[line_start:line_end]:
            if "lattice_vector   " in line:
                cell.append([float(inp) for inp in line.split()[1:]])
            elif "atom   " in line:
                line_split = line.split()
                atoms.append(Atom(line_split[4], tuple(
                    float(inp) for inp in line_split[1:4])))
            elif "velocity   " in line:
                velocities.append([float(inp) for inp in line.split()[1:]])

        assert len(atoms) == self.n_atoms
        assert (len(velocities) == self.n_atoms) or (len(velocities) == 0)
        if len(cell) == 3:
            atoms.set_cell(np.array(cell))
            atoms.set_pbc([True, True, True])
        elif len(cell) != 0:
            raise AimsParseError(
                "Parsed geometry has incorrect number of lattice vectors."
            )

        if len(velocities) > 0:
            atoms.set_velocities(np.array(velocities))
        atoms.set_constraint(self.constraints)

        return atoms

    @cached_property
    def forces(self):
        """Parse the forces from the aims.out file"""
        line_start = self.reverse_search_for(["Total atomic forces"])
        if line_start == LINE_NOT_FOUND:
            return None

        line_start += 1

        return np.array(
            [
                [float(inp) for inp in line.split()[-3:]]
                for line in self.lines[line_start:line_start + self.n_atoms]
            ]
        )

    @cached_property
    def stresses(self):
        """Parse the stresses from the aims.out file"""
        line_start = self.reverse_search_for(
            ["Per atom stress (eV) used for heat flux calculation"]
        )
        if line_start == LINE_NOT_FOUND:
            return None
        line_start += 3
        stresses = []
        for line in self.lines[line_start:line_start + self.n_atoms]:
            xx, yy, zz, xy, xz, yz = (float(d) for d in line.split()[2:8])
            stresses.append([xx, yy, zz, yz, xz, xy])

        return np.array(stresses)

    @cached_property
    def stress(self):
        """Parse the stress from the aims.out file"""
        from ase.stress import full_3x3_to_voigt_6_stress

        line_start = self.reverse_search_for(
            [
                "Analytical stress tensor - Symmetrized",
                "Numerical stress tensor",
            ]

        )  # Offest to relevant lines
        if line_start == LINE_NOT_FOUND:
            return None

        stress = [
            [float(inp) for inp in line.split()[2:5]]
            for line in self.lines[line_start + 5:line_start + 8]
        ]
        return full_3x3_to_voigt_6_stress(stress)

    @cached_property
    def is_metallic(self):
        """Checks the outputfile to see if the chunk corresponds
        to a metallic system"""
        line_start = self.reverse_search_for(
            ["material is metallic within the approximate finite "
             "broadening function (occupation_type)"])
        return line_start != LINE_NOT_FOUND

    @cached_property
    def total_energy(self):
        """Parse the energy from the aims.out file"""
        if np.all(self._atoms.pbc) and self.is_metallic:
            line_ind = self.reverse_search_for(["Total energy corrected"])
        else:
            line_ind = self.reverse_search_for(["Total energy uncorrected"])
        if line_ind == LINE_NOT_FOUND:
            raise AimsParseError("No energy is associated with the structure.")

        return float(self.lines[line_ind].split()[5])

    @cached_property
    def dipole(self):
        """Parse the electric dipole moment from the aims.out file."""
        line_start = self.reverse_search_for(["Total dipole moment [eAng]"])
        if line_start == LINE_NOT_FOUND:
            return None

        line = self.lines[line_start]
        return np.array([float(inp) for inp in line.split()[6:9]])

    @cached_property
    def dielectric_tensor(self):
        """Parse the dielectric tensor from the aims.out file"""
        line_start = self.reverse_search_for(
            ["DFPT for dielectric_constant:--->",
             "PARSE DFPT_dielectric_tensor"],
        )
        if line_start == LINE_NOT_FOUND:
            return None

        # we should find the tensor in the next three lines:
        lines = self.lines[line_start + 1:line_start + 4]

        # make ndarray and return
        return np.array([np.fromstring(line, sep=' ') for line in lines])

    @cached_property
    def polarization(self):
        """ Parse the polarization vector from the aims.out file"""
        line_start = self.reverse_search_for(["| Cartesian Polarization"])
        if line_start == LINE_NOT_FOUND:
            return None
        line = self.lines[line_start]
        return np.array([float(s) for s in line.split()[-3:]])

    @cached_property
    def _hirshfeld(self):
        """Parse the Hirshfled charges volumes, and dipole moments from the
        ouput"""
        line_start = self.reverse_search_for(
            ["Performing Hirshfeld analysis of fragment charges and moments."]
        )
        if line_start == LINE_NOT_FOUND:
            return {
                "charges": None,
                "volumes": None,
                "atomic_dipoles": None,
                "dipole": None,
            }

        line_inds = self.search_for_all("Hirshfeld charge", line_start, -1)
        hirshfeld_charges = np.array(
            [float(self.lines[ind].split(":")[1]) for ind in line_inds]
        )

        line_inds = self.search_for_all("Hirshfeld volume", line_start, -1)
        hirshfeld_volumes = np.array(
            [float(self.lines[ind].split(":")[1]) for ind in line_inds]
        )

        line_inds = self.search_for_all(
            "Hirshfeld dipole vector", line_start, -1)
        hirshfeld_atomic_dipoles = np.array(
            [
                [float(inp) for inp in self.lines[ind].split(":")[1].split()]
                for ind in line_inds
            ]
        )

        if not np.any(self._atoms.pbc):
            positions = self._atoms.get_positions()
            hirshfeld_dipole = np.sum(
                hirshfeld_charges.reshape((-1, 1)) * positions,
                axis=1,
            )
        else:
            hirshfeld_dipole = None
        return {
            "charges": hirshfeld_charges,
            "volumes": hirshfeld_volumes,
            "atomic_dipoles": hirshfeld_atomic_dipoles,
            "dipole": hirshfeld_dipole,
        }

    @cached_property
    def _eigenvalues(self):
        """Parse the eigenvalues and occupancies of the system. If eigenvalue
        for a particular k-point is not present in the output file
        then set it to np.nan
        """

        line_start = self.reverse_search_for(["Writing Kohn-Sham eigenvalues."])
        if line_start == LINE_NOT_FOUND:
            return {"eigenvalues": None, "occupancies": None}

        line_end_1 = self.reverse_search_for(
            ["Self-consistency cycle converged."], line_start
        )
        line_end_2 = self.reverse_search_for(
            [
                "What follows are estimated values for band gap, "
                "HOMO, LUMO, etc.",
                "Current spin moment of the entire structure :",
                "Highest occupied state (VBM)"
            ],
            line_start,
        )
        if line_end_1 == LINE_NOT_FOUND:
            line_end = line_end_2
        elif line_end_2 == LINE_NOT_FOUND:
            line_end = line_end_1
        else:
            line_end = min(line_end_1, line_end_2)

        n_kpts = self.n_k_points if np.all(self._atoms.pbc) else 1
        if n_kpts is None:
            return {"eigenvalues": None, "occupancies": None}

        eigenvalues = np.full((n_kpts, self.n_bands, self.n_spins), np.nan)
        occupancies = np.full((n_kpts, self.n_bands, self.n_spins), np.nan)

        occupation_block_start = self.search_for_all(
            "State    Occupation    Eigenvalue [Ha]    Eigenvalue [eV]",
            line_start,
            line_end,
        )
        kpt_def = self.search_for_all("K-point: ", line_start, line_end)

        if len(kpt_def) > 0:
            kpt_inds = [int(self.lines[ll].split()[1]) - 1 for ll in kpt_def]
        elif (self.n_k_points is None) or (self.n_k_points == 1):
            kpt_inds = [0]
        else:
            raise ParseError("Cannot find k-point definitions")

        assert len(kpt_inds) == len(occupation_block_start)
        spins = [0] * len(occupation_block_start)

        if self.n_spins == 2:
            spin_def = self.search_for_all("Spin-", line_start, line_end)
            assert len(spin_def) == len(occupation_block_start)

            spins = [int("Spin-down eigenvalues:" in self.lines[ll])
                     for ll in spin_def]

        for occ_start, kpt_ind, spin in zip(
                occupation_block_start, kpt_inds, spins):
            for ll, line in enumerate(
                self.lines[occ_start + 1:occ_start + self.n_bands + 1]
            ):
                if "***" in line:
                    warn_msg = f"The {ll + 1}th eigenvalue for the "
                    "{kpt_ind+1}th k-point and {spin}th channels could "
                    "not be read (likely too large to be printed "
                    "in the output file)"
                    warnings.warn(warn_msg)
                    continue
                split_line = line.split()
                eigenvalues[kpt_ind, ll, spin] = float(split_line[3])
                occupancies[kpt_ind, ll, spin] = float(split_line[1])
        return {"eigenvalues": eigenvalues, "occupancies": occupancies}

    @cached_property
    def atoms(self):
        """Convert AimsOutChunk to Atoms object and add all non-standard
outputs to atoms.info"""
        atoms = self._atoms

        atoms.calc = SinglePointDFTCalculator(
            atoms,
            energy=self.free_energy,
            free_energy=self.free_energy,
            forces=self.forces,
            stress=self.stress,
            stresses=self.stresses,
            magmom=self.magmom,
            dipole=self.dipole,
            dielectric_tensor=self.dielectric_tensor,
            polarization=self.polarization,
        )
        return atoms

    @property
    def results(self):
        """Convert an AimsOutChunk to a Results Dictionary"""
        results = {
            "energy": self.free_energy,
            "free_energy": self.free_energy,
            "total_energy": self.total_energy,
            "forces": self.forces,
            "stress": self.stress,
            "stresses": self.stresses,
            "magmom": self.magmom,
            "dipole": self.dipole,
            "fermi_energy": self.E_f,
            "n_iter": self.n_iter,
            "hirshfeld_charges": self.hirshfeld_charges,
            "hirshfeld_dipole": self.hirshfeld_dipole,
            "hirshfeld_volumes": self.hirshfeld_volumes,
            "hirshfeld_atomic_dipoles": self.hirshfeld_atomic_dipoles,
            "eigenvalues": self.eigenvalues,
            "occupancies": self.occupancies,
            "dielectric_tensor": self.dielectric_tensor,
            "polarization": self.polarization,
        }

        return {
            key: value for key,
            value in results.items() if value is not None}

    @property
    def initial_atoms(self):
        """The initial structure defined in the geoemtry.in file"""
        return self._header["initial_atoms"]

    @property
    def initial_cell(self):
        """The initial lattice vectors defined in the geoemtry.in file"""
        return self._header["initial_cell"]

    @property
    def constraints(self):
        """The relaxation constraints for the calculation"""
        return self._header["constraints"]

    @property
    def n_atoms(self):
        """The number of atoms for the material"""
        return self._header["n_atoms"]

    @property
    def n_bands(self):
        """The number of Kohn-Sham states for the chunk"""
        return self._header["n_bands"]

    @property
    def n_electrons(self):
        """The number of electrons for the chunk"""
        return self._header["n_electrons"]

    @property
    def n_spins(self):
        """The number of spin channels for the chunk"""
        return self._header["n_spins"]

    @property
    def electronic_temperature(self):
        """The electronic temperature for the chunk"""
        return self._header["electronic_temperature"]

    @property
    def n_k_points(self):
        """The number of electrons for the chunk"""
        return self._header["n_k_points"]

    @property
    def k_points(self):
        """The number of spin channels for the chunk"""
        return self._header["k_points"]

    @property
    def k_point_weights(self):
        """k_point_weights electronic temperature for the chunk"""
        return self._header["k_point_weights"]

    @cached_property
    def free_energy(self):
        """The free energy for the chunk"""
        return self.parse_scalar("free_energy")

    @cached_property
    def n_iter(self):
        """The number of SCF iterations needed to converge the SCF cycle for
the chunk"""
        return self.parse_scalar("number_of_iterations")

    @cached_property
    def magmom(self):
        """The magnetic moment for the chunk"""
        return self.parse_scalar("magnetic_moment")

    @cached_property
    def E_f(self):
        """The Fermi energy for the chunk"""
        return self.parse_scalar("fermi_energy")

    @cached_property
    def converged(self):
        """True if the chunk is a fully converged final structure"""
        return (len(self.lines) > 0) and ("Have a nice day." in self.lines[-5:])

    @property
    def hirshfeld_charges(self):
        """The Hirshfeld charges for the chunk"""
        return self._hirshfeld["charges"]

    @property
    def hirshfeld_atomic_dipoles(self):
        """The Hirshfeld atomic dipole moments for the chunk"""
        return self._hirshfeld["atomic_dipoles"]

    @property
    def hirshfeld_volumes(self):
        """The Hirshfeld volume for the chunk"""
        return self._hirshfeld["volumes"]

    @property
    def hirshfeld_dipole(self):
        """The Hirshfeld systematic dipole moment for the chunk"""
        if not np.any(self._atoms.pbc):
            return self._hirshfeld["dipole"]

        return None

    @property
    def eigenvalues(self):
        """All outputted eigenvalues for the system"""
        return self._eigenvalues["eigenvalues"]

    @property
    def occupancies(self):
        """All outputted occupancies for the system"""
        return self._eigenvalues["occupancies"]


def get_header_chunk(fd):
    """Returns the header information from the aims.out file"""
    header = []
    line = ""

    # Stop the header once the first SCF cycle begins
    while (
        "Convergence:    q app. |  density  | eigen (eV) | Etot (eV)"
            not in line
            and "Convergence:    q app. |  density,  spin     | eigen (eV) |"
            not in line
            and "Begin self-consistency iteration #" not in line
    ):
        try:
            line = next(fd).strip()  # Raises StopIteration on empty file
        except StopIteration:
            raise ParseError(
                "No SCF steps present, calculation failed at setup."
            )

        header.append(line)
    return AimsOutHeaderChunk(header)


def get_aims_out_chunks(fd, header_chunk):
    """Yield unprocessed chunks (header, lines) for each AimsOutChunk image."""
    try:
        line = next(fd).strip()  # Raises StopIteration on empty file
    except StopIteration:
        return

    # If the calculation is relaxation the updated structural information
    # occurs before the re-initialization
    if header_chunk.is_relaxation:
        chunk_end_line = (
            "Geometry optimization: Attempting to predict improved coordinates."
        )
    else:
        chunk_end_line = "Begin self-consistency loop: Re-initialization"

    # If SCF is not converged then do not treat the next chunk_end_line as a
    # new chunk until after the SCF is re-initialized
    ignore_chunk_end_line = False
    while True:
        try:
            line = next(fd).strip()  # Raises StopIteration on empty file
        except StopIteration:
            break

        lines = []
        while chunk_end_line not in line or ignore_chunk_end_line:
            lines.append(line)
            # If SCF cycle not converged or numerical stresses are requested,
            # don't end chunk on next Re-initialization
            patterns = [
                (
                    "Self-consistency cycle not yet converged -"
                    " restarting mixer to attempt better convergence."
                ),
                (
                    "Components of the stress tensor (for mathematical "
                    "background see comments in numerical_stress.f90)."
                ),
                "Calculation of numerical stress completed",
            ]
            if any(pattern in line for pattern in patterns):
                ignore_chunk_end_line = True
            elif "Begin self-consistency loop: Re-initialization" in line:
                ignore_chunk_end_line = False

            try:
                line = next(fd).strip()
            except StopIteration:
                break

        yield AimsOutCalcChunk(lines, header_chunk)


def check_convergence(chunks, non_convergence_ok=False):
    """Check if the aims output file is for a converged calculation

    Parameters
    ----------
    chunks: list of AimsOutChunks
        The list of chunks for the aims calculations
    non_convergence_ok: bool
        True if it is okay for the calculation to not be converged

    Returns
    -------
    bool
        True if the calculation is converged
    """
    if not non_convergence_ok and not chunks[-1].converged:
        raise ParseError("The calculation did not complete successfully")
    return True


@reader
def read_aims_output(fd, index=-1, non_convergence_ok=False):
    """Import FHI-aims output files with all data available, i.e.
    relaxations, MD information, force information etc etc etc."""
    header_chunk = get_header_chunk(fd)
    chunks = list(get_aims_out_chunks(fd, header_chunk))
    check_convergence(chunks, non_convergence_ok)

    # Relaxations have an additional footer chunk due to how it is split
    if header_chunk.is_relaxation:
        images = [chunk.atoms for chunk in chunks[:-1]]
    else:
        images = [chunk.atoms for chunk in chunks]
    return images[index]


@reader
def read_aims_results(fd, index=-1, non_convergence_ok=False):
    """Import FHI-aims output files and summarize all relevant information
    into a dictionary"""
    header_chunk = get_header_chunk(fd)
    chunks = list(get_aims_out_chunks(fd, header_chunk))
    check_convergence(chunks, non_convergence_ok)

    # Relaxations have an additional footer chunk due to how it is split
    if header_chunk.is_relaxation and (index == -1):
        return chunks[-2].results

    return chunks[index].results
