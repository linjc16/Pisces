# fmt: off

import re
import warnings
from copy import deepcopy
from os.path import dirname, isfile
from pathlib import Path

import numpy as np

from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointDFTCalculator
from ase.cell import Cell
from ase.units import Bohr

no_positions_error = (
    "no positions can be read from this onetep output "
    "if you wish to use ASE to read onetep outputs "
    "please use uppercase block positions in your calculations"
)

unable_to_read = "unable to read this onetep output file, ending"

# taken from onetep source code,
# does not seem to be from any known NIST data
units = {"Hartree": 27.2116529, "Bohr": 1 / 1.889726134583548707935}

# Want to add a functionality? add a global constant below
ONETEP_START = re.compile(
    r"(?i)^\s*\|\s*Linear-Scaling\s*Ab\s*"
    r"Initio\s*Total\s*Energy\s*Program\s*\|\s*$"
)
ONETEP_STOP = re.compile(r"(?i)^\s*-+\s*TIMING\s*INFORMATION\s*-+\s*$")
ONETEP_TOTAL_ENERGY = re.compile(
    r"(?i)^\s*\|\s*\*{3}\s*NGWF\s*" r"optimisation\s*converged\s*\*{3}\s*\|\s*$"
)
ONETEP_FORCE = re.compile(r"(?i)^\s*\*+\s*Forces\s*\*+\s*$")
ONETEP_MULLIKEN = re.compile(r"(?i)^\s*Mulliken\s*Atomic\s*Populations\s*$")
ONETEP_SPIN = re.compile(r"(?i)^\s*Down\s*spin\s*density")
ONETEP_POSITION = re.compile(r"(?i)^\s*Cell\s*Contents\s*$")
ONETEP_FIRST_POSITION = re.compile(
    r"^\s*%BLOCK\s*POSITIONS\s*_?\s*(ABS|FRAC)\s*:?\s*([*#!].*)?$"
)
ONETEP_WRONG_FIRST_POSITION = re.compile(
    r"^\s*%block\s*positions\s*_?\s*(abs|frac)\s*:?\s*([*#!].*)?$"
)
ONETEP_RESUMING_GEOM = re.compile(
    r"(?i)^\s*<{16}\s*Resuming\s*previous"
    r"\s*ONETEP\s*Geometry\s*Optimisation\s*>{16}\s*$"
)

ONETEP_ATOM_COUNT = re.compile(r"(?i)^\s*Totals\s*:\s*(\d+\s*)*$")
ONETEP_IBFGS_ITER = re.compile(r"(?i)^\s*BFGS\s*:\s*starting\s*iteration")
ONETEP_IBFGS_IMPROVE = re.compile(r"(?i)^\s*BFGS\s*:\s*improving\s*iteration")
ONETEP_START_GEOM = re.compile(
    r"(?i)^<+\s*Starting\s*ONETEP\s*Geometry\s*Optimisation\s*>+$"
)
ONETEP_END_GEOM = re.compile(r"(?i)^\s*BFGS\s*:\s*Final\s*Configuration:\s*$")

ONETEP_SPECIES = re.compile(r"(?i)^\s*%BLOCK\s*SPECIES\s*:?\s*([*#!].*)?$")

ONETEP_FIRST_CELL = re.compile(
    r"(?i)^\s*%BLOCK\s*LATTICE\s*_?\s*CART\s*:?\s*([*#!].*)?$"
)
ONETEP_STRESS_CELL = re.compile(
    r"(?i)^\s*stress_calculation:\s*cell\s*geometry\s*$"
)


def get_onetep_keywords(path):
    if isinstance(path, str):
        with open(path) as fd:
            results = read_onetep_in(fd, only_keywords=True)
    else:
        results = read_onetep_in(path, only_keywords=True)

    # If there is an include file, the entire
    # file keyword's will be included in the dict
    # and the include_file keyword will be deleted
    if "include_file" in results["keywords"]:
        warnings.warn("include_file will be deleted from the dict")
        del results["keywords"]["include_file"]
    return results["keywords"]


def read_onetep_in(fd, **kwargs):
    """
    Read a single ONETEP input.

    This function can be used to visually check ONETEP inputs,
    using the ase gui. It can also be used to get access to
    the input parameters attached to the ONETEP calculator
    returned

    The function should work on inputs which contain
    'include_file' command(s), (possibly recursively
    but untested)

    The function should work on input which contains
    exotic element(s) name(s) if the specie block is
    present to map them back to real element(s)

    Parameters
    ----------
    fd : io-object
        File to read.

    Return
    ------
    structure: Atoms
        Atoms object with cell and a Onetep calculator
        attached which contains the keywords dictionary
    """

    fdi_lines = fd.readlines()

    try:
        fd_path = Path(fd.name).resolve()
        fd_parent = fd_path.parent
        include_files = [fd_path]
    except AttributeError:
        # We are in a StringIO or something similar
        fd_path = Path().cwd()
        fd_parent = fd_path
        include_files = [Path().cwd()]

    def clean_lines(lines):
        """
        Remove indesirable line from the input
        """
        new_lines = []
        for line in lines:
            sep = re.split(r"[!#]", line.strip())[0]
            if sep:
                new_lines.append(sep)
        return new_lines

    # Skip comments and empty lines
    fdi_lines = clean_lines(fdi_lines)

    # Are we in a block?
    block_start = 0

    keywords = {}
    atoms = Atoms()
    cell = np.zeros((3, 3))
    fractional = False
    positions = False
    symbols = False

    # Main loop reading the input
    for n, line in enumerate(fdi_lines):
        line_lower = line.lower()
        if re.search(r"^\s*%block", line_lower):
            block_start = n + 1
            if re.search(r"lattice_cart$", line_lower):
                if re.search(r"^\s*ang\s*$", fdi_lines[block_start]):
                    cell = np.loadtxt(fdi_lines[n + 2: n + 5])
                else:
                    cell = np.loadtxt(fdi_lines[n + 1: n + 4])
                    cell *= Bohr

        if not block_start:
            if "devel_code" in line_lower:
                warnings.warn("devel_code is not supported")
                continue
            # Splits line on any valid onetep separator
            sep = re.split(r"[:=\s]+", line)
            keywords[sep[0]] = " ".join(sep[1:])
            # If include_file is used, we open the included file
            # and insert it in the current fdi_lines...
            # ONETEP does not work with cascade
            # and this SHOULD NOT work with cascade
            if re.search(r"^\s*include_file$", sep[0]):
                name = sep[1].replace("'", "")
                name = name.replace('"', "")
                new_path = fd_parent / name
                for path in include_files:
                    if new_path.samefile(path):
                        raise ValueError("invalid/recursive include_file")
                new_fd = open(new_path)
                new_lines = new_fd.readlines()
                new_lines = clean_lines(new_lines)
                for include_line in new_lines:
                    sep = re.split(r"[:=\s]+", include_line)
                    if re.search(r"^\s*include_file$", sep[0]):
                        raise ValueError("nested include_file")
                fdi_lines[:] = (
                    fdi_lines[: n + 1] + new_lines + fdi_lines[n + 1:]
                )
                include_files.append(new_path)
                continue

        if re.search(r"^\s*%endblock", line_lower):
            if re.search(r"\s*positions_", line_lower):
                head = re.search(r"(?i)^\s*(\S*)\s*$", fdi_lines[block_start])
                head = head.group(1).lower() if head else ""
                conv = 1 if head == "ang" else units["Bohr"]
                # Skip one line if head is True
                to_read = fdi_lines[block_start + int(bool(head)): n]
                positions = np.loadtxt(to_read, usecols=(1, 2, 3))
                positions *= conv
                symbols = np.loadtxt(to_read, usecols=(0), dtype="str")
                if re.search(r".*frac$", line_lower):
                    fractional = True
            elif re.search(r"^\s*%endblock\s*species$", line_lower):
                els = fdi_lines[block_start:n]
                species = {}
                for el in els:
                    sep = el.split()
                    species[sep[0]] = sep[1]
                to_read = [i.strip() for i in fdi_lines[block_start:n]]
                keywords["species"] = to_read
            elif re.search(r"lattice_cart$", line_lower):
                pass
            else:
                to_read = [i.strip() for i in fdi_lines[block_start:n]]
                block_title = line_lower.replace("%endblock", "").strip()
                keywords[block_title] = to_read
            block_start = 0

    # We don't need a fully valid onetep
    # input to read the keywords, just
    # the keywords
    if kwargs.get("only_keywords", False):
        return {"keywords": keywords}
    # Necessary if we have only one atom
    # Check if the cell is valid (3D)
    if not cell.any(axis=1).all():
        raise ValueError("invalid cell specified")

    if positions is False:
        raise ValueError("invalid position specified")

    if symbols is False:
        raise ValueError("no symbols found")

    positions = positions.reshape(-1, 3)
    symbols = symbols.reshape(-1)
    tags = []
    info = {"onetep_species": []}
    for symbol in symbols:
        label = symbol.replace(species[symbol], "")
        if label.isdigit():
            tags.append(int(label))
        else:
            tags.append(0)
        info["onetep_species"].append(symbol)
    atoms = Atoms(
        [species[i] for i in symbols], cell=cell, pbc=True, tags=tags, info=info
    )
    if fractional:
        atoms.set_scaled_positions(positions / units["Bohr"])
    else:
        atoms.set_positions(positions)
    results = {"atoms": atoms, "keywords": keywords}
    return results


def write_onetep_in(
    fd,
    atoms,
    edft=False,
    xc="PBE",
    ngwf_count=-1,
    ngwf_radius=9.0,
    keywords={},
    pseudopotentials={},
    pseudo_path=".",
    pseudo_suffix=None,
    **kwargs
):
    """
    Write a single ONETEP input.

    This function will be used by ASE to perform
    various workflows (Opt, NEB...) or can be used
    manually to quickly create ONETEP input file(s).

    The function will first write keywords in
    alphabetic order in lowercase. Secondly, blocks
    will be written in alphabetic order in uppercase.

    Two ways to work with the function:

        - By providing only (simple) keywords present in
          the parameters. ngwf_count and ngwf_radius
          accept multiple types as described in the Parameters
          section.

        - If the keywords parameters is provided as a dictionary
          these keywords will be used to write the input file and
          will take priority.

    If no pseudopotentials are provided in the parameters and
    the function will try to look for suitable pseudopotential
    in the pseudo_path.

    Parameters
    ----------
    fd : file
        File to write.
    atoms: Atoms
        Atoms including Cell object to write.
    edft: Bool
        Activate EDFT.
    xc: str
        DFT xc to use e.g (PBE, RPBE, ...)
    ngwf_count: int|list|dict
        Behaviour depends on the type:
            int: every species will have this amount
            of ngwfs.
            list: list of int, will be attributed
            alphabetically to species:
            dict: keys are species name(s),
            value are their number:
    ngwf_radius: int|list|dict
        Behaviour depends on the type:
            float: every species will have this radius.
            list: list of float, will be attributed
            alphabetically to species:
            [10.0, 9.0]
            dict: keys are species name(s),
            value are their radius:
            {'Na': 9.0, 'Cl': 10.0}
    keywords: dict
        Dictionary with ONETEP keywords to write,
        keywords with lists as values will be
        treated like blocks, with each element
        of list being a different line.
    pseudopotentials: dict
        Behaviour depends on the type:
            keys are species name(s) their
            value are the pseudopotential file to use:
            {'Na': 'Na.usp', 'Cl': 'Cl.usp'}
    pseudo_path: str
        Where to look for pseudopotential, correspond
        to the pseudo_path keyword of ONETEP.
    pseudo_suffix: str
        Suffix for the pseudopotential filename
        to look for, useful if you have multiple sets of
        pseudopotentials in pseudo_path.
    """

    label = kwargs.get("label", "onetep")
    try:
        directory = kwargs.get("directory", Path(dirname(fd.name)))
    except AttributeError:
        directory = "."
    autorestart = kwargs.get("autorestart", False)
    elements = np.array(atoms.symbols)
    tags = np.array(atoms.get_tags())
    species_maybe = atoms.info.get("onetep_species", False)
    #  We look if the atom.info contains onetep species information
    # If it does, we use it, as it might contains character
    #  which are not allowed in ase tags, if not we fall back
    # to tags and use them instead.
    if species_maybe:
        if set(species_maybe) != set(elements):
            species = np.array(species_maybe)
        else:
            species = elements
    else:
        formatted_tags = np.array(["" if i == 0 else str(i) for i in tags])
        species = np.char.add(elements, formatted_tags)
    numbers = np.array(atoms.numbers)
    tmp = np.argsort(species)
    # We sort both Z and name the same
    numbers = np.take_along_axis(numbers, tmp, axis=0)
    # u_elements = np.take_along_axis(elements, tmp, axis=0)
    u_species = np.take_along_axis(species, tmp, axis=0)
    elements = np.take_along_axis(elements, tmp, axis=0)
    # We want to keep unique but without sort: small trick with index
    idx = np.unique(u_species, return_index=True)[1]
    elements = elements[idx]
    # Unique species
    u_species = u_species[idx]
    numbers = numbers[idx]
    n_sp = len(u_species)

    if isinstance(ngwf_count, int):
        ngwf_count = dict(zip(u_species, [ngwf_count] * n_sp))
    elif isinstance(ngwf_count, list):
        ngwf_count = dict(zip(u_species, ngwf_count))
    elif isinstance(ngwf_count, dict):
        pass
    else:
        raise TypeError("ngwf_count can only be int|list|dict")

    if isinstance(ngwf_radius, float):
        ngwf_radius = dict(zip(u_species, [ngwf_radius] * n_sp))
    elif isinstance(ngwf_radius, list):
        ngwf_radius = dict(zip(u_species, ngwf_radius))
    elif isinstance(ngwf_radius, dict):
        pass
    else:
        raise TypeError("ngwf_radius can only be float|list|dict")

    pp_files = re.sub("'|\"", "", keywords.get("pseudo_path", pseudo_path))
    pp_files = Path(pp_files).glob("*")
    pp_files = [i for i in pp_files if i.is_file()]

    if pseudo_suffix:
        common_suffix = [pseudo_suffix]
    else:
        common_suffix = [".usp", ".recpot", ".upf", ".paw", ".psp", ".pspnc"]

    if keywords.get("species_pot", False):
        pp_list = keywords["species_pot"]
    elif isinstance(pseudopotentials, dict):
        pp_list = []
        for idx, el in enumerate(u_species):
            if el in pseudopotentials:
                pp_list.append(f"{el} {pseudopotentials[el]}")
            else:
                for i in pp_files:
                    reg_el_candidate = re.split(r"[-_.:= ]+", i.stem)[0]
                    if (
                        elements[idx] == reg_el_candidate.title()
                        and i.suffix.lower() in common_suffix
                    ):
                        pp_list.append(f"{el} {i.name}")
    else:
        raise TypeError("pseudopotentials object can only be dict")

    default_species = []
    for idx, el in enumerate(u_species):
        tmp = ""
        tmp += u_species[idx] + " " + elements[idx] + " "
        tmp += str(numbers[idx]) + " "
        try:
            tmp += str(ngwf_count[el]) + " "
        except KeyError:
            tmp += str(ngwf_count[elements[idx]]) + " "
        try:
            tmp += str(ngwf_radius[el])
        except KeyError:
            tmp += str(ngwf_radius[elements[idx]])
        default_species.append(tmp)

    positions_abs = ["ang"]
    for s, p in zip(species, atoms.get_positions()):
        line = "{s:>5} {0:>12.6f} {1:>12.6f} {2:>12.6f}".format(s=s, *p)
        positions_abs.append(line)

    lattice_cart = ["ang"]
    for axis in atoms.get_cell():
        line = "{:>16.8f} {:>16.8f} {:>16.8f}".format(*axis)
        lattice_cart.append(line)

    # Default keywords if not provided by the user,
    # most of them are ONETEP default, except write_forces
    # which is always turned on.
    default_keywords = {
        "xc_functional": xc,
        "edft": edft,
        "cutoff_energy": 20,
        "paw": False,
        "task": "singlepoint",
        "output_detail": "normal",
        "species": default_species,
        "pseudo_path": pseudo_path,
        "species_pot": pp_list,
        "positions_abs": positions_abs,
        "lattice_cart": lattice_cart,
        "write_forces": True,
        "forces_output_detail": "verbose",
    }

    # Main loop, fill the keyword dictionary
    keywords = {key.lower(): value for key, value in keywords.items()}
    for value in default_keywords:
        if not keywords.get(value, None):
            keywords[value] = default_keywords[value]

    # No pseudopotential provided, we look for them in pseudo_path
    # If autorestart is True, we look for restart files,
    # and turn on relevant keywords...
    if autorestart:
        keywords["read_denskern"] = isfile(directory / (label + ".dkn"))
        keywords["read_tightbox_ngwfs"] = isfile(
            directory / (label + ".tightbox_ngwfs")
        )
        keywords["read_hamiltonian"] = isfile(directory / (label + ".ham"))

    # If not EDFT, hamiltonian is irrelevant.
    # print(keywords.get('edft', False))
    # keywords['read_hamiltonian'] = \
    # keywords.get('read_hamiltonian', False) & keywords.get('edft', False)

    keywords = dict(sorted(keywords.items()))

    lines = []
    block_lines = []

    for key, value in keywords.items():
        if isinstance(value, (list, np.ndarray)):
            if not all(isinstance(_, str) for _ in value):
                raise TypeError("list values for blocks must be strings only")
            block_lines.append(("\n%block " + key).upper())
            block_lines.extend(value)
            block_lines.append(("%endblock " + key).upper())
        elif isinstance(value, bool):
            lines.append(str(key) + " : " + str(value)[0])
        elif isinstance(value, (str, int, float)):
            lines.append(str(key) + " : " + str(value))
        else:
            raise TypeError("keyword values must be list|str|bool")
    input_header = (
        "!"
        + "-" * 78
        + "!\n"
        + "!"
        + "-" * 33
        + " INPUT FILE "
        + "-" * 33
        + "!\n"
        + "!"
        + "-" * 78
        + "!\n\n"
    )

    input_footer = (
        "\n!"
        + "-" * 78
        + "!\n"
        + "!"
        + "-" * 32
        + " END OF INPUT "
        + "-" * 32
        + "!\n"
        + "!"
        + "-" * 78
        + "!"
    )

    fd.write(input_header)
    fd.writelines(line + "\n" for line in lines)
    fd.writelines(b_line + "\n" for b_line in block_lines)

    if "devel_code" in kwargs:
        warnings.warn("writing devel code as it is, at the end of the file")
        fd.writelines("\n" + line for line in kwargs["devel_code"])

    fd.write(input_footer)


def read_onetep_out(fd, index=-1, improving=False, **kwargs):
    """
    Read ONETEP output(s).

    !!!
    This function will be used by ASE when performing
    various workflows (Opt, NEB...)
    !!!

    Parameters
    ----------
    fd : file
        File to read.
    index: slice
        Which atomic configuration to read
    improving: Bool
        If the output is a geometry optimisation,
        improving = True will keep line search
        configuration from BFGS

    Yields
    ------
    structure: Atoms|list of Atoms
    """
    # Put everything in memory
    fdo_lines = fd.readlines()
    n_lines = len(fdo_lines)

    freg = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+\-]?\d+)?")

    # Used to store index of important elements
    output = {
        ONETEP_START: [],
        ONETEP_STOP: [],
        ONETEP_TOTAL_ENERGY: [],
        ONETEP_FORCE: [],
        ONETEP_SPIN: [],
        ONETEP_MULLIKEN: [],
        ONETEP_POSITION: [],
        ONETEP_FIRST_POSITION: [],
        ONETEP_WRONG_FIRST_POSITION: [],
        ONETEP_ATOM_COUNT: [],
        ONETEP_IBFGS_IMPROVE: [],
        ONETEP_IBFGS_ITER: [],
        ONETEP_START_GEOM: [],
        ONETEP_RESUMING_GEOM: [],
        ONETEP_END_GEOM: [],
        ONETEP_SPECIES: [],
        ONETEP_FIRST_CELL: [],
        ONETEP_STRESS_CELL: [],
    }

    # Index will be treated to get rid of duplicate or improving iterations
    output_corr = deepcopy(output)

    # Core properties that will be used in Yield
    properties = [
        ONETEP_TOTAL_ENERGY,
        ONETEP_FORCE,
        ONETEP_MULLIKEN,
        ONETEP_FIRST_CELL,
    ]

    # Find all matches append them to the dictionary
    breg = "|".join([i.pattern.replace("(?i)", "") for i in output.keys()])
    prematch = {}

    for idx, line in enumerate(fdo_lines):
        matches = re.search(breg, line)
        if matches:
            prematch[idx] = matches.group(0)

    for key, value in prematch.items():
        for reg in output.keys():
            if re.search(reg, value):
                output[reg].append(key)
                break

    output = {key: np.array(value) for key, value in output.items()}
    # Conveniance notation (pointers: no overhead, no additional memory)
    ibfgs_iter = np.hstack((output[ONETEP_IBFGS_ITER], output[ONETEP_END_GEOM]))
    ibfgs_start = output[ONETEP_START_GEOM]
    ibfgs_improve = output[ONETEP_IBFGS_IMPROVE]
    ibfgs_resume = output[ONETEP_RESUMING_GEOM]

    onetep_start = output[ONETEP_START]
    onetep_stop = output[ONETEP_STOP]

    bfgs_keywords = np.hstack((ibfgs_improve, ibfgs_resume, ibfgs_iter))
    bfgs_keywords = np.sort(bfgs_keywords)

    core_keywords = np.hstack(
        (
            ibfgs_iter,
            ibfgs_start,
            ibfgs_improve,
            ibfgs_resume,
            ibfgs_iter,
            onetep_start,
            onetep_stop,
        )
    )
    core_keywords = np.sort(core_keywords)

    i_first_positions = output[ONETEP_FIRST_POSITION]
    is_frac_positions = [i for i in i_first_positions if "FRAC" in fdo_lines[i]]

    # In onetep species can have arbritary names,
    # We want to map them to real element names
    # Via the species block
    # species = np.concatenate((output[ONETEP_SPECIES],
    #                          output[ONETEP_SPECIESL])).astype(np.int32)
    species = output[ONETEP_SPECIES]

    icells = np.hstack((output[ONETEP_FIRST_CELL], output[ONETEP_STRESS_CELL]))
    icells = icells.astype(np.int32)
    # Using the fact that 0 == False and > 0 == True
    has_bfgs = (
        len(ibfgs_iter)
        + len(output[ONETEP_START_GEOM])
        + len(output[ONETEP_RESUMING_GEOM])
    )

    # When the input block position is written in lowercase
    # ONETEP does not print the initial position but a hash
    # of it, might be needed
    has_hash = len(output[ONETEP_WRONG_FIRST_POSITION])

    def is_in_bfgs(idx):
        """
        Check if a given index is in a BFGS block
        """
        for past, future in zip(
            output[ONETEP_START],
            np.hstack((output[ONETEP_START][1:], [n_lines])),
        ):
            if past < idx < future:
                if np.any(
                    (past < ibfgs_start) & (ibfgs_start < future)
                ) or np.any((past < ibfgs_resume) & (ibfgs_resume < future)):
                    return True
        return False

    def where_in_bfgs(idx):
        for past, future in zip(
            core_keywords, np.hstack((core_keywords[1:], [n_lines]))
        ):
            if past < idx < future:
                if past in onetep_start:
                    if future in ibfgs_start or future in ibfgs_resume:
                        return "resume"
                    continue
                # Are we in start or resume or improve
                if past in ibfgs_start:
                    return "start"
                elif past in ibfgs_resume:
                    return "resume"
                elif past in ibfgs_improve:
                    return "improve"

        return False

    ipositions = np.hstack((output[ONETEP_POSITION], i_first_positions)).astype(
        np.int32
    )
    ipositions = np.sort(ipositions)

    n_pos = len(ipositions)

    # Some ONETEP files will not have any positions
    # due to how the software is coded. As a last
    # resort we look for a geom file with the same label.

    if n_pos == 0:
        if has_hash:
            raise RuntimeError(no_positions_error)
        raise RuntimeError(unable_to_read)

    to_del = []

    # Important loop which:
    # - Get rid of improving BFGS iteration if improving == False
    # - Append None to properties to make sure each properties will
    # have the same length and each index correspond to the right
    # atomic configuration (hopefully).
    # Past is the index of the current atomic conf, future is the
    # index of the next one.

    for idx, (past, future) in enumerate(
        zip(ipositions, np.hstack((ipositions[1:], [n_lines])))
    ):
        if has_bfgs:
            which_bfgs = where_in_bfgs(past)

            if which_bfgs == "resume":
                to_del.append(idx)
                continue

            if not improving:
                if which_bfgs == "improve":
                    to_del.append(idx)
                    continue

        # We append None if no properties in contained for
        # one specific atomic configurations.
        for prop in properties:
            (tmp,) = np.where((past < output[prop]) & (output[prop] <= future))
            if len(tmp) == 0:
                output_corr[prop].append(None)
            else:
                output_corr[prop].extend(output[prop][tmp[:1]])

    if to_del and len(to_del) != n_pos:
        new_indices = np.setdiff1d(np.arange(n_pos), to_del)
        ipositions = ipositions[new_indices]

    # Bunch of methods to grep properties from output.
    def parse_cell(idx):
        a, b, c = np.loadtxt([fdo_lines[idx + 2]]) * units["Bohr"]
        al, be, ga = np.loadtxt([fdo_lines[idx + 4]])
        cell = Cell.fromcellpar([a, b, c, al, be, ga])
        return np.array(cell)

    def parse_charge(idx):
        n = 0
        offset = 4
        while idx + n < len(fdo_lines):
            if not fdo_lines[idx + n].strip():
                tmp_charges = np.loadtxt(
                    fdo_lines[idx + offset: idx + n - 1], usecols=3
                )
                return np.reshape(tmp_charges, -1)
            n += 1
        return None

    #  In ONETEP there is no way to differentiate electronic entropy
    #  and entropy due to solvent, therefore there is no way to
    # extrapolate the energy at 0 K. We return the last energy
    #  instead.

    def parse_energy(idx):
        n = 0
        while idx + n < len(fdo_lines):
            if re.search(r"^\s*\|\s*Total\s*:.*\|\s*$", fdo_lines[idx + n]):
                energy_str = re.search(freg, fdo_lines[idx + n]).group(0)
                return float(energy_str) * units["Hartree"]
            n += 1
        return None

    def parse_fermi_level(idx):
        n = 0
        fermi_levels = None
        while idx + n < len(fdo_lines):
            if "Fermi_level" in fdo_lines[idx + n]:
                tmp = "\n".join(fdo_lines[idx + n: idx + n + 1])
                fermi_level = re.findall(freg, tmp)
                fermi_levels = [
                    float(i) * units["Hartree"] for i in fermi_level
                ]
            if re.search(
                r"^\s*<{5}\s*CALCULATION\s*SUMMARY\s*>{5}\s*$",
                fdo_lines[idx + n],
            ):
                return fermi_levels
            n += 1
        return None

    def parse_first_cell(idx):
        n = 0
        offset = 1
        while idx + n < len(fdo_lines):
            if re.search(
                r'(?i)^\s*"?\s*ang\s*"?\s*([*#!].*)?$', fdo_lines[idx + n]
            ):
                offset += 1
            if re.search(
                r"(?i)^\s*%ENDBLOCK\s*LATTICE"
                r"\s*_?\s*CART\s*:?\s*([*#!].*)?$",
                fdo_lines[idx + n],
            ):
                cell = np.loadtxt(fdo_lines[idx + offset: idx + n])
                return cell if offset == 2 else cell * units["Bohr"]
            n += 1
        return None

    def parse_first_positions(idx):
        n = 0
        offset = 1
        while idx + n < len(fdo_lines):
            if re.search(
                r'(?i)^\s*"?\s*ang\s*"?\s*([*#!].*)?$', fdo_lines[idx + n]
            ):
                offset += 1
            if re.search(r"^\s*%ENDBLOCK\s*POSITIONS_", fdo_lines[idx + n]):
                if "FRAC" in fdo_lines[idx + n]:
                    conv_factor = 1
                else:
                    conv_factor = units["Bohr"]
                tmp = np.loadtxt(
                    fdo_lines[idx + offset: idx + n], dtype="str"
                ).reshape(-1, 4)
                els = np.char.array(tmp[:, 0])
                if offset == 2:
                    pos = tmp[:, 1:].astype(np.float64)
                else:
                    pos = tmp[:, 1:].astype(np.float64) * conv_factor
                try:
                    atoms = Atoms(els, pos)
                # ASE doesn't recognize names used in ONETEP
                # as chemical symbol: dig deeper
                except KeyError:
                    tags, real_elements = find_correct_species(
                        els, idx, first=True
                    )
                    atoms = Atoms(real_elements, pos)
                    atoms.set_tags(tags)
                    atoms.info["onetep_species"] = list(els)
                return atoms
            n += 1
        return None

    def parse_force(idx):
        n = 0
        while idx + n < len(fdo_lines):
            if re.search(r"(?i)^\s*\*\s*TOTAL:.*\*\s*$", fdo_lines[idx + n]):
                tmp = np.loadtxt(
                    fdo_lines[idx + 6: idx + n - 2],
                    dtype=np.float64,
                    usecols=(3, 4, 5),
                )
                return tmp * units["Hartree"] / units["Bohr"]
            n += 1
        return None

    def parse_positions(idx):
        n = 0
        offset = 7
        stop = 0
        while idx + n < len(fdo_lines):
            if re.search(r"^\s*x{60,}\s*$", fdo_lines[idx + n]):
                stop += 1
            if stop == 2:
                tmp = np.loadtxt(
                    fdo_lines[idx + offset: idx + n],
                    dtype="str",
                    usecols=(1, 3, 4, 5),
                )
                els = np.char.array(tmp[:, 0])
                pos = tmp[:, 1:].astype(np.float64) * units["Bohr"]
                try:
                    atoms = Atoms(els, pos)
                # ASE doesn't recognize names used in ONETEP
                # as chemical symbol: dig deeper
                except KeyError:
                    tags, real_elements = find_correct_species(els, idx)
                    atoms = Atoms(real_elements, pos)
                    atoms.set_tags(tags)
                    atoms.info["onetep_species"] = list(els)
                return atoms
            n += 1
        return None

    def parse_species(idx):
        n = 1
        element_map = {}
        while idx + n < len(fdo_lines):
            sep = fdo_lines[idx + n].split()
            if re.search(
                r"(?i)^\s*%ENDBLOCK\s*SPECIES\s*:?\s*([*#!].*)?$",
                fdo_lines[idx + n],
            ):
                return element_map
            to_skip = re.search(
                r"(?i)^\s*(ang|bohr)\s*([*#!].*)?$", fdo_lines[idx + n]
            )
            if not to_skip:
                element_map[sep[0]] = sep[1]
            n += 1
        return None

    def parse_spin(idx):
        n = 0
        offset = 4
        while idx + n < len(fdo_lines):
            if not fdo_lines[idx + n].strip():
                # If no spin is present we return None
                try:
                    tmp_spins = np.loadtxt(
                        fdo_lines[idx + offset: idx + n - 1], usecols=4
                    )
                    return np.reshape(tmp_spins, -1)
                except ValueError:
                    return None
            n += 1
        return None

    # This is needed if ASE doesn't recognize the element
    def find_correct_species(els, idx, first=False):
        real_elements = []
        tags = []
        # Find nearest species block in case of
        # multi-output file with different species blocks.
        if first:
            closest_species = np.argmin(abs(idx - species))
        else:
            tmp = idx - species
            tmp[tmp < 0] = 9999999999
            closest_species = np.argmin(tmp)
        elements_map = real_species[closest_species]
        for el in els:
            real_elements.append(elements_map[el])
            tag_maybe = el.replace(elements_map[el], "")
            if tag_maybe.isdigit():
                tags.append(int(tag_maybe))
            else:
                tags.append(False)
        return tags, real_elements

    cells = []
    for idx in icells:
        if idx in output[ONETEP_STRESS_CELL]:
            cell = parse_cell(idx) if idx else None
        else:
            cell = parse_first_cell(idx) if idx else None
        cells.append(cell)

    charges = []
    for idx in output_corr[ONETEP_MULLIKEN]:
        charge = parse_charge(idx) if idx else None
        charges.append(charge)

    energies = []
    for idx in output_corr[ONETEP_TOTAL_ENERGY]:
        energy = parse_energy(idx) if idx else None
        energies.append(energy)

    fermi_levels = []
    for idx in output_corr[ONETEP_TOTAL_ENERGY]:
        fermi_level = parse_fermi_level(idx) if idx else None
        fermi_levels.append(fermi_level)

    magmoms = []
    for idx in output_corr[ONETEP_MULLIKEN]:
        magmom = parse_spin(idx) if idx else None
        magmoms.append(magmom)

    real_species = []
    for idx in species:
        real_specie = parse_species(idx)
        real_species.append(real_specie)

    positions, forces = [], []
    for idx in ipositions:
        if idx in i_first_positions:
            position = parse_first_positions(idx)
        else:
            position = parse_positions(idx)
        if position:
            positions.append(position)
        else:
            n_pos -= 1
            break
    for idx in output_corr[ONETEP_FORCE]:
        force = parse_force(idx) if idx else None
        forces.append(force)

    n_pos = len(positions)

    # Numpy trick to get rid of configuration that are essentially the same
    # in a regular geometry optimisation with internal BFGS, the first
    # configuration is printed three time, we get rid of it
    properties = [energies, forces, charges, magmoms]

    if has_bfgs:
        tmp = [i.positions for i in positions]
        to_del = []
        for i in range(len(tmp[:-1])):
            if is_in_bfgs(ipositions[i]):
                if np.array_equal(tmp[i], tmp[i + 1]):
                    # If the deleted configuration has a property
                    # we want to keep it
                    for prop in properties:
                        if prop[i + 1] is not None and prop[i] is None:
                            prop[i] = prop[i + 1]
                    to_del.append(i + 1)
        c = np.full((len(tmp)), True)
        c[to_del] = False
        energies = [energies[i] for i in range(n_pos) if c[i]]
        forces = [forces[i] for i in range(n_pos) if c[i]]
        charges = [charges[i] for i in range(n_pos) if c[i]]
        magmoms = [magmoms[i] for i in range(n_pos) if c[i]]
        positions = [positions[i] for i in range(n_pos) if c[i]]
        ipositions = [ipositions[i] for i in range(n_pos) if c[i]]

    n_pos = len(positions)

    # We take care of properties that only show up at
    # the beginning of onetep calculation
    whole = np.append(output[ONETEP_START], n_lines)

    if n_pos == 0:
        raise RuntimeError(unable_to_read)

    spin = np.full((n_pos), 1)
    for sp in output[ONETEP_SPIN]:
        tmp = zip(whole, whole[1:])
        for past, future in tmp:
            if past < sp < future:
                p = (past < ipositions) & (ipositions < future)
                spin[p] = 2

    cells_all = np.ones((n_pos, 3, 3))
    for idx, cell in enumerate(output[ONETEP_FIRST_CELL]):
        tmp = zip(whole, whole[1:])
        for past, future in tmp:
            if past < cell < future:
                p = (past < ipositions) & (ipositions < future)
                cells_all[p] = cells[idx]

    # Prepare atom objects with all the properties
    if isinstance(index, int):
        indices = [range(n_pos)[index]]
    else:
        indices = range(n_pos)[index]

    for idx in indices:
        positions[idx].set_cell(cells_all[idx])
        if ipositions[idx] in is_frac_positions:
            positions[idx].set_scaled_positions(positions[idx].get_positions())
        positions[idx].set_initial_charges(charges[idx])
        calc = SinglePointDFTCalculator(
            positions[idx],
            energy=energies[idx] if energies else None,
            free_energy=energies[idx] if energies else None,
            forces=forces[idx] if forces else None,
            charges=charges[idx] if charges else None,
            magmoms=magmoms[idx] if magmoms else None,
        )
        # calc.kpts = [(0, 0, 0) for _ in range(spin[idx])]
        positions[idx].calc = calc
        yield positions[idx]
