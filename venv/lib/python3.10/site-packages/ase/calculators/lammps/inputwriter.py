# fmt: off

"""
Stream input commands to lammps to perform desired simulations
"""
from ase.calculators.lammps.unitconvert import convert
from ase.parallel import paropen

# "End mark" used to indicate that the calculation is done
CALCULATION_END_MARK = "__end_of_ase_invoked_calculation__"


def lammps_create_atoms(fileobj, parameters, atoms, prismobj):
    """Create atoms in lammps with 'create_box' and 'create_atoms'

    :param fileobj: open stream for lammps input
    :param parameters: dict of all lammps parameters
    :type parameters: dict
    :param atoms: Atoms object
    :type atoms: Atoms
    :param prismobj: coordinate transformation between ase and lammps
    :type prismobj: Prism

    """
    if parameters["verbose"]:
        fileobj.write("## Original ase cell\n")
        fileobj.write(
            "".join(
                [
                    "# {:.16} {:.16} {:.16}\n".format(*x)
                    for x in atoms.get_cell()
                ]
            )
        )

    fileobj.write("lattice sc 1.0\n")

    # Get cell parameters and convert from ASE units to LAMMPS units
    xhi, yhi, zhi, xy, xz, yz = convert(prismobj.get_lammps_prism(),
                                        "distance", "ASE", parameters.units)

    if parameters["always_triclinic"] or prismobj.is_skewed():
        fileobj.write(
            "region asecell prism 0.0 {} 0.0 {} 0.0 {} ".format(
                xhi, yhi, zhi
            )
        )
        fileobj.write(
            f"{xy} {xz} {yz} side in units box\n"
        )
    else:
        fileobj.write(
            "region asecell block 0.0 {} 0.0 {} 0.0 {} "
            "side in units box\n".format(xhi, yhi, zhi)
        )

    symbols = atoms.get_chemical_symbols()
    try:
        # By request, specific atom type ordering
        species = parameters["specorder"]
    except AttributeError:
        # By default, atom types in alphabetic order
        species = sorted(set(symbols))

    species_i = {s: i + 1 for i, s in enumerate(species)}

    fileobj.write(
        "create_box {} asecell\n" "".format(len(species))
    )
    for sym, pos in zip(symbols, atoms.get_positions()):
        # Convert position from ASE units to LAMMPS units
        pos = convert(pos, "distance", "ASE", parameters.units)
        if parameters["verbose"]:
            fileobj.write(
                "# atom pos in ase cell: {:.16} {:.16} {:.16}\n"
                "".format(*tuple(pos))
            )
        fileobj.write(
            "create_atoms {} single {} {} {} remap yes units box\n".format(
                *((species_i[sym],) + tuple(prismobj.vector_to_lammps(pos)))
            )
        )


def write_lammps_in(lammps_in, parameters, atoms, prismobj,
                    lammps_trj=None, lammps_data=None):
    """Write a LAMMPS in_ file with run parameters and settings."""

    def write_model_post_and_masses(fileobj, parameters):
        # write additional lines needed for some LAMMPS potentials
        if 'model_post' in parameters:
            mlines = parameters['model_post']
            for ii in range(len(mlines)):
                fileobj.write(mlines[ii])

        if "masses" in parameters:
            for mass in parameters["masses"]:
                # Note that the variable mass is a string containing
                # the type number and value of mass separated by a space
                fileobj.write(f"mass {mass} \n")

    if isinstance(lammps_in, str):
        fileobj = paropen(lammps_in, "w")
        close_in_file = True
    else:
        # Expect lammps_in to be a file-like object
        fileobj = lammps_in
        close_in_file = False

    if parameters["verbose"]:
        fileobj.write("# (written by ASE)\n")

    # Write variables
    fileobj.write(
        (
            "clear\n"
            'variable dump_file string "{}"\n'
            'variable data_file string "{}"\n'
        ).format(lammps_trj, lammps_data)
    )

    if "package" in parameters:
        fileobj.write(
            "\n".join(
                [f"package {p}" for p in parameters["package"]]
            )
            + "\n"
        )

    # setup styles except 'pair_style'
    for style_type in ("atom", "bond", "angle",
                       "dihedral", "improper", "kspace"):
        style = style_type + "_style"
        if style in parameters:
            fileobj.write(
                '{} {} \n'.format(
                    style,
                    parameters[style]))

    # write initialization lines needed for some LAMMPS potentials
    if 'model_init' in parameters:
        mlines = parameters['model_init']
        for ii in range(len(mlines)):
            fileobj.write(mlines[ii])

    # write units
    if 'units' in parameters:
        units_line = 'units ' + parameters['units'] + '\n'
        fileobj.write(units_line)
    else:
        fileobj.write('units metal\n')

    pbc = atoms.get_pbc()
    if "boundary" in parameters:
        fileobj.write(
            "boundary {} \n".format(parameters["boundary"])
        )
    else:
        fileobj.write(
            "boundary {} {} {} \n".format(
                *tuple("sp"[int(x)] for x in pbc)
            )
        )
    # Prior to version 22Dec2022, `box tilt large` is necessary to run systems
    # with large tilts. Since version 22Dec2022, this command is ignored, and
    # systems with large tilts can be run by default.
    # https://docs.lammps.org/Commands_removed.html#box-command
    # This command does not affect the efficiency for systems with small tilts
    # and therefore worth written always.
    fileobj.write("box tilt large \n")
    fileobj.write("atom_modify sort 0 0.0 \n")
    for key in ("neighbor", "newton"):
        if key in parameters:
            fileobj.write(
                f"{key} {parameters[key]} \n"
            )
    fileobj.write("\n")

    # write the simulation box and the atoms
    if not lammps_data:
        lammps_create_atoms(fileobj, parameters, atoms, prismobj)
    # or simply refer to the data-file
    else:
        fileobj.write(f"read_data {lammps_data}\n")

    # Write interaction stuff
    fileobj.write("\n### interactions\n")
    if "kim_interactions" in parameters:
        fileobj.write(
            "{}\n".format(
                parameters["kim_interactions"]))
        write_model_post_and_masses(fileobj, parameters)

    elif ("pair_style" in parameters) and ("pair_coeff" in parameters):
        pair_style = parameters["pair_style"]
        fileobj.write(f"pair_style {pair_style} \n")
        for pair_coeff in parameters["pair_coeff"]:
            fileobj.write(
                "pair_coeff {} \n" "".format(pair_coeff)
            )
        write_model_post_and_masses(fileobj, parameters)

    else:
        # simple default parameters
        # that should always make the LAMMPS calculation run
        fileobj.write(
            "pair_style lj/cut 2.5 \n"
            "pair_coeff * * 1 1 \n"
            "mass * 1.0 \n"
        )

    if "group" in parameters:
        fileobj.write(
            "\n".join([f"group {p}" for p in parameters["group"]])
            + "\n"
        )

    fileobj.write("\n### run\n" "fix fix_nve all nve\n")

    if "fix" in parameters:
        fileobj.write(
            "\n".join([f"fix {p}" for p in parameters["fix"]])
            + "\n"
        )

    fileobj.write(
        "dump dump_all all custom {1} {0} id type x y z vx vy vz "
        "fx fy fz\n"
        "".format(lammps_trj, parameters["dump_period"])
    )
    fileobj.write(
        "thermo_style custom {}\n"
        "thermo_modify flush yes format float %23.16g\n"
        "thermo 1\n".format(" ".join(parameters["thermo_args"]))
    )

    if "timestep" in parameters:
        fileobj.write(
            "timestep {}\n".format(parameters["timestep"])
        )

    if "minimize" in parameters:
        fileobj.write(
            "minimize {}\n".format(parameters["minimize"])
        )
    if "run" in parameters:
        fileobj.write("run {}\n".format(parameters["run"]))
    if "minimize" not in parameters and "run" not in parameters:
        fileobj.write("run 0\n")

    fileobj.write(
        f'print "{CALCULATION_END_MARK}" \n'
    )
    # Force LAMMPS to flush log
    fileobj.write("log /dev/stdout\n")

    fileobj.flush()
    if close_in_file:
        fileobj.close()
