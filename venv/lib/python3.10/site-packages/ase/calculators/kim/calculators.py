# fmt: off

import os
import re

from ase.calculators.lammps import convert
from ase.calculators.lammpslib import LAMMPSlib
from ase.calculators.lammpsrun import LAMMPS
from ase.data import atomic_masses, atomic_numbers

from .exceptions import KIMCalculatorError
from .kimmodel import KIMModelCalculator


def KIMCalculator(model_name, options, debug):
    """
    Used only for Portable Models
    """

    options_not_allowed = ["modelname", "debug"]

    _check_conflict_options(options, options_not_allowed,
                            simulator="kimmodel")

    return KIMModelCalculator(model_name, debug=debug, **options)


def LAMMPSRunCalculator(
    model_name, model_type, supported_species, options, debug, **kwargs
):
    """Used for Portable Models or LAMMPS Simulator Models if
    specifically requested"""

    def get_params(model_name, supported_units, supported_species, atom_style):
        """
        Extract parameters for LAMMPS calculator from model definition lines.
        Returns a dictionary with entries for "pair_style" and "pair_coeff".
        Expects there to be only one "pair_style" line. There can be multiple
        "pair_coeff" lines (result is returned as a list).
        """
        parameters = {}

        # In case the SM supplied its own atom_style in its model-init
        # -- only needed because lammpsrun writes data files and needs
        # to know the proper format
        if atom_style:
            parameters["atom_style"] = atom_style

        # Set units to prevent them from defaulting to metal
        parameters["units"] = supported_units

        parameters["model_init"] = [
            f"kim_init {model_name} {supported_units}{os.linesep}"
        ]

        parameters["kim_interactions"] = "kim_interactions {}{}".format(
            (" ").join(supported_species), os.linesep
        )

        # For every species in "supported_species", add an entry to the
        # "masses" key in dictionary "parameters".
        parameters["masses"] = []
        for i, species in enumerate(supported_species):
            if species not in atomic_numbers:
                raise KIMCalculatorError(
                    "Could not determine mass of unknown species "
                    "{} listed as supported by model".format(species)
                )
            massstr = str(
                convert(
                    atomic_masses[atomic_numbers[species]],
                    "mass",
                    "ASE",
                    supported_units,
                )
            )
            parameters["masses"].append(str(i + 1) + " " + massstr)

        return parameters

    options_not_allowed = ["parameters", "files", "specorder",
                           "keep_tmp_files"]

    _check_conflict_options(options, options_not_allowed,
                            simulator="lammpsrun")

    # If no atom_style kwarg is passed, lammpsrun will default to
    # atom_style atomic, which is what we want for KIM Portable Models
    atom_style = kwargs.get("atom_style", None)

    # Simulator Models will supply their own units from their
    # metadata. For Portable Models, we use "metal" units.
    supported_units = kwargs.get("supported_units", "metal")

    # Set up kim_init and kim_interactions lines
    parameters = get_params(
        model_name,
        supported_units,
        supported_species,
        atom_style)

    return LAMMPS(
        **parameters, specorder=supported_species, keep_tmp_files=debug,
        **options
    )


def LAMMPSLibCalculator(model_name, supported_species,
                        supported_units, options):
    """
    Only used for LAMMPS Simulator Models
    """
    options_not_allowed = [
        "lammps_header",
        "lmpcmds",
        "atom_types",
        "log_file",
        "keep_alive",
    ]

    _check_conflict_options(options, options_not_allowed,
                            simulator="lammpslib")
    # Set up LAMMPS header commands lookup table

    # This units command actually has no effect, but is necessary because
    # LAMMPSlib looks in the header lines for units in order to set them
    # internally
    model_init = ["units " + supported_units + os.linesep]

    model_init.append(
        f"kim_init {model_name} {supported_units}{os.linesep}"
    )
    model_init.append("atom_modify map array sort 0 0" + os.linesep)

    # Assign atom types to species
    atom_types = {s: i_s + 1 for i_s, s in enumerate(supported_species)}
    kim_interactions = [
        "kim_interactions {}".format(
            (" ").join(supported_species))]

    # Return LAMMPSlib calculator
    return LAMMPSlib(
        lammps_header=model_init,
        lammps_name=None,
        lmpcmds=kim_interactions,
        post_changebox_cmds=kim_interactions,
        atom_types=atom_types,
        log_file="lammps.log",
        keep_alive=True,
        **options
    )


def ASAPCalculator(model_name, model_type, options, **kwargs):
    """
    Can be used with either Portable Models or Simulator Models
    """
    import asap3

    options_not_allowed = {"pm": ["name", "verbose"], "sm": ["Params"]}

    _check_conflict_options(
        options,
        options_not_allowed[model_type],
        simulator="asap")

    if model_type == "pm":

        return asap3.OpenKIMcalculator(
            name=model_name, verbose=kwargs["verbose"], **options
        )

    elif model_type == "sm":
        model_defn = kwargs["model_defn"]
        supported_units = kwargs["supported_units"]

        # Verify units (ASAP models are expected to work with "ase" units)
        if supported_units != "ase":
            raise KIMCalculatorError(
                'KIM Simulator Model units are "{}", but expected to '
                'be "ase" for ASAP.'.format(supported_units)
            )

        # Check model_defn to make sure there's only one element in it
        # that is a non-empty string
        if len(model_defn) == 0:
            raise KIMCalculatorError(
                "model-defn is an empty list in metadata file of "
                "Simulator Model {}".format(model_name)
            )
        elif len(model_defn) > 1:
            raise KIMCalculatorError(
                "model-defn should contain only one entry for an ASAP "
                "model (found {} lines)".format(len(model_defn))
            )

        if "" in model_defn:
            raise KIMCalculatorError(
                "model-defn contains an empty string in metadata "
                "file of Simulator "
                "Model {}".format(model_name)
            )

        model_defn = model_defn[0].strip()

        # Instantiate calculator from ASAP.  Currently, this must be one of:
        # (1) EMT
        # (2) EMT(EMTRasmussenParameters)
        # (3) EMT(EMTMetalGlassParameters)
        model_defn_is_valid = False
        if model_defn.startswith("EMT"):
            # Pull out potential parameters
            mobj = re.search(r"\(([A-Za-z0-9_\(\)]+)\)", model_defn)
            if mobj is None:
                asap_calc = asap3.EMT()
            else:
                pp = mobj.group(1)

                # Currently we only supported two specific EMT models
                # that are built into ASAP
                if pp.startswith("EMTRasmussenParameters"):
                    asap_calc = asap3.EMT(
                        parameters=asap3.EMTRasmussenParameters())
                    model_defn_is_valid = True
                elif pp.startswith("EMTMetalGlassParameters"):
                    asap_calc = asap3.EMT(
                        parameters=asap3.EMTMetalGlassParameters())
                    model_defn_is_valid = True

        if not model_defn_is_valid:
            raise KIMCalculatorError(
                'Unknown model "{}" requested for simulator asap.'.format(
                    model_defn)
            )

        # Disable undocumented feature for the EMT self.calculators to
        # take the energy of an isolated atoms as zero. (Otherwise it
        # is taken to be that of perfect FCC.)
        asap_calc.set_subtractE0(False)

        return asap_calc


def _check_conflict_options(options, options_not_allowed, simulator):
    """Check whether options intended to be passed to a given calculator
    are allowed.  Some options are not allowed because they must be
    set internally in this package."""
    s1 = set(options)
    s2 = set(options_not_allowed)
    common = s1.intersection(s2)

    if common:
        options_in_not_allowed = ", ".join([f'"{s}"' for s in common])

        msg = (
            'Simulator "{}" does not support argument(s): '
            '{} provided in "options", '
            "because it is (they are) determined internally within the KIM "
            "calculator".format(simulator, options_in_not_allowed)
        )

        raise KIMCalculatorError(msg)
