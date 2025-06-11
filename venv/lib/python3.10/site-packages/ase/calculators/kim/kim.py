# fmt: off

"""
Knowledgebase of Interatomic Models (KIM) Calculator for ASE written by:

Ellad B. Tadmor
Mingjian Wen
Daniel S. Karls
University of Minnesota

This calculator functions as a wrapper that selects an appropriate
calculator for a given KIM model depending on whether it supports the
KIM application programming interface (API) or not. For more information
on KIM, visit https://openkim.org.
"""

from . import kimpy_wrappers
from .calculators import (
    ASAPCalculator,
    KIMCalculator,
    LAMMPSLibCalculator,
    LAMMPSRunCalculator,
)
from .exceptions import KIMCalculatorError


def KIM(model_name, simulator=None, options=None, debug=False):
    """Calculator wrapper for OpenKIM models

    Returns a suitable calculator that can be used with any model
    archived in the Open Knowledgebase of Interatomic Models (OpenKIM)
    at https://openkim.org.  There are two kinds of models in KIM:
    Portable Models (PMs), which can be used with any KIM API-compliant
    simulator, and Simulator Models (SMs), which are essentially just
    wrappers around native commands in a specific simulator (often
    combined with values for the model parameters).  PMs published on
    openkim.org contain the string '__MO_' in their name, while SMs
    published on openkim.org contain the string '__SM_' in their name.

    Parameters
    ----------
    model_name : str
        The name of the KIM model installed on your system.  KIM models
        published on openkim.org follow a specific naming scheme (see
        https://openkim.org/doc/schema/kim-ids).

    simulator : str, optional
        Used to identify the ASE calculator that will be used.
        Currently supported values include 'kimmodel', 'lammpslib',
        'lammpsrun' and 'asap', and correspond to different calculators
        as follows:

        - kimmodel (default for PMs)
          : :py:mod:`ase.calculators.kim.kimmodel.KIMModelCalculator`

        - lammpsrun (PMs or LAMMPS SMs)
          : :py:mod:`ase.calculators.lammpsrun.LAMMPS`

        - lammpslib (default for LAMMPS SMs)
          : :py:mod:`ase.calculators.lammpslib.LAMMPSlib`

        - asap (PMs)
          : :py:mod:`asap3.Internal.OpenKIMcalculator.OpenKIMcalculator`

        - asap (ASAP SMs)
          : :py:mod:`asap3.Internal.BuiltinPotentials.EMT`

        In general, this argument should be omitted, in which case a
        calculator compatible with the specified model will
        automatically be determined.

    options : dict, optional
        Additional options passed to the initializer of the selected
        calculator.  If ``simulator`` == 'kimmodel', possible options are:

        - ase_neigh (bool)
          : Whether to use the kimpy neighbor list library (False) or
          use ASE's internal neighbor list mechanism (True). Usually
          kimpy's neighbor list library will be faster.  (Default:
          False)

        - neigh_skin_ratio (float)
          : The skin distance used for neighbor list construction,
          expressed as a fraction of the model cutoff (Default: 0.2)

        - release_GIL (bool)
          : Whether to release python GIL.  Releasing the GIL allows a KIM
          model to run with multiple concurrent threads. (Default: False)

        See the ASE LAMMPS calculators doc page
        (https://wiki.fysik.dtu.dk/ase/ase/calculators/lammps.html) for
        available options for the lammpslib and lammpsrun calculators.

    debug : bool, optional
        If True, detailed information is printed to stdout.  If the
        lammpsrun calculator is being used, this also serves as the
        value of the ``keep_tmp_files`` option. (Default: False)

    Returns
    -------
    ase.calculators.calculator.Calculator
        An ASE-compatible calculator.  Currently, this will be an instance of
        KIMModelCalculator, LAMMPS (the lammpsrun calculator), or LAMMPSlib,
        which are all defined in the ASE codebase, or an instance of either
        OpenKIMcalculator or EMT defined in the asap3 codebase.

    Raises
    ------
    KIMCalculatorError
        Indicates an error occurred in initializing the calculator,
        e.g. due to incompatible combinations of argument values
    """

    if options is None:
        options = {}

    # If this is a KIM Portable Model (supports KIM API), return
    # support through a KIM-compliant simulator
    model_type = "pm" if _is_portable_model(model_name) else "sm"

    if model_type == "pm":
        if simulator is None:  # Default
            simulator = "kimmodel"

        if simulator == "kimmodel":
            return KIMCalculator(model_name, options, debug)

        elif simulator == "asap":
            return ASAPCalculator(
                model_name, model_type, options=options, verbose=debug
            )

        elif simulator == "lammpsrun":
            supported_species = get_model_supported_species(model_name)

            # Return LAMMPS calculator
            return LAMMPSRunCalculator(
                model_name, model_type, supported_species, options, debug
            )

        elif simulator == "lammpslib":
            raise KIMCalculatorError(
                '"lammpslib" calculator does not support KIM Portable Models. '
                'Try using the "lammpsrun" calculator.'
            )
        else:
            raise KIMCalculatorError(
                'Unsupported simulator "{}" requested to run KIM '
                'Portable Model.'.format(simulator)
            )

    #######################################################
    # If we get to here, the model is a KIM Simulator Model
    #######################################################
    with kimpy_wrappers.SimulatorModel(model_name) as sm:

        # Handle default behavior for 'simulator'
        if simulator is None:
            if sm.simulator_name == "ASAP":
                simulator = "asap"
            elif sm.simulator_name == "LAMMPS":
                simulator = "lammpslib"

        if sm.simulator_name == "ASAP":

            return ASAPCalculator(
                model_name,
                model_type,
                options=options,
                model_defn=sm.model_defn,
                verbose=debug,
                supported_units=sm.supported_units,
            )

        elif sm.simulator_name == "LAMMPS":

            if simulator == "lammpsrun":

                return LAMMPSRunCalculator(
                    model_name,
                    model_type,
                    sm.supported_species,
                    options,
                    debug,
                    atom_style=sm.atom_style,
                    supported_units=sm.supported_units,
                )

            elif simulator == "lammpslib":
                return LAMMPSLibCalculator(
                    model_name, sm.supported_species, sm.supported_units,
                    options
                )

            else:
                raise KIMCalculatorError(
                    f'Unknown LAMMPS calculator: "{simulator}".'
                )

        else:
            raise KIMCalculatorError(
                f'Unsupported simulator: "{sm.simulator_name}".'
            )


def _is_portable_model(model_name):
    """
    Returns True if the model specified is a KIM Portable Model (if it
    is not, then it must be a KIM Simulator Model -- there are no other
    types of models in KIM)
    """
    with kimpy_wrappers.ModelCollections() as col:
        model_type = col.get_item_type(model_name)

    return (model_type ==
            kimpy_wrappers.wrappers.collection_item_type_portableModel)


def get_model_supported_species(model_name):
    if _is_portable_model(model_name):
        with kimpy_wrappers.PortableModel(model_name, debug=False) as pm:
            supported_species, _ = pm.get_model_supported_species_and_codes()
    else:
        with kimpy_wrappers.SimulatorModel(model_name) as sm:
            supported_species = sm.supported_species

    return supported_species
