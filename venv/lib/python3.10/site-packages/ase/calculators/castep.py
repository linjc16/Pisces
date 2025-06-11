# fmt: off

"""This module defines an interface to CASTEP for
    use by the ASE (Webpage: http://wiki.fysik.dtu.dk/ase)

Authors:
    Max Hoffmann, max.hoffmann@ch.tum.de
    Joerg Meyer, joerg.meyer@ch.tum.de
    Simon P. Rittmeyer, simon.rittmeyer@tum.de

Contributors:
    Juan M. Lorenzi, juan.lorenzi@tum.de
    Georg S. Michelitsch, georg.michelitsch@tch.tum.de
    Reinhard J. Maurer, reinhard.maurer@yale.edu
    Simone Sturniolo, simone.sturniolo@stfc.ac.uk
"""

import difflib
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from collections import namedtuple
from copy import deepcopy
from itertools import product
from pathlib import Path

import numpy as np

from ase import Atoms
from ase.calculators.calculator import (
    BaseCalculator,
    compare_atoms,
    kpts2sizeandoffsets,
)
from ase.config import cfg
from ase.dft.kpoints import BandPath
from ase.io.castep import read_bands, read_param
from ase.io.castep.castep_input_file import CastepCell, CastepParam
from ase.io.castep.castep_reader import read_castep_castep
from ase.parallel import paropen

__all__ = [
    'Castep',
    'CastepCell',
    'CastepParam',
    'create_castep_keywords']

# A convenient table to avoid the previously used "eval"
_tf_table = {
    '': True,  # Just the keyword is equivalent to True
    'True': True,
    'False': False}


def _self_getter(getf):
    # A decorator that makes it so that if no 'atoms' argument is passed to a
    # getter function, self.atoms is used instead

    def decor_getf(self, atoms=None, *args, **kwargs):

        if atoms is None:
            atoms = self.atoms

        return getf(self, atoms, *args, **kwargs)

    return decor_getf


class Castep(BaseCalculator):
    r"""
CASTEP Interface Documentation


Introduction
============

CASTEP_ [1]_ W_ is a software package which uses density functional theory to
provide a good atomic-level description of all manner of materials and
molecules. CASTEP can give information about total energies, forces and
stresses on an atomic system, as well as calculating optimum geometries, band
structures, optical spectra, phonon spectra and much more. It can also perform
molecular dynamics simulations.

The CASTEP calculator interface class offers intuitive access to all CASTEP
settings and most results. All CASTEP specific settings are accessible via
attribute access (*i.e*. ``calc.param.keyword = ...`` or
``calc.cell.keyword = ...``)


Getting Started:
================

Set the environment variables appropriately for your system::

    export CASTEP_COMMAND=' ... '
    export CASTEP_PP_PATH=' ... '

Note: alternatively to CASTEP_PP_PATH one can set PSPOT_DIR
as CASTEP consults this by default, i.e.::

    export PSPOT_DIR=' ... '


Running the Calculator
======================

The default initialization command for the CASTEP calculator is

.. class:: Castep(directory='CASTEP', label='castep')

To do a minimal run one only needs to set atoms, this will use all
default settings of CASTEP, meaning LDA, singlepoint, etc..

With a generated *castep_keywords.json* in place all options are accessible
by inspection, *i.e.* tab-completion. This works best when using ``ipython``.
All options can be accessed via ``calc.param.<TAB>`` or ``calc.cell.<TAB>``
and documentation is printed with ``calc.param.<keyword> ?`` or
``calc.cell.<keyword> ?``. All options can also be set directly
using ``calc.keyword = ...`` or ``calc.KEYWORD = ...`` or even
``ialc.KeYwOrD`` or directly as named arguments in the call to the constructor
(*e.g.* ``Castep(task='GeometryOptimization')``).
If using this calculator on a machine without CASTEP, one might choose to copy
a *castep_keywords.json* file generated elsewhere in order to access this
feature: the file will be used if located in the working directory,
*$HOME/.ase/* or *ase/ase/calculators/* within the ASE library. The file should
be generated the first time it is needed, but you can generate a new keywords
file in the currect directory with ``python -m ase.calculators.castep``.

All options that go into the ``.param`` file are held in an ``CastepParam``
instance, while all options that go into the ``.cell`` file and don't belong
to the atoms object are held in an ``CastepCell`` instance. Each instance can
be created individually and can be added to calculators by attribute
assignment, *i.e.* ``calc.param = param`` or ``calc.cell = cell``.

All internal variables of the calculator start with an underscore (_).
All cell attributes that clearly belong into the atoms object are blocked.
Setting ``calc.atoms_attribute`` (*e.g.* ``= positions``) is sent directly to
the atoms object.


Arguments:
==========

=========================  ====================================================
Keyword                    Description
=========================  ====================================================
``directory``              The relative path where all input and output files
                           will be placed. If this does not exist, it will be
                           created.  Existing directories will be moved to
                           directory-TIMESTAMP unless self._rename_existing_dir
                           is set to false.

``label``                  The prefix of .param, .cell, .castep, etc. files.

``castep_command``         Command to run castep. Can also be set via the bash
                           environment variable ``CASTEP_COMMAND``. If none is
                           given or found, will default to ``castep``

``check_castep_version``   Boolean whether to check if the installed castep
                           version matches the version from which the available
                           options were deduced. Defaults to ``False``.

``castep_pp_path``         The path where the pseudopotentials are stored. Can
                           also be set via the bash environment variables
                           ``PSPOT_DIR`` (preferred) and ``CASTEP_PP_PATH``.
                           Will default to the current working directory if
                           none is given or found. Note that pseudopotentials
                           may be generated on-the-fly if they are not found.

``find_pspots``            Boolean whether to search for pseudopotentials in
                           ``<castep_pp_path>`` or not. If activated, files in
                           this directory will be checked for typical names. If
                           files are not found, they will be generated on the
                           fly, depending on the ``_build_missing_pspots``
                           value.  A RuntimeError will be raised in case
                           multiple files per element are found. Defaults to
                           ``False``.
``keyword_tolerance``      Integer to indicate the level of tolerance to apply
                           validation of any parameters set in the CastepCell
                           or CastepParam objects against the ones found in
                           castep_keywords. Levels are as following:

                           0 = no tolerance, keywords not found in
                           castep_keywords will raise an exception

                           1 = keywords not found will be accepted but produce
                           a warning (default)

                           2 = keywords not found will be accepted silently

                           3 = no attempt is made to look for
                           castep_keywords.json at all
``castep_keywords``        Can be used to pass a CastepKeywords object that is
                           then used with no attempt to actually load a
                           castep_keywords.json file. Most useful for debugging
                           and testing purposes.

=========================  ====================================================


Additional Settings
===================

=========================  ====================================================
Internal Setting           Description
=========================  ====================================================
``_castep_command``        (``=castep``): the actual shell command used to
                           call CASTEP.

``_check_checkfile``       (``=True``): this makes write_param() only
                           write a continue or reuse statement if the
                           addressed .check or .castep_bin file exists in the
                           directory.

``_copy_pspots``           (``=False``): if set to True the calculator will
                           actually copy the needed pseudo-potential (\*.usp)
                           file, usually it will only create symlinks.

``_link_pspots``           (``=True``): if set to True the calculator will
                           actually will create symlinks to the needed pseudo
                           potentials. Set this option (and ``_copy_pspots``)
                           to False if you rather want to access your pseudo
                           potentials using the PSPOT_DIR environment variable
                           that is read by CASTEP.
                           *Note:* This option has no effect if ``copy_pspots``
                           is True..

``_build_missing_pspots``  (``=True``): if set to True, castep will generate
                           missing pseudopotentials on the fly. If not, a
                           RuntimeError will be raised if not all files were
                           found.

``_export_settings``       (``=True``): if this is set to
                           True, all calculator internal settings shown here
                           will be included in the .param in a comment line (#)
                           and can be read again by merge_param. merge_param
                           can be forced to ignore this directive using the
                           optional argument ``ignore_internal_keys=True``.

``_force_write``           (``=True``): this controls wether the \*cell and
                           \*param will be overwritten.

``_prepare_input_only``    (``=False``): If set to True, the calculator will
                           create \*cell und \*param file but not
                           start the calculation itself.
                           If this is used to prepare jobs locally
                           and run on a remote cluster it is recommended
                           to set ``_copy_pspots = True``.

``_castep_pp_path``        (``='.'``) : the place where the calculator
                           will look for pseudo-potential files.

``_find_pspots``           (``=False``): if set to True, the calculator will
                           try to find the respective pseudopotentials from
                           <_castep_pp_path>. As long as there are no multiple
                           files per element in this directory, the auto-detect
                           feature should be very robust. Raises a RuntimeError
                           if required files are not unique (multiple files per
                           element). Non existing pseudopotentials will be
                           generated, though this could be dangerous.

``_rename_existing_dir``   (``=True``) : when using a new instance
                           of the calculator, this will move directories out of
                           the way that would be overwritten otherwise,
                           appending a date string.

``_set_atoms``             (``=False``) : setting this to True will overwrite
                           any atoms object previously attached to the
                           calculator when reading a \.castep file.  By de-
                           fault, the read() function will only create a new
                           atoms object if none has been attached and other-
                           wise try to assign forces etc. based on the atom's
                           positions.  ``_set_atoms=True`` could be necessary
                           if one uses CASTEP's internal geometry optimization
                           (``calc.param.task='GeometryOptimization'``)
                           because then the positions get out of sync.
                           *Warning*: this option is generally not recommended
                           unless one knows one really needs it. There should
                           never be any need, if CASTEP is used as a
                           single-point calculator.

``_track_output``          (``=False``) : if set to true, the interface
                           will append a number to the label on all input
                           and output files, where n is the number of calls
                           to this instance. *Warning*: this setting may con-
                           sume a lot more disk space because of the additio-
                           nal \*check files.

``_try_reuse``             (``=_track_output``) : when setting this, the in-
                           terface will try to fetch the reuse file from the
                           previous run even if _track_output is True. By de-
                           fault it is equal to _track_output, but may be
                           overridden.

                           Since this behavior may not always be desirable for
                           single-point calculations. Regular reuse for *e.g.*
                           a geometry-optimization can be achieved by setting
                           ``calc.param.reuse = True``.

``_pedantic``              (``=False``) if set to true, the calculator will
                           inform about settings probably wasting a lot of CPU
                           time or causing numerical inconsistencies.

=========================  ====================================================

Special features:
=================


``.dryrun_ok()``
  Runs ``castep_command seed -dryrun`` in a temporary directory return True if
  all variables initialized ok. This is a fast way to catch errors in the
  input. Afterwards _kpoints_used is set.

``.merge_param()``
  Takes a filename or filehandler of a .param file or CastepParam instance and
  merges it into the current calculator instance, overwriting current settings

``.keyword.clear()``
  Can be used on any option like ``calc.param.keyword.clear()`` or
  ``calc.cell.keyword.clear()`` to return to the CASTEP default.

``.initialize()``
  Creates all needed input in the ``_directory``. This can then copied to and
  run in a place without ASE or even python.

``.set_pspot('<library>')``
  This automatically sets the pseudo-potential for all present species to
  ``<Species>_<library>.usp``. Make sure that ``_castep_pp_path`` is set
  correctly. Note that there is no check, if the file actually exists. If it
  doesn't castep will crash! You may want to use ``find_pspots()`` instead.

``.find_pspots(pspot=<library>, suffix=<suffix>)``
  This automatically searches for pseudopotentials of type
  ``<Species>_<library>.<suffix>`` or ``<Species>-<library>.<suffix>`` in
  ``castep_pp_path` (make sure this is set correctly). Note that ``<Species>``
  will be searched for case insensitive.  Regular expressions are accepted, and
  arguments ``'*'`` will be regarded as bash-like wildcards. Defaults are any
  ``<library>`` and any ``<suffix>`` from ``['usp', 'UPF', 'recpot']``. If you
  have well-organized folders with pseudopotentials of one kind, this should
  work with the defaults.

``print(calc)``
  Prints a short summary of the calculator settings and atoms.

``ase.io.castep.read_seed('path-to/seed')``
  Given you have a combination of seed.{param,cell,castep} this will return an
  atoms object with the last ionic positions in the .castep file and all other
  settings parsed from the .cell and .param file. If no .castep file is found
  the positions are taken from the .cell file. The output directory will be
  set to the same directory, only the label is preceded by 'copy_of\_'  to
  avoid overwriting.

``.set_kpts(kpoints)``
  This is equivalent to initialising the calculator with
  ``calc = Castep(kpts=kpoints)``. ``kpoints`` can be specified in many
  convenient forms: simple Monkhorst-Pack grids can be specified e.g.
  ``(2, 2, 3)`` or ``'2 2 3'``; lists of specific weighted k-points can be
  given in reciprocal lattice coordinates e.g.
  ``[[0, 0, 0, 0.25], [0.25, 0.25, 0.25, 0.75]]``; a dictionary syntax is
  available for more complex requirements e.g.
  ``{'size': (2, 2, 2), 'gamma': True}`` will give a Gamma-centered 2x2x2 M-P
  grid, ``{'density': 10, 'gamma': False, 'even': False}`` will give a mesh
  with density of at least 10 Ang (based on the unit cell of currently-attached
  atoms) with an odd number of points in each direction and avoiding the Gamma
  point.

``.set_bandpath(bandpath)``
  This is equivalent to initialialising the calculator with
  ``calc=Castep(bandpath=bandpath)`` and may be set simultaneously with *kpts*.
  It allows an electronic band structure path to be set up using ASE BandPath
  objects. This enables a band structure calculation to be set up conveniently
  using e.g. calc.set_bandpath(atoms.cell.bandpath().interpolate(npoints=200))

``.band_structure(bandfile=None)``
  Read a band structure from _seedname.bands_ file. This returns an ase
  BandStructure object which may be plotted with e.g.
  ``calc.band_structure().plot()``

Notes/Issues:
==============

* Currently *only* the FixAtoms *constraint* is fully supported for reading and
  writing. There is some experimental support for the FixCartesian constraint.

* There is no support for the CASTEP *unit system*. Units of eV and Angstrom
  are used throughout. In particular when converting total energies from
  different calculators, one should check that the same CODATA_ version is
  used for constants and conversion factors, respectively.

.. _CASTEP: http://www.castep.org/

.. _W: https://en.wikipedia.org/wiki/CASTEP

.. _CODATA: https://physics.nist.gov/cuu/Constants/index.html

.. [1] S. J. Clark, M. D. Segall, C. J. Pickard, P. J. Hasnip, M. J. Probert,
       K. Refson, M. C. Payne Zeitschrift für Kristallographie 220(5-6)
       pp.567- 570 (2005) PDF_.

.. _PDF: http://www.tcm.phy.cam.ac.uk/castep/papers/ZKristallogr_2005.pdf


End CASTEP Interface Documentation
    """

    # Class attributes !
    # keys set through atoms object
    atoms_keys = [
        'charges',
        'ionic_constraints',
        'lattice_abs',
        'lattice_cart',
        'positions_abs',
        'positions_abs_final',
        'positions_abs_intermediate',
        'positions_frac',
        'positions_frac_final',
        'positions_frac_intermediate']

    atoms_obj_keys = [
        'dipole',
        'energy_free',
        'energy_zero',
        'fermi',
        'forces',
        'nbands',
        'positions',
        'stress',
        'pressure']

    internal_keys = [
        '_castep_command',
        '_check_checkfile',
        '_copy_pspots',
        '_link_pspots',
        '_find_pspots',
        '_build_missing_pspots',
        '_directory',
        '_export_settings',
        '_force_write',
        '_label',
        '_prepare_input_only',
        '_castep_pp_path',
        '_rename_existing_dir',
        '_set_atoms',
        '_track_output',
        '_try_reuse',
        '_pedantic']

    implemented_properties = [
        'energy',
        'free_energy',
        'forces',
        'stress',
        'charges',
        'magmoms',
    ]

    # specific to this calculator
    implemented_properties += [
        'energy_without_dispersion_correction',
        'free_energy_without_dispersion_correction',
        'energy_zero_without_dispersion_correction',
        'energy_with_dispersion_correction',
        'free_energy_with_dispersion_correction',
        'energy_zero_with_dispersion_correction',
        'energy_with_finite_basis_set_correction',
        'pressure',
        'hirshfeld_volume_ratios',
        'hirshfeld_charges',
        'hirshfeld_magmoms',
    ]

    def __init__(self, directory='CASTEP', label='castep',
                 castep_command=None, check_castep_version=False,
                 castep_pp_path=None, find_pspots=False, keyword_tolerance=1,
                 castep_keywords=None, **kwargs):

        self.results = {}

        from ase.io.castep import write_castep_cell
        self._write_cell = write_castep_cell

        if castep_keywords is None:
            castep_keywords = CastepKeywords(make_param_dict(),
                                             make_cell_dict(),
                                             [],
                                             [],
                                             0)
            if keyword_tolerance < 3:
                try:
                    castep_keywords = import_castep_keywords(castep_command)
                except CastepVersionError as e:
                    if keyword_tolerance == 0:
                        raise e
                    else:
                        warnings.warn(str(e))

        self._kw_tol = keyword_tolerance
        keyword_tolerance = max(keyword_tolerance, 2)  # 3 not accepted below
        self.param = CastepParam(castep_keywords,
                                 keyword_tolerance=keyword_tolerance)
        self.cell = CastepCell(castep_keywords,
                               keyword_tolerance=keyword_tolerance)

        ###################################
        # Calculator state variables      #
        ###################################
        self._calls = 0
        self._castep_version = castep_keywords.castep_version

        # collects content from *.err file
        self._error = None
        # warnings raised by the ASE interface
        self._interface_warnings = []

        # store to check if recalculation is necessary
        self._old_atoms = None
        self._old_cell = None
        self._old_param = None

        ###################################
        # Internal keys                   #
        # Allow to tweak the behavior     #
        ###################################
        self._opt = {}
        self._castep_command = get_castep_command(castep_command)
        self._castep_pp_path = get_castep_pp_path(castep_pp_path)
        self._check_checkfile = True
        self._copy_pspots = False
        self._link_pspots = True
        self._find_pspots = find_pspots
        self._build_missing_pspots = True
        self._directory = os.path.abspath(directory)
        self._export_settings = True
        self._force_write = True
        self._label = label
        self._prepare_input_only = False
        self._rename_existing_dir = True
        self._set_atoms = False
        self._track_output = False
        self._try_reuse = False

        # turn off the pedantic user warnings
        self._pedantic = False

        # will be set on during runtime
        self._seed = None

        ###################################
        # (Physical) result variables     #
        ###################################
        self.atoms = None
        # initialize result variables
        self._eigenvalues = None
        self._efermi = None
        self._ibz_kpts = None
        self._ibz_weights = None
        self._band_structure = None

        self._number_of_cell_constraints = None
        self._output_verbosity = None
        self._unit_cell = None
        self._kpoints = None

        # pointers to other files used at runtime
        self._check_file = None
        self._castep_bin_file = None

        # plane wave cutoff energy (may be derived during PP generation)
        self._cut_off_energy = None

        # runtime information
        self._total_time = None
        self._peak_memory = None

        # check version of CASTEP options module against current one
        if check_castep_version:
            local_castep_version = get_castep_version(self._castep_command)
            if not hasattr(self, '_castep_version'):
                warnings.warn('No castep version found')
                return
            if local_castep_version != self._castep_version:
                warnings.warn(
                    'The options module was generated from version %s '
                    'while your are currently using CASTEP version %s' %
                    (self._castep_version,
                     get_castep_version(self._castep_command)))
                self._castep_version = local_castep_version

        # processes optional arguments in kw style
        for keyword, value in kwargs.items():
            # first fetch special keywords issued by ASE CLI
            if keyword == 'kpts':
                self.set_kpts(value)
            elif keyword == 'bandpath':
                self.set_bandpath(value)
            elif keyword == 'xc':
                self.xc_functional = value
            elif keyword == 'ecut':
                self.cut_off_energy = value
            else:  # the general case
                self.__setattr__(keyword, value)

        # TODO: to be self.use_cache = True after revising `__setattr__`
        self.__dict__['use_cache'] = True

    def set_atoms(self, atoms):
        self.atoms = atoms

    def get_atoms(self):
        if self.atoms is None:
            raise ValueError('Calculator has no atoms')
        atoms = self.atoms.copy()
        atoms.calc = self
        return atoms

    def _get_name(self) -> str:
        return self.__class__.__name__

    def band_structure(self, bandfile=None):
        from ase.spectrum.band_structure import BandStructure

        if bandfile is None:
            bandfile = os.path.join(self._directory, self._seed) + '.bands'

        if not os.path.exists(bandfile):
            raise ValueError(f'Cannot find band file "{bandfile}".')

        kpts, _weights, eigenvalues, efermi = read_bands(bandfile)

        # Get definitions of high-symmetry points
        special_points = self.atoms.cell.bandpath(npoints=0).special_points
        bandpath = BandPath(self.atoms.cell,
                            kpts=kpts,
                            special_points=special_points)
        return BandStructure(bandpath, eigenvalues, reference=efermi)

    def set_bandpath(self, bandpath):
        """Set a band structure path from ase.dft.kpoints.BandPath object

        This will set the bs_kpoint_list block with a set of specific points
        determined in ASE. bs_kpoint_spacing will not be used; to modify the
        number of points, consider using e.g. bandpath.resample(density=20) to
        obtain a new dense path.

        Args:
            bandpath (:obj:`ase.dft.kpoints.BandPath` or None):
                Set to None to remove list of band structure points. Otherwise,
                sampling will follow BandPath parameters.

        """

        def clear_bs_keywords():
            bs_keywords = product({'bs_kpoint', 'bs_kpoints'},
                                  {'path', 'path_spacing',
                                   'list',
                                   'mp_grid', 'mp_spacing', 'mp_offset'})
            for bs_tag in bs_keywords:
                setattr(self.cell, '_'.join(bs_tag), None)

        if bandpath is None:
            clear_bs_keywords()
        elif isinstance(bandpath, BandPath):
            clear_bs_keywords()
            self.cell.bs_kpoint_list = [' '.join(map(str, row))
                                        for row in bandpath.kpts]
        else:
            raise TypeError('Band structure path must be an '
                            'ase.dft.kpoint.BandPath object')

    def set_kpts(self, kpts):
        """Set k-point mesh/path using a str, tuple or ASE features

        Args:
            kpts (None, tuple, str, dict):

        This method will set the CASTEP parameters kpoints_mp_grid,
        kpoints_mp_offset and kpoints_mp_spacing as appropriate. Unused
        parameters will be set to None (i.e. not included in input files.)

        If kpts=None, all these parameters are set as unused.

        The simplest useful case is to give a 3-tuple of integers specifying
        a Monkhorst-Pack grid. This may also be formatted as a string separated
        by spaces; this is the format used internally before writing to the
        input files.

        A more powerful set of features is available when using a python
        dictionary with the following allowed keys:

        - 'size' (3-tuple of int) mesh of mesh dimensions
        - 'density' (float) for BZ sampling density in points per recip. Ang
          ( kpoint_mp_spacing = 1 / (2pi * density) ). An explicit MP mesh will
          be set to allow for rounding/centering.
        - 'spacing' (float) for BZ sampling density for maximum space between
          sample points in reciprocal space. This is numerically equivalent to
          the inbuilt ``calc.cell.kpoint_mp_spacing``, but will be converted to
          'density' to allow for rounding/centering.
        - 'even' (bool) to round each direction up to the nearest even number;
          set False for odd numbers, leave as None for no odd/even rounding.
        - 'gamma' (bool) to offset the Monkhorst-Pack grid to include
          (0, 0, 0); set False to offset each direction avoiding 0.
        """

        def clear_mp_keywords():
            mp_keywords = product({'kpoint', 'kpoints'},
                                  {'mp_grid', 'mp_offset',
                                   'mp_spacing', 'list'})
            for kp_tag in mp_keywords:
                setattr(self.cell, '_'.join(kp_tag), None)

        # Case 1: Clear parameters with set_kpts(None)
        if kpts is None:
            clear_mp_keywords()

        # Case 2: list of explicit k-points with weights
        # e.g. [[ 0,    0,   0,    0.125],
        #       [ 0,   -0.5, 0,    0.375],
        #       [-0.5,  0,  -0.5,  0.375],
        #       [-0.5, -0.5, -0.5, 0.125]]

        elif (isinstance(kpts, (tuple, list))
              and isinstance(kpts[0], (tuple, list))):

            if not all(map((lambda row: len(row) == 4), kpts)):
                raise ValueError(
                    'In explicit kpt list each row should have 4 elements')

            clear_mp_keywords()
            self.cell.kpoint_list = [' '.join(map(str, row)) for row in kpts]

        # Case 3: list of explicit kpts formatted as list of str
        # i.e. the internal format of calc.kpoint_list split on \n
        # e.g. ['0 0 0 0.125', '0 -0.5 0 0.375', '-0.5 0 -0.5 0.375']
        elif isinstance(kpts, (tuple, list)) and isinstance(kpts[0], str):

            if not all(map((lambda row: len(row.split()) == 4), kpts)):
                raise ValueError(
                    'In explicit kpt list each row should have 4 elements')

            clear_mp_keywords()
            self.cell.kpoint_list = kpts

        # Case 4: list or tuple of MP samples e.g. [3, 3, 2]
        elif isinstance(kpts, (tuple, list)) and isinstance(kpts[0], int):
            if len(kpts) != 3:
                raise ValueError('Monkhorst-pack grid should have 3 values')
            clear_mp_keywords()
            self.cell.kpoint_mp_grid = '%d %d %d' % tuple(kpts)

        # Case 5: str representation of Case 3 e.g. '3 3 2'
        elif isinstance(kpts, str):
            self.set_kpts([int(x) for x in kpts.split()])

        # Case 6: dict of options e.g. {'size': (3, 3, 2), 'gamma': True}
        # 'spacing' is allowed but transformed to 'density' to get mesh/offset
        elif isinstance(kpts, dict):
            kpts = kpts.copy()

            if (kpts.get('spacing') is not None
                    and kpts.get('density') is not None):
                raise ValueError(
                    'Cannot set kpts spacing and density simultaneously.')
            else:
                if kpts.get('spacing') is not None:
                    kpts = kpts.copy()
                    spacing = kpts.pop('spacing')
                    kpts['density'] = 1 / (2 * np.pi * spacing)

                clear_mp_keywords()
                size, offsets = kpts2sizeandoffsets(atoms=self.atoms, **kpts)
                self.cell.kpoint_mp_grid = '%d %d %d' % tuple(size)
                self.cell.kpoint_mp_offset = '%f %f %f' % tuple(offsets)

        # Case 7: some other iterator. Try treating as a list:
        elif hasattr(kpts, '__iter__'):
            self.set_kpts(list(kpts))

        # Otherwise, give up
        else:
            raise TypeError('Cannot interpret kpts of this type')

    def todict(self, skip_default=True):
        """Create dict with settings of .param and .cell"""
        dct = {}
        dct['param'] = self.param.get_attr_dict()
        dct['cell'] = self.cell.get_attr_dict()

        return dct

    def check_state(self, atoms, tol=1e-15):
        """Check for system changes since last calculation."""
        return compare_atoms(self._old_atoms, atoms)

    def read(self, castep_file):
        """Read a castep file into the current instance."""

        atoms = read_castep_castep(castep_file)

        self.results = atoms.calc.results

        self._cut_off_energy = atoms.calc._cut_off_energy
        for k, v in atoms.calc._parameters_header.items():
            setattr(self.param, k, v)

        if self.atoms and not self._set_atoms:
            # compensate for internal reordering of atoms by CASTEP
            # using the fact that the order is kept within each species

            indices = _get_indices_to_sort_back(
                self.atoms.symbols,
                atoms.symbols,
            )
            positions_frac_atoms = atoms.get_scaled_positions()[indices]
            self.atoms.set_scaled_positions(positions_frac_atoms)
            keys = [
                'forces',
                'charges',
                'magmoms',
                'hirshfeld_volume_ratios',
                'hirshfeld_charges',
                'hirshfeld_magmoms',
            ]
            for k in keys:
                if k not in self.results:
                    continue
                self.results[k] = self.results[k][indices]

        else:
            atoms.set_initial_charges(self.results.get('charges'))
            atoms.set_initial_magnetic_moments(self.results.get('magmoms'))
            atoms.calc = self

        self._kpoints = atoms.calc._kpoints

        self.cell.species_pot = atoms.calc._species_pot

        self._total_time = atoms.calc._total_time
        self._peak_memory = atoms.calc._peak_memory

        # Read in eigenvalues from bands file
        bands_file = castep_file[:-7] + '.bands'
        if (self.param.task.value is not None
                and self.param.task.value.lower() == 'bandstructure'):
            self._band_structure = self.band_structure(bandfile=bands_file)
        else:
            try:
                (self._ibz_kpts,
                 self._ibz_weights,
                 self._eigenvalues,
                 self._efermi) = read_bands(bands_file)
            except FileNotFoundError:
                warnings.warn('Could not load .bands file, eigenvalues and '
                              'Fermi energy are unknown')

    # TODO: deprecate once inheriting BaseCalculator
    def get_hirsh_volrat(self):
        """
        Return the Hirshfeld volume ratios.
        """
        return self.results.get('hirshfeld_volume_ratios')

    # TODO: deprecate once inheriting BaseCalculator
    def get_spins(self):
        """
        Return the spins from a plane-wave Mulliken analysis.
        """
        return self.results['magmoms']

    # TODO: deprecate once inheriting BaseCalculator
    def get_mulliken_charges(self):
        """
        Return the charges from a plane-wave Mulliken analysis.
        """
        return self.results['charges']

    # TODO: deprecate once inheriting BaseCalculator
    def get_hirshfeld_charges(self):
        """
        Return the charges from a Hirshfeld analysis.
        """
        return self.results.get('hirshfeld_charges')

    def get_total_time(self):
        """
        Return the total runtime
        """
        return self._total_time

    def get_peak_memory(self):
        """
        Return the peak memory usage
        """
        return self._peak_memory

    def set_label(self, label):
        """The label is part of each seed, which in turn is a prefix
        in each CASTEP related file.
        """
        # we may think about changing this in future to set `self._directory`
        # and `self._label`, as one would expect
        self._label = label

    def set_pspot(self, pspot, elems=None,
                  notelems=None,
                  clear=True,
                  suffix='usp'):
        """Quickly set all pseudo-potentials: Usually CASTEP psp are named
        like <Elem>_<pspot>.<suffix> so this function function only expects
        the <LibraryName>. It then clears any previous pseudopotential
        settings apply the one with <LibraryName> for each element in the
        atoms object. The optional elems and notelems arguments can be used
        to exclusively assign to some species, or to exclude with notelemens.

        Parameters ::

            - elems (None) : set only these elements
            - notelems (None): do not set the elements
            - clear (True): clear previous settings
            - suffix (usp): PP file suffix
        """
        if self._find_pspots:
            if self._pedantic:
                warnings.warn('Warning: <_find_pspots> = True. '
                              'Do you really want to use `set_pspots()`? '
                              'This does not check whether the PP files exist. '
                              'You may rather want to use `find_pspots()` with '
                              'the same <pspot>.')

        if clear and not elems and not notelems:
            self.cell.species_pot.clear()
        for elem in set(self.atoms.get_chemical_symbols()):
            if elems is not None and elem not in elems:
                continue
            if notelems is not None and elem in notelems:
                continue
            self.cell.species_pot = (elem, f'{elem}_{pspot}.{suffix}')

    def find_pspots(self, pspot='.+', elems=None,
                    notelems=None, clear=True, suffix='(usp|UPF|recpot)'):
        r"""Quickly find and set all pseudo-potentials by searching in
        castep_pp_path:

        This one is more flexible than set_pspots, and also checks if the files
        are actually available from the castep_pp_path.

        Essentially, the function parses the filenames in <castep_pp_path> and
        does a regex matching. The respective pattern is:

            r"^(<elem>|<elem.upper()>|elem.lower()>(_|-)<pspot>\.<suffix>$"

        In most cases, it will be sufficient to not specify anything, if you
        use standard CASTEP USPPs with only one file per element in the
        <castep_pp_path>.

        The function raises a `RuntimeError` if there is some ambiguity
        (multiple files per element).

        Parameters ::

            - pspots ('.+') : as defined above, will be a wildcard if not
                              specified.
            - elems (None) : set only these elements
            - notelems (None): do not set the elements
            - clear (True): clear previous settings
            - suffix (usp|UPF|recpot): PP file suffix
        """
        if clear and not elems and not notelems:
            self.cell.species_pot.clear()

        if not os.path.isdir(self._castep_pp_path):
            if self._pedantic:
                warnings.warn(
                    'Cannot search directory: {} Folder does not exist'
                    .format(self._castep_pp_path))
            return

        # translate the bash wildcard syntax to regex
        if pspot == '*':
            pspot = '.*'
        if suffix == '*':
            suffix = '.*'
        if pspot == '*':
            pspot = '.*'

        # GBRV USPPs have a strnage naming schme
        pattern = r'^({elem}|{elem_upper}|{elem_lower})(_|-){pspot}\.{suffix}$'

        for elem in set(self.atoms.get_chemical_symbols()):
            if elems is not None and elem not in elems:
                continue
            if notelems is not None and elem in notelems:
                continue
            p = pattern.format(elem=elem,
                               elem_upper=elem.upper(),
                               elem_lower=elem.lower(),
                               pspot=pspot,
                               suffix=suffix)
            pps = []
            for f in os.listdir(self._castep_pp_path):
                if re.match(p, f):
                    pps.append(f)
            if not pps:
                if self._pedantic:
                    warnings.warn('Pseudopotential for species {} not found!'
                                  .format(elem))
            elif len(pps) != 1:
                raise RuntimeError(
                    'Pseudopotential for species ''{} not unique!\n'
                    .format(elem)
                    + 'Found the following files in {}\n'
                    .format(self._castep_pp_path)
                    + '\n'.join([f'    {pp}' for pp in pps]) +
                    '\nConsider a stricter search pattern in `find_pspots()`.')
            else:
                self.cell.species_pot = (elem, pps[0])

    @_self_getter
    def get_total_energy(self, atoms):
        """Run CASTEP calculation if needed and return total energy."""
        self.update(atoms)
        return self.results.get('energy_without_dispersion_correction')

    @_self_getter
    def get_total_energy_corrected(self, atoms):
        """Run CASTEP calculation if needed and return total energy."""
        self.update(atoms)
        return self.results.get('energy_with_finite_basis_set_correction')

    @_self_getter
    def get_free_energy(self, atoms):
        """Run CASTEP calculation if needed and return free energy.
           Only defined with smearing."""
        self.update(atoms)
        return self.results.get('free_energy_without_dispersion_correction')

    @_self_getter
    def get_0K_energy(self, atoms):
        """Run CASTEP calculation if needed and return 0K energy.
           Only defined with smearing."""
        self.update(atoms)
        return self.results.get('energy_zero_without_dispersion_correction')

    @_self_getter
    def get_pressure(self, atoms):
        """Return the pressure."""
        self.update(atoms)
        return self.results.get('pressure')

    @_self_getter
    def get_unit_cell(self, atoms):
        """Return the unit cell."""
        self.update(atoms)
        return self._unit_cell

    @_self_getter
    def get_kpoints(self, atoms):
        """Return the kpoints."""
        self.update(atoms)
        return self._kpoints

    @_self_getter
    def get_number_cell_constraints(self, atoms):
        """Return the number of cell constraints."""
        self.update(atoms)
        return self._number_of_cell_constraints

    def update(self, atoms):
        """Checks if atoms object or calculator changed and
        runs calculation if so.
        """
        if self.calculation_required(atoms, None):
            self.calculate(atoms, [], [])

    def calculation_required(self, atoms, properties):
        """Checks wether anything changed in the atoms object or CASTEP
        settings since the last calculation using this instance.
        """
        # SPR: what happens with the atoms parameter here? Why don't we use it?
        # from all that I can tell we need to compare against atoms instead of
        # self.atoms
        # if not self.atoms == self._old_atoms:
        if atoms != self._old_atoms:
            return True
        if self._old_param is None or self._old_cell is None:
            return True
        if self.param._options != self._old_param._options:
            return True
        if self.cell._options != self._old_cell._options:
            return True
        return False

    def calculate(self, atoms, properties, system_changes):
        """Write all necessary input file and call CASTEP."""
        self.prepare_input_files(atoms, force_write=self._force_write)
        if not self._prepare_input_only:
            self.run()
            if self._seed is None:
                basename = os.path.basename(self._castep_file)
                self._seed = os.path.splitext(basename)[0]
            err_file = f'{self._seed}.0001.err'
            if os.path.exists(err_file):
                err_file = paropen(err_file)
                self._error = err_file.read()
                err_file.close()
            self.read(self._castep_file)

            # we need to push the old state here!
            # although run() pushes it, read() may change the atoms object
            # again.
            # yet, the old state is supposed to be the one AFTER read()
            self.push_oldstate()

    def push_oldstate(self):
        """This function pushes the current state of the (CASTEP) Atoms object
        onto the previous state. Or in other words after calling this function,
        calculation_required will return False and enquiry functions just
        report the current value, e.g. get_forces(), get_potential_energy().
        """
        # make a snapshot of all current input
        # to be able to test if recalculation
        # is necessary
        self._old_atoms = self.atoms.copy()
        self._old_param = deepcopy(self.param)
        self._old_cell = deepcopy(self.cell)

    def initialize(self, *args, **kwargs):
        """Just an alias for prepar_input_files to comply with standard
        function names in ASE.
        """
        self.prepare_input_files(*args, **kwargs)

    def prepare_input_files(self, atoms=None, force_write=None):
        """Only writes the input .cell and .param files and return
        This can be useful if one quickly needs to prepare input files
        for a cluster where no python or ASE is available. One can than
        upload the file manually and read out the results using
        Castep().read().
        """

        if self.param.reuse.value is None:
            if self._pedantic:
                warnings.warn(
                    'You have not set e.g. calc.param.reuse = True. '
                    'Reusing a previous calculation may save CPU time! '
                    'The interface will make sure by default, .check exists. '
                    'file before adding this statement to the .param file.')
        if self.param.num_dump_cycles.value is None:
            if self._pedantic:
                warnings.warn(
                    'You have not set e.g. calc.param.num_dump_cycles = 0. '
                    'This can save you a lot of disk space. One only needs '
                    '*wvfn* if electronic convergence is not achieved.')
        from ase.io.castep import write_param

        if atoms is None:
            atoms = self.atoms
        else:
            self.atoms = atoms

        if force_write is None:
            force_write = self._force_write

        # if we have new instance of the calculator,
        # move existing results out of the way, first
        if (os.path.isdir(self._directory)
                and self._calls == 0
                and self._rename_existing_dir):
            if os.listdir(self._directory) == []:
                os.rmdir(self._directory)
            else:
                # rename appending creation date of the directory
                ctime = time.localtime(os.lstat(self._directory).st_ctime)
                os.rename(self._directory, '%s.bak-%s' %
                          (self._directory,
                           time.strftime('%Y%m%d-%H%M%S', ctime)))

        # create work directory
        if not os.path.isdir(self._directory):
            os.makedirs(self._directory, 0o775)

        # we do this every time, not only upon first call
        # if self._calls == 0:
        self._fetch_pspots()

        # if _try_reuse is requested and this
        # is not the first run, we try to find
        # the .check file from the previous run
        # this is only necessary if _track_output
        # is set to true
        if self._try_reuse and self._calls > 0:
            if os.path.exists(self._abs_path(self._check_file)):
                self.param.reuse = self._check_file
            elif os.path.exists(self._abs_path(self._castep_bin_file)):
                self.param.reuse = self._castep_bin_file
        self._seed = self._build_castep_seed()
        self._check_file = f'{self._seed}.check'
        self._castep_bin_file = f'{self._seed}.castep_bin'
        self._castep_file = self._abs_path(f'{self._seed}.castep')

        # write out the input file
        magnetic_moments = ('initial' if
                            self.param.spin_polarized.value == 'TRUE'
                            else None)
        self._write_cell(self._abs_path(f'{self._seed}.cell'),
                         self.atoms, castep_cell=self.cell,
                         magnetic_moments=magnetic_moments)

        if self._export_settings:
            interface_options = self._opt
        else:
            interface_options = None
        write_param(self._abs_path(f'{self._seed}.param'), self.param,
                    check_checkfile=self._check_checkfile,
                    force_write=force_write,
                    interface_options=interface_options,)

    def _build_castep_seed(self):
        """Abstracts to construction of the final castep <seed>
        with and without _tracking_output.
        """
        if self._track_output:
            return '%s-%06d' % (self._label, self._calls)
        else:
            return f'{(self._label)}'

    def _abs_path(self, path):
        # Create an absolute path for a file to put in the working directory
        return os.path.join(self._directory, path)

    def run(self):
        """Simply call castep. If the first .err file
        contains text, this will be printed to the screen.
        """
        # change to target directory
        self._calls += 1

        # run castep itself
        stdout, stderr = shell_stdouterr('{} {}'.format(self._castep_command,
                                                        self._seed),
                                         cwd=self._directory)
        if stdout:
            print(f'castep call stdout:\n{stdout}')
        if stderr:
            print(f'castep call stderr:\n{stderr}')

        # shouldn't it be called after read()???
        # self.push_oldstate()

        # check for non-empty error files
        err_file = self._abs_path(f'{self._seed}.0001.err')
        if os.path.exists(err_file):
            with open(err_file) as err_file:
                self._error = err_file.read()
        if self._error:
            raise RuntimeError(self._error)

    def __repr__(self):
        """Returns generic, fast to capture representation of
        CASTEP settings along with atoms object.
        """
        expr = ''
        expr += '-----------------Atoms--------------------\n'
        if self.atoms is not None:
            expr += str('%20s\n' % self.atoms)
        else:
            expr += 'None\n'

        expr += '-----------------Param keywords-----------\n'
        expr += str(self.param)
        expr += '-----------------Cell keywords------------\n'
        expr += str(self.cell)
        expr += '-----------------Internal keys------------\n'
        for key in self.internal_keys:
            expr += '%20s : %s\n' % (key, self._opt[key])

        return expr

    def __getattr__(self, attr):
        """___getattr___ gets overloaded to reroute the internal keys
        and to be able to easily store them in in the param so that
        they can be read in again in subsequent calls.
        """
        if attr in self.internal_keys:
            return self._opt[attr]
        if attr in ['__repr__', '__str__']:
            raise AttributeError
        elif attr not in self.__dict__:
            raise AttributeError(f'Attribute {attr} not found')
        else:
            return self.__dict__[attr]

    def __setattr__(self, attr, value):
        """We overload the settattr method to make value assignment
        as pythonic as possible. Internal values all start with _.
        Value assigment is case insensitive!
        """

        if attr.startswith('_'):
            # internal variables all start with _
            # let's check first if they are close but not identical
            # to one of the switches, that the user accesses directly
            similars = difflib.get_close_matches(attr, self.internal_keys,
                                                 cutoff=0.9)
            if attr not in self.internal_keys and similars:
                warnings.warn(
                    'Warning: You probably tried one of: '
                    f'{similars} but typed {attr}')
            if attr in self.internal_keys:
                self._opt[attr] = value
                if attr == '_track_output':
                    if value:
                        self._try_reuse = True
                        if self._pedantic:
                            warnings.warn(
                                'You switched _track_output on. This will '
                                'consume a lot of disk-space. The interface '
                                'also switched _try_reuse on, which will '
                                'try to find the last check file. Set '
                                '_try_reuse = False, if you need '
                                'really separate calculations')
                    elif '_try_reuse' in self._opt and self._try_reuse:
                        self._try_reuse = False
                        if self._pedantic:
                            warnings.warn('_try_reuse is set to False, too')
            else:
                self.__dict__[attr] = value
            return
        elif attr in ['atoms', 'cell', 'param', 'results']:
            if value is not None:
                if attr == 'atoms' and not isinstance(value, Atoms):
                    raise TypeError(
                        f'{value} is not an instance of Atoms.')
                elif attr == 'cell' and not isinstance(value, CastepCell):
                    raise TypeError(
                        f'{value} is not an instance of CastepCell.')
                elif attr == 'param' and not isinstance(value, CastepParam):
                    raise TypeError(
                        f'{value} is not an instance of CastepParam.')
            # These 3 are accepted right-away, no matter what
            self.__dict__[attr] = value
            return
        elif attr in self.atoms_obj_keys:
            # keywords which clearly belong to the atoms object are
            # rerouted to go there
            self.atoms.__dict__[attr] = value
            return
        elif attr in self.atoms_keys:
            # CASTEP keywords that should go into the atoms object
            # itself are blocked
            warnings.warn('Ignoring setings of "%s", since this has to be set '
                          'through the atoms object' % attr)
            return

        attr = attr.lower()
        if attr not in (list(self.cell._options.keys())
                        + list(self.param._options.keys())):
            # what is left now should be meant to be a castep keyword
            # so we first check if it defined, and if not offer some error
            # correction
            if self._kw_tol == 0:
                similars = difflib.get_close_matches(
                    attr,
                    self.cell._options.keys() + self.param._options.keys())
                if similars:
                    raise RuntimeError(
                        f'Option "{attr}" not known! You mean "{similars[0]}"?')
                else:
                    raise RuntimeError(f'Option "{attr}" is not known!')
            else:
                warnings.warn('Option "%s" is not known - please set any new'
                              ' options directly in the .cell or .param '
                              'objects' % attr)
                return

        # here we know it must go into one of the component param or cell
        # so we first determine which one
        if attr in self.param._options.keys():
            comp = 'param'
        elif attr in self.cell._options.keys():
            comp = 'cell'
        else:
            raise RuntimeError('Programming error: could not attach '
                               'the keyword to an input file')

        self.__dict__[comp].__setattr__(attr, value)

    def merge_param(self, param, overwrite=True, ignore_internal_keys=False):
        """Parse a param file and merge it into the current parameters."""
        if isinstance(param, CastepParam):
            for key, option in param._options.items():
                if option.value is not None:
                    self.param.__setattr__(key, option.value)
            return

        elif isinstance(param, str):
            param_file = open(param)
            _close = True

        else:
            # in this case we assume that we have a fileobj already, but check
            # for attributes in order to avoid extended EAFP blocks.
            param_file = param

            # look before you leap...
            attributes = ['name',
                          'close'
                          'readlines']

            for attr in attributes:
                if not hasattr(param_file, attr):
                    raise TypeError('"param" is neither CastepParam nor str '
                                    'nor valid fileobj')

            param = param_file.name
            _close = False

        self, int_opts = read_param(fd=param_file, calc=self,
                                    get_interface_options=True)

        # Add the interface options
        for k, val in int_opts.items():
            if (k in self.internal_keys and not ignore_internal_keys):
                if val in _tf_table:
                    val = _tf_table[val]
                self._opt[k] = val

        if _close:
            param_file.close()

    def dryrun_ok(self, dryrun_flag='-dryrun'):
        """Starts a CASTEP run with the -dryrun flag [default]
        in a temporary and check wether all variables are initialized
        correctly. This is recommended for every bigger simulation.
        """
        from ase.io.castep import write_param

        temp_dir = tempfile.mkdtemp()
        self._fetch_pspots(temp_dir)
        seed = 'dryrun'

        magnetic_moments = ('initial' if
                            self.param.spin_polarized.value == 'TRUE'
                            else None)
        self._write_cell(os.path.join(temp_dir, f'{seed}.cell'),
                         self.atoms, castep_cell=self.cell,
                         magnetic_moments=magnetic_moments)
        # This part needs to be modified now that we rely on the new formats.py
        # interface
        if not os.path.isfile(os.path.join(temp_dir, f'{seed}.cell')):
            warnings.warn(f'{seed}.cell not written - aborting dryrun')
            return None
        write_param(os.path.join(temp_dir, f'{seed}.param'), self.param, )

        stdout, stderr = shell_stdouterr(('{} {} {}'.format(
            self._castep_command,
            seed,
            dryrun_flag)),
            cwd=temp_dir)

        if stdout:
            print(stdout)
        if stderr:
            print(stderr)
        with open(os.path.join(temp_dir, f'{seed}.castep')) as result_file:
            txt = result_file.read()
            ok_string = (r'.*DRYRUN finished.*No problems found with input '
                         r'files.*')
            match = re.match(ok_string, txt, re.DOTALL)

            m = re.search(r'Number of kpoints used =\s*([0-9]+)', txt)
            if m:
                self._kpoints = int(m.group(1))
            else:
                warnings.warn(
                    'Couldn\'t fetch number of kpoints from dryrun CASTEP file')

            err_file = os.path.join(temp_dir, f'{seed}.0001.err')
            if match is None and os.path.exists(err_file):
                with open(err_file) as err_file:
                    self._error = err_file.read()
        shutil.rmtree(temp_dir)

        # re.match return None is the string does not match
        return match is not None

    def _fetch_pspots(self, directory=None):
        """Put all specified pseudo-potentials into the working directory.
        """
        # should be a '==' right? Otherwise setting _castep_pp_path is not
        # honored.
        if (not cfg.get('PSPOT_DIR', None)
                and self._castep_pp_path == os.path.abspath('.')):
            # By default CASTEP consults the environment variable
            # PSPOT_DIR. If this contains a list of colon separated
            # directories it will check those directories for pseudo-
            # potential files if not in the current directory.
            # Thus if PSPOT_DIR is set there is nothing left to do.
            # If however PSPOT_DIR was been accidentally set
            # (e.g. with regards to a different program)
            # setting CASTEP_PP_PATH to an explicit value will
            # still be honored.
            return

        if directory is None:
            directory = self._directory
        if not os.path.isdir(self._castep_pp_path):
            warnings.warn(f'PSPs directory {self._castep_pp_path} not found')
        pspots = {}
        if self._find_pspots:
            self.find_pspots()
        if self.cell.species_pot.value is not None:
            for line in self.cell.species_pot.value.split('\n'):
                line = line.split()
                if line:
                    pspots[line[0]] = line[1]
        for species in self.atoms.get_chemical_symbols():
            if not pspots or species not in pspots.keys():
                if self._build_missing_pspots:
                    if self._pedantic:
                        warnings.warn(
                            'Warning: you have no PP specified for %s. '
                            'CASTEP will now generate an on-the-fly '
                            'potentials. '
                            'For sake of numerical consistency and efficiency '
                            'this is discouraged.' % species)
                else:
                    raise RuntimeError(
                        f'Warning: you have no PP specified for {species}.')
        if self.cell.species_pot.value:
            for (species, pspot) in pspots.items():
                orig_pspot_file = os.path.join(self._castep_pp_path, pspot)
                cp_pspot_file = os.path.join(directory, pspot)
                if (os.path.exists(orig_pspot_file)
                        and not os.path.exists(cp_pspot_file)):
                    if self._copy_pspots:
                        shutil.copy(orig_pspot_file, directory)
                    elif self._link_pspots:
                        os.symlink(orig_pspot_file, cp_pspot_file)
                    else:
                        if self._pedantic:
                            warnings.warn(ppwarning)


ppwarning = ('Warning: PP files have neither been '
             'linked nor copied to the working directory. Make '
             'sure to set the evironment variable PSPOT_DIR '
             'accordingly!')


def _get_indices_to_sort_back(symbols, species):
    """Get indices to sort spicies in .castep back to atoms.symbols."""
    uniques = np.unique(symbols)
    indices = np.full(len(symbols), -1, dtype=int)
    for unique in uniques:
        where_symbols = [i for i, s in enumerate(symbols) if s == unique]
        where_species = [j for j, s in enumerate(species) if s == unique]
        for i, j in zip(where_symbols, where_species):
            indices[i] = j
    if -1 in indices:
        not_assigned = [_ for _ in indices if _ == -1]
        raise RuntimeError(f'Atoms {not_assigned} where not assigned.')
    return indices


def get_castep_version(castep_command):
    """This returns the version number as printed in the CASTEP banner.
       For newer CASTEP versions ( > 6.1) the --version command line option
       has been added; this will be attempted first.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        return _get_castep_version(castep_command, temp_dir)


def _get_castep_version(castep_command, temp_dir):
    jname = 'dummy_jobname'
    stdout, stderr = '', ''
    fallback_version = 16.  # CASTEP 16.0 and 16.1 report version wrongly
    try:
        stdout, stderr = subprocess.Popen(
            castep_command.split() + ['--version'],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE, cwd=temp_dir,
            universal_newlines=True).communicate()
        if 'CASTEP version' not in stdout:
            stdout, stderr = subprocess.Popen(
                castep_command.split() + [jname],
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE, cwd=temp_dir,
                universal_newlines=True).communicate()
    except Exception:  # XXX Which kind of exception?
        msg = ''
        msg += 'Could not determine the version of your CASTEP binary \n'
        msg += 'This usually means one of the following \n'
        msg += '   * you do not have CASTEP installed \n'
        msg += '   * you have not set the CASTEP_COMMAND to call it \n'
        msg += '   * you have provided a wrong CASTEP_COMMAND. \n'
        msg += '     Make sure it is in your PATH\n\n'
        msg += stdout
        msg += stderr
        raise CastepVersionError(msg)
    if 'CASTEP version' in stdout:
        output_txt = stdout.split('\n')
        version_re = re.compile(r'CASTEP version:\s*([0-9\.]*)')
    else:
        with open(os.path.join(temp_dir, f'{jname}.castep')) as output:
            output_txt = output.readlines()
        version_re = re.compile(r'(?<=CASTEP version )[0-9.]*')
    # shutil.rmtree(temp_dir)
    for line in output_txt:
        if 'CASTEP version' in line:
            try:
                return float(version_re.findall(line)[0])
            except ValueError:
                # Fallback for buggy --version on CASTEP 16.0, 16.1
                return fallback_version


def create_castep_keywords(castep_command, filename='castep_keywords.json',
                           force_write=True, path='.', fetch_only=None):
    """This function allows to fetch all available keywords from stdout
    of an installed castep binary. It furthermore collects the documentation
    to harness the power of (ipython) inspection and type for some basic
    type checking of input. All information is stored in a JSON file that is
    not distributed by default to avoid breaking the license of CASTEP.
    """
    # Takes a while ...
    # Fetch all allowed parameters
    # fetch_only : only fetch that many parameters (for testsuite only)
    suffixes = ['cell', 'param']

    filepath = os.path.join(path, filename)

    if os.path.exists(filepath) and not force_write:
        warnings.warn('CASTEP Options Module file exists. '
                      'You can overwrite it by calling '
                      'python castep.py -f [CASTEP_COMMAND].')
        return False

    # Not saving directly to file her to prevent half-generated files
    # which will cause problems on future runs

    castep_version = get_castep_version(castep_command)

    help_all, _ = shell_stdouterr(f'{castep_command} -help all')

    # Filter out proper keywords
    try:
        # The old pattern does not math properly as in CASTEP as of v8.0 there
        # are some keywords for the semi-empircal dispersion correction (SEDC)
        # which also include numbers.
        if castep_version < 7.0:
            pattern = r'((?<=^ )[A-Z_]{2,}|(?<=^)[A-Z_]{2,})'
        else:
            pattern = r'((?<=^ )[A-Z_\d]{2,}|(?<=^)[A-Z_\d]{2,})'

        raw_options = re.findall(pattern, help_all, re.MULTILINE)
    except Exception:
        warnings.warn(f'Problem parsing: {help_all}')
        raise

    types = set()
    levels = set()

    processed_n = 0
    to_process = len(raw_options[:fetch_only])

    processed_options = {sf: {} for sf in suffixes}

    for o_i, option in enumerate(raw_options[:fetch_only]):
        doc, _ = shell_stdouterr(f'{castep_command} -help {option}')

        # Stand Back! I know regular expressions (http://xkcd.com/208/) :-)
        match = re.match(r'(?P<before_type>.*)Type: (?P<type>.+?)\s+'
                         + r'Level: (?P<level>[^ ]+)\n\s*\n'
                         + r'(?P<doc>.*?)(\n\s*\n|$)', doc, re.DOTALL)

        processed_n += 1

        if match is not None:
            match = match.groupdict()

            # JM: uncomment lines in following block to debug issues
            #     with keyword assignment during extraction process from CASTEP
            suffix = None
            if re.findall(r'PARAMETERS keywords:\n\n\s?None found', doc):
                suffix = 'cell'
            if re.findall(r'CELL keywords:\n\n\s?None found', doc):
                suffix = 'param'
            if suffix is None:
                warnings.warn('%s -> not assigned to either'
                              ' CELL or PARAMETERS keywords' % option)

            option = option.lower()
            mtyp = match.get('type', None)
            mlvl = match.get('level', None)
            mdoc = match.get('doc', None)

            if mtyp is None:
                warnings.warn(f'Found no type for {option}')
                continue
            if mlvl is None:
                warnings.warn(f'Found no level for {option}')
                continue
            if mdoc is None:
                warnings.warn(f'Found no doc string for {option}')
                continue

            types = types.union([mtyp])
            levels = levels.union([mlvl])

            processed_options[suffix][option] = {
                'keyword': option,
                'option_type': mtyp,
                'level': mlvl,
                'docstring': mdoc
            }

            processed_n += 1

            frac = (o_i + 1.0) / to_process
            sys.stdout.write('\rProcessed: [{}] {:>3.0f}%'.format(
                             '#' * int(frac * 20) + ' '
                             * (20 - int(frac * 20)),
                             100 * frac))
            sys.stdout.flush()

        else:
            warnings.warn(f'create_castep_keywords: Could not process {option}')

    sys.stdout.write('\n')
    sys.stdout.flush()

    processed_options['types'] = list(types)
    processed_options['levels'] = list(levels)
    processed_options['castep_version'] = castep_version

    json.dump(processed_options, open(filepath, 'w'), indent=4)

    warnings.warn(f'CASTEP v{castep_version}, fetched {processed_n} keywords')
    return True


CastepKeywords = namedtuple('CastepKeywords',
                            ['CastepParamDict', 'CastepCellDict',
                             'types', 'levels', 'castep_version'])

# We keep this just for naming consistency with older versions


def make_cell_dict(data=None):
    from ase.io.castep.castep_input_file import CastepOptionDict

    data = data if data is not None else {}

    class CastepCellDict(CastepOptionDict):
        def __init__(self):
            CastepOptionDict.__init__(self, data)

    return CastepCellDict


def make_param_dict(data=None):
    from ase.io.castep.castep_input_file import CastepOptionDict

    data = data if data is not None else {}

    class CastepParamDict(CastepOptionDict):
        def __init__(self):
            CastepOptionDict.__init__(self, data)

    return CastepParamDict


class CastepVersionError(Exception):
    """No special behaviour, works to signal when Castep can not be found"""


def get_castep_pp_path(castep_pp_path=''):
    """Abstract the quest for a CASTEP PSP directory."""
    if castep_pp_path:
        return os.path.abspath(os.path.expanduser(castep_pp_path))
    elif 'PSPOT_DIR' in cfg:
        return cfg['PSPOT_DIR']
    elif 'CASTEP_PP_PATH' in cfg:
        return cfg['CASTEP_PP_PATH']
    else:
        return os.path.abspath('.')


def get_castep_command(castep_command=''):
    """Abstract the quest for a castep_command string."""
    if castep_command:
        return castep_command
    elif 'CASTEP_COMMAND' in cfg:
        return cfg['CASTEP_COMMAND']
    else:
        return 'castep'


def shell_stdouterr(raw_command, cwd=None):
    """Abstracts the standard call of the commandline, when
    we are only interested in the stdout and stderr
    """
    stdout, stderr = subprocess.Popen(raw_command,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE,
                                      universal_newlines=True,
                                      shell=True, cwd=cwd).communicate()
    return stdout.strip(), stderr.strip()


def import_castep_keywords(castep_command='',
                           filename='castep_keywords.json',
                           path='.'):
    """Search for castep keywords JSON in multiple paths"""

    config_paths = ('~/.ase', '~/.config/ase')
    searchpaths = [path] + [os.path.expanduser(config_path)
                            for config_path in config_paths]
    try:
        keywords_file = sum(
            (glob.glob(os.path.join(sp, filename)) for sp in searchpaths), []
        )[0]
    except IndexError:
        warnings.warn("""Generating CASTEP keywords JSON file... hang on.
    The CASTEP keywords JSON file contains abstractions for CASTEP input
    parameters (for both .cell and .param input files), including some
    format checks and descriptions. The latter are extracted from the
    internal online help facility of a CASTEP binary, thus allowing to
    easily keep the calculator synchronized with (different versions of)
    the CASTEP code. Consequently, avoiding licensing issues (CASTEP is
    distributed commercially by Biovia), we consider it wise not to
    provide the file in the first place.""")
        create_castep_keywords(get_castep_command(castep_command),
                               filename=filename, path=path)
        keywords_file = Path(path).absolute() / filename

        warnings.warn(
            f'Stored castep keywords dictionary as {keywords_file}. '
            f'Copy it to {Path(config_paths[0]).expanduser() / filename} for '
            r'user installation.')

    # Now create the castep_keywords object proper
    with open(keywords_file) as fd:
        kwdata = json.load(fd)

    # This is a bit awkward, but it's necessary for backwards compatibility
    param_dict = make_param_dict(kwdata['param'])
    cell_dict = make_cell_dict(kwdata['cell'])

    castep_keywords = CastepKeywords(param_dict, cell_dict,
                                     kwdata['types'], kwdata['levels'],
                                     kwdata['castep_version'])

    return castep_keywords


if __name__ == '__main__':
    warnings.warn(
        'When called directly this calculator will fetch all available '
        'keywords from the binarys help function into a '
        'castep_keywords.json in the current directory %s '
        'For system wide usage, it can be copied into an ase installation '
        'at ASE/calculators. '
        'This castep_keywords.json usually only needs to be generated once '
        'for a CASTEP binary/CASTEP version.' % os.getcwd())

    import optparse
    parser = optparse.OptionParser()
    parser.add_option(
        '-f', '--force-write', dest='force_write',
        help='Force overwriting existing castep_keywords.json', default=False,
        action='store_true')
    (options, args) = parser.parse_args()

    if args:
        opt_castep_command = ''.join(args)
    else:
        opt_castep_command = ''
    generated = create_castep_keywords(get_castep_command(opt_castep_command),
                                       force_write=options.force_write)

    if generated:
        try:
            with open('castep_keywords.json') as fd:
                json.load(fd)
        except Exception as e:
            warnings.warn(
                f'{e} Ooops, something went wrong with the CASTEP keywords')
        else:
            warnings.warn('Import works. Looking good!')
