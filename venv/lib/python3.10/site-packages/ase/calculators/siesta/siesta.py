# fmt: off

"""
This module defines the ASE interface to SIESTA.

Written by Mads Engelund (see www.espeem.com)

Home of the SIESTA package:
http://www.uam.es/departamentos/ciencias/fismateriac/siesta

2017.04 - Pedro Brandimarte: changes for python 2-3 compatible

"""

import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ase import Atoms
from ase.calculators.calculator import FileIOCalculator, Parameters, ReadError
from ase.calculators.siesta.import_ion_xml import get_ion
from ase.calculators.siesta.parameters import PAOBasisBlock, format_fdf
from ase.data import atomic_numbers
from ase.io.siesta import read_siesta_xv
from ase.io.siesta_input import SiestaInput
from ase.units import Ry, eV
from ase.utils import deprecated

meV = 0.001 * eV


def parse_siesta_version(output: bytes) -> str:
    match = re.search(rb'Version\s*:\s*(\S+)', output)

    if match is None:
        raise RuntimeError('Could not get Siesta version info from output '
                           '{!r}'.format(output))

    string = match.group(1).decode('ascii')
    return string


def get_siesta_version(executable: str) -> str:
    """ Return SIESTA version number.

    Run the command, for instance 'siesta' and
    then parse the output in order find the
    version number.
    """
    # XXX We need a test of this kind of function.  But Siesta().command
    # is not enough to tell us how to run Siesta, because it could contain
    # all sorts of mpirun and other weird parts.

    temp_dirname = tempfile.mkdtemp(prefix='siesta-version-check-')
    try:
        from subprocess import PIPE, Popen
        proc = Popen([executable],
                     stdin=PIPE,
                     stdout=PIPE,
                     stderr=PIPE,
                     cwd=temp_dirname)
        output, _ = proc.communicate()
        # We are not providing any input, so Siesta will give us a failure
        # saying that it has no Chemical_species_label and exit status 1
        # (as of siesta-4.1-b4)
    finally:
        shutil.rmtree(temp_dirname)

    return parse_siesta_version(output)


def format_block(name, block):
    lines = [f'%block {name}']
    for row in block:
        data = ' '.join(str(obj) for obj in row)
        lines.append(f'    {data}')
    lines.append(f'%endblock {name}')
    return '\n'.join(lines)


def bandpath2bandpoints(path):
    return '\n'.join([
        'BandLinesScale ReciprocalLatticeVectors',
        format_block('BandPoints', path.kpts)])


class SiestaParameters(Parameters):
    def __init__(
            self,
            label='siesta',
            mesh_cutoff=200 * Ry,
            energy_shift=100 * meV,
            kpts=None,
            xc='LDA',
            basis_set='DZP',
            spin='non-polarized',
            species=(),
            pseudo_qualifier=None,
            pseudo_path=None,
            symlink_pseudos=None,
            atoms=None,
            restart=None,
            fdf_arguments=None,
            atomic_coord_format='xyz',
            bandpath=None):
        kwargs = locals()
        kwargs.pop('self')
        Parameters.__init__(self, **kwargs)


def _nonpolarized_alias(_: List, kwargs: Dict[str, Any]) -> bool:
    if kwargs.get("spin", None) == "UNPOLARIZED":
        kwargs["spin"] = "non-polarized"
        return True
    return False


class Siesta(FileIOCalculator):
    """Calculator interface to the SIESTA code.
    """
    allowed_xc = {
        'LDA': ['PZ', 'CA', 'PW92'],
        'GGA': ['PW91', 'PBE', 'revPBE', 'RPBE',
                'WC', 'AM05', 'PBEsol', 'PBEJsJrLO',
                'PBEGcGxLO', 'PBEGcGxHEG', 'BLYP'],
        'VDW': ['DRSLL', 'LMKLL', 'KBM', 'C09', 'BH', 'VV']}

    name = 'siesta'
    _legacy_default_command = 'siesta < PREFIX.fdf > PREFIX.out'
    implemented_properties = [
        'energy',
        'free_energy',
        'forces',
        'stress',
        'dipole',
        'eigenvalues',
        'density',
        'fermi_energy']

    # Dictionary of valid input vaiables.
    default_parameters = SiestaParameters()

    # XXX Not a ASE standard mechanism (yet).  We need to communicate to
    # ase.spectrum.band_structure.calculate_band_structure() that we expect
    # it to use the bandpath keyword.
    accepts_bandpath_keyword = True

    fileio_rules = FileIOCalculator.ruleset(
        configspec=dict(pseudo_path=None),
        stdin_name='{prefix}.fdf',
        stdout_name='{prefix}.out')

    def __init__(self, command=None, profile=None, directory='.', **kwargs):
        """ASE interface to the SIESTA code.

        Parameters:
           - label        : The basename of all files created during
                            calculation.
           - mesh_cutoff  : Energy in eV.
                            The mesh cutoff energy for determining number of
                            grid points in the matrix-element calculation.
           - energy_shift : Energy in eV
                            The confining energy of the basis set generation.
           - kpts         : Tuple of 3 integers, the k-points in different
                            directions.
           - xc           : The exchange-correlation potential. Can be set to
                            any allowed value for either the Siesta
                            XC.funtional or XC.authors keyword. Default "LDA"
           - basis_set    : "SZ"|"SZP"|"DZ"|"DZP"|"TZP", strings which specify
                            the type of functions basis set.
           - spin         : "non-polarized"|"collinear"|
                            "non-collinear|spin-orbit".
                            The level of spin description to be used.
           - species      : None|list of Species objects. The species objects
                            can be used to to specify the basis set,
                            pseudopotential and whether the species is ghost.
                            The tag on the atoms object and the element is used
                            together to identify the species.
           - pseudo_path  : None|path. This path is where
                            pseudopotentials are taken from.
                            If None is given, then then the path given
                            in $SIESTA_PP_PATH will be used.
           - pseudo_qualifier: None|string. This string will be added to the
                            pseudopotential path that will be retrieved.
                            For hydrogen with qualifier "abc" the
                            pseudopotential "H.abc.psf" will be retrieved.
           - symlink_pseudos: None|bool
                            If true, symlink pseudopotentials
                            into the calculation directory, else copy them.
                            Defaults to true on Unix and false on Windows.
           - atoms        : The Atoms object.
           - restart      : str.  Prefix for restart file.
                            May contain a directory.
                            Default is  None, don't restart.
           - fdf_arguments: Explicitly given fdf arguments. Dictonary using
                            Siesta keywords as given in the manual. List values
                            are written as fdf blocks with each element on a
                            separate line, while tuples will write each element
                            in a single line.  ASE units are assumed in the
                            input.
           - atomic_coord_format: "xyz"|"zmatrix", strings to switch between
                            the default way of entering the system's geometry
                            (via the block AtomicCoordinatesAndAtomicSpecies)
                            and a recent method via the block Zmatrix. The
                            block Zmatrix allows to specify basic geometry
                            constrains such as realized through the ASE classes
                            FixAtom, FixedLine and FixedPlane.
        """

        # Put in the default arguments.
        parameters = self.default_parameters.__class__(**kwargs)

        # Call the base class.
        FileIOCalculator.__init__(
            self,
            command=command,
            profile=profile,
            directory=directory,
            **parameters)

    def __getitem__(self, key):
        """Convenience method to retrieve a parameter as
        calculator[key] rather than calculator.parameters[key]

            Parameters:
                -key       : str, the name of the parameters to get.
        """
        return self.parameters[key]

    def species(self, atoms):
        """Find all relevant species depending on the atoms object and
        species input.

            Parameters :
                - atoms : An Atoms object.
        """
        return SiestaInput.get_species(
            atoms, list(self['species']), self['basis_set'])

    @deprecated(
        "The keyword 'UNPOLARIZED' has been deprecated,"
        "and replaced by 'non-polarized'",
        category=FutureWarning,
        callback=_nonpolarized_alias,
    )
    def set(self, **kwargs):
        """Set all parameters.

            Parameters:
                -kwargs  : Dictionary containing the keywords defined in
                           SiestaParameters.

        .. deprecated:: 3.18.2
            The keyword 'UNPOLARIZED' has been deprecated and replaced by
            'non-polarized'
        """

        # XXX Inserted these next few lines because set() would otherwise
        # discard all previously set keywords to their defaults!  --askhl
        current = self.parameters.copy()
        current.update(kwargs)
        kwargs = current

        # Find not allowed keys.
        default_keys = list(self.__class__.default_parameters)
        offending_keys = set(kwargs) - set(default_keys)
        if len(offending_keys) > 0:
            mess = "'set' does not take the keywords: %s "
            raise ValueError(mess % list(offending_keys))

        # Use the default parameters.
        parameters = self.__class__.default_parameters.copy()
        parameters.update(kwargs)
        kwargs = parameters

        # Check energy inputs.
        for arg in ['mesh_cutoff', 'energy_shift']:
            value = kwargs.get(arg)
            if value is None:
                continue
            if not (isinstance(value, (float, int)) and value > 0):
                mess = "'{}' must be a positive number(in eV), \
                    got '{}'".format(arg, value)
                raise ValueError(mess)

        # Check the functional input.
        xc = kwargs.get('xc', 'LDA')
        if isinstance(xc, (tuple, list)) and len(xc) == 2:
            functional, authors = xc
            if functional.lower() not in [k.lower() for k in self.allowed_xc]:
                mess = f"Unrecognized functional keyword: '{functional}'"
                raise ValueError(mess)

            lsauthorslower = [a.lower() for a in self.allowed_xc[functional]]
            if authors.lower() not in lsauthorslower:
                mess = "Unrecognized authors keyword for %s: '%s'"
                raise ValueError(mess % (functional, authors))

        elif xc in self.allowed_xc:
            functional = xc
            authors = self.allowed_xc[xc][0]
        else:
            found = False
            for key, value in self.allowed_xc.items():
                if xc in value:
                    found = True
                    functional = key
                    authors = xc
                    break

            if not found:
                raise ValueError(f"Unrecognized 'xc' keyword: '{xc}'")
        kwargs['xc'] = (functional, authors)

        # Check fdf_arguments.
        if kwargs['fdf_arguments'] is None:
            kwargs['fdf_arguments'] = {}

        if not isinstance(kwargs['fdf_arguments'], dict):
            raise TypeError("fdf_arguments must be a dictionary.")

        # Call baseclass.
        FileIOCalculator.set(self, **kwargs)

    def set_fdf_arguments(self, fdf_arguments):
        """ Set the fdf_arguments after the initialization of the
            calculator.
        """
        self.validate_fdf_arguments(fdf_arguments)
        FileIOCalculator.set(self, fdf_arguments=fdf_arguments)

    def validate_fdf_arguments(self, fdf_arguments):
        """ Raises error if the fdf_argument input is not a
            dictionary of allowed keys.
        """
        # None is valid
        if fdf_arguments is None:
            return

        # Type checking.
        if not isinstance(fdf_arguments, dict):
            raise TypeError("fdf_arguments must be a dictionary.")

    def write_input(self, atoms, properties=None, system_changes=None):
        """Write input (fdf)-file.
        See calculator.py for further details.

        Parameters:
            - atoms        : The Atoms object to write.
            - properties   : The properties which should be calculated.
            - system_changes : List of properties changed since last run.
        """

        super().write_input(
            atoms=atoms,
            properties=properties,
            system_changes=system_changes)

        filename = self.getpath(ext='fdf')

        more_fdf_args = {}

        # Use the saved density matrix if only 'cell' and 'positions'
        # have changed.
        if (system_changes is None or
            ('numbers' not in system_changes and
             'initial_magmoms' not in system_changes and
             'initial_charges' not in system_changes)):

            more_fdf_args['DM.UseSaveDM'] = True

        if 'density' in properties:
            more_fdf_args['SaveRho'] = True

        species, species_numbers = self.species(atoms)

        pseudo_path = (self['pseudo_path']
                       or self.profile.configvars.get('pseudo_path')
                       or self.cfg.get('SIESTA_PP_PATH'))

        if not pseudo_path:
            raise Exception(
                'Please configure pseudo_path or SIESTA_PP_PATH envvar')

        species_info = SpeciesInfo(
            atoms=atoms,
            pseudo_path=Path(pseudo_path),
            pseudo_qualifier=self.pseudo_qualifier(),
            species=species)

        writer = FDFWriter(
            name=self.prefix,
            xc=self['xc'],
            spin=self['spin'],
            mesh_cutoff=self['mesh_cutoff'],
            energy_shift=self['energy_shift'],
            fdf_user_args=self['fdf_arguments'],
            more_fdf_args=more_fdf_args,
            species_numbers=species_numbers,
            atomic_coord_format=self['atomic_coord_format'].lower(),
            kpts=self['kpts'],
            bandpath=self['bandpath'],
            species_info=species_info,
        )

        with open(filename, 'w') as fd:
            writer.write(fd)

        writer.link_pseudos_into_directory(
            symlink_pseudos=self['symlink_pseudos'],
            directory=Path(self.directory))

    def read(self, filename):
        """Read structural parameters from file .XV file
           Read other results from other files
           filename : siesta.XV
        """

        fname = self.getpath(filename)
        if not fname.exists():
            raise ReadError(f"The restart file '{fname}' does not exist")
        with fname.open() as fd:
            self.atoms = read_siesta_xv(fd)
        self.read_results()

    def getpath(self, fname=None, ext=None):
        """ Returns the directory/fname string """
        if fname is None:
            fname = self.prefix
        if ext is not None:
            fname = f'{fname}.{ext}'
        return Path(self.directory) / fname

    def pseudo_qualifier(self):
        """Get the extra string used in the middle of the pseudopotential.
        The retrieved pseudopotential for a specific element will be
        'H.xxx.psf' for the element 'H' with qualifier 'xxx'. If qualifier
        is set to None then the qualifier is set to functional name.
        """
        if self['pseudo_qualifier'] is None:
            return self['xc'][0].lower()
        else:
            return self['pseudo_qualifier']

    def read_results(self):
        """Read the results."""
        from ase.io.siesta_output import OutputReader
        reader = OutputReader(prefix=self.prefix,
                              directory=Path(self.directory),
                              bandpath=self['bandpath'])
        results = reader.read_results()
        self.results.update(results)

        self.results['ion'] = self.read_ion(self.atoms)

    def read_ion(self, atoms):
        """
        Read the ion.xml file of each specie
        """
        species, _species_numbers = self.species(atoms)

        ion_results = {}
        for species_number, spec in enumerate(species, start=1):
            symbol = spec['symbol']
            atomic_number = atomic_numbers[symbol]

            if spec['pseudopotential'] is None:
                if self.pseudo_qualifier() == '':
                    label = symbol
                else:
                    label = f"{symbol}.{self.pseudo_qualifier()}"
                pseudopotential = self.getpath(label, 'psf')
            else:
                pseudopotential = Path(spec['pseudopotential'])
                label = pseudopotential.stem

            name = f"{label}.{species_number}"
            if spec['ghost']:
                name = f"{name}.ghost"
                atomic_number = -atomic_number

            if name not in ion_results:
                fname = self.getpath(name, 'ion.xml')
                if fname.is_file():
                    ion_results[name] = get_ion(str(fname))

        return ion_results

    def band_structure(self):
        return self.results['bandstructure']

    def get_fermi_level(self):
        return self.results['fermi_energy']

    def get_k_point_weights(self):
        return self.results['kpoint_weights']

    def get_ibz_k_points(self):
        return self.results['kpoints']

    def get_eigenvalues(self, kpt=0, spin=0):
        return self.results['eigenvalues'][spin, kpt]

    def get_number_of_spins(self):
        return self.results['eigenvalues'].shape[0]


def generate_atomic_coordinates(atoms: Atoms, species_numbers,
                                atomic_coord_format: str):
    """Write atomic coordinates.

    Parameters
    ----------
    fd : IO
        An open file object.
    atoms : Atoms
        An atoms object.
    """
    if atomic_coord_format == 'xyz':
        return generate_atomic_coordinates_xyz(atoms, species_numbers)
    elif atomic_coord_format == 'zmatrix':
        return generate_atomic_coordinates_zmatrix(atoms, species_numbers)
    else:
        raise RuntimeError(
            f'Unknown atomic_coord_format: {atomic_coord_format}')


def generate_atomic_coordinates_zmatrix(atoms: Atoms, species_numbers):
    """Write atomic coordinates in Z-matrix format.

    Parameters
    ----------
    fd : IO
        An open file object.
    atoms : Atoms
        An atoms object.
    """
    yield '\n'
    yield var('ZM.UnitsLength', 'Ang')
    yield '%block Zmatrix\n'
    yield '  cartesian\n'

    fstr = "{:5d}" + "{:20.10f}" * 3 + "{:3d}" * 3 + "{:7d} {:s}\n"
    a2constr = SiestaInput.make_xyz_constraints(atoms)
    a2p, a2s = atoms.get_positions(), atoms.symbols
    for ia, (sp, xyz, ccc, sym) in enumerate(
            zip(species_numbers, a2p, a2constr, a2s)):
        yield fstr.format(
            sp, xyz[0], xyz[1], xyz[2], ccc[0],
            ccc[1], ccc[2], ia + 1, sym)
    yield '%endblock Zmatrix\n'

    # origin = tuple(-atoms.get_celldisp().flatten())
    # yield block('AtomicCoordinatesOrigin', [origin])


def generate_atomic_coordinates_xyz(atoms: Atoms, species_numbers):
    """Write atomic coordinates.

    Parameters
    ----------
    fd : IO
        An open file object.
    atoms : Atoms
        An atoms object.
    """
    yield '\n'
    yield var('AtomicCoordinatesFormat', 'Ang')
    yield block('AtomicCoordinatesAndAtomicSpecies',
                [[*atom.position, number]
                 for atom, number in zip(atoms, species_numbers)])
    yield '\n'

    # origin = tuple(-atoms.get_celldisp().flatten())
    # yield block('AtomicCoordinatesOrigin', [origin])


@dataclass
class SpeciesInfo:
    atoms: Atoms
    pseudo_path: Path
    pseudo_qualifier: str
    species: dict  # actually a kind of Parameters object, should refactor

    def __post_init__(self):
        pao_basis = []
        chemical_labels = []
        basis_sizes = []
        file_instructions = []

        for species_number, spec in enumerate(self.species, start=1):
            symbol = spec['symbol']
            atomic_number = atomic_numbers[symbol]

            if spec['pseudopotential'] is None:
                if self.pseudo_qualifier == '':
                    label = symbol
                else:
                    label = f"{symbol}.{self.pseudo_qualifier}"
                src_path = self.pseudo_path / f"{label}.psf"
            else:
                src_path = Path(spec['pseudopotential'])

            if not src_path.is_absolute():
                src_path = self.pseudo_path / src_path
            if not src_path.exists():
                src_path = self.pseudo_path / f"{symbol}.psml"

            name = src_path.name
            name = name.split('.')
            name.insert(-1, str(species_number))
            if spec['ghost']:
                name.insert(-1, 'ghost')
                atomic_number = -atomic_number

            name = '.'.join(name)

            instr = FileInstruction(src_path, name)
            file_instructions.append(instr)

            label = '.'.join(np.array(name.split('.'))[:-1])
            pseudo_name = ''
            if src_path.suffix != '.psf':
                pseudo_name = f'{label}{src_path.suffix}'
            string = '    %d %d %s %s' % (species_number, atomic_number, label,
                                          pseudo_name)
            chemical_labels.append(string)
            if isinstance(spec['basis_set'], PAOBasisBlock):
                pao_basis.append(spec['basis_set'].script(label))
            else:
                basis_sizes.append(("    " + label, spec['basis_set']))

        self.file_instructions = file_instructions
        self.chemical_labels = chemical_labels
        self.pao_basis = pao_basis
        self.basis_sizes = basis_sizes

    def generate_text(self):
        yield var('NumberOfSpecies', len(self.species))
        yield var('NumberOfAtoms', len(self.atoms))

        yield var('ChemicalSpecieslabel', self.chemical_labels)
        yield '\n'
        yield var('PAO.Basis', self.pao_basis)
        yield var('PAO.BasisSizes', self.basis_sizes)
        yield '\n'


@dataclass
class FileInstruction:
    src_path: Path
    targetname: str

    def copy_to(self, directory):
        self._link(shutil.copy, directory)

    def symlink_to(self, directory):
        self._link(os.symlink, directory)

    def _link(self, file_operation, directory):
        dst_path = directory / self.targetname
        if self.src_path == dst_path:
            return

        dst_path.unlink(missing_ok=True)
        file_operation(self.src_path, dst_path)


@dataclass
class FDFWriter:
    name: str
    xc: str
    fdf_user_args: dict
    more_fdf_args: dict
    mesh_cutoff: float
    energy_shift: float
    spin: str
    species_numbers: object  # ?
    atomic_coord_format: str
    kpts: object  # ?
    bandpath: object  # ?
    species_info: object

    def write(self, fd):
        for chunk in self.generate_text():
            fd.write(chunk)

    def generate_text(self):
        yield var('SystemName', self.name)
        yield var('SystemLabel', self.name)
        yield "\n"

        # Write explicitly given options first to
        # allow the user to override anything.
        fdf_arguments = self.fdf_user_args
        for key in sorted(fdf_arguments):
            yield var(key, fdf_arguments[key])

        # Force siesta to return error on no convergence.
        # as default consistent with ASE expectations.
        if 'SCFMustConverge' not in fdf_arguments:
            yield var('SCFMustConverge', True)
        yield '\n'

        yield var('Spin', self.spin)
        # Spin backwards compatibility.
        if self.spin == 'collinear':
            key = 'SpinPolarized'
        elif self.spin == 'non-collinear':
            key = 'NonCollinearSpin'
        else:
            key = None

        if key is not None:
            yield var(key, (True, '# Backwards compatibility.'))

        # Write functional.
        functional, authors = self.xc
        yield var('XC.functional', functional)
        yield var('XC.authors', authors)
        yield '\n'

        # Write mesh cutoff and energy shift.
        yield var('MeshCutoff', (self.mesh_cutoff, 'eV'))
        yield var('PAO.EnergyShift', (self.energy_shift, 'eV'))
        yield '\n'

        yield from self.species_info.generate_text()
        yield from self.generate_atoms_text(self.species_info.atoms)

        for key, value in self.more_fdf_args.items():
            yield var(key, value)

        if self.kpts is not None:
            kpts = np.array(self.kpts)
            yield from SiestaInput.generate_kpts(kpts)

        if self.bandpath is not None:
            lines = bandpath2bandpoints(self.bandpath)
            assert isinstance(lines, str)  # rename this variable?
            yield lines
            yield '\n'

    def generate_atoms_text(self, atoms: Atoms):
        """Translate the Atoms object to fdf-format."""

        cell = atoms.cell
        yield '\n'

        if cell.rank in [1, 2]:
            raise ValueError('Expected 3D unit cell or no unit cell.  You may '
                             'wish to add vacuum along some directions.')

        if np.any(cell):
            yield var('LatticeConstant', '1.0 Ang')
            yield block('LatticeVectors', cell)

        yield from generate_atomic_coordinates(
            atoms, self.species_numbers, self.atomic_coord_format)

        # Write magnetic moments.
        magmoms = atoms.get_initial_magnetic_moments()

        # The DM.InitSpin block must be written to initialize to
        # no spin. SIESTA default is FM initialization, if the
        # block is not written, but  we must conform to the
        # atoms object.
        if len(magmoms) == 0:
            yield '#Empty block forces ASE initialization.\n'

        yield '%block DM.InitSpin\n'
        if len(magmoms) != 0 and isinstance(magmoms[0], np.ndarray):
            for n, M in enumerate(magmoms):
                if M[0] != 0:
                    yield ('    %d %.14f %.14f %.14f \n'
                           % (n + 1, M[0], M[1], M[2]))
        elif len(magmoms) != 0 and isinstance(magmoms[0], float):
            for n, M in enumerate(magmoms):
                if M != 0:
                    yield '    %d %.14f \n' % (n + 1, M)
        yield '%endblock DM.InitSpin\n'
        yield '\n'

    def link_pseudos_into_directory(self, *, symlink_pseudos=None, directory):
        if symlink_pseudos is None:
            symlink_pseudos = os.name != 'nt'

        for instruction in self.species_info.file_instructions:
            if symlink_pseudos:
                instruction.symlink_to(directory)
            else:
                instruction.copy_to(directory)


# Utilities for generating bits of strings.
#
# We are re-aliasing format_fdf and format_block in the anticipation
# that they may change, or we might move this onto a Formatter object
# which applies consistent spacings etc.
def var(key, value):
    return format_fdf(key, value)


def block(name, data):
    return format_block(name, data)
