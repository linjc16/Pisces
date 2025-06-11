# fmt: off

"""
The ASE Calculator for OpenMX <http://www.openmx-square.org>: Python interface
to the software package for nano-scale material simulations based on density
functional theories.
    Copyright (C) 2018 JaeHwan Shim and JaeJun Yu

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 2.1 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with ASE.  If not, see <http://www.gnu.org/licenses/>.
"""
import os

import numpy as np

from ase.calculators.calculator import kpts2sizeandoffsets
from ase.calculators.openmx import parameters as param
from ase.calculators.openmx.reader import (
    get_file_name,
    get_standard_key,
    read_electron_valency,
)
from ase.config import cfg
from ase.units import Bohr, Ha, Ry, fs, m, s

keys = [param.tuple_integer_keys, param.tuple_float_keys,
        param.tuple_bool_keys, param.integer_keys, param.float_keys,
        param.string_keys, param.bool_keys, param.list_int_keys,
        param.list_bool_keys, param.list_float_keys, param.matrix_keys]


def write_openmx(label=None, atoms=None, parameters=None, properties=None,
                 system_changes=None):
    """
    From atom image, 'images', write '.dat' file.
    First, set
    Write input (dat)-file.
    See calculator.py for further details.

    Parameters:
        - atoms        : The Atoms object to write.
        - properties   : The properties which should be calculated.
        - system_changes : List of properties changed since last run.
    """
    from ase.calculators.openmx import parameters as param
    filtered_keywords = parameters_to_keywords(label=label, atoms=atoms,
                                               parameters=parameters,
                                               properties=properties,
                                               system_changes=system_changes)
    keys = ['string', 'bool', 'integer', 'float',
            'tuple_integer', 'tuple_float', 'tuple_bool',
            'matrix', 'list_int', 'list_bool', 'list_float']
    # Start writing the file
    filename = get_file_name('.dat', label)
    with open(filename, 'w') as fd:
        # Write 1-line keywords
        for fltrd_keyword in filtered_keywords.keys():
            for key in keys:
                openmx_keywords = getattr(param, key + '_keys')
                write = globals()['write_' + key]
                for omx_keyword in openmx_keywords:
                    if fltrd_keyword == get_standard_key(omx_keyword):
                        write(fd, omx_keyword, filtered_keywords[fltrd_keyword])


def parameters_to_keywords(label=None, atoms=None, parameters=None,
                           properties=None, system_changes=None):
    """
    Before writing `label.dat` file, set up the ASE variables to OpenMX
    keywords. First, It initializes with given openmx keywords and reconstruct
    dictionary using standard parameters. If standard parameters and openmx
    keywords are contradict to each other, ignores openmx keyword.
     It includes,

    For aesthetical purpose, sequnece of writing input file is specified.
    """
    from collections import OrderedDict

    from ase.calculators.openmx.parameters import matrix_keys, unit_dat_keywords
    keywords = OrderedDict()
    sequence = [
        'system_currentdirectory', 'system_name', 'data_path',
        'level_of_fileout',
        'species_number', 'definition_of_atomic_species',
        'atoms_number', 'atoms_speciesandcoordinates_unit',
        'atoms_speciesandcoordinates', 'atoms_unitvectors_unit',
        'atoms_unitvectors', 'band_dispersion', 'band_nkpath',
        'band_kpath']

    _directory, prefix = os.path.split(label)
    curdir = os.path.join(os.getcwd(), prefix)
    counterparts = {
        'system_currentdirectory': curdir,
        'system_name': prefix,
        'data_path': cfg.get('OPENMX_DFT_DATA_PATH'),
        'species_number': len(get_species(atoms.get_chemical_symbols())),
        'atoms_number': len(atoms),
        'scf_restart': 'restart',
        'scf_maxiter': 'maxiter',
        'scf_xctype': 'xc',
        'scf_energycutoff': 'energy_cutoff',
        'scf_criterion': 'convergence',
        'scf_external_fields': 'external',
        'scf_mixing_type': 'mixer',
        'scf_electronic_temperature': 'smearing',
        'scf_system_charge': 'charge',
        'scf_eigenvaluesolver': 'eigensolver'
    }
    standard_units = {'eV': 1, 'Ha': Ha, 'Ry': Ry, 'Bohr': Bohr, 'fs': fs,
                      'K': 1, 'GV / m': 1e9 / 1.6e-19 / m,
                      'Ha/Bohr': Ha / Bohr,
                      'm/s': m / s, '_amu': 1, 'Tesla': 1}
    unit_dict = {get_standard_key(k): v for k, v in unit_dat_keywords.items()}

    for key in sequence:
        keywords[key] = None
    for key in parameters:
        if 'scf' in key:
            keywords[key] = None
    for key in parameters:
        if 'md' in key:
            keywords[key] = None

    # Initializes keywords to to given parameters
    for key in parameters.keys():
        keywords[key] = parameters[key]

    def parameter_overwrites(openmx_keyword):
        """
        In a situation conflicting ASE standard parameters and OpenMX keywords,
        ASE parameters overrides to OpenMX keywords. While doing so, units are
        converted to OpenMX unit.
        However, if both parameters and keyword are not given, we fill up that
        part in suitable manner
          openmx_keyword : key |  Name of key used in OpenMX
          keyword : value | value corresponds to openmx_keyword
          ase_parameter : key | Name of parameter used in ASE
          parameter : value | value corresponds to ase_parameter
        """
        ase_parameter = counterparts[openmx_keyword]
        keyword = parameters.get(openmx_keyword)
        parameter = parameters.get(ase_parameter)
        if parameter is not None:
            # Handles the unit
            unit = standard_units.get(unit_dict.get(openmx_keyword))
            if unit is not None:
                return parameter / unit
            return parameter
        elif keyword is not None:
            return keyword
        elif 'scf' in openmx_keyword:
            return None
        else:
            return counterparts[openmx_keyword]

    # Overwrites openmx keyword using standard parameters
    for openmx_keyword in counterparts:
        keywords[openmx_keyword] = parameter_overwrites(openmx_keyword)

    # keywords['scf_stress_tensor'] = 'stress' in properties
    # This is not working due to the UnitCellFilter method.
    if 'energies' in properties:
        keywords['energy_decomposition'] = True
    if 'stress' in properties:
        keywords['scf_stress_tensor'] = True

    keywords['scf_xctype'] = get_xc(keywords['scf_xctype'])
    keywords['scf_kgrid'] = get_scf_kgrid(atoms, parameters)
    keywords['scf_spinpolarization'] = get_spinpol(atoms, parameters)

    if parameters.get('band_kpath') is not None:
        keywords['band_dispersion'] = True
    keywords['band_kpath'] = parameters.get('band_kpath')
    if parameters.get('band_nkpath') is not None:
        keywords['band_nkpath'] = len(keywords['band_kpath'])

    # Set up Wannier Environment
    if parameters.get('wannier_func_calc') is not None:
        keywords['species_number'] *= 2

    # Set up some parameters for the later use
    parameters['_xc'] = keywords['scf_xctype']
    parameters['_data_path'] = keywords['data_path']
    parameters['_year'] = get_dft_data_year(parameters)

    # Set up the matrix-type OpenMX keywords
    for key in matrix_keys:
        get_matrix_key = globals()['get_' + get_standard_key(key)]
        keywords[get_standard_key(key)] = get_matrix_key(atoms, parameters)
    return OrderedDict([(k, v)for k, v in keywords.items()
                        if not (v is None or
                                (isinstance(v, list) and v == []))])


def get_species(symbols):
    species = []
    [species.append(s) for s in symbols if s not in species]
    return species


def get_xc(xc):
    """
    Change the name of xc appropriate to OpenMX format
    """
    xc = xc.upper()
    assert xc.upper() in param.OpenMXParameters().allowed_xc
    if xc in ['PBE', 'GGA', 'GGA-PBE']:
        return 'GGA-PBE'
    elif xc in ['LDA']:
        return 'LDA'
    elif xc in ['CA', 'PW']:
        return 'LSDA-' + xc
    elif xc in ['LSDA', 'LSDA-CA']:
        return 'LSDA-CA'
    elif xc in ['LSDA-PW']:
        return 'LSDA-PW'
    else:
        return 'LDA'


def get_vps(xc):
    if xc in ['GGA-PBE']:
        return 'PBE'
    else:
        return 'CA'


def get_scf_kgrid(atoms, parameters):
    kpts, scf_kgrid = parameters.get('kpts'), parameters.get('scf_kgrid')
    if isinstance(kpts, (tuple, list, np.ndarray)) and len(
            kpts) == 3 and isinstance(kpts[0], int):
        return kpts
    elif isinstance(kpts, (float, int)):
        return tuple(kpts2sizeandoffsets(atoms=atoms, density=kpts)[0])
    else:
        return scf_kgrid


def get_definition_of_atomic_species(atoms, parameters):
    """
    Using atoms and parameters, Returns the list `definition_of_atomic_species`
    where matrix of strings contains the information between keywords.
    For example,
     definition_of_atomic_species =
         [['H','H5.0-s1>1p1>1','H_CA13'],
          ['C','C5.0-s1>1p1>1','C_CA13']]
    Goes to,
      <Definition.of.Atomic.Species
        H   H5.0-s1>1p1>1      H_CA13
        C   C5.0-s1>1p1>1      C_CA13
      Definition.of.Atomic.Species>
    Further more, you can specify the wannier information here.
    A. Define local functions for projectors
      Since the pseudo-atomic orbitals are used for projectors,
      the specification of them is the same as for the basis functions.
      An example setting, for silicon in diamond structure, is as following:
   Species.Number          2
      <Definition.of.Atomic.Species
        Si       Si7.0-s2p2d1    Si_CA13
        proj1    Si5.5-s1p1d1f1  Si_CA13
      Definition.of.Atomic.Species>
    """
    if parameters.get('definition_of_atomic_species') is not None:
        return parameters['definition_of_atomic_species']

    definition_of_atomic_species = []
    xc = parameters.get('_xc')
    year = parameters.get('_year')

    chem = atoms.get_chemical_symbols()
    species = get_species(chem)
    for element in species:
        rad_orb = get_cutoff_radius_and_orbital(element=element)
        suffix = get_pseudo_potential_suffix(element=element, xc=xc, year=year)
        definition_of_atomic_species.append([element, rad_orb, suffix])
    # Put the same orbital and radii with chemical symbol.
    wannier_projectors = parameters.get('definition_of_wannier_projectors', [])
    for i, projector in enumerate(wannier_projectors):
        full_projector = definition_of_atomic_species[i]
        full_projector[0] = projector
        definition_of_atomic_species.append(full_projector)
    return definition_of_atomic_species


def get_dft_data_year(parameters):
    """
    It seems there is no version or potential year checker in openmx, thus we
    implemented one. It parse the pesudo potential path variable such as
    `~/PATH/TO/OPENMX/openmx3.9/DFT_DATA19/` or `.../openmx3.8/DFT_DATA13/`.
    By spliting this string, we harness the number of the year that generated
    such pseudo potential path.
    """
    if parameters.get('dft_data_year') is not None:
        return str(parameters.get('dft_data_year'))
    data_path = parameters['_data_path']
    year = data_path.split('DFT_DATA')[1][:2]
    if year is not None:
        return year
    else:
        raise ValueError('DFT_DATA year can not be found. Please specify '
                         '`dft_data_year` as year of pseudo potential relesed')


def get_cutoff_radius_and_orbital(element=None, orbital=None):
    """
    For a given element, retruns the string specifying cutoff radius and
    orbital using default_settings.py. For example,
       'Si'   ->   'Si.7.0-s2p2d1'
    If one wannts to change the atomic radius for a special purpose, one should
    change the default_settings.py directly.
    """
    from ase.calculators.openmx import default_settings
    orbital = element
    orbital_letters = ['s', 'p', 'd', 'f', 'g', 'h']
    default_dictionary = default_settings.default_dictionary
    orbital_numbers = default_dictionary[element]['orbitals used']
    cutoff_radius = default_dictionary[element]['cutoff radius']
    orbital += "%.1f" % float(cutoff_radius) + '-'
    for i, orbital_number in enumerate(orbital_numbers):
        orbital += orbital_letters[i] + str(orbital_number)
    return orbital


def get_pseudo_potential_suffix(element=None, xc=None, year='13'):
    """
    For a given element, returns the string specifying pseudo potential suffix.
    For example,
        'Si'   ->   'Si_CA13'
    or
        'Si'   ->   'Si_CA19'
    depending on pseudo potential generation year
    """
    from ase.calculators.openmx import default_settings
    default_dictionary = default_settings.default_dictionary
    pseudo_potential_suffix = element
    vps = get_vps(xc)
    suffix = default_dictionary[element]['pseudo-potential suffix']
    pseudo_potential_suffix += '_' + vps + year + suffix
    return pseudo_potential_suffix


def get_atoms_speciesandcoordinates(atoms, parameters):
    """
    The atomic coordinates and the number of spin charge are given by the
    keyword
    'Atoms.SpeciesAndCoordinates' as follows:
    <Atoms.SpeciesAndCoordinates
     1  Mn    0.00000   0.00000   0.00000   8.0  5.0  45.0 0.0 45.0 0.0  1 on
     2  O     1.70000   0.00000   0.00000   3.0  3.0  45.0 0.0 45.0 0.0  1 on
    Atoms.SpeciesAndCoordinates>
    to know more, link <http://www.openmx-square.org/openmx_man3.7/node85.html>
    """
    atoms_speciesandcoordinates = []
    xc = parameters.get('_xc')
    year = parameters.get('_year')
    data_pth = parameters.get('_data_path')
    # Appending number and elemental symbol
    elements = atoms.get_chemical_symbols()
    for i, element in enumerate(elements):
        atoms_speciesandcoordinates.append([str(i + 1), element])
    # Appending positions
    unit = parameters.get('atoms_speciesandcoordinates_unit', 'ang').lower()
    if unit == 'ang':
        positions = atoms.get_positions()
    elif unit == 'frac':
        positions = atoms.get_scaled_positions(wrap=False)
    elif unit == 'au':
        positions = atoms.get_positions() / Bohr
    for i, position in enumerate(positions):
        atoms_speciesandcoordinates[i].extend(position)

    # Even if 'atoms_speciesandcoordinates_unit' exists, `positions` goes first
    if parameters.get('atoms_speciesandcoordinates') is not None:
        atoms_spncrd = parameters['atoms_speciesandcoordinates'].copy()
        for i in range(len(atoms)):
            atoms_spncrd[i][2] = atoms_speciesandcoordinates[i][2]
            atoms_spncrd[i][3] = atoms_speciesandcoordinates[i][3]
            atoms_spncrd[i][4] = atoms_speciesandcoordinates[i][4]
        return atoms_spncrd

    # Appending magnetic moment
    magmoms = atoms.get_initial_magnetic_moments()
    for i, magmom in enumerate(magmoms):
        up_down_spin = get_up_down_spin(magmom, elements[i], xc, data_pth, year)
        atoms_speciesandcoordinates[i].extend(up_down_spin)
    # Appending magnetic field Spin magnetic moment theta phi
    spin_directions = get_spin_direction(magmoms)
    for i, spin_direction in enumerate(spin_directions):
        atoms_speciesandcoordinates[i].extend(spin_direction)
    # Appending magnetic field for Orbital magnetic moment theta phi
    orbital_directions = get_orbital_direction()
    for i, orbital_direction in enumerate(orbital_directions):
        atoms_speciesandcoordinates[i].extend(orbital_direction)
    # Appending Noncolinear schem switch
    noncollinear_switches = get_noncollinear_switches()
    for i, noncollinear_switch in enumerate(noncollinear_switches):
        atoms_speciesandcoordinates[i].extend(noncollinear_switch)
    # Appending orbital_enhancement_switch
    lda_u_switches = get_lda_u_switches()
    for i, lda_u_switch in enumerate(lda_u_switches):
        atoms_speciesandcoordinates[i].extend(lda_u_switch)
    return atoms_speciesandcoordinates


def get_up_down_spin(magmom, element, xc, data_path, year):
    # for magmom with single number (collinear spin) skip  the normalization
    if isinstance(magmom, (int, float)):
        # Collinear spin
        magmom = float(magmom)
    else:
        # Non-collinear spin
        magmom = np.linalg.norm(magmom)
    suffix = get_pseudo_potential_suffix(element, xc, year)
    filename = os.path.join(data_path, 'VPS/' + suffix + '.vps')
    valence_electron = float(read_electron_valency(filename))
    return [valence_electron / 2 + magmom / 2,
            valence_electron / 2 - magmom / 2]


def get_spin_direction(magmoms):
    '''
    From atoms.magmom, returns the spin direction of phi and theta
    '''
    if np.array(magmoms).dtype == float or \
       np.array(magmoms).dtype is np.float64:
        return []
    else:
        magmoms = np.array(magmoms)
        return magmoms / np.linalg.norm(magmoms, axis=1)


def get_orbital_direction():
    orbital_direction = []
    # print("Not Implemented Yet")
    return orbital_direction


def get_noncollinear_switches():
    noncolinear_switches = []
    # print("Not Implemented Yet")
    return noncolinear_switches


def get_lda_u_switches():
    lda_u_switches = []
    # print("Not Implemented Yet")
    return lda_u_switches


def get_spinpol(atoms, parameters):
    ''' Judgeds the keyword 'scf.SpinPolarization'
     If the keyword is not None, spinpol gets the keyword by following priority
       1. standard_spinpol
       2. scf_spinpolarization
       3. magnetic moments of atoms
    '''
    standard_spinpol = parameters.get('spinpol', None)
    scf_spinpolarization = parameters.get('scf_spinpolarization', None)
    m = atoms.get_initial_magnetic_moments()
    syn = {True: 'On', False: None, 'on': 'On', 'off': None,
           None: None, 'nc': 'NC'}
    spinpol = np.any(m >= 0.1)
    if scf_spinpolarization is not None:
        spinpol = scf_spinpolarization
    if standard_spinpol is not None:
        spinpol = standard_spinpol
    if isinstance(spinpol, str):
        spinpol = spinpol.lower()
    return syn[spinpol]


def get_atoms_unitvectors(atoms, parameters):
    zero_vec = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    if np.all(atoms.get_cell() == zero_vec) is True:
        default_cell = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        return parameters.get('atoms_unitvectors', default_cell)
    unit = parameters.get('atoms_unitvectors_unit', 'ang').lower()
    if unit == 'ang':
        atoms_unitvectors = atoms.get_cell()
    elif unit == 'au':
        atoms_unitvectors = atoms.get_cell() / Bohr
    return atoms_unitvectors


def get_hubbard_u_values(atoms, parameters):
    return parameters.get('hubbard_u_values', [])


def get_atoms_cont_orbitals(atoms, parameters):
    return parameters.get('atoms_cont_orbitals', [])


def get_md_fixed_xyz(atoms, parameters):
    return parameters.get('md_fixed_xyz', [])


def get_md_tempcontrol(atoms, parameters):
    return parameters.get('md_tempcontrol', [])


def get_md_init_velocity(atoms, parameters):
    return parameters.get('md_init_velocity', [])


def get_band_kpath_unitcell(atoms, parameters):
    return parameters.get('band_kpath_unitcell', [])


def get_band_kpath(atoms, parameters):
    kpts = parameters.get('kpts')
    if isinstance(kpts, list) and len(kpts) > 3:
        return get_kpath(kpts=kpts)
    else:
        return parameters.get('band_kpath', [])


def get_mo_kpoint(atoms, parameters):
    return parameters.get('mo_kpoint', [])


def get_wannier_initial_projectors(atoms, parameters):
    """
    B. Specify the orbital, central position and orientation of a projector
    Wannier.Initial.Projectos will be used to specify the projector name,
    local orbital function, center of local orbital, and the local z-axis and
    x-axis for orbital orientation.

    An example setting is shown here:
    wannier_initial_projectors=
    [['proj1-sp3','0.250','0.250','0.25','-1.0','0.0','0.0','0.0','0.0','-1.0']
    ,['proj1-sp3','0.000','0.000','0.00','0.0','0.0','1.0','1.0','0.0','0.0']]
    Goes to,
        <Wannier.Initial.Projectors
           proj1-sp3   0.250  0.250  0.250   -1.0 0.0 0.0    0.0  0.0 -1.0
           proj1-sp3   0.000  0.000  0.000    0.0 0.0 1.0    1.0  0.0  0.0
        Wannier.Initial.Projectors>
    """
    return parameters.get('wannier_initial_projectors', [])


def get_kpath(self, kpts=None, symbols=None, band_kpath=None, eps=1e-5):
    """
    Convert band_kpath <-> kpts. Symbols will be guess automatically
    by using dft space group method
    For example,
    kpts  = [(0, 0, 0), (0.125, 0, 0) ... (0.875, 0, 0),
             (1, 0, 0), (1, 0.0625, 0) .. (1, 0.4375,0),
             (1, 0.5,0),(0.9375, 0.5,0).. (    ...    ),
             (0.5, 0.5, 0.5) ...               ...     ,
                ...          ...               ...     ,
                ...        (0.875, 0, 0),(1.0, 0.0, 0.0)]
    band_kpath =
    [['15','0.0','0.0','0.0','1.0','0.0','0.0','g','X'],
     ['15','1.0','0.0','0.0','1.0','0.5','0.0','X','W'],
     ['15','1.0','0.5','0.0','0.5','0.5','0.5','W','L'],
     ['15','0.5','0.5','0.5','0.0','0.0','0.0','L','g'],
     ['15','0.0','0.0','0.0','1.0','0.0','0.0','g','X']]
    where, it will be written as
     <Band.kpath
      15  0.0 0.0 0.0   1.0 0.0 0.0   g X
      15  1.0 0.0 0.0   1.0 0.5 0.0   X W
      15  1.0 0.5 0.0   0.5 0.5 0.5   W L
      15  0.5 0.5 0.5   0.0 0.0 0.0   L g
      15  0.0 0.0 0.0   1.0 0.0 0.0   g X
     Band.kpath>
    """
    if kpts is None:
        kx_linspace = np.linspace(band_kpath[0]['start_point'][0],
                                  band_kpath[0]['end_point'][0],
                                  band_kpath[0][0])
        ky_linspace = np.linspace(band_kpath[0]['start_point'][1],
                                  band_kpath[0]['end_point'][1],
                                  band_kpath[0]['kpts'])
        kz_linspace = np.linspace(band_kpath[0]['start_point'][2],
                                  band_kpath[0]['end_point'][2],
                                  band_kpath[0]['kpts'])
        kpts = np.array([kx_linspace, ky_linspace, kz_linspace]).T
        for path in band_kpath[1:]:
            kx_linspace = np.linspace(path['start_point'][0],
                                      path['end_point'][0],
                                      path['kpts'])
            ky_linspace = np.linspace(path['start_point'][1],
                                      path['end_point'][1],
                                      path['kpts'])
            kz_linspace = np.linspace(path['start_point'][2],
                                      path['end_point'][2],
                                      path['kpts'])
            k_lin = np.array([kx_linspace, ky_linspace, kz_linspace]).T
            kpts = np.append(kpts, k_lin, axis=0)
        return kpts
    elif band_kpath is None:
        band_kpath = []
        points = np.asarray(kpts)
        diffs = points[1:] - points[:-1]
        kinks = abs(diffs[1:] - diffs[:-1]).sum(1) > eps
        N = len(points)
        indices = [0]
        indices.extend(np.arange(1, N - 1)[kinks])
        indices.append(N - 1)
        for start, end, s_sym, e_sym in zip(indices[1:], indices[:-1],
                                            symbols[1:], symbols[:-1]):
            band_kpath.append({'start_point': start, 'end_point': end,
                               'kpts': 20,
                               'path_symbols': (s_sym, e_sym)})
    else:
        raise KeyError('You should specify band_kpath or kpts')
        return band_kpath


def write_string(fd, key, value):
    fd.write("        ".join([key, value]))
    fd.write("\n")


def write_tuple_integer(fd, key, value):
    fd.write("        ".join([key, "%d %d %d" % value]))
    fd.write("\n")


def write_tuple_float(fd, key, value):
    fd.write("        ".join([key, "%.4f %.4f %.4f" % value]))
    fd.write("\n")


def write_tuple_bool(fd, key, value):
    omx_bl = {True: 'On', False: 'Off'}
    fd.write("        ".join([key, "%s %s %s" % [omx_bl[bl] for bl in value]]))
    fd.write("\n")


def write_integer(fd, key, value):
    fd.write("        ".join([key, "%d" % value]))
    fd.write("\n")


def write_float(fd, key, value):
    fd.write("        ".join([key, "%.8g" % value]))
    fd.write("\n")


def write_bool(fd, key, value):
    omx_bl = {True: 'On', False: 'Off'}
    fd.write("        ".join([key, f"{omx_bl[value]}"]))
    fd.write("\n")


def write_list_int(fd, key, value):
    fd.write("".join(key) + ' ' + "     ".join(map(str, value)))


def write_list_bool(fd, key, value):
    omx_bl = {True: 'On', False: 'Off'}
    fd.write("".join(key) + ' ' + "     ".join([omx_bl[bl] for bl in value]))


def write_list_float(fd, key, value):
    fd.write("".join(key) + ' ' + "     ".join(map(str, value)))


def write_matrix(fd, key, value):
    fd.write('<' + key)
    fd.write("\n")
    for line in value:
        fd.write("    " + "  ".join(map(str, line)))
        fd.write("\n")
    fd.write(key + '>')
    fd.write("\n\n")


def get_openmx_key(key):
    """
    For the writing purpose, we need to know Original OpenMX keyword format.
    By comparing keys in the parameters.py, restore the original key
    """
    for openmx_key in keys:
        for openmx_keyword in openmx_key:
            if key == get_standard_key(openmx_keyword):
                return openmx_keyword
