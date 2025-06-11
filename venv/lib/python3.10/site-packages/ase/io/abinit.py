# fmt: off

import os
import re
from glob import glob
from os.path import join
from pathlib import Path

import numpy as np

from ase import Atoms
from ase.calculators.calculator import all_properties
from ase.calculators.singlepoint import SinglePointCalculator
from ase.data import chemical_symbols
from ase.units import Bohr, Hartree, fs


def read_abinit_in(fd):
    """Import ABINIT input file.

    Reads cell, atom positions, etc. from abinit input file
    """

    tokens = []

    for line in fd:
        meat = line.split('#', 1)[0]
        tokens += meat.lower().split()

    # note that the file can not be scanned sequentially

    index = tokens.index("acell")
    unit = 1.0
    if tokens[index + 4].lower()[:3] != 'ang':
        unit = Bohr
    acell = [unit * float(tokens[index + 1]),
             unit * float(tokens[index + 2]),
             unit * float(tokens[index + 3])]

    index = tokens.index("natom")
    natom = int(tokens[index + 1])

    index = tokens.index("ntypat")
    ntypat = int(tokens[index + 1])

    index = tokens.index("typat")
    typat = []
    while len(typat) < natom:
        token = tokens[index + 1]
        if '*' in token:  # e.g. typat 4*1 3*2 ...
            nrepeat, typenum = token.split('*')
            typat += [int(typenum)] * int(nrepeat)
        else:
            typat.append(int(token))
        index += 1
    assert natom == len(typat)

    index = tokens.index("znucl")
    znucl = []
    for i in range(ntypat):
        znucl.append(int(tokens[index + 1 + i]))

    index = tokens.index("rprim")
    rprim = []
    for i in range(3):
        rprim.append([acell[i] * float(tokens[index + 3 * i + 1]),
                      acell[i] * float(tokens[index + 3 * i + 2]),
                      acell[i] * float(tokens[index + 3 * i + 3])])

    # create a list with the atomic numbers
    numbers = []
    for i in range(natom):
        ii = typat[i] - 1
        numbers.append(znucl[ii])

    # now the positions of the atoms
    if "xred" in tokens:
        index = tokens.index("xred")
        xred = []
        for i in range(natom):
            xred.append([float(tokens[index + 3 * i + 1]),
                         float(tokens[index + 3 * i + 2]),
                         float(tokens[index + 3 * i + 3])])
        atoms = Atoms(cell=rprim, scaled_positions=xred, numbers=numbers,
                      pbc=True)
    else:
        if "xcart" in tokens:
            index = tokens.index("xcart")
            unit = Bohr
        elif "xangst" in tokens:
            unit = 1.0
            index = tokens.index("xangst")
        else:
            raise OSError(
                "No xred, xcart, or xangs keyword in abinit input file")

        xangs = []
        for i in range(natom):
            xangs.append([unit * float(tokens[index + 3 * i + 1]),
                          unit * float(tokens[index + 3 * i + 2]),
                          unit * float(tokens[index + 3 * i + 3])])
        atoms = Atoms(cell=rprim, positions=xangs, numbers=numbers, pbc=True)

    try:
        ii = tokens.index('nsppol')
    except ValueError:
        nsppol = None
    else:
        nsppol = int(tokens[ii + 1])

    if nsppol == 2:
        index = tokens.index('spinat')
        magmoms = [float(tokens[index + 3 * i + 3]) for i in range(natom)]
        atoms.set_initial_magnetic_moments(magmoms)

    assert len(atoms) == natom
    return atoms


keys_with_units = {
    'toldfe': 'eV',
    'tsmear': 'eV',
    'paoenergyshift': 'eV',
    'zmunitslength': 'Bohr',
    'zmunitsangle': 'rad',
    'zmforcetollength': 'eV/Ang',
    'zmforcetolangle': 'eV/rad',
    'zmmaxdispllength': 'Ang',
    'zmmaxdisplangle': 'rad',
    'ecut': 'eV',
    'pawecutdg': 'eV',
    'dmenergytolerance': 'eV',
    'electronictemperature': 'eV',
    'oneta': 'eV',
    'onetaalpha': 'eV',
    'onetabeta': 'eV',
    'onrclwf': 'Ang',
    'onchemicalpotentialrc': 'Ang',
    'onchemicalpotentialtemperature': 'eV',
    'mdmaxcgdispl': 'Ang',
    'mdmaxforcetol': 'eV/Ang',
    'mdmaxstresstol': 'eV/Ang**3',
    'mdlengthtimestep': 'fs',
    'mdinitialtemperature': 'eV',
    'mdtargettemperature': 'eV',
    'mdtargetpressure': 'eV/Ang**3',
    'mdnosemass': 'eV*fs**2',
    'mdparrinellorahmanmass': 'eV*fs**2',
    'mdtaurelax': 'fs',
    'mdbulkmodulus': 'eV/Ang**3',
    'mdfcdispl': 'Ang',
    'warningminimumatomicdistance': 'Ang',
    'rcspatial': 'Ang',
    'kgridcutoff': 'Ang',
    'latticeconstant': 'Ang'}


def write_abinit_in(fd, atoms, param=None, species=None, pseudos=None):
    from ase.calculators.calculator import kpts2mp

    if param is None:
        param = {}

    if species is None:
        species = sorted(set(atoms.numbers))

    inp = dict(param)
    xc = inp.pop('xc', 'LDA')
    for key in ['smearing', 'kpts', 'pps', 'raw']:
        inp.pop(key, None)

    smearing = param.get('smearing')
    if 'tsmear' in param or 'occopt' in param:
        assert smearing is None

    if smearing is not None:
        inp['occopt'] = {'fermi-dirac': 3,
                         'gaussian': 7}[smearing[0].lower()]
        inp['tsmear'] = smearing[1]

    inp['natom'] = len(atoms)

    if 'nbands' in param:
        inp['nband'] = param['nbands']
        del inp['nbands']

    # ixc is set from paw/xml file. Ignore 'xc' setting then.
    if param.get('pps') not in ['pawxml']:
        if 'ixc' not in param:
            inp['ixc'] = {'LDA': 7,
                          'PBE': 11,
                          'revPBE': 14,
                          'RPBE': 15,
                          'WC': 23}[xc]

    magmoms = atoms.get_initial_magnetic_moments()
    if magmoms.any():
        inp['nsppol'] = 2
        fd.write('spinat\n')
        for n, M in enumerate(magmoms):
            fd.write(f'{0:.14f} {0:.14f} {M:.14f}\n')
    else:
        inp['nsppol'] = 1

    if param.get('kpts') is not None:
        mp = kpts2mp(atoms, param['kpts'])
        fd.write('kptopt 1\n')
        fd.write('ngkpt %d %d %d\n' % tuple(mp))
        fd.write('nshiftk 1\n')
        fd.write('shiftk\n')
        fd.write('%.1f %.1f %.1f\n' % tuple((np.array(mp) + 1) % 2 * 0.5))

    valid_lists = (list, np.ndarray)
    for key in sorted(inp):
        value = inp[key]
        unit = keys_with_units.get(key)
        if unit is not None:
            if 'fs**2' in unit:
                value /= fs**2
            elif 'fs' in unit:
                value /= fs
        if isinstance(value, valid_lists):
            if isinstance(value[0], valid_lists):
                fd.write(f"{key}\n")
                for dim in value:
                    write_list(fd, dim, unit)
            else:
                fd.write(f"{key}\n")
                write_list(fd, value, unit)
        else:
            if unit is None:
                fd.write(f"{key} {value}\n")
            else:
                fd.write(f"{key} {value} {unit}\n")

    if param.get('raw') is not None:
        if isinstance(param['raw'], str):
            raise TypeError('The raw parameter is a single string; expected '
                            'a sequence of lines')
        for line in param['raw']:
            if isinstance(line, tuple):
                fd.write(' '.join([f'{x}' for x in line]) + '\n')
            else:
                fd.write(f'{line}\n')

    fd.write('#Definition of the unit cell\n')
    fd.write('acell\n')
    fd.write(f'{1.0:.14f} {1.0:.14f} {1.0:.14f} Angstrom\n')
    fd.write('rprim\n')
    if atoms.cell.rank != 3:
        raise RuntimeError('Abinit requires a 3D cell, but cell is {}'
                           .format(atoms.cell))
    for v in atoms.cell:
        fd.write('%.14f %.14f %.14f\n' % tuple(v))

    fd.write('chkprim 0 # Allow non-primitive cells\n')

    fd.write('#Definition of the atom types\n')
    fd.write('ntypat %d\n' % (len(species)))
    fd.write('znucl {}\n'.format(' '.join(str(Z) for Z in species)))
    fd.write('#Enumerate different atomic species\n')
    fd.write('typat')
    fd.write('\n')

    types = []
    for Z in atoms.numbers:
        for n, Zs in enumerate(species):
            if Z == Zs:
                types.append(n + 1)
    n_entries_int = 20  # integer entries per line
    for n, type in enumerate(types):
        fd.write(' %d' % (type))
        if n > 1 and ((n % n_entries_int) == 1):
            fd.write('\n')
    fd.write('\n')

    if pseudos is not None:
        listing = ',\n'.join(pseudos)
        line = f'pseudos "{listing}"\n'
        fd.write(line)

    fd.write('#Definition of the atoms\n')
    fd.write('xcart\n')
    for pos in atoms.positions / Bohr:
        fd.write('%.14f %.14f %.14f\n' % tuple(pos))

    fd.write('chkexit 1 # abinit.exit file in the running '
             'directory terminates after the current SCF\n')


def write_list(fd, value, unit):
    for element in value:
        fd.write(f"{element} ")
    if unit is not None:
        fd.write(f"{unit}")
    fd.write("\n")


def read_stress(fd):
    # sigma(1 1)=  4.02063464E-04  sigma(3 2)=  0.00000000E+00
    # sigma(2 2)=  4.02063464E-04  sigma(3 1)=  0.00000000E+00
    # sigma(3 3)=  4.02063464E-04  sigma(2 1)=  0.00000000E+00
    pat = re.compile(r'\s*sigma\(\d \d\)=\s*(\S+)\s*sigma\(\d \d\)=\s*(\S+)')
    stress = np.empty(6)
    for i in range(3):
        line = next(fd)
        m = pat.match(line)
        if m is None:
            # Not a real value error.  What should we raise?
            raise ValueError('Line {!r} does not match stress pattern {!r}'
                             .format(line, pat))
        stress[i] = float(m.group(1))
        stress[i + 3] = float(m.group(2))
    unit = Hartree / Bohr**3
    return stress * unit


def consume_multiline(fd, headerline, nvalues, dtype):
    """Parse abinit-formatted "header + values" sections.

    Example:

        typat 1 1 1 1 1
              1 1 1 1
    """
    tokens = headerline.split()
    assert tokens[0].isalpha()

    values = tokens[1:]
    while len(values) < nvalues:
        line = next(fd)
        values.extend(line.split())
    assert len(values) == nvalues
    return np.array(values).astype(dtype)


def read_abinit_out(fd):
    results = {}

    def skipto(string):
        for line in fd:
            if string in line:
                return line
        raise RuntimeError(f'Not found: {string}')

    line = skipto('Version')
    m = re.match(r'\.*?Version\s+(\S+)\s+of ABINIT', line)
    assert m is not None
    version = m.group(1)
    results['version'] = version

    use_v9_format = int(version.split('.', 1)[0]) >= 9

    shape_vars = {}

    skipto('echo values of preprocessed input variables')

    for line in fd:
        if '===============' in line:
            break

        tokens = line.split()
        if not tokens:
            continue

        for key in ['natom', 'nkpt', 'nband', 'ntypat']:
            if tokens[0] == key:
                shape_vars[key] = int(tokens[1])

        if line.lstrip().startswith('typat'):  # Avoid matching ntypat
            types = consume_multiline(fd, line, shape_vars['natom'], int)

        if 'znucl' in line:
            znucl = consume_multiline(fd, line, shape_vars['ntypat'], float)

        if 'rprim' in line:
            cell = consume_multiline(fd, line, 9, float)
            cell = cell.reshape(3, 3)

    natoms = shape_vars['natom']

    # Skip ahead to results:
    for line in fd:
        if 'was not enough scf cycles to converge' in line:
            raise RuntimeError(line)
        if 'iterations are completed or convergence reached' in line:
            break
    else:
        raise RuntimeError('Cannot find results section')

    def read_array(fd, nlines):
        arr = []
        for i in range(nlines):
            line = next(fd)
            arr.append(line.split()[1:])
        arr = np.array(arr).astype(float)
        return arr

    if use_v9_format:
        energy_header = '--- !EnergyTerms'
        total_energy_name = 'total_energy_eV'

        def parse_energy(line):
            return float(line.split(':')[1].strip())
    else:
        energy_header = 'Components of total free energy (in Hartree) :'
        total_energy_name = '>>>>>>>>> Etotal'

        def parse_energy(line):
            return float(line.rsplit('=', 2)[1]) * Hartree

    for line in fd:
        if 'cartesian coordinates (angstrom) at end' in line:
            positions = read_array(fd, natoms)
        if 'cartesian forces (eV/Angstrom) at end' in line:
            results['forces'] = read_array(fd, natoms)
        if 'Cartesian components of stress tensor (hartree/bohr^3)' in line:
            results['stress'] = read_stress(fd)

        if line.strip() == energy_header:
            # Header not to be confused with EnergyTermsDC,
            # therefore we don't use .startswith()
            energy = None
            for line in fd:
                # Which of the listed energies should we include?
                if total_energy_name in line:
                    energy = parse_energy(line)
                    break
            if energy is None:
                raise RuntimeError('No energy found in output')
            results['energy'] = results['free_energy'] = energy

        if 'END DATASET(S)' in line:
            break

    znucl_int = znucl.astype(int)
    znucl_int[znucl_int != znucl] = 0  # (Fractional Z)
    numbers = znucl_int[types - 1]

    atoms = Atoms(numbers=numbers,
                  positions=positions,
                  cell=cell,
                  pbc=True)

    calc_results = {name: results[name] for name in
                    set(all_properties) & set(results)}
    atoms.calc = SinglePointCalculator(atoms,
                                       **calc_results)
    atoms.calc.name = "abinit"

    results['atoms'] = atoms
    return results


def match_kpt_header(line):
    headerpattern = (r'\s*kpt#\s*\S+\s*'
                     r'nband=\s*(\d+),\s*'
                     r'wtk=\s*(\S+?),\s*'
                     r'kpt=\s*(\S+)+\s*(\S+)\s*(\S+)')
    m = re.match(headerpattern, line)
    assert m is not None, line
    nbands = int(m.group(1))
    weight = float(m.group(2))
    kvector = np.array(m.group(3, 4, 5)).astype(float)
    return nbands, weight, kvector


def read_eigenvalues_for_one_spin(fd, nkpts):

    kpoint_weights = []
    kpoint_coords = []

    eig_kn = []
    for _ in range(nkpts):
        header = next(fd)
        nbands, weight, kvector = match_kpt_header(header)
        kpoint_coords.append(kvector)
        kpoint_weights.append(weight)

        eig_n = []
        while len(eig_n) < nbands:
            line = next(fd)
            tokens = line.split()
            values = np.array(tokens).astype(float) * Hartree
            eig_n.extend(values)
        assert len(eig_n) == nbands
        eig_kn.append(eig_n)
        assert nbands == len(eig_kn[0])

    kpoint_weights = np.array(kpoint_weights)
    kpoint_coords = np.array(kpoint_coords)
    eig_kn = np.array(eig_kn)
    return kpoint_coords, kpoint_weights, eig_kn


def read_eig(fd):
    line = next(fd)
    results = {}
    m = re.match(r'\s*Fermi \(or HOMO\) energy \(hartree\)\s*=\s*(\S+)', line)
    if m is not None:
        results['fermilevel'] = float(m.group(1)) * Hartree
        line = next(fd)

    nspins = 1

    m = re.match(r'\s*Magnetization \(Bohr magneton\)=\s*(\S+)', line)
    if m is not None:
        nspins = 2
        magmom = float(m.group(1))
        results['magmom'] = magmom
        line = next(fd)

    if 'Total spin up' in line:
        assert nspins == 2
        line = next(fd)

    m = re.match(r'\s*Eigenvalues \(hartree\) for nkpt\s*='
                 r'\s*(\S+)\s*k\s*points', line)
    if 'SPIN' in line or 'spin' in line:
        # If using spinpol with fixed magmoms, we don't get the magmoms
        # listed before now.
        nspins = 2
    assert m is not None
    nkpts = int(m.group(1))

    eig_skn = []

    kpts, weights, eig_kn = read_eigenvalues_for_one_spin(fd, nkpts)
    nbands = eig_kn.shape[1]

    eig_skn.append(eig_kn)
    if nspins == 2:
        line = next(fd)
        assert 'SPIN DOWN' in line
        _, _, eig_kn = read_eigenvalues_for_one_spin(fd, nkpts)
        assert eig_kn.shape == (nkpts, nbands)
        eig_skn.append(eig_kn)
    eig_skn = np.array(eig_skn)

    eigshape = (nspins, nkpts, nbands)
    assert eig_skn.shape == eigshape, (eig_skn.shape, eigshape)

    results['ibz_kpoints'] = kpts
    results['kpoint_weights'] = weights
    results['eigenvalues'] = eig_skn
    return results


def get_default_abinit_pp_paths():
    return os.environ.get('ABINIT_PP_PATH', '.').split(':')


def prepare_abinit_input(directory, atoms, properties, parameters,
                         pp_paths=None,
                         raise_exception=True):
    directory = Path(directory)
    species = sorted(set(atoms.numbers))
    if pp_paths is None:
        pp_paths = get_default_abinit_pp_paths()
    ppp = get_ppp_list(atoms, species,
                       raise_exception=raise_exception,
                       xc=parameters['xc'],
                       pps=parameters['pps'],
                       search_paths=pp_paths)

    inputfile = directory / 'abinit.in'

    # XXX inappropriate knowledge about choice of outputfile
    outputfile = directory / 'abinit.abo'

    # Abinit will write to label.txtA if label.txt already exists,
    # so we remove it if it's there:
    if outputfile.exists():
        outputfile.unlink()

    with open(inputfile, 'w') as fd:
        write_abinit_in(fd, atoms, param=parameters, species=species,
                        pseudos=ppp)


def read_abinit_outputs(directory, label):
    directory = Path(directory)
    textfilename = directory / f'{label}.abo'
    results = {}
    with open(textfilename) as fd:
        dct = read_abinit_out(fd)
        results.update(dct)

    # The eigenvalues section in the main file is shortened to
    # a limited number of kpoints.  We read the complete one from
    # the EIG file then:
    with open(directory / f'{label}o_EIG') as fd:
        dct = read_eig(fd)
        results.update(dct)
    return results


def read_abinit_gsr(filename):
    import netCDF4
    data = netCDF4.Dataset(filename)
    data.set_auto_mask(False)
    version = data.abinit_version

    typat = data.variables['atom_species'][:]
    cell = data.variables['primitive_vectors'][:] * Bohr
    znucl = data.variables['atomic_numbers'][:]
    xred = data.variables['reduced_atom_positions'][:]

    znucl_int = znucl.astype(int)
    znucl_int[znucl_int != znucl] = 0  # (Fractional Z)
    numbers = znucl_int[typat - 1]

    atoms = Atoms(numbers=numbers,
                  scaled_positions=xred,
                  cell=cell,
                  pbc=True)

    results = {}

    def addresult(name, abinit_name, unit=1):
        if abinit_name not in data.variables:
            return
        values = data.variables[abinit_name][:]
        # Within the netCDF4 dataset, the float variables return a array(float)
        # The tolist() is here to ensure that the result is of type float
        if not values.shape:
            values = values.tolist()
        results[name] = values * unit

    addresult('energy', 'etotal', Hartree)
    addresult('free_energy', 'etotal', Hartree)
    addresult('forces', 'cartesian_forces', Hartree / Bohr)
    addresult('stress', 'cartesian_stress_tensor', Hartree / Bohr**3)

    atoms.calc = SinglePointCalculator(atoms, **results)
    atoms.calc.name = 'abinit'
    results['atoms'] = atoms

    addresult('fermilevel', 'fermie', Hartree)
    addresult('ibz_kpoints', 'reduced_coordinates_of_kpoints')
    addresult('eigenvalues', 'eigenvalues', Hartree)
    addresult('occupations', 'occupations')
    addresult('kpoint_weights', 'kpoint_weights')
    results['version'] = version

    return results


def get_ppp_list(atoms, species, raise_exception, xc, pps,
                 search_paths):
    ppp_list = []

    xcname = 'GGA' if xc != 'LDA' else 'LDA'
    for Z in species:
        number = abs(Z)
        symbol = chemical_symbols[number]

        names = []
        for s in [symbol, symbol.lower()]:
            for xcn in [xcname, xcname.lower()]:
                if pps in ['paw']:
                    hghtemplate = '%s-%s-%s.paw'  # E.g. "H-GGA-hard-uspp.paw"
                    names.append(hghtemplate % (s, xcn, '*'))
                    names.append(f'{s}[.-_]*.paw')
                elif pps in ['pawxml']:
                    hghtemplate = '%s.%s%s.xml'  # E.g. "H.GGA_PBE-JTH.xml"
                    names.append(hghtemplate % (s, xcn, '*'))
                    names.append(f'{s}[.-_]*.xml')
                elif pps in ['hgh.k']:
                    hghtemplate = '%s-q%s.hgh.k'  # E.g. "Co-q17.hgh.k"
                    names.append(hghtemplate % (s, '*'))
                    names.append(f'{s}[.-_]*.hgh.k')
                    names.append(f'{s}[.-_]*.hgh')
                elif pps in ['tm']:
                    hghtemplate = '%d%s%s.pspnc'  # E.g. "44ru.pspnc"
                    names.append(hghtemplate % (number, s, '*'))
                    names.append(f'{s}[.-_]*.pspnc')
                elif pps in ['psp8']:
                    hghtemplate = '%s.psp8'  # E.g. "Si.psp8"
                    names.append(hghtemplate % (s))
                elif pps in ['hgh', 'hgh.sc']:
                    hghtemplate = '%d%s.%s.hgh'  # E.g. "42mo.6.hgh"
                    # There might be multiple files with different valence
                    # electron counts, so we must choose between
                    # the ordinary and the semicore versions for some elements.
                    #
                    # Therefore we first use glob to get all relevant files,
                    # then pick the correct one afterwards.
                    names.append(hghtemplate % (number, s, '*'))
                    names.append('%d%s%s.hgh' % (number, s, '*'))
                    names.append(f'{s}[.-_]*.hgh')
                else:  # default extension
                    names.append('%02d-%s.%s.%s' % (number, s, xcn, pps))
                    names.append('%02d[.-_]%s*.%s' % (number, s, pps))
                    names.append('%02d%s*.%s' % (number, s, pps))
                    names.append(f'{s}[.-_]*.{pps}')

        found = False
        for name in names:        # search for file names possibilities
            for path in search_paths:  # in all available directories
                filenames = glob(join(path, name))
                if not filenames:
                    continue
                if pps == 'paw':
                    # warning: see download.sh in
                    # abinit-pseudopotentials*tar.gz for additional
                    # information!
                    #
                    # XXXX This is probably buggy, max(filenames) uses
                    # an lexicographic order so 14 < 8, and it's
                    # untested so if I change it I'm sure things will
                    # just be inconsistent.  --askhl
                    filenames[0] = max(filenames)  # Semicore or hard
                elif pps == 'hgh':
                    # Lowest valence electron count
                    filenames[0] = min(filenames)
                elif pps == 'hgh.k':
                    # Semicore - highest electron count
                    filenames[0] = max(filenames)
                elif pps == 'tm':
                    # Semicore - highest electron count
                    filenames[0] = max(filenames)
                elif pps == 'hgh.sc':
                    # Semicore - highest electron count
                    filenames[0] = max(filenames)

                if filenames:
                    found = True
                    ppp_list.append(filenames[0])
                    break
            if found:
                break

        if not found:
            ppp_list.append("Provide {}.{}.{}?".format(symbol, '*', pps))
            if raise_exception:
                msg = ('Could not find {} pseudopotential {} for {} in {}'
                       .format(xcname.lower(), pps, symbol, search_paths))
                raise RuntimeError(msg)

    return ppp_list
