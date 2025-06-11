# fmt: off

"""Functions to read from control file and from turbomole standard output"""

import os
import re
import subprocess
import warnings

import numpy as np

from ase import Atom, Atoms
from ase.calculators.calculator import ReadError
from ase.units import Bohr, Ha


def execute_command(args):
    """execute commands like sdg, eiger"""
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, encoding='ASCII')
    stdout, _stderr = proc.communicate()
    return stdout


def read_data_group(data_group):
    """read a turbomole data group from control file"""
    return execute_command(['sdg', data_group]).strip()


def parse_data_group(dgr, dg_name):
    """parse a data group"""
    if len(dgr) == 0:
        return None
    dg_key = '$' + dg_name
    if not dgr.startswith(dg_key):
        raise ValueError(f'data group does not start with {dg_key}')
    ndg = dgr.replace(dg_key, '').strip()
    ndg = re.sub(r'=\s+', '=', re.sub(r'\s+=', '=', ndg))
    if all(c not in ndg for c in ('\n', ' ', '=')):
        return ndg
    lsep = '\n' if '\n' in dgr else ' '
    result = {}
    lines = ndg.split(lsep)
    for line in lines:
        if len(line) == 0:
            continue
        ksep = '=' if '=' in line else None
        fields = line.strip().split(ksep)
        if len(fields) == 2:
            result[fields[0]] = fields[1]
        elif len(fields) == 1:
            result[fields[0]] = True
    return result


def read_output(regex, path):
    """collects all matching strings from the output"""
    hitlist = []
    checkfiles = []
    for filename in os.listdir(path):
        if filename.startswith('job.') or filename.endswith('.out'):
            checkfiles.append(filename)
    for filename in checkfiles:
        with open(filename) as f:
            lines = f.readlines()
            for line in lines:
                match = re.search(regex, line)
                if match:
                    hitlist.append(match.group(1))
    return hitlist


def read_version(path):
    """read the version from the tm output if stored in a file"""
    versions = read_output(r'TURBOMOLE\s+V(\d+\.\d+)\s+', path)
    if len(set(versions)) > 1:
        warnings.warn('different turbomole versions detected')
        version = list(set(versions))
    elif len(versions) == 0:
        warnings.warn('no turbomole version detected')
        version = None
    else:
        version = versions[0]
    return version


def read_datetime(path):
    """read the datetime of the most recent calculation
    from the tm output if stored in a file
    """
    datetimes = read_output(
        r'(\d{4}-[01]\d-[0-3]\d([T\s][0-2]\d:[0-5]'
        r'\d:[0-5]\d\.\d+)?([+-][0-2]\d:[0-5]\d|Z)?)', path)
    if len(datetimes) == 0:
        warnings.warn('no turbomole datetime detected')
        datetime = None
    else:
        # take the most recent time stamp
        datetime = sorted(datetimes, reverse=True)[0]
    return datetime


def read_runtime(path):
    """read the total runtime of calculations"""
    hits = read_output(r'total wall-time\s+:\s+(\d+.\d+)\s+seconds', path)
    if len(hits) == 0:
        warnings.warn('no turbomole runtimes detected')
        runtime = None
    else:
        runtime = np.sum([float(a) for a in hits])
    return runtime


def read_hostname(path):
    """read the hostname of the computer on which the calc has run"""
    hostnames = read_output(r'hostname is\s+(.+)', path)
    if len(set(hostnames)) > 1:
        warnings.warn('runs on different hosts detected')
        hostname = list(set(hostnames))
    else:
        hostname = hostnames[0]
    return hostname


def read_convergence(restart, parameters):
    """perform convergence checks"""
    if restart:
        if bool(len(read_data_group('restart'))):
            return False
        if bool(len(read_data_group('actual'))):
            return False
        if not bool(len(read_data_group('energy'))):
            return False
        if (os.path.exists('job.start') and
                os.path.exists('GEO_OPT_FAILED')):
            return False
        return True

    if parameters['task'] in ['optimize', 'geometry optimization']:
        if os.path.exists('GEO_OPT_CONVERGED'):
            return True
        elif os.path.exists('GEO_OPT_FAILED'):
            # check whether a failed scf convergence is the reason
            checkfiles = []
            for filename in os.listdir('.'):
                if filename.startswith('job.'):
                    checkfiles.append(filename)
            for filename in checkfiles:
                for line in open(filename):
                    if 'SCF FAILED TO CONVERGE' in line:
                        # scf did not converge in some jobex iteration
                        if filename == 'job.last':
                            raise RuntimeError('scf failed to converge')
                        else:
                            warnings.warn('scf failed to converge')
            warnings.warn('geometry optimization failed to converge')
            return False
        else:
            raise RuntimeError('error during geometry optimization')
    else:
        if os.path.isfile('dscf_problem'):
            raise RuntimeError('scf failed to converge')
        else:
            return True


def read_run_parameters(results):
    """read parameters set by define and not in self.parameters"""

    if 'calculation parameters' not in results.keys():
        results['calculation parameters'] = {}
    parameters = results['calculation parameters']
    dg = read_data_group('symmetry')
    parameters['point group'] = str(dg.split()[1])
    parameters['uhf'] = '$uhf' in read_data_group('uhf')
    # Gaussian function type
    gt = read_data_group('pople')
    if gt == '':
        parameters['gaussian type'] = 'spherical harmonic'
    else:
        gt = gt.split()[1]
        if gt == 'AO':
            parameters['gaussian type'] = 'spherical harmonic'
        elif gt == 'CAO':
            parameters['gaussian type'] = 'cartesian'
        else:
            parameters['gaussian type'] = None

    nvibro = read_data_group('nvibro')
    if nvibro:
        parameters['nuclear degrees of freedom'] = int(nvibro.split()[1])


def read_energy(results, post_HF):
    """Read energy from Turbomole energy file."""
    try:
        with open('energy') as enf:
            text = enf.read().lower()
    except OSError:
        raise ReadError('failed to read energy file')
    if text == '':
        raise ReadError('empty energy file')

    lines = iter(text.split('\n'))

    for line in lines:
        if line.startswith('$end'):
            break
        elif line.startswith('$'):
            pass
        else:
            energy_tmp = float(line.split()[1])
            if post_HF:
                energy_tmp += float(line.split()[4])
    # update energy units
    e_total = energy_tmp * Ha
    results['total energy'] = e_total


def read_occupation_numbers(results):
    """read occupation numbers with module 'eiger' """
    if 'molecular orbitals' not in results.keys():
        return
    mos = results['molecular orbitals']
    lines = execute_command(['eiger', '--all', '--pview']).split('\n')
    for line in lines:
        regex = (
            r'^\s+(\d+)\.\s([\sab])\s*(\d+)\s?(\w+)'
            r'\s+(\d*\.*\d*)\s+([-+]?\d+\.\d*)'
        )
        match = re.search(regex, line)
        if match:
            orb_index = int(match.group(3))
            if match.group(2) == 'a':
                spin = 'alpha'
            elif match.group(2) == 'b':
                spin = 'beta'
            else:
                spin = None
            ar_index = next(
                index for (index, molecular_orbital) in enumerate(mos)
                if (molecular_orbital['index'] == orb_index and
                    molecular_orbital['spin'] == spin)
            )
            mos[ar_index]['index by energy'] = int(match.group(1))
            irrep = str(match.group(4))
            mos[ar_index]['irreducible representation'] = irrep
            if match.group(5) != '':
                mos[ar_index]['occupancy'] = float(match.group(5))
            else:
                mos[ar_index]['occupancy'] = float(0)


def read_mos(results):
    """read the molecular orbital coefficients and orbital energies
    from files mos, alpha and beta"""

    results['molecular orbitals'] = []
    mos = results['molecular orbitals']
    keywords = ['scfmo', 'uhfmo_alpha', 'uhfmo_beta']
    spin = [None, 'alpha', 'beta']
    converged = None

    for index, keyword in enumerate(keywords):
        flen = None
        mo = {}
        orbitals_coefficients_line = []
        mo_string = read_data_group(keyword)
        if mo_string == '':
            continue
        mo_string += '\n$end'
        lines = mo_string.split('\n')
        for line in lines:
            if re.match(r'^\s*#', line):
                continue
            if 'eigenvalue' in line:
                if len(orbitals_coefficients_line) != 0:
                    mo['eigenvector'] = orbitals_coefficients_line
                    mos.append(mo)
                    mo = {}
                    orbitals_coefficients_line = []
                regex = (r'^\s*(\d+)\s+(\S+)\s+'
                         r'eigenvalue=([\+\-\d\.\w]+)\s')
                match = re.search(regex, line)
                mo['index'] = int(match.group(1))
                mo['irreducible representation'] = str(match.group(2))
                eig = float(re.sub('[dD]', 'E', match.group(3))) * Ha
                mo['eigenvalue'] = eig
                mo['spin'] = spin[index]
                mo['degeneracy'] = 1
                continue
            if keyword in line:
                # e.g. format(4d20.14)
                regex = r'format\(\d+[a-zA-Z](\d+)\.\d+\)'
                match = re.search(regex, line)
                if match:
                    flen = int(match.group(1))
                if ('scfdump' in line or 'expanded' in line or
                        'scfconv' not in line):
                    converged = False
                continue
            if '$end' in line:
                if len(orbitals_coefficients_line) != 0:
                    mo['eigenvector'] = orbitals_coefficients_line
                    mos.append(mo)
                break
            sfields = [line[i:i + flen]
                       for i in range(0, len(line), flen)]
            ffields = [float(f.replace('D', 'E').replace('d', 'E'))
                       for f in sfields]
            orbitals_coefficients_line += ffields
    return converged


def read_basis_set(results):
    """read the basis set"""
    results['basis set'] = []
    results['basis set formatted'] = {}
    bsf = read_data_group('basis')
    results['basis set formatted']['turbomole'] = bsf
    lines = bsf.split('\n')
    basis_set = {}
    functions = []
    function = {}
    primitives = []
    read_tag = False
    read_data = False
    for line in lines:
        if len(line.strip()) == 0:
            continue
        if '$basis' in line:
            continue
        if '$end' in line:
            break
        if re.match(r'^\s*#', line):
            continue
        if re.match(r'^\s*\*', line):
            if read_tag:
                read_tag = False
                read_data = True
            else:
                if read_data:
                    # end primitives
                    function['primitive functions'] = primitives
                    function['number of primitives'] = len(primitives)
                    primitives = []
                    functions.append(function)
                    function = {}
                    # end contracted
                    basis_set['functions'] = functions
                    functions = []
                    results['basis set'].append(basis_set)
                    basis_set = {}
                    read_data = False
                read_tag = True
            continue
        if read_tag:
            match = re.search(r'^\s*(\w+)\s+(.+)', line)
            if match:
                basis_set['element'] = match.group(1)
                basis_set['nickname'] = match.group(2)
            else:
                raise RuntimeError('error reading basis set')
        else:
            match = re.search(r'^\s+(\d+)\s+(\w+)', line)
            if match:
                if len(primitives) > 0:
                    # end primitives
                    function['primitive functions'] = primitives
                    function['number of primitives'] = len(primitives)
                    primitives = []
                    functions.append(function)
                    function = {}
                    # begin contracted
                function['shell type'] = str(match.group(2))
                continue
            regex = (
                r'^\s*([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)'
                r'\s+([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)'
            )
            match = re.search(regex, line)
            if match:
                exponent = float(match.group(1))
                coefficient = float(match.group(3))
                primitives.append(
                    {'exponent': exponent, 'coefficient': coefficient}
                )


def read_ecps(results):
    """read the effective core potentials"""
    ecpf = read_data_group('ecp')
    if not bool(len(ecpf)):
        results['ecps'] = None
        results['ecps formatted'] = None
        return
    results['ecps'] = []
    results['ecps formatted'] = {}
    results['ecps formatted']['turbomole'] = ecpf
    lines = ecpf.split('\n')
    ecp = {}
    groups = []
    group = {}
    terms = []
    read_tag = False
    read_data = False
    for line in lines:
        if len(line.strip()) == 0:
            continue
        if '$ecp' in line:
            continue
        if '$end' in line:
            break
        if re.match(r'^\s*#', line):
            continue
        if re.match(r'^\s*\*', line):
            if read_tag:
                read_tag = False
                read_data = True
            else:
                if read_data:
                    # end terms
                    group['terms'] = terms
                    group['number of terms'] = len(terms)
                    terms = []
                    groups.append(group)
                    group = {}
                    # end group
                    ecp['groups'] = groups
                    groups = []
                    results['ecps'].append(ecp)
                    ecp = {}
                    read_data = False
                read_tag = True
            continue
        if read_tag:
            match = re.search(r'^\s*(\w+)\s+(.+)', line)
            if match:
                ecp['element'] = match.group(1)
                ecp['nickname'] = match.group(2)
            else:
                raise RuntimeError('error reading ecp')
        else:
            regex = r'ncore\s*=\s*(\d+)\s+lmax\s*=\s*(\d+)'
            match = re.search(regex, line)
            if match:
                ecp['number of core electrons'] = int(match.group(1))
                ecp['maximum angular momentum number'] = \
                    int(match.group(2))
                continue
            match = re.search(r'^(\w(\-\w)?)', line)
            if match:
                if len(terms) > 0:
                    # end terms
                    group['terms'] = terms
                    group['number of terms'] = len(terms)
                    terms = []
                    groups.append(group)
                    group = {}
                    # begin group
                group['title'] = str(match.group(1))
                continue
            regex = (r'^\s*([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s+'
                     r'(\d)\s+([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)')
            match = re.search(regex, line)
            if match:
                terms.append(
                    {
                        'coefficient': float(match.group(1)),
                        'power of r': float(match.group(3)),
                        'exponent': float(match.group(4))
                    }
                )


def read_forces(results, natoms):
    """Read forces from Turbomole gradient file."""
    dg = read_data_group('grad')
    if len(dg) == 0:
        return None
    with open('gradient') as file:
        lines = file.readlines()
    forces = np.array([[0, 0, 0]])

    nline = len(lines)
    iline = -1

    for i in range(nline):
        if 'cycle' in lines[i]:
            iline = i

    if iline < 0:
        raise RuntimeError('Please check TURBOMOLE gradients')

    # next line
    iline += natoms + 1
    # $end line
    nline -= 1
    # read gradients
    for i in range(iline, nline):
        line = lines[i].replace('D', 'E')
        tmp = np.array([[float(f) for f in line.split()[0:3]]])
        forces = np.concatenate((forces, tmp))
    # Note the '-' sign for turbomole, to get forces
    forces = -np.delete(forces, np.s_[0:1], axis=0) * Ha / Bohr
    results['energy gradient'] = (-forces).tolist()
    return forces


def read_gradient(results):
    """read all information in file 'gradient'"""
    grad_string = read_data_group('grad')
    if len(grad_string) == 0:
        return
#       try to reuse ase:
#       structures = read('gradient', index=':')
    lines = grad_string.split('\n')
    history = []
    image = {}
    gradient = []
    atoms = Atoms()
    (cycle, energy, norm) = (None, None, None)
    for line in lines:
        # cycle lines
        regex = (
            r'^\s*cycle =\s*(\d+)\s+'
            r'SCF energy =\s*([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s+'
            r'\|dE\/dxyz\| =\s*([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)'
        )
        match = re.search(regex, line)
        if match:
            if len(atoms):
                image['optimization cycle'] = cycle
                image['total energy'] = energy
                image['gradient norm'] = norm
                image['energy gradient'] = gradient
                history.append(image)
                image = {}
                atoms = Atoms()
                gradient = []
            cycle = int(match.group(1))
            energy = float(match.group(2)) * Ha
            norm = float(match.group(4)) * Ha / Bohr
            continue
        # coordinate lines
        regex = (
            r'^\s*([-+]?[0-9]*\.?[0-9]+([eEdD][-+]?[0-9]+)?)'
            r'\s+([-+]?[0-9]*\.?[0-9]+([eEdD][-+]?[0-9]+)?)'
            r'\s+([-+]?[0-9]*\.?[0-9]+([eEdD][-+]?[0-9]+)?)'
            r'\s+(\w+)'
        )
        match = re.search(regex, line)
        if match:
            x = float(match.group(1)) * Bohr
            y = float(match.group(3)) * Bohr
            z = float(match.group(5)) * Bohr
            symbol = str(match.group(7)).capitalize()

            if symbol == 'Q':
                symbol = 'X'
            atoms += Atom(symbol, (x, y, z))

            continue
        # gradient lines
        regex = (
            r'^\s*([-+]?[0-9]*\.?[0-9]+([eEdD][-+]?[0-9]+)?)'
            r'\s+([-+]?[0-9]*\.?[0-9]+([eEdD][-+]?[0-9]+)?)'
            r'\s+([-+]?[0-9]*\.?[0-9]+([eEdD][-+]?[0-9]+)?)'
        )
        match = re.search(regex, line)
        if match:
            gradx = float(match.group(1).replace('D', 'E')) * Ha / Bohr
            grady = float(match.group(3).replace('D', 'E')) * Ha / Bohr
            gradz = float(match.group(5).replace('D', 'E')) * Ha / Bohr
            gradient.append([gradx, grady, gradz])

    image['optimization cycle'] = cycle
    image['total energy'] = energy
    image['gradient norm'] = norm
    image['energy gradient'] = gradient
    history.append(image)
    results['geometry optimization history'] = history


def read_hessian(results, noproj=False):
    """Read in the hessian matrix"""
    results['hessian matrix'] = {}
    results['hessian matrix']['array'] = []
    results['hessian matrix']['units'] = '?'
    results['hessian matrix']['projected'] = True
    results['hessian matrix']['mass weighted'] = True
    dg = read_data_group('nvibro')
    if len(dg) == 0:
        return
    nvibro = int(dg.split()[1])
    results['hessian matrix']['dimension'] = nvibro
    row = []
    key = 'hessian'
    if noproj:
        key = 'npr' + key
        results['hessian matrix']['projected'] = False
    lines = read_data_group(key).split('\n')
    for line in lines:
        if key in line:
            continue
        fields = line.split()
        row.extend(fields[2:len(fields)])
        if len(row) == nvibro:
            # check whether it is mass-weighted
            float_row = [float(element) for element in row]
            results['hessian matrix']['array'].append(float_row)
            row = []


def read_normal_modes(results, noproj=False):
    """Read in vibrational normal modes"""
    results['normal modes'] = {}
    results['normal modes']['array'] = []
    results['normal modes']['projected'] = True
    results['normal modes']['mass weighted'] = True
    results['normal modes']['units'] = '?'
    dg = read_data_group('nvibro')
    if len(dg) == 0:
        return
    nvibro = int(dg.split()[1])
    results['normal modes']['dimension'] = nvibro
    row = []
    key = 'vibrational normal modes'
    if noproj:
        key = 'npr' + key
        results['normal modes']['projected'] = False
    lines = read_data_group(key).split('\n')
    for line in lines:
        if key in line:
            continue
        if '$end' in line:
            break
        fields = line.split()
        row.extend(fields[2:len(fields)])
        if len(row) == nvibro:
            # check whether it is mass-weighted
            float_row = [float(element) for element in row]
            results['normal modes']['array'].append(float_row)
            row = []


def read_vibrational_reduced_masses(results):
    """Read vibrational reduced masses"""
    results['vibrational reduced masses'] = []
    dg = read_data_group('vibrational reduced masses')
    if len(dg) == 0:
        return
    lines = dg.split('\n')
    for line in lines:
        if '$vibrational' in line:
            continue
        if '$end' in line:
            break
        fields = [float(element) for element in line.split()]
        results['vibrational reduced masses'].extend(fields)


def read_vibrational_spectrum(results, noproj=False):
    """Read the vibrational spectrum"""
    results['vibrational spectrum'] = []
    key = 'vibrational spectrum'
    if noproj:
        key = 'npr' + key
    lines = read_data_group(key).split('\n')
    for line in lines:
        dictionary = {}
        regex = (
            r'^\s+(\d+)\s+(\S*)\s+([-+]?\d+\.\d*)'
            r'\s+(\d+\.\d*)\s+(\S+)\s+(\S+)'
        )
        match = re.search(regex, line)
        if match:
            dictionary['mode number'] = int(match.group(1))
            dictionary['irreducible representation'] = str(match.group(2))
            dictionary['frequency'] = {
                'units': 'cm^-1',
                'value': float(match.group(3))
            }
            dictionary['infrared intensity'] = {
                'units': 'km/mol',
                'value': float(match.group(4))
            }

            if match.group(5) == 'YES':
                dictionary['infrared active'] = True
            elif match.group(5) == 'NO':
                dictionary['infrared active'] = False
            else:
                dictionary['infrared active'] = None

            if match.group(6) == 'YES':
                dictionary['Raman active'] = True
            elif match.group(6) == 'NO':
                dictionary['Raman active'] = False
            else:
                dictionary['Raman active'] = None

            results['vibrational spectrum'].append(dictionary)


def read_ssquare(results):
    """Read the expectation value of S^2 operator"""
    s2_string = read_data_group('ssquare from dscf')
    if s2_string == '':
        return
    string = s2_string.split('\n')[1]
    ssquare = float(re.search(r'^\s*(\d+\.*\d*)', string).group(1))
    results['ssquare from scf calculation'] = ssquare


def read_dipole_moment(results):
    """Read the dipole moment"""
    dip_string = read_data_group('dipole')
    if dip_string == '':
        return
    lines = dip_string.split('\n')
    for line in lines:
        regex = (
            r'^\s+x\s+([-+]?\d+\.\d*)\s+y\s+([-+]?\d+\.\d*)'
            r'\s+z\s+([-+]?\d+\.\d*)\s+a\.u\.'
        )
        match = re.search(regex, line)
        if match:
            dip_vec = [float(match.group(c)) for c in range(1, 4)]
        regex = r'^\s+\| dipole \| =\s+(\d+\.*\d*)\s+debye'
        match = re.search(regex, line)
        if match:
            dip_abs_val = float(match.group(1))
    results['electric dipole moment'] = {}
    results['electric dipole moment']['vector'] = {
        'array': dip_vec,
        'units': 'a.u.'
    }
    results['electric dipole moment']['absolute value'] = {
        'value': dip_abs_val,
        'units': 'Debye'
    }


def read_charges(filename, natoms):
    """read partial charges on atoms from an ESP fit"""
    charges = None
    if os.path.exists(filename):
        with open(filename) as infile:
            lines = infile.readlines()
        oklines = None
        for n, line in enumerate(lines):
            if 'atom  radius/au   charge' in line:
                oklines = lines[n + 1:n + natoms + 1]
        if oklines is not None:
            qm_charges = [float(line.split()[3]) for line in oklines]
            charges = np.array(qm_charges)
    return charges


def read_point_charges():
    """read point charges from previous calculation"""
    pcs = read_data_group('point_charges')
    lines = pcs.split('\n')[1:]
    (charges, positions) = ([], [])
    for line in lines:
        columns = [float(col) for col in line.strip().split()]
        positions.append([col * Bohr for col in columns[0:3]])
        charges.append(columns[3])
    return charges, positions
