# fmt: off

import os
import random
import string
import warnings
from copy import deepcopy
from typing import List, Optional, Tuple

import numpy as np

from ase.calculators.calculator import KPoints, kpts2kpts

_special_kws = ['center', 'autosym', 'autoz', 'theory', 'basis', 'xc', 'task',
                'set', 'symmetry', 'label', 'geompar', 'basispar', 'kpts',
                'bandpath', 'restart_kw', 'pretasks', 'charge']

_system_type = {1: 'polymer', 2: 'surface', 3: 'crystal'}


def _render_geom(atoms, params: dict) -> List[str]:
    """Generate the geometry block

    Parameters
    ----------
    atoms : Atoms
        Geometry for the computation
    params : dict
        Parameter set for the computation

    Returns
    -------
    geom : [str]
        Geometry block to use in the computation
    """
    geom_header = ['geometry units angstrom']
    for geomkw in ['center', 'autosym', 'autoz']:
        geom_header.append(geomkw if params.get(geomkw) else 'no' + geomkw)
    if 'geompar' in params:
        geom_header.append(params['geompar'])
    geom = [' '.join(geom_header)]

    outpos = atoms.get_positions()
    pbc = atoms.pbc
    if np.any(pbc):
        scpos = atoms.get_scaled_positions()
        for i, pbci in enumerate(pbc):
            if pbci:
                outpos[:, i] = scpos[:, i]
        npbc = pbc.sum()
        cellpars = atoms.cell.cellpar()
        geom.append(f'  system {_system_type[npbc]} units angstrom')
        if npbc == 3:
            geom.append('    lattice_vectors')
            for row in atoms.cell:
                geom.append('      {:20.16e} {:20.16e} {:20.16e}'.format(*row))
        else:
            if pbc[0]:
                geom.append(f'    lat_a {cellpars[0]:20.16e}')
            if pbc[1]:
                geom.append(f'    lat_b {cellpars[1]:20.16e}')
            if pbc[2]:
                geom.append(f'    lat_c {cellpars[2]:20.16e}')
            if pbc[1] and pbc[2]:
                geom.append(f'    alpha {cellpars[3]:20.16e}')
            if pbc[0] and pbc[2]:
                geom.append(f'    beta {cellpars[4]:20.16e}')
            if pbc[1] and pbc[0]:
                geom.append(f'    gamma {cellpars[5]:20.16e}')
        geom.append('  end')

    for i, atom in enumerate(atoms):
        geom.append('  {:<2} {:20.16e} {:20.16e} {:20.16e}'
                    ''.format(atom.symbol, *outpos[i]))
    symm = params.get('symmetry')
    if symm is not None:
        geom.append(f'  symmetry {symm}')
    geom.append('end')
    return geom


def _render_basis(theory, params: dict) -> List[str]:
    """Infer the basis set block

    Arguments
    ---------
    theory : str
        Name of the theory used for the calculation
    params : dict
        Parameters for the computation

    Returns
    -------
    [str]
        List of input file lines for the basis block
    """

    # Break if no basis set is provided and non is applicable
    if 'basis' not in params:
        if theory in ['pspw', 'band', 'paw']:
            return []

    # Write the header section
    if 'basispar' in params:
        header = 'basis {} noprint'.format(params['basispar'])
    else:
        header = 'basis noprint'
    basis_out = [header]

    # Write the basis set for each atom type
    basis_in = params.get('basis', '3-21G')
    if isinstance(basis_in, str):
        basis_out.append(f'   * library {basis_in}')
    else:
        for symbol, ibasis in basis_in.items():
            basis_out.append(f'{symbol:>4} library {ibasis}')
    basis_out.append('end')
    return basis_out


_special_keypairs = [('nwpw', 'simulation_cell'),
                     ('nwpw', 'carr-parinello'),
                     ('nwpw', 'brillouin_zone'),
                     ('tddft', 'grad'),
                     ]


def _render_brillouin_zone(array, name=None) -> List[str]:
    out = ['  brillouin_zone']
    if name is not None:
        out += [f'    zone_name {name}']
    template = '    kvector' + ' {:20.16e}' * array.shape[1]
    for row in array:
        out.append(template.format(*row))
    out.append('  end')
    return out


def _render_bandpath(bp) -> List[str]:
    if bp is None:
        return []
    out = ['nwpw']
    out += _render_brillouin_zone(bp.kpts, name=bp.path)
    out += [f'  zone_structure_name {bp.path}',
            'end',
            'task band structure']
    return out


def _format_line(key, val) -> str:
    if val is None:
        return key
    if isinstance(val, bool):
        return f'{key} .{str(val).lower()}.'
    else:
        return ' '.join([key, str(val)])


def _format_block(key, val, nindent=0) -> List[str]:
    prefix = '  ' * nindent
    prefix2 = '  ' * (nindent + 1)
    if val is None:
        return [prefix + key]

    if not isinstance(val, dict):
        return [prefix + _format_line(key, val)]

    out = [prefix + key]
    for subkey, subval in val.items():
        if (key, subkey) in _special_keypairs:
            if (key, subkey) == ('nwpw', 'brillouin_zone'):
                out += _render_brillouin_zone(subval)
            else:
                out += _format_block(subkey, subval, nindent + 1)
        else:
            if isinstance(subval, dict):
                subval = ' '.join([_format_line(a, b)
                                   for a, b in subval.items()])
            out.append(prefix2 + ' '.join([_format_line(subkey, subval)]))
    out.append(prefix + 'end')
    return out


def _render_other(params) -> List[str]:
    """Render other commands

    Parameters
    ----------
    params : dict
        Parameter set to be rendered

    Returns
    -------
    out : [str]
        Block defining other commands
    """
    out = []
    for kw, block in params.items():
        if kw in _special_kws:
            continue
        out += _format_block(kw, block)
    return out


def _render_set(set_params) -> List[str]:
    """Render the commands for the set parameters

    Parameters
    ----------
    set_params : dict
        Parameters being set

    Returns
    -------
    out : [str]
        Block defining set commands
    """
    return ['set ' + _format_line(key, val) for key, val in set_params.items()]


_gto_theories = ['tce', 'ccsd', 'tddft', 'scf', 'dft',
                 'direct_mp2', 'mp2', 'rimp2']
_pw_theories = ['band', 'pspw', 'paw']
_all_theories = _gto_theories + _pw_theories


def _get_theory(params: dict) -> str:
    """Infer the theory given the user-provided settings

    Parameters
    ----------
    params : dict
        Parameters for the computation

    Returns
    -------
    theory : str
        Theory directive to use
    """
    # Default: user-provided theory
    theory = params.get('theory')
    if theory is not None:
        return theory

    # Check if the user passed a theory to xc
    xc = params.get('xc')
    if xc in _all_theories:
        return xc

    # Check for input blocks that correspond to a particular level of
    # theory. Correlated theories (e.g. CCSD) are checked first.
    for kw in _gto_theories:
        if kw in params:
            return kw

    # If the user passed an 'nwpw' block, then they want a plane-wave
    # calculation, but what kind? If they request k-points, then
    # they want 'band', otherwise assume 'pspw' (if the user wants
    # to use 'paw', they will have to ask for it specifically).
    nwpw = params.get('nwpw')
    if nwpw is not None:
        if 'monkhorst-pack' in nwpw or 'brillouin_zone' in nwpw:
            return 'band'
        return 'pspw'

    # When all else fails, default to dft.
    return 'dft'


_xc_conv = dict(lda='slater pw91lda',
                pbe='xpbe96 cpbe96',
                revpbe='revpbe cpbe96',
                rpbe='rpbe cpbe96',
                pw91='xperdew91 perdew91',
                )


def _update_mult(magmom_tot: int, params: dict) -> None:
    """Update parameters for multiplicity given the magnetic moment

    For example, sets the number of open shells for SCF calculations
    and the multiplicity for DFT calculations.

    Parameters
    ----------
    magmom_tot : int
        Total magnetic moment of the system
    params : dict
        Current set of parameters, will be modified
    """
    # Determine theory and multiplicity
    theory = params['theory']
    if magmom_tot == 0:
        magmom_mult = 1
    else:
        magmom_mult = np.sign(magmom_tot) * (abs(magmom_tot) + 1)

    # Adjust the kwargs for each type of theory
    if 'scf' in params:
        for kw in ['nopen', 'singlet', 'doublet', 'triplet', 'quartet',
                   'quintet', 'sextet', 'septet', 'octet']:
            if kw in params['scf']:
                break
        else:
            params['scf']['nopen'] = magmom_tot
    elif theory in ['scf', 'mp2', 'direct_mp2', 'rimp2', 'ccsd', 'tce']:
        params['scf'] = dict(nopen=magmom_tot)

    if 'dft' in params:
        if 'mult' not in params['dft']:
            params['dft']['mult'] = magmom_mult
    elif theory in ['dft', 'tddft']:
        params['dft'] = dict(mult=magmom_mult)

    if 'nwpw' in params:
        if 'mult' not in params['nwpw']:
            params['nwpw']['mult'] = magmom_mult
    elif theory in ['pspw', 'band', 'paw']:
        params['nwpw'] = dict(mult=magmom_mult)


def _update_kpts(atoms, params) -> None:
    """Converts top-level 'kpts' argument to native keywords

    Parameters
    ----------
    atoms : Atoms
        Input structure
    params : dict
        Current parameter set, will be updated
    """
    kpts = params.get('kpts')
    if kpts is None:
        return

    nwpw = params.get('nwpw', {})

    if 'monkhorst-pack' in nwpw or 'brillouin_zone' in nwpw:
        raise ValueError("Redundant k-points specified!")

    if isinstance(kpts, KPoints):
        nwpw['brillouin_zone'] = kpts.kpts
    elif isinstance(kpts, dict):
        if kpts.get('gamma', False) or 'size' not in kpts:
            nwpw['brillouin_zone'] = kpts2kpts(kpts, atoms).kpts
        else:
            nwpw['monkhorst-pack'] = ' '.join(map(str, kpts['size']))
    elif isinstance(kpts, np.ndarray):
        nwpw['brillouin_zone'] = kpts
    else:
        nwpw['monkhorst-pack'] = ' '.join(map(str, kpts))

    params['nwpw'] = nwpw


def _render_pretask(
        this_step: dict,
        previous_basis: Optional[List[str]],
        wfc_path: str,
        next_steps: List[dict],
) -> Tuple[List[str], List[str]]:
    """Generate input file lines that perform a cheaper method first

    Parameters
    ----------
    this_step: dict
        Input parameters used to define the computation
    previous_basis: [str], optional
        Basis set block used in the previous step
    wfc_path: str
        Name of the wavefunction path
    next_steps: [dict]
        Parameters for the next steps in the calculation.
        This function will adjust the next steps to read
        and project from the wave functions written to disk by this step
        if the basis set changes between this step and the next.

    Returns
    -------
    output: [str]
        Output lines for this task
    this_basis: [str]
        Basis set block used for this task
    """

    # Get the theory for the next step
    next_step = next_steps[0]
    next_theory = _get_theory(next_step)
    next_step['theory'] = next_theory
    out = []
    if next_theory not in ['dft', 'mp2', 'direct_mp2', 'rimp2', 'scf']:
        raise ValueError(f'Initial guesses not supported for {next_theory}')

    # Determine the theory for this step
    this_theory = _get_theory(this_step)
    this_step['theory'] = this_theory
    if this_theory not in ['dft', 'scf']:
        raise ValueError('Initial guesses must use either dft or scf')

    # Determine which basis set to use for this step. Our priorities
    #  1. Basis defined explicitly in this step
    #  2. Basis set of the previous step
    #  3. Basis set of the target computation level
    if 'basis' in this_step:
        this_basis = _render_basis(this_theory, this_step)
    elif previous_basis is not None:
        this_basis = previous_basis.copy()
    else:
        # Use the basis for the last step
        this_basis = _render_basis(next_theory, next_steps[-1])

    # Determine the basis for the next step
    #  If not defined, it'll be the same as this step
    if 'basis' in next_step:
        next_basis = _render_basis(next_theory, next_step)
    else:
        next_basis = this_basis

    # Set up projections if needed
    if this_basis == next_basis:
        out.append('\n'.join(this_basis))
        proj_from = None  # We do not need to project basis
    else:
        # Check for known limitations of NWChem
        if this_theory != next_theory:
            msg = 'Theories must be the same if basis are different. ' \
                  f'This step: {this_theory}//{this_basis} ' \
                  f'Next step: {next_theory}//{next_basis}'
            if 'basis' not in this_step:
                msg += f". Consider specifying basis in {this_step}"
            raise ValueError(msg)
        if not any('* library' in x for x in this_basis):
            raise ValueError('We can only support projecting from systems '
                             'where all atoms share the same basis')

        # Append a new name to this basis function by
        #  appending it as the first argument of the basis block
        proj_from = f"smb_{len(next_steps)}"
        this_basis[0] = f'basis {proj_from} {this_basis[0][6:]}'
        out.append('\n'.join(this_basis))

        # Point ao basis (the active basis set) to this new basis set
        out.append(f'set "ao basis" {proj_from}')

    # Insert a command to write wfcs from this computation to disk
    if this_theory not in this_step:
        this_step[this_theory] = {}
    if 'vectors' not in this_step[this_theory]:
        this_step[this_theory]['vectors'] = {}
    this_step[this_theory]['vectors']['output'] = wfc_path

    # Check if the initial theory changes
    if this_theory != next_theory and \
            'lindep:n_dep' not in this_step.get('set', {}):
        warnings.warn('Loading initial guess may fail if you do not specify'
                      ' the number of linearly-dependent basis functions.'
                      ' Consider adding {"set": {"lindep:n_dep": 0}} '
                      f' to the step: {this_step}.')

    # Add this to the input file along with a "task * ignore" command
    out.extend(['\n'.join(_render_other(this_step)),
                '\n'.join(_render_set(this_step.get('set', {}))),
                f'task {this_theory} ignore'])

    # Command to read the wavefunctions in the next step
    #  Theory used to get the wavefunctions may be different (mp2 uses SCF)
    wfc_theory = 'scf' if 'mp2' in next_theory else next_theory
    next_step = next_step
    if wfc_theory not in next_step:
        next_step[wfc_theory] = {}
    if 'vectors' not in next_step[wfc_theory]:
        next_step[wfc_theory]['vectors'] = {}

    if proj_from is None:
        # No need for projection
        next_step[wfc_theory]['vectors']['input'] = wfc_path
    else:
        # Define that we should project from our basis set
        next_step[wfc_theory]['vectors']['input'] \
            = f'project {proj_from} {wfc_path}'

        # Replace the name of the basis set to the default
        out.append('set "ao basis" "ao basis"')

    return out, this_basis


def write_nwchem_in(fd, atoms, properties=None, echo=False, **params):
    """Writes NWChem input file.

    See :class:`~ase.calculators.nwchem.NWChem` for available params.

    Parameters
    ----------
    fd
        file descriptor
    atoms
        atomic configuration
    properties
        list of properties to compute; by default only the
        calculation of the energy is requested
    echo
        if True include the `echo` keyword at the top of the file,
        which causes the content of the input file to be included
        in the output file
    params
        dict of instructions blocks to be included
    """
    # Copy so we can alter params w/o changing it for the function caller
    params = deepcopy(params)

    if properties is None:
        properties = ['energy']

    if 'stress' in properties:
        if 'set' not in params:
            params['set'] = {}
        params['set']['includestress'] = True

    task = params.get('task')
    if task is None:
        if 'stress' in properties or 'forces' in properties:
            task = 'gradient'
        else:
            task = 'energy'

    _update_kpts(atoms, params)

    # Determine the theory for each step
    #  We determine the theory ahead of time because it is
    #  used when generating other parts of the input file (e.g., _get_mult)
    theory = _get_theory(params)
    params['theory'] = theory

    for pretask in params.get('pretasks', []):
        pretask['theory'] = _get_theory(pretask)

    if 'xc' in params:
        xc = _xc_conv.get(params['xc'].lower(), params['xc'])
        if theory in ['dft', 'tddft']:
            if 'dft' not in params:
                params['dft'] = {}
            params['dft']['xc'] = xc
        elif theory in ['pspw', 'band', 'paw']:
            if 'nwpw' not in params:
                params['nwpw'] = {}
            params['nwpw']['xc'] = xc

    # Update the multiplicity given the charge of the system
    magmom_tot = int(atoms.get_initial_magnetic_moments().sum())
    _update_mult(magmom_tot, params)
    for pretask in params.get('pretasks', []):
        _update_mult(magmom_tot, pretask)

    # Determine the header options
    label = params.get('label', 'nwchem')
    perm = os.path.abspath(params.pop('perm', label))
    scratch = os.path.abspath(params.pop('scratch', label))
    restart_kw = params.get('restart_kw', 'start')
    if restart_kw not in ('start', 'restart'):
        raise ValueError("Unrecognised restart keyword: {}!"
                         .format(restart_kw))
    short_label = label.rsplit('/', 1)[-1]
    if echo:
        out = ['echo']
    else:
        out = []

    # Defines the geometry and global options
    out.extend([f'title "{short_label}"',
                f'permanent_dir {perm}',
                f'scratch_dir {scratch}',
                f'{restart_kw} {short_label}',
                '\n'.join(_render_set(params.get('set', {}))),
                '\n'.join(_render_geom(atoms, params))])

    # Add the charge if provided
    if 'charge' in params:
        out.append(f'charge {params["charge"]}')

    # Define the memory separately from the other options so that it
    #  is defined before any initial wfc guesses are performed
    memory_opts = params.pop('memory', None)
    if memory_opts is not None:
        out.extend(_render_other(dict(memory=memory_opts)))

    # If desired, output commands to generate the initial wavefunctions
    if 'pretasks' in params:
        wfc_path = f'tmp-{"".join(random.choices(string.hexdigits, k=8))}.wfc'
        pretasks = params['pretasks']
        previous_basis = None
        for this_ind, this_step in enumerate(pretasks):
            new_out, previous_basis = _render_pretask(
                this_step,
                previous_basis,
                wfc_path,
                pretasks[this_ind + 1:] + [params]
            )
            out.extend(new_out)

    # Finish output file with the commands to perform the desired computation
    out.extend(['\n'.join(_render_basis(theory, params)),
                '\n'.join(_render_other(params)),
                f'task {theory} {task}',
                '\n'.join(_render_bandpath(params.get('bandpath', None)))])

    fd.write('\n\n'.join(out))
