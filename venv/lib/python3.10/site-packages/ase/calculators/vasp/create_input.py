# fmt: off

# Copyright (C) 2008 CSC - Scientific Computing Ltd.
"""This module defines an ASE interface to VASP.

Developed on the basis of modules by Jussi Enkovaara and John
Kitchin.  The path of the directory containing the pseudopotential
directories (potpaw,potpaw_GGA, potpaw_PBE, ...) should be set
by the environmental flag $VASP_PP_PATH.

The user should also set the environmental flag $VASP_SCRIPT pointing
to a python script looking something like::

   import os
   exitcode = os.system('vasp')

Alternatively, user can set the environmental flag $VASP_COMMAND pointing
to the command use the launch vasp e.g. 'vasp' or 'mpirun -n 16 vasp'

http://cms.mpi.univie.ac.at/vasp/
"""

import os
import shutil
import warnings
from os.path import isfile, islink, join
from typing import List, Sequence, Tuple

import numpy as np

import ase
from ase.calculators.calculator import kpts2ndarray
from ase.calculators.vasp.setups import get_default_setups
from ase.config import cfg
from ase.io.vasp_parsers.incar_writer import write_incar

FLOAT_FORMAT = '5.6f'
EXP_FORMAT = '5.2e'


def check_ichain(ichain, ediffg, iopt):
    ichain_dct = {}
    if ichain > 0:
        ichain_dct['ibrion'] = 3
        ichain_dct['potim'] = 0.0
        if iopt is None:
            warnings.warn(
                'WARNING: optimization is set to LFBGS (IOPT = 1)')
            ichain_dct['iopt'] = 1
        if ediffg is None or float(ediffg > 0.0):
            raise RuntimeError('Please set EDIFFG < 0')
    return ichain_dct


def set_magmom(ispin, spinpol, atoms, magmom_input, sorting):
    """Helps to set the magmom tag in the INCAR file with correct formatting"""
    magmom_dct = {}
    if magmom_input is not None:
        if len(magmom_input) != len(atoms):
            msg = ('Expected length of magmom tag to be'
                   ' {}, i.e. 1 value per atom, but got {}').format(
                len(atoms), len(magmom_input))
            raise ValueError(msg)

            # Check if user remembered to specify ispin
            # note: we do not overwrite ispin if ispin=1
        if not ispin:
            spinpol = True
            # note that ispin is an int key, but for the INCAR it does not
            # matter
            magmom_dct['ispin'] = 2
        magmom = np.array(magmom_input)
        magmom = magmom[sorting]
    elif (spinpol and atoms.get_initial_magnetic_moments().any()):
        # We don't want to write magmoms if they are all 0.
        # but we could still be doing a spinpol calculation
        if not ispin:
            magmom_dct['ispin'] = 2
        # Write out initial magnetic moments
        magmom = atoms.get_initial_magnetic_moments()[sorting]
        # unpack magmom array if three components specified
        if magmom.ndim > 1:
            magmom = [item for sublist in magmom for item in sublist]
    else:
        return spinpol, {}
    # Compactify the magmom list to symbol order
    lst = [[1, magmom[0]]]
    for n in range(1, len(magmom)):
        if magmom[n] == magmom[n - 1]:
            lst[-1][0] += 1
        else:
            lst.append([1, magmom[n]])
    line = ' '.join(['{:d}*{:.4f}'.format(mom[0], mom[1])
                     for mom in lst])
    magmom_dct['magmom'] = line
    return spinpol, magmom_dct


def set_ldau(ldau_param, luj_params, symbol_count):
    """Helps to set the ldau tag in the INCAR file with correct formatting"""
    ldau_dct = {}
    if ldau_param is None:
        ldau_dct['ldau'] = '.TRUE.'
    llist = []
    ulist = []
    jlist = []
    for symbol in symbol_count:
        #  default: No +U
        luj = luj_params.get(
            symbol[0],
            {'L': -1, 'U': 0.0, 'J': 0.0}
        )
        llist.append(int(luj['L']))
        ulist.append(f'{luj["U"]:{".3f"}}')
        jlist.append(f'{luj["J"]:{".3f"}}')
    ldau_dct['ldaul'] = llist
    ldau_dct['ldauu'] = ulist
    ldau_dct['ldauj'] = jlist
    return ldau_dct


def test_nelect_charge_compitability(nelect, charge, nelect_from_ppp):
    # We need to determine the nelect resulting from a given
    # charge in any case if it's != 0, but if nelect is
    # additionally given explicitly, then we need to determine it
    # even for net charge of 0 to check for conflicts
    if charge is not None and charge != 0:
        nelect_from_charge = nelect_from_ppp - charge
        if nelect and nelect != nelect_from_charge:
            raise ValueError('incompatible input parameters: '
                             f'nelect={nelect}, but charge={charge} '
                             '(neutral nelect is '
                             f'{nelect_from_ppp})')
        print(nelect_from_charge)
        return nelect_from_charge
    else:
        return nelect


def get_pp_setup(setup) -> Tuple[dict, Sequence[int]]:
    """
    Get the pseudopotential mapping based on the "setpus" input.

    Parameters
    ----------
    setup : [str, dict]
        The setup to use for the calculation. This can be a string
        shortcut, or a dict of atom identities and suffixes.
        In the dict version it is also possible to select a base setup
        e.g.: {'base': 'minimal', 'Ca': '_sv', 2: 'O_s'}
        If the key is an integer, this means an atom index.
        For the string version, 'minimal', 'recommended' and 'GW' are
        available. The default is 'minimal

    Returns
    -------
    setups : dict
        The setup dictionary, with atom indices as keys and suffixes
        as values.
    special_setups : list
        A list of atom indices that have a special setup.
    """
    special_setups = []

    # Avoid mutating the module dictionary, so we use a copy instead
    # Note, it is a nested dict, so a regular copy is not enough
    setups_defaults = get_default_setups()

    # Default to minimal basis
    if setup is None:
        setup = {'base': 'minimal'}

    # String shortcuts are initialised to dict form
    elif isinstance(setup, str):
        if setup.lower() in setups_defaults.keys():
            setup = {'base': setup}

    # Dict form is then queried to add defaults from setups.py.
    if 'base' in setup:
        setups = setups_defaults[setup['base'].lower()]
    else:
        setups = {}

    # Override defaults with user-defined setups
    if setup is not None:
        setups.update(setup)

    for m in setups:
        try:
            special_setups.append(int(m))
        except ValueError:
            pass
    return setups, special_setups


def format_kpoints(kpts, atoms, reciprocal=False, gamma=False):
    tokens = []
    append = tokens.append

    append('KPOINTS created by Atomic Simulation Environment\n')

    if isinstance(kpts, dict):
        kpts = kpts2ndarray(kpts, atoms=atoms)
        reciprocal = True

    shape = np.array(kpts).shape

    # Wrap scalar in list if necessary
    if shape == ():
        kpts = [kpts]
        shape = (1, )

    if len(shape) == 1:
        append('0\n')
        if shape == (1, ):
            append('Auto\n')
        elif gamma:
            append('Gamma\n')
        else:
            append('Monkhorst-Pack\n')
        append(' '.join(f'{kpt:d}' for kpt in kpts))
        append('\n0 0 0\n')
    elif len(shape) == 2:
        append('%i \n' % (len(kpts)))
        if reciprocal:
            append('Reciprocal\n')
        else:
            append('Cartesian\n')
        for n in range(len(kpts)):
            [append('%f ' % kpt) for kpt in kpts[n]]
            if shape[1] == 4:
                append('\n')
            elif shape[1] == 3:
                append('1.0 \n')
    return ''.join(tokens)


# Parameters that can be set in INCAR. The values which are None
# are not written and default parameters of VASP are used for them.

float_keys = [
    'aexx',  # Fraction of exact/DFT exchange
    'aggac',  # Fraction of gradient correction to correlation
    'aggax',  # Fraction of gradient correction to exchange
    'aldac',  # Fraction of LDA correlation energy
    'amin',  #
    'amix',  #
    'amix_mag',  #
    'bmix',  # tags for mixing
    'bmix_mag',  #
    'cshift',  # Complex shift for dielectric tensor calculation (LOPTICS)
    'deper',  # relative stopping criterion for optimization of eigenvalue
    'ebreak',  # absolute stopping criterion for optimization of eigenvalues
    # (EDIFF/N-BANDS/4)
    'efield',  # applied electrostatic field
    'emax',  # energy-range for DOSCAR file
    'emin',  #
    'enaug',  # Density cutoff
    'encut',  # Planewave cutoff
    'encutgw',  # energy cutoff for response function
    'encutfock',  # FFT grid in the HF related routines
    'hfscreen',  # attribute to change from PBE0 to HSE
    'kspacing',  # determines the number of k-points if the KPOINTS
    # file is not present. KSPACING is the smallest
    # allowed spacing between k-points in units of
    # $\AA$^{-1}$.
    'potim',  # time-step for ion-motion (fs)
    'nelect',  # total number of electrons
    'param1',  # Exchange parameter
    'param2',  # Exchange parameter
    'pomass',  # mass of ions in am
    'pstress',  # add this stress to the stress tensor, and energy E = V *
    # pstress
    'sigma',  # broadening in eV
    'smass',  # Nose mass-parameter (am)
    'spring',  # spring constant for NEB
    'time',  # special control tag
    'weimin',  # maximum weight for a band to be considered empty
    'zab_vdw',  # vdW-DF parameter
    'zval',  # ionic valence
    # The next keywords pertain to the VTST add-ons from Graeme Henkelman's
    # group at UT Austin
    'jacobian',  # Weight of lattice to atomic motion
    'ddr',  # (DdR) dimer separation
    'drotmax',  # (DRotMax) number of rotation steps per translation step
    'dfnmin',  # (DFNMin) rotational force below which dimer is not rotated
    'dfnmax',  # (DFNMax) rotational force below which dimer rotation stops
    'sltol',  # convergence ratio for minimum eigenvalue
    'sdr',  # finite difference for setting up Lanczos matrix and step
    # size when translating
    'maxmove',  # Max step for translation for IOPT > 0
    'invcurv',  # Initial curvature for LBFGS (IOPT = 1)
    'timestep',  # Dynamical timestep for IOPT = 3 and IOPT = 7
    'sdalpha',  # Ratio between force and step size for IOPT = 4
    # The next keywords pertain to IOPT = 7 (i.e. FIRE)
    'ftimemax',  # Max time step
    'ftimedec',  # Factor to dec. dt
    'ftimeinc',  # Factor to inc. dt
    'falpha',  # Parameter for velocity damping
    'falphadec',  # Factor to dec. alpha
    'clz',  # electron count for core level shift
    'vdw_radius',  # Cutoff radius for Grimme's DFT-D2 and DFT-D3 and
    # Tkatchenko and Scheffler's DFT-TS dispersion corrections
    'vdw_scaling',  # Global scaling parameter for Grimme's DFT-D2 dispersion
    # correction
    'vdw_d',  # Global damping parameter for Grimme's DFT-D2 and Tkatchenko
    # and Scheffler's DFT-TS dispersion corrections
    'vdw_cnradius',  # Cutoff radius for calculating coordination number in
    # Grimme's DFT-D3 dispersion correction
    'vdw_s6',  # Damping parameter for Grimme's DFT-D2 and DFT-D3 and
    # Tkatchenko and Scheffler's DFT-TS dispersion corrections
    'vdw_s8',  # Damping parameter for Grimme's DFT-D3 dispersion correction
    'vdw_sr',  # Scaling parameter for Grimme's DFT-D2 and DFT-D3 and
    # Tkatchenko and Scheffler's DFT-TS dispersion correction
    'vdw_a1',  # Damping parameter for Grimme's DFT-D3 dispersion correction
    'vdw_a2',  # Damping parameter for Grimme's DFT-D3 dispersion correction
    'eb_k',  # solvent permitivity in Vaspsol
    'tau',  # surface tension parameter in Vaspsol
    'langevin_gamma_l',  # Friction for lattice degrees of freedom
    'pmass',  # Mass for latice degrees of freedom
    'bparam',  # B parameter for nonlocal VV10 vdW functional
    'cparam',  # C parameter for nonlocal VV10 vdW functional
    'aldax',  # Fraction of LDA exchange (for hybrid calculations)
    'tebeg',  #
    'teend',  # temperature during run
    'andersen_prob',  # Probability of collision in Andersen thermostat
    'apaco',  # Distance cutoff for pair correlation function calc.
    'auger_ecblo',  # Undocumented parameter for Auger calculations
    'auger_edens',  # Density of electrons in conduction band
    'auger_hdens',  # Density of holes in valence band
    'auger_efermi',  # Fixed Fermi level for Auger calculations
    'auger_evbhi',  # Upper bound for valence band maximum
    'auger_ewidth',  # Half-width of energy window function
    'auger_occ_fac_eeh',  # Undocumented parameter for Auger calculations
    'auger_occ_fac_ehh',  # Undocumented parameter for Auger calculations
    'auger_temp',  # Temperature for Auger calculation
    'dq',  # Finite difference displacement magnitude (NMR)
    'avgap',  # Average gap (Model GW)
    'ch_sigma',  # Broadening of the core electron absorption spectrum
    'bpotim',  # Undocumented Bond-Boost parameter (GH patches)
    'qrr',  # Undocumented Bond-Boost parameter (GH patches)
    'prr',  # Undocumented Bond-Boost parameter (GH patches)
    'rcut',  # Undocumented Bond-Boost parameter (GH patches)
    'dvmax',  # Undocumented Bond-Boost parameter (GH patches)
    'bfgsinvcurv',  # Initial curvature for BFGS (GH patches)
    'damping',  # Damping parameter for LBFGS (GH patches)
    'efirst',  # Energy of first NEB image (GH patches)
    'elast',  # Energy of final NEB image (GH patches)
    'fmagval',  # Force magnitude convergence criterion (GH patches)
    'cmbj',  # Modified Becke-Johnson MetaGGA c-parameter
    'cmbja',  # Modified Becke-Johnson MetaGGA alpha-parameter
    'cmbjb',  # Modified Becke-Johnson MetaGGA beta-parameter
    'sigma_nc_k',  # Width of ion gaussians (VASPsol)
    'sigma_k',  # Width of dielectric cavidty (VASPsol)
    'nc_k',  # Cavity turn-on density (VASPsol)
    'lambda_d_k',  # Debye screening length (VASPsol)
    'ediffsol',  # Tolerance for solvation convergence (VASPsol)
    'soltemp',  # Solvent temperature for isol 2 in Vaspsol++
    'a_k',  # Smoothing length for FFT for isol 2 in Vaspsol++
    'r_cav',  # Offset for solute surface area for isol 2 in Vaspsol++
    'epsilon_inf',  # Bulk optical dielectric for isol 2 in Vaspsol++
    'n_mol',  # Solvent density for isol 2 in Vaspsol++
    'p_mol',  # Solvent dipole moment for isol 2 in Vaspsol++
    'r_solv',  # Solvent radius for isol 2 in Vaspsol++
    'r_diel',  # Dielectric radius for isol 2 in Vaspsol++
    'r_b',  # Bound charge smearing length for isol 2 in Vaspsol++
    'c_molar',  # Electrolyte concentration for isol 2 in Vaspsol++
    'zion',  # Electrolyte ion valency for isol 2 in Vaspsol++
    'd_ion',  # Packing diameter of electrolyte ions for isol 2 in Vaspsol++
    'r_ion',  # Ionic radius of electrolyte ions for isol 2 in Vaspsol++
    'efermi_ref',  # Potential vs vacuum for isol 2 in Vaspsol++
    'capacitance_init',  # Initial guess for isol 2 in Vaspsol++
    'deg_threshold',  # Degeneracy threshold
    'omegamin',  # Minimum frequency for dense freq. grid
    'omegamax',  # Maximum frequency for dense freq. grid
    'rtime',  # Undocumented parameter
    'wplasma',  # Undocumented parameter
    'wplasmai',  # Undocumented parameter
    'dfield',  # Undocumented parameter
    'omegatl',  # Maximum frequency for coarse freq. grid
    'encutgwsoft',  # Soft energy cutoff for response kernel
    'encutlf',  # Undocumented parameter
    'scissor',  # Scissor correction for GW/BSE calcs
    'dimer_dist',  # Distance between dimer images
    'step_size',  # Step size for finite difference in dimer calculation
    'step_max',  # Maximum step size for dimer calculation
    'minrot',  # Minimum rotation allowed in dimer calculation
    'dummy_mass',  # Mass of dummy atom(s?)
    'shaketol',  # Tolerance for SHAKE algorithm
    'shaketolsoft',  # Soft tolerance for SHAKE algorithm
    'shakesca',  # Scaling of each step taken in SHAKE algorithm
    'hills_stride',  # Undocumented metadynamics parameter
    'hills_h',  # Height (in eV) of gaussian bias for metadynamics
    'hills_w',  # Width of gaussian bias for metadynamics
    'hills_k',  # Force constant coupling dummy&real for metadynamics
    'hills_m',  # Mass of dummy particle for use in metadynamics
    'hills_temperature',  # Temp. of dummy particle for metadynamics
    'hills_andersen_prob',  # Probability of thermostat coll. for metadynamics
    'hills_sqq',  # Nose-hoover particle mass for metadynamics
    'dvvdelta0',  # Undocumented parameter
    'dvvvnorm0',  # Undocumented parameter
    'dvvminpotim',  # Undocumented parameter
    'dvvmaxpotim',  # Undocumented parameter
    'enchg',  # Undocumented charge fitting parameter
    'tau0',  # Undocumented charge fitting parameter
    'encut4o',  # Cutoff energy for 4-center integrals (HF)
    'param3',  # Undocumented HF parameter
    'model_eps0',  # Undocumented HF parameter
    'model_alpha',  # Undocumented HF parameter
    'qmaxfockae',  # Undocumented HF parameter
    'hfscreenc',  # Range-separated screening length for correlations
    'hfrcut',  # Cutoff radius for HF potential kernel
    'encutae',  # Undocumented parameter for all-electron density calc.
    'encutsubrotscf',  # Undocumented subspace rotation SCF parameter
    'enini',  # Cutoff energy for wavefunctions (?)
    'wc',  # Undocumented mixing parameter
    'enmax',  # Cutoff energy for wavefunctions (?)
    'scalee',  # Undocumented parameter
    'eref',  # Reference energy
    'epsilon',  # Dielectric constant of bulk charged cells
    'rcmix',  # Mixing parameter for core density in rel. core calcs.
    'esemicore',  # Energetic lower bound for states considered "semicore"
    'external_pressure',  # Pressure for NPT calcs., equivalent to PSTRESS
    'lj_radius',  # Undocumented classical vdW parameter
    'lj_epsilon',  # Undocumented classical vdW parameter
    'lj_sigma',  # Undocumented classical vdW parameter
    'mbd_beta',  # TS MBD vdW correction damping parameter
    'scsrad',  # Cutoff radius for dipole-dipole interaction tensor in SCS
    'hitoler',  # Iterative Hirschfeld partitioning tolerance
    'lambda',  # "Spring constant" for magmom constraint calcs.
    'kproj_threshold',  # Threshold for k-point projection scheme
    'maxpwamp',  # Undocumented HF parameter
    'vcutoff',  # Undocumented parameter
    'mdtemp',  # Temperature for AIMD
    'mdgamma',  # Undocumented AIMD parameter
    'mdalpha',  # Undocumented AIMD parameter
    'ofield_kappa',  # Bias potential strength for interface pinning method
    'ofield_q6_near',  # Steinhardt-Nelson Q6 parameters for interface pinning
    'ofield_q6_far',  # Steinhardt-Nelson Q6 parameters for interface pinning
    'ofield_a',  # Target order parameter for interface pinning method
    'pthreshold',  # Don't print timings for routines faster than this value
    'qltol',  # Eigenvalue tolerance for Lanczos iteration (instanton)
    'qdr',  # Step size for building Lanczos matrix & CG (instanton)
    'qmaxmove',  # Max step size (instanton)
    'qdt',  # Timestep for quickmin minimization (instanton)
    'qtpz',  # Temperature (instanton)
    'qftol',  # Tolerance (instanton)
    'nupdown',  # fix spin moment to specified value
]

exp_keys = [
    'ediff',  # stopping-criterion for electronic upd.
    'ediffg',  # stopping-criterion for ionic upd.
    'symprec',  # precession in symmetry routines
    # The next keywords pertain to the VTST add-ons from Graeme Henkelman's
    # group at UT Austin
    'fdstep',  # Finite diference step for IOPT = 1 or 2
]

string_keys = [
    'algo',  # algorithm: Normal (Davidson) | Fast | Very_Fast (RMM-DIIS)
    'gga',  # xc-type: PW PB LM or 91 (LDA if not set)
    'metagga',  #
    'prec',  # Precission of calculation (Low, Normal, Accurate)
    'system',  # name of System
    'precfock',  # FFT grid in the HF related routines
    'radeq',  # Which type of radial equations to use for rel. core calcs.
    'localized_basis',  # Basis to use in CRPA
    'proutine',  # Select profiling routine
    'efermi',  # Sets the FERMI level in VASP 6.4.0+
]

int_keys = [
    'ialgo',  # algorithm: use only 8 (CG) or 48 (RMM-DIIS)
    'ibrion',  # ionic relaxation: 0-MD 1-quasi-New 2-CG
    'icharg',  # charge: 0-WAVECAR 1-CHGCAR 2-atom 10-const
    'idipol',  # monopol/dipol and quadropole corrections
    'images',  # number of images for NEB calculation
    'imix',  # specifies density mixing
    'iniwav',  # initial electr wf. : 0-lowe 1-rand
    'isif',  # calculate stress and what to relax
    'ismear',  # part. occupancies: -5 Blochl -4-tet -1-fermi 0-gaus >0 MP
    'ispin',  # spin-polarized calculation
    'istart',  # startjob: 0-new 1-cont 2-samecut
    'isym',  # symmetry: 0-nonsym 1-usesym 2-usePAWsym
    'iwavpr',  # prediction of wf.: 0-non 1-charg 2-wave 3-comb
    'kpar',  # k-point parallelization paramater
    'ldauprint',  # 0-silent, 1-occ. matrix written to OUTCAR, 2-1+pot. matrix
    # written
    'ldautype',  # L(S)DA+U: 1-Liechtenstein 2-Dudarev 4-Liechtenstein(LDAU)
    'lmaxmix',  #
    'lorbit',  # create PROOUT
    'maxmix',  #
    'ngx',  # FFT mesh for wavefunctions, x
    'ngxf',  # FFT mesh for charges x
    'ngy',  # FFT mesh for wavefunctions, y
    'ngyf',  # FFT mesh for charges y
    'ngz',  # FFT mesh for wavefunctions, z
    'ngzf',  # FFT mesh for charges z
    'nbands',  # Number of bands
    'nblk',  # blocking for some BLAS calls (Sec. 6.5)
    'nbmod',  # specifies mode for partial charge calculation
    'nelm',  # nr. of electronic steps (default 60)
    'nelmdl',  # nr. of initial electronic steps
    'nelmgw',  # nr. of self-consistency cycles for GW
    'nelmin',
    'nfree',  # number of steps per DOF when calculting Hessian using
    # finite differences
    'nkred',  # define sub grid of q-points for HF with
    # nkredx=nkredy=nkredz
    'nkredx',  # define sub grid of q-points in x direction for HF
    'nkredy',  # define sub grid of q-points in y direction for HF
    'nkredz',  # define sub grid of q-points in z direction for HF
    'nomega',  # number of frequency points
    'nomegar',  # number of frequency points on real axis
    'npar',  # parallelization over bands
    'nsim',  # evaluate NSIM bands simultaneously if using RMM-DIIS
    'nsw',  # number of steps for ionic upd.
    'nwrite',  # verbosity write-flag (how much is written)
    'vdwgr',  # extra keyword for Andris program
    'vdwrn',  # extra keyword for Andris program
    'voskown',  # use Vosko, Wilk, Nusair interpolation
    # The next keywords pertain to the VTST add-ons from Graeme Henkelman's
    # group at UT Austin
    'ichain',  # Flag for controlling which method is being used (0=NEB,
    # 1=DynMat, 2=Dimer, 3=Lanczos) if ichain > 3, then both
    # IBRION and POTIM are automatically set in the INCAR file
    'iopt',  # Controls which optimizer to use.  for iopt > 0, ibrion = 3
    # and potim = 0.0
    'snl',  # Maximum dimentionality of the Lanczos matrix
    'lbfgsmem',  # Steps saved for inverse Hessian for IOPT = 1 (LBFGS)
    'fnmin',  # Max iter. before adjusting dt and alpha for IOPT = 7 (FIRE)
    'icorelevel',  # core level shifts
    'clnt',  # species index
    'cln',  # main quantum number of excited core electron
    'cll',  # l quantum number of excited core electron
    'ivdw',  # Choose which dispersion correction method to use
    'nbandsgw',  # Number of bands for GW
    'nbandso',  # Number of occupied bands for electron-hole treatment
    'nbandsv',  # Number of virtual bands for electron-hole treatment
    'ncore',  # Number of cores per band, equal to number of cores divided
    # by npar
    'mdalgo',  # Determines which MD method of Tomas Bucko to use
    'nedos',  # Number of grid points in DOS
    'turbo',  # Ewald, 0 = Normal, 1 = PME
    'omegapar',  # Number of groups for response function calc.
    # (Possibly Depricated) Number of groups in real time for
    # response function calc.
    'taupar',
    'ntaupar',  # Number of groups in real time for response function calc.
    'antires',  # How to treat antiresonant part of response function
    'magatom',  # Index of atom at which to place magnetic field (NMR)
    'jatom',  # Index of atom at which magnetic moment is evaluated (NMR)
    'ichibare',  # chi_bare stencil size (NMR)
    'nbas',  # Undocumented Bond-Boost parameter (GH patches)
    'rmds',  # Undocumented Bond-Boost parameter (GH patches)
    'ilbfgsmem',  # Number of histories to store for LBFGS (GH patches)
    'vcaimages',  # Undocumented parameter (GH patches)
    'ntemper',  # Undocumented subspace diagonalization param. (GH patches)
    'ncshmem',  # Share memory between this many cores on each process
    'lmaxtau',  # Undocumented MetaGGA parameter (prob. max ang.mom. for tau)
    'kinter',  # Additional finer grid (?)
    'ibse',  # Type of BSE calculation
    'nbseeig',  # Number of BSE wfns to write
    'naturalo',  # Use NATURALO (?)
    'nbandsexact',  # Undocumented parameter
    'nbandsgwlow',  # Number of bands for which shifts are calculated
    'nbandslf',  # Number of bands included in local field effect calc.
    'omegagrid',  # Undocumented parameter
    'telescope',  # Undocumented parameter
    'maxmem',  # Amount of memory to allocate per core in MB
    'nelmhf',  # Number of iterations for HF part (GW)
    'dim',  # Undocumented parameter
    'nkredlf',  # Reduce k-points for local field effects
    'nkredlfx',  # Reduce k-points for local field effects in X
    'nkredlfy',  # Reduce k-points for local field effects in Y
    'nkredlfz',  # Reduce k-points for local field effects in Z
    'lmaxmp2',  # Undocumented parameter
    'switch',  # Undocumented dimer parameter
    'findiff',  # Use forward (1) or central (2) finite difference for dimer
    'engine',  # Undocumented dimer parameter
    'restartcg',  # Undocumented dimer parameter
    'thermostat',  # Deprecated parameter for selecting MD method (use MDALGO)
    'scaling',  # After how many steps velocities should be rescaled
    'shakemaxiter',  # Maximum # of iterations in SHAKE algorithm
    'equi_regime',  # Number of steps to equilibrate for
    'hills_bin',  # Update metadynamics bias after this many steps
    'hills_maxstride',  # Undocumented metadynamics parameter
    'dvvehistory',  # Undocumented parameter
    'ipead',  # Undocumented parameter
    'ngaus',  # Undocumented charge fitting parameter
    'exxoep',  # Undocumented HF parameter
    'fourorbit',  # Undocumented HF parameter
    'model_gw',  # Undocumented HF parameter
    'hflmax',  # Maximum L quantum number for HF calculation
    'lmaxfock',  # Maximum L quantum number for HF calc. (same as above)
    'lmaxfockae',  # Undocumented HF parameter
    'nmaxfockae',  # Undocumented HF parameter
    'nblock_fock',  # Undocumented HF parameter
    'idiot',  # Determines which warnings/errors to print
    'nrmm',  # Number of RMM-DIIS iterations
    'mremove',  # Undocumented mixing parameter
    'inimix',  # Undocumented mixing parameter
    'mixpre',  # Undocumented mixing parameter
    'nelmall',  # Undocumented parameter
    'nblock',  # How frequently to write data
    'kblock',  # How frequently to write data
    'npaco',  # Undocumented pair correlation function parameter
    'lmaxpaw',  # Max L quantum number for on-site charge expansion
    'irestart',  # Undocumented parameter
    'nreboot',  # Undocumented parameter
    'nmin',  # Undocumented parameter
    'nlspline',  # Undocumented parameter
    'ispecial',  # "Select undocumented and unsupported special features"
    'rcrep',  # Number of steps between printing relaxed core info
    'rcndl',  # Wait this many steps before updating core density
    'rcstrd',  # Relax core density after this many SCF steps
    'vdw_idampf',  # Select type of damping function for TS vdW
    'i_constrained_m',  # Select type of magmom. constraint to use
    'igpar',  # "G parallel" direction for Berry phase calculation
    'nppstr',  # Number of kpts in "igpar' direction for Berry phase calc.
    'nbands_out',  # Undocumented QP parameter
    'kpts_out',  # Undocumented QP parameter
    'isp_out',  # Undocumented QP parameter
    'nomega_out',  # Undocumented QP parameter
    'maxiter_ft',  # Max iterations for sloppy Remez algorithm
    'nmaxalt',  # Max sample points for alternant in Remez algorithms
    'itmaxlsq',  # Max iterations in LSQ search algorithm
    'ndatalsq',  # Number of sample points for LSQ search algorithm
    'ncore_in_image1',  # Undocumented parameter
    'kimages',  # Undocumented parameter
    'ncores_per_band',  # Undocumented parameter
    'maxlie',  # Max iterations in CRPA diagonalization routine
    'ncrpalow',  # Undocumented CRPA parameter
    'ncrpahigh',  # Undocumented CRPA parameter
    'nwlow',  # Undocumented parameter
    'nwhigh',  # Undocumented parameter
    'nkopt',  # Number of k-points to include in Optics calculation
    'nkoffopt',  # K-point "counter offset" for Optics
    'nbvalopt',  # Number of valence bands to write in OPTICS file
    'nbconopt',  # Number of conduction bands to write in OPTICS file
    'ch_nedos',  # Number dielectric function calculation grid points for XAS
    'plevel',  # No timings for routines with "level" higher than this
    'qnl',  # Lanczos matrix size (instanton)
    'isol',  # vaspsol++ flag 1 linear, 2 nonlinear
]

bool_keys = [
    'addgrid',  # finer grid for augmentation charge density
    'kgamma',  # The generated kpoint grid (from KSPACING) is either
    # centred at the $\Gamma$
    # point (e.g. includes the $\Gamma$ point)
    # (KGAMMA=.TRUE.)
    'laechg',  # write AECCAR0/AECCAR1/AECCAR2
    'lasph',  # non-spherical contributions to XC energy (and pot for
    # VASP.5.X)
    'lasync',  # overlap communcation with calculations
    'lcharg',  #
    'lcorr',  # Harris-correction to forces
    'ldau',  # L(S)DA+U
    'ldiag',  # algorithm: perform sub space rotation
    'ldipol',  # potential correction mode
    'lelf',  # create ELFCAR
    'lepsilon',  # enables to calculate and to print the BEC tensors
    'lhfcalc',  # switch to turn on Hartree Fock calculations
    'loptics',  # calculate the frequency dependent dielectric matrix
    'lpard',  # evaluate partial (band and/or k-point) decomposed charge
    # density
    'lplane',  # parallelisation over the FFT grid
    'lscalapack',  # switch off scaLAPACK
    'lscalu',  # switch of LU decomposition
    'lsepb',  # write out partial charge of each band separately?
    'lsepk',  # write out partial charge of each k-point separately?
    'lthomas',  #
    'luse_vdw',  # Invoke vdW-DF implementation by Klimes et. al
    'lvdw',  # Invoke DFT-D2 method of Grimme
    'lvhar',  # write Hartree potential to LOCPOT (vasp 5.x)
    'lvtot',  # create WAVECAR/CHGCAR/LOCPOT
    'lwave',  #
    # The next keywords pertain to the VTST add-ons from Graeme Henkelman's
    # group at UT Austin
    'lclimb',  # Turn on CI-NEB
    'ltangentold',  # Old central difference tangent
    'ldneb',  # Turn on modified double nudging
    'lnebcell',  # Turn on SS-NEB
    'lglobal',  # Optmize NEB globally for LBFGS (IOPT = 1)
    'llineopt',  # Use force based line minimizer for translation (IOPT = 1)
    'lbeefens',  # Switch on print of BEE energy contributions in OUTCAR
    'lbeefbas',  # Switch off print of all BEEs in OUTCAR
    'lcalcpol',  # macroscopic polarization (vasp5.2). 'lcalceps'
    'lcalceps',  # Macroscopic dielectric properties and Born effective charge
    # tensors (vasp 5.2)
    'lvdw',  # Turns on dispersion correction
    'lvdw_ewald',  # Turns on Ewald summation for Grimme's DFT-D2 and
    # Tkatchenko and Scheffler's DFT-TS dispersion correction
    'lspectral',  # Use the spectral method to calculate independent particle
    # polarizability
    'lrpa',  # Include local field effects on the Hartree level only
    'lwannier90',  # Switches on the interface between VASP and WANNIER90
    'lsorbit',  # Enable spin-orbit coupling
    'lsol',  # turn on solvation for Vaspsol
    'lnldiel',  # turn on nonlinear dielectric in Vaspsol++
    'lnlion',  # turn on nonlinear ionic in Vaspsol++
    'lsol_scf',  # turn on solvation in SCF cycle in Vaspsol++
    'lautoscale',  # automatically calculate inverse curvature for VTST LBFGS
    'interactive',  # Enables interactive calculation for VaspInteractive
    'lauger',  # Perform Auger calculation (Auger)
    'lauger_eeh',  # Calculate EEH processes (Auger)
    'lauger_ehh',  # Calculate EHH processes (Auger)
    'lauger_collect',  # Collect wfns before looping over k-points (Auger)
    'lauger_dhdk',  # Auto-determine E. window width from E. derivs. (Auger)
    'lauger_jit',  # Distribute wavefunctions for k1-k4 (Auger)
    'orbitalmag',  # Enable orbital magnetization (NMR)
    'lchimag',  # Use linear response for shielding tensor (NMR)
    'lwrtcur',  # Write response of current to mag. field to file (NMR)
    'lnmr_sym_red',  # Reduce symmetry for finite difference (NMR)
    'lzora',  # Use ZORA approximation in linear-response NMR (NMR)
    'lbone',  # Use B-component in AE one-center terms for LR NMR (NMR)
    'lmagbloch',  # Use Bloch summations to obtain orbital magnetization (NMR)
    'lgauge',  # Use gauge transformation for zero moment terms (NMR)
    'lbfconst',  # Use constant B-field with sawtooth vector potential (NMR)
    'nucind',  # Use nuclear independent calculation (NMR)
    'lnicsall',  # Use all grid points for 'nucind' calculation (NMR)
    'llraug',  # Use two-center corrections for induced B-field (NMR)
    'lbbm',  # Undocumented Bond-Boost parameter (GH patches)
    'lnoncollinear',  # Do non-collinear spin polarized calculation
    'bfgsdfp',  # Undocumented BFGS parameter (GH patches)
    'linemin',  # Use line minimization (GH patches)
    'ldneborg',  # Undocumented NEB parameter (GH patches)
    'dseed',  # Undocumented dimer parameter (GH patches)
    'linteract',  # Undocumented parameter (GH patches)
    'lmpmd',  # Undocumented parameter (GH patches)
    'ltwodim',  # Makes stress tensor two-dimensional (GH patches)
    'fmagflag',  # Use force magnitude as convergence criterion (GH patches)
    'ltemper',  # Use subspace diagonalization (?) (GH patches)
    'qmflag',  # Undocumented FIRE parameter (GH patches)
    'lmixtau',  # Undocumented MetaGGA parameter
    'ljdftx',  # Undocumented VASPsol parameter (VASPsol)
    'lrhob',  # Write the bound charge density (VASPsol)
    'lrhoion',  # Write the ionic charge density (VASPsol)
    'lnabla',  # Undocumented parameter
    'linterfast',  # Interpolate in K using linear response routines
    'lvel',  # Undocumented parameter
    'lrpaforce',  # Calculate RPA forces
    'lhartree',  # Use IP approx. in BSE (testing only)
    'ladder',  # Use ladder diagrams
    'lfxc',  # Use approximate ladder diagrams
    'lrsrpa',  # Undocumented parameter
    'lsingles',  # Calculate HF singles
    'lfermigw',  # Iterate Fermi level
    'ltcte',  # Undocumented parameter
    'ltete',  # Undocumented parameter
    'ltriplet',  # Undocumented parameter
    'lfxceps',  # Undocumented parameter
    'lfxheg',  # Undocumented parameter
    'l2order',  # Undocumented parameter
    'lmp2lt',  # Undocumented parameter
    'lgwlf',  # Undocumented parameter
    'lusew',  # Undocumented parameter
    'selfenergy',  # Undocumented parameter
    'oddonlygw',  # Avoid gamma point in response function calc.
    'evenonlygw',  # Avoid even points in response function calc.
    'lspectralgw',  # More accurate self-energy calculation
    'ch_lspec',  # Calculate matrix elements btw. core and conduction states
    'fletcher_reeves',  # Undocumented dimer parameter
    'lidm_selective',  # Undocumented dimer parameter
    'lblueout',  # Write output of blue-moon algorithm
    'hills_variable_w',  # Enable variable-width metadynamics bias
    'dvvminus',  # Undocumented parameter
    'lpead',  # Calculate cell-periodic orbital derivs. using finite diff.
    'skip_edotp',  # Skip updating elec. polarization during scf
    'skip_scf',  # Skip calculation w/ local field effects
    'lchgfit',  # Turn on charge fitting
    'lgausrc',  # Undocumented charge fitting parameter
    'lstockholder',  # Enable ISA charge fitting (?)
    'lsymgrad',  # Restore symmetry of gradient (HF)
    'lhfone',  # Calculate one-center terms (HF)
    'lrscor',  # Include long-range correlation (HF)
    'lrhfcalc',  # Include long-range HF (HF)
    'lmodelhf',  # Model HF calculation (HF)
    'shiftred',  # Undocumented HF parameter
    'hfkident',  # Undocumented HF parameter
    'oddonly',  # Undocumented HF parameter
    'evenonly',  # Undocumented HF parameter
    'lfockaedft',  # Undocumented HF parameter
    'lsubrot',  # Enable subspace rotation diagonalization
    'mixfirst',  # Mix before diagonalization
    'lvcader',  # Calculate derivs. w.r.t. VCA parameters
    'lcompat',  # Enable "full compatibility"
    'lmusic',  # "Joke" parameter
    'ldownsample',  # Downsample WAVECAR to fewer k-points
    'lscaaware',  # Disable ScaLAPACK for some things but not all
    'lorbitalreal',  # Undocumented parameter
    'lmetagga',  # Undocumented parameter
    'lspiral',  # Undocumented parameter
    'lzeroz',  # Undocumented parameter
    'lmono',  # Enable "monopole" corrections
    'lrelcore',  # Perform relaxed core calculation
    'lmimicfc',  # Mimic frozen-core calcs. for relaxed core calcs.
    'lmatchrw',  # Match PS partial waves at RWIGS? (otherwise PAW cutoff)
    'ladaptelin',  # Linearize core state energies to avoid divergences
    'lonlysemicore',  # Only linearize semi-core state energies
    'gga_compat',  # Enable backwards-compatible symmetrization of GGA derivs.
    'lrelvol',  # Undocumented classical vdW parameter
    'lj_only',  # Undocumented classical vdW parameter
    'lvdwscs',  # Include self-consistent screening in TS vdW correction
    'lcfdm',  # Use coupled fluctuating dipoles model for TS vdW
    'lvdw_sametype',  # Include interactions between atoms of the same type
    'lrescaler0',  # Rescale damping parameters in SCS vdW correction
    'lscsgrad',  # Calculate gradients for TS+SCS vdW correction energies
    'lvdwexpansion',  # Write 2-6 body contribs. to MBD vdW correction energy
    'lvdw_relvolone',  # Undocumented classical vdW parameter
    'lberry',  # Enable Berry-phase calculation
    'lpade_fit',  # Undocumented QP parameter
    'lkproj',  # Enable projection onto k-points
    'l_wr_moments',  # Undocumented parameter
    'l_wr_density',  # Undocumented parameter
    'lkotani',  # Undocumented parameter
    'ldyson',  # Undocumented parameter
    'laddherm',  # Undocumented parameter
    'lcrpaplot',  # Plot bands used in CRPA response func. calc.
    'lplotdis',  # Plot disentangled bands in CRPA response func. calc.
    'ldisentangle',  # Disentangle bands in CRPA
    'lweighted',  # "Weighted" CRPA approach
    'luseorth_lcaos',  # Use orthogonalized LCAOs in CRPA
    'lfrpa',  # Use full RPA in CRPA
    'lregularize',  # Regularize projectors in CRPA
    'ldrude',  # Include Drude term in CRPA
    'ldmatrix',  # Undocumented parameter
    'lefg',  # Calculate electric field gradient at atomic nuclei
    'lhyperfine',  # Enable Hyperfine calculation
    'lwannier',  # Enable Wannier interface
    'localize',  # Undocumented Wannier parameter
    'lintpol_wpot',  # Interpolate WPOT for Wannier
    'lintpol_orb',  # Interpolate orbitals for Wannier
    'lintpol_kpath',  # Interpolate bandstructure on given kpath for Wannier
    'lintpol_kpath_orb',  # Interpolate orbitals on given kpath for Wannier
    'lread_eigenvalues',  # Use Eigenvalues from EIGENVALUES.INT file
    'lintpol_velocity',  # Interpolate electron velocity for Wannier
    'lintpol_conductivity',  # Interpolate conductivity for Wannier
    'lwannierinterpol',  # Undocumented Wannier parameter
    'wanproj',  # Undocumented Wannier parameter
    'lorbmom',  # Undocumented LDA+U parameter
    'lwannier90_run',  # Undocumented WANNIER90 parameter
    'lwrite_wanproj',  # Write UWAN files for WANNIER90
    'lwrite_unk',  # Write UNK files for WANNIER90
    'lwrite_mmn_amn',  # Write MMN and AMN files for WANNIER90
    'lread_amn',  # Read AMN files instead of recomputing (WANNIER90)
    'lrhfatm',  # Undocumented HF parameter
    'lvpot',  # Calculate unscreened potential
    'lwpot',  # Calculate screened potential
    'lwswq',  # Undocumented parameter
    'pflat',  # Only print "flat" timings to OUTCAR
    'qifcg',  # Use CG instead of quickmin (instanton)
    'qdo_ins',  # Find instanton
    'qdo_pre',  # Calculate prefactor (instanton)
    # The next keyword pertains to the periodic NBO code of JR Schmidt's group
    # at UW-Madison (https://github.com/jrschmidt2/periodic-NBO)
    'lnbo',  # Enable NBO analysis
]

list_int_keys = [
    'iband',  # bands to calculate partial charge for
    'kpuse',  # k-point to calculate partial charge for
    'ldaul',  # DFT+U parameters, overruled by dict key 'ldau_luj'
    'random_seed',  # List of ints used to seed RNG for advanced MD routines
    # (Bucko)
    'auger_bmin_eeh',  # 4 ints | Various undocumented parameters for Auger
    'auger_bmax_eeh',  # 4 ints | calculations
    'auger_bmin_ehh',  # 4 ints |
    'auger_bmax_ehh',  # 4 ints |
    'balist',  # nbas ints | Undocumented Bond-Boost parameter (GH patches)
    'kpoint_bse',  # 4 ints | Undocumented parameter
    'nsubsys',  # <=3 ints | Last atom # for each of up to 3 thermostats
    'vdw_refstate',  # ntyp ints | Undocumented classical vdW parameter
    'vdw_mbd_size',  # 3 ints | Supercell size for TS MBD vdW correction
    'nbands_index',  # nbands_out ints | Undocumented QP parameter
    'kpts_index',  # kpts_out ints | Undocumented QP parameter
    'isp_index',  # isp_out ints | Undocumented QP parameter
    'nomega_index',  # nomega_out ints | Undocumented QP parameter
    'ntarget_states',  # nbands ints | Undocumented CRPA parameter
    'wanproj_i',  # nions ints | Undocumented Wannier parameter
    'wanproj_l',  # ? ints | Undocumented Wannier parameter
]

list_bool_keys = [
    'lattice_constraints',  # 3 bools | Undocumented advanced MD parameter
    'lrctype',  # ntyp bools | Enable relaxed-core calc. for these atoms
    'lvdw_onecell',  # 3 bools | Enable periodicity in A, B, C vector for vdW
]

list_float_keys = [
    'dipol',  # center of cell for dipol
    'eint',  # energy range to calculate partial charge for
    'ferwe',  # Fixed band occupation (spin-paired)
    'ferdo',  # Fixed band occupation (spin-plarized)
    'magmom',  # initial magnetic moments
    'ropt',  # number of grid points for non-local proj in real space
    'rwigs',  # Wigner-Seitz radii
    'ldauu',  # ldau parameters, has potential to redundant w.r.t. dict
    'ldauj',  # key 'ldau_luj', but 'ldau_luj' can't be read direct from
    # the INCAR (since it needs to know information about atomic
    # species. In case of conflict 'ldau_luj' gets written out
    # when a calculation is set up
    'vdw_c6',  # List of floats of C6 parameters (J nm^6 mol^-1) for each
    # species (DFT-D2 and DFT-TS)
    'vdw_c6au',  # List of floats of C6 parameters (a.u.) for each species
    # (DFT-TS)
    'vdw_r0',  # List of floats of R0 parameters (angstroms) for each
    # species (DFT-D2 and DFT-TS)
    'vdw_r0au',  # List of floats of R0 parameters (a.u.) for each species
    # (DFT-TS)
    'vdw_alpha',  # List of floats of free-atomic polarizabilities for each
    # species (DFT-TS)
    'langevin_gamma',  # List of floats for langevin friction coefficients
    'auger_emin_eeh',  # 4 floats | Various undocumented parameters for Auger
    'auger_emax_eeh',  # 4 floats | calculations
    'auger_emin_ehh',  # 4 floats |
    'auger_emax_ehh',  # 4 floats |
    'avecconst',  # 3 floats | magnitude of magnetic moment (NMR)
    'magdipol',  # 3 floats | magnitude of magnetic dipole (NMR)
    'bconst',  # 3 floats | magnitude of constant magnetic field (NMR)
    'magpos',  # 3 floats | position for magnetic moment w/ 'nucind' (NMR)
    'bext',  # 3 floats | Undocumented (probably external magnetic field)
    'core_c',  # ntyp floats | pseudo-core charge magnitude (VASPsol)
    'sigma_rc_k',  # ntyp floats | width of pseudo-core gaussians (VASPsol)
    'darwinr',  # ntypd (?) floats | Undocumented parameter
    'darwinv',  # ntypd (?) floats | Undocumented parameter
    'dummy_k',  # ? floats | Force const. connecting dummy atoms to sys.
    'dummy_r0',  # ? floats | Minimum dist., ang., etc. for dummy atom DOFs
    'dummy_positions',  # 3 floats | Position of dummy atom(s?)
    'psubsys',  # <=3 floats | Coll. prob. for each of up to 3 thermostats
    'tsubsys',  # <=3 floats | Temp. for each of up to 3 thermostats
    'increm',  # ? floats | Undocumented advanced MD parameter
    'value_min',  # ? floats | Undocumented advanced MD parameter
    'value_max',  # ? floats | Undocumented advanced MD parameter
    'hills_position',  # ? floats | Dummy particle(s) pos. for metadynamics
    'hills_velocity',  # ? floats | Dummy particle(s) vel. for metadynamics
    'spring_k',  # ? floats | Spring constant for harmonic constraints
    'spring_r0',  # ? floats | Spring minima for harmonic constraints
    'spring_v0',  # ? floats | Initial velocity of harmonic constraints
    'hills_wall_lower',  # ? floats | Undocumented metadynamics parameter
    'hills_wall_upper',  # ? floats | Undocumented metadynamics parameter
    'efield_pead',  # 3 floats | homogeneous electric field for PEAD calc.
    'zct',  # ? floats | Undocumented charge fitting parameter
    'rgaus',  # ? floats | Undocumented charge fitting parameter
    'hfalpha',  # 10 floats | Undocumented HF parameter
    'mcalpha',  # 10 floats | Undocumented HF parameter
    'saxis',  # 3 floats | Coordinate for collinear spin calculations
    'vca',  # ? floats | Atom weight for VCA calculations
    'stm',  # 7 floats | "range for STM data"
    'qspiral',  # 3 floats | Undocumented parameter
    'external_stress',  # 6 floats | Target stress (adds w/ external_pressure)
    'm_constr',  # 3*nions floats | Local magmom assigned to each spin DOF
    'quad_efg',  # ntyp floats | Nuclear quadrupole moments
    'ngyromag',  # ntyp floats | Nuclear gyromagnetic ratios
    'rcrhocut',  # ntyp floats | Core density cutoff rad. for HF relcore calc
    'ofield_k',  # 3 floats | Undocumented parameter
    'paripot',  # ? floats | Undocumented parameter
    'smearings',  # ? floats | ismear,sigma smearing params to loop over
    'wanproj_e',  # 2 floats | Undocumented Wannier parameter
]

special_keys = [
    'lreal',  # non-local projectors in real space
]

dict_keys = [
    'ldau_luj',  # dictionary with L(S)DA+U parameters, e.g. {'Fe':{'L':2,
    # 'U':4.0, 'J':0.9}, ...}
]

keys: List[str] = [
    # 'NBLOCK' and KBLOCK       inner block; outer block
    # 'NPACO' and APACO         distance and nr. of slots for P.C.
    # 'WEIMIN, EBREAK, DEPER    special control tags
]


class GenerateVaspInput:
    # Parameters corresponding to 'xc' settings.  This may be modified
    # by the user in-between loading calculators.vasp submodule and
    # instantiating the calculator object with calculators.vasp.Vasp()
    xc_defaults = {
        'lda': {
            'pp': 'LDA'
        },
        # GGAs
        'blyp': {  # https://www.vasp.at/forum/viewtopic.php?p=17234
            'pp': 'PBE',
            'gga': 'B5',
            'aldax': 1.00,
            'aggax': 1.00,
            'aggac': 1.00,
            'aldac': 0.00
        },
        'pw91': {
            'pp': 'PW91',
            'gga': '91'
        },
        'pbe': {
            'pp': 'PBE',
            'gga': 'PE'
        },
        'pbesol': {
            'gga': 'PS'
        },
        'revpbe': {
            'gga': 'RE'
        },
        'rpbe': {
            'gga': 'RP'
        },
        'am05': {
            'gga': 'AM'
        },
        # Meta-GGAs
        'tpss': {
            'metagga': 'TPSS'
        },
        'revtpss': {
            'metagga': 'RTPSS'
        },
        'm06l': {
            'metagga': 'M06L'
        },
        'ms0': {
            'metagga': 'MS0'
        },
        'ms1': {
            'metagga': 'MS1'
        },
        'ms2': {
            'metagga': 'MS2'
        },
        'scan': {
            'metagga': 'SCAN'
        },
        'rscan': {
            'metagga': 'RSCAN'
        },
        'r2scan': {
            'metagga': 'R2SCAN'
        },
        'scan-rvv10': {
            'metagga': 'SCAN',
            'luse_vdw': True,
            'bparam': 15.7
        },
        'mbj': {
            # Modified Becke-Johnson
            'metagga': 'MBJ',
        },
        'tb09': {
            # Alias for MBJ
            'metagga': 'MBJ',
        },
        # vdW-DFs
        'vdw-df': {
            'gga': 'RE',
            'luse_vdw': True,
            'aggac': 0.
        },
        'vdw-df-cx': {
            'gga': 'CX',
            'luse_vdw': True,
            'aggac': 0.
        },
        'vdw-df-cx0p': {
            'gga': 'CX',
            'luse_vdw': True,
            'aggac': 0.,
            'lhfcalc': True,
            'aexx': 0.2,
            'aggax': 0.8
        },
        'optpbe-vdw': {
            'gga': 'OR',
            'luse_vdw': True,
            'aggac': 0.0
        },
        'optb88-vdw': {
            'gga': 'BO',
            'luse_vdw': True,
            'aggac': 0.0,
            'param1': 1.1 / 6.0,
            'param2': 0.22
        },
        'optb86b-vdw': {
            'gga': 'MK',
            'luse_vdw': True,
            'aggac': 0.0,
            'param1': 0.1234,
            'param2': 1.0
        },
        'vdw-df2': {
            'gga': 'ML',
            'luse_vdw': True,
            'aggac': 0.0,
            'zab_vdw': -1.8867
        },
        'rev-vdw-df2': {
            'gga': 'MK',
            'luse_vdw': True,
            'param1': 0.1234,
            'param2': 0.711357,
            'zab_vdw': -1.8867,
            'aggac': 0.0
        },
        'beef-vdw': {
            'gga': 'BF',
            'luse_vdw': True,
            'zab_vdw': -1.8867
        },
        # Hartree-Fock and hybrids
        'hf': {
            'lhfcalc': True,
            'aexx': 1.0,
            'aldac': 0.0,
            'aggac': 0.0
        },
        'b3lyp': {
            'gga': 'B3',
            'lhfcalc': True,
            'aexx': 0.2,
            'aggax': 0.72,
            'aggac': 0.81,
            'aldac': 0.19
        },
        'pbe0': {
            'gga': 'PE',
            'lhfcalc': True
        },
        'hse03': {
            'gga': 'PE',
            'lhfcalc': True,
            'hfscreen': 0.3
        },
        'hse06': {
            'gga': 'PE',
            'lhfcalc': True,
            'hfscreen': 0.2
        },
        'hsesol': {
            'gga': 'PS',
            'lhfcalc': True,
            'hfscreen': 0.2
        },
        # MN-VFM functionals
        'sogga': {
            'gga': 'SA'
        },
        'sogga11': {
            'gga': 'S1'
        },
        'sogga11-x': {
            'gga': 'SX',
            'lhfcalc': True,
            'aexx': 0.401
        },
        'n12': {
            'gga': 'N2'
        },
        'n12-sx': {
            'gga': 'NX',
            'lhfcalc': True,
            'lhfscreen': 0.2
        },
        'mn12l': {
            'metagga': 'MN12L'
        },
        'gam': {
            'gga': 'GA'
        },
        'mn15l': {
            'metagga': 'MN15L'
        },
        'hle17': {
            'metagga': 'HLE17'
        },
        'revm06l': {
            'metagga': 'revM06L'
        },
        'm06sx': {
            'metagga': 'M06SX',
            'lhfcalc': True,
            'hfscreen': 0.189,
            'aexx': 0.335
        }
    }

    # environment variable for PP paths
    VASP_PP_PATH = 'VASP_PP_PATH'

    def __init__(self, restart=None):
        self.float_params = {}
        self.exp_params = {}
        self.string_params = {}
        self.int_params = {}
        self.bool_params = {}
        self.list_bool_params = {}
        self.list_int_params = {}
        self.list_float_params = {}
        self.special_params = {}
        self.dict_params = {}
        self.atoms = None
        for key in float_keys:
            self.float_params[key] = None
        for key in exp_keys:
            self.exp_params[key] = None
        for key in string_keys:
            self.string_params[key] = None
        for key in int_keys:
            self.int_params[key] = None
        for key in bool_keys:
            self.bool_params[key] = None
        for key in list_bool_keys:
            self.list_bool_params[key] = None
        for key in list_int_keys:
            self.list_int_params[key] = None
        for key in list_float_keys:
            self.list_float_params[key] = None
        for key in special_keys:
            self.special_params[key] = None
        for key in dict_keys:
            self.dict_params[key] = None

        # Initialize internal dictionary of input parameters which are
        # not regular VASP keys
        self.input_params = {
            'xc': None,  # Exchange-correlation recipe (e.g. 'B3LYP')
            'pp': None,  # Pseudopotential file (e.g. 'PW91')
            'setups': None,  # Special setups (e.g pv, sv, ...)
            'txt': '-',  # Where to send information
            'kpts': (1, 1, 1),  # k-points
            # Option to use gamma-sampling instead of Monkhorst-Pack:
            'gamma': False,
            # number of points between points in band structures:
            'kpts_nintersections': None,
            # Option to write explicit k-points in units
            # of reciprocal lattice vectors:
            'reciprocal': False,
            # Switch to disable writing constraints to POSCAR
            'ignore_constraints': False,
            # Net charge for the whole system; determines nelect if not 0
            'charge': None,
            # Deprecated older parameter which works just like "charge" but
            # with the sign flipped
            'net_charge': None,
            # Custom key-value pairs, written to INCAR with *no* type checking
            'custom': {},
        }
        # warning message for pw91
        self.pw91_warning_msg =\
            "The PW91 (potpaw_GGA) pseudopotential set is " \
            "from 2006 and not recommended for use.\nWe will " \
            "remove support for it in a future release, " \
            "and use the current PBE (potpaw_PBE) set instead.\n" \
            "Note that this still allows for PW91 calculations, " \
            "since VASP recalculates the exchange-correlation\n" \
            "energy inside the PAW sphere and corrects the atomic " \
            "energies given by the POTCAR file."

    def set_xc_params(self, xc):
        """Set parameters corresponding to XC functional"""
        xc = xc.lower()
        if xc is None:
            pass
        elif xc not in self.xc_defaults:
            xc_allowed = ', '.join(self.xc_defaults.keys())
            raise ValueError('{} is not supported for xc! Supported xc values'
                             'are: {}'.format(xc, xc_allowed))
        else:
            # print future warning in case pw91 is selected:
            if xc == 'pw91':
                warnings.warn(
                    self.pw91_warning_msg, FutureWarning
                )
            # XC defaults to PBE pseudopotentials
            if 'pp' not in self.xc_defaults[xc]:
                self.set(pp='PBE')
            self.set(**self.xc_defaults[xc])

    def set(self, **kwargs):

        if (('ldauu' in kwargs) and ('ldaul' in kwargs) and ('ldauj' in kwargs)
                and ('ldau_luj' in kwargs)):
            raise NotImplementedError(
                'You can either specify ldaul, ldauu, and ldauj OR '
                'ldau_luj. ldau_luj is not a VASP keyword. It is a '
                'dictionary that specifies L, U and J for each '
                'chemical species in the atoms object. '
                'For example for a water molecule:'
                '''ldau_luj={'H':{'L':2, 'U':4.0, 'J':0.9},
                      'O':{'L':2, 'U':4.0, 'J':0.9}}''')

        if 'xc' in kwargs:
            self.set_xc_params(kwargs['xc'])
        for key, value in kwargs.items():
            if key in self.float_params:
                self.float_params[key] = value
            elif key in self.exp_params:
                self.exp_params[key] = value
            elif key in self.string_params:
                self.string_params[key] = value
            elif key in self.int_params:
                self.int_params[key] = value
            elif key in self.bool_params:
                self.bool_params[key] = value
            elif key in self.list_bool_params:
                self.list_bool_params[key] = value
            elif key in self.list_int_params:
                self.list_int_params[key] = value
            elif key in self.list_float_params:
                self.list_float_params[key] = value
            elif key in self.special_params:
                self.special_params[key] = value
            elif key in self.dict_params:
                self.dict_params[key] = value
            elif key in self.input_params:
                self.input_params[key] = value
            elif isinstance(value, str):
                self.string_params[key] = value
            # `bool` is a subclass of `int` and should be checked earlier.
            # https://docs.python.org/3/c-api/bool.html
            elif isinstance(value, bool):
                self.bool_params[key] = value
            elif isinstance(value, int):
                self.int_params[key] = value
            elif isinstance(value, float):
                self.float_params[key] = value
            elif isinstance(value, list):
                if len(value) == 0:
                    msg = f'empty list is given for {key}'
                    raise ValueError(msg)
                if isinstance(value[0], bool):
                    self.list_bool_params[key] = value
                elif isinstance(value[0], int):
                    self.list_int_params[key] = value
                elif isinstance(value[0], float):
                    self.list_float_params[key] = value
                else:
                    msg = f'cannot handle type of value for {key} = {value!r}'
                    raise TypeError(msg)
            else:
                msg = f'cannot handle type of value for {key} = {value!r}'
                raise TypeError(msg)

    def check_xc(self):
        """Make sure the calculator has functional & pseudopotentials set up

        If no XC combination, GGA functional or POTCAR type is specified,
        default to PW91. Otherwise, try to guess the desired pseudopotentials.
        """

        p = self.input_params

        # There is no way to correctly guess the desired
        # set of pseudopotentials without 'pp' being set.
        # Usually, 'pp' will be set by 'xc'.
        if 'pp' not in p or p['pp'] is None:
            if self.string_params['gga'] is None:
                p.update({'pp': 'lda'})
            elif self.string_params['gga'] == '91':
                p.update({'pp': 'pw91'})
                warnings.warn(
                    self.pw91_warning_msg, FutureWarning
                )

            elif self.string_params['gga'] == 'PE':
                p.update({'pp': 'pbe'})
            else:
                raise NotImplementedError(
                    "Unable to guess the desired set of pseudopotential"
                    "(POTCAR) files. Please do one of the following: \n"
                    "1. Use the 'xc' parameter to define your XC functional."
                    "These 'recipes' determine the pseudopotential file as "
                    "well as setting the INCAR parameters.\n"
                    "2. Use the 'gga' settings None (default), 'PE' or '91'; "
                    "these correspond to LDA, PBE and PW91 respectively.\n"
                    "3. Set the POTCAR explicitly with the 'pp' flag. The "
                    "value should be the name of a folder on the VASP_PP_PATH"
                    ", and the aliases 'LDA', 'PBE' and 'PW91' are also"
                    "accepted.\n")

        if (p['xc'] is not None and p['xc'].lower() == 'lda'
                and p['pp'].lower() != 'lda'):
            warnings.warn("XC is set to LDA, but PP is set to "
                          "{0}. \nThis calculation is using the {0} "
                          "POTCAR set. \n Please check that this is "
                          "really what you intended!"
                          "\n".format(p['pp'].upper()))

    def _make_sort(
        self, atoms: ase.Atoms, special_setups: Sequence[int] = ()
    ) -> Tuple[List[int], List[int]]:
        symbols, _ = count_symbols(atoms, exclude=special_setups)

        # Create sorting list
        srt = []  # type: List[int]
        srt.extend(special_setups)

        for symbol in symbols:
            for m, atom in enumerate(atoms):
                if m in special_setups:
                    continue
                if atom.symbol == symbol:
                    srt.append(m)
        # Create the resorting list
        resrt = list(range(len(srt)))
        for n in range(len(resrt)):
            resrt[srt[n]] = n
        return srt, resrt

    def _set_spinpol(self, atoms):
        if self.int_params['ispin'] is None:
            self.spinpol = atoms.get_initial_magnetic_moments().any()
        else:
            # VASP runs non-spin-polarized calculations when `ispin=1`,
            # regardless if `magmom` is specified or not.
            self.spinpol = (self.int_params['ispin'] == 2)

    def _build_pp_list(self,
                       atoms,
                       setups=None,
                       special_setups: Sequence[int] = ()):
        """Build the pseudopotential lists"""

        p = self.input_params

        if setups is None:
            setups, special_setups = get_pp_setup(p['setups'])

        symbols, _ = count_symbols(atoms, exclude=special_setups)

        # Potpaw folders may be identified by an alias or full name
        for pp_alias, pp_folder in (('lda', 'potpaw'), ('pw91', 'potpaw_GGA'),
                                    ('pbe', 'potpaw_PBE')):
            if p['pp'].lower() == pp_alias:
                break
        else:
            pp_folder = p['pp']

        if self.VASP_PP_PATH in cfg:
            pppaths = cfg[self.VASP_PP_PATH].split(':')
        else:
            pppaths = []
        ppp_list = []
        # Setting the pseudopotentials, first special setups and
        # then according to symbols
        for m in special_setups:
            if m in setups:
                special_setup_index = m
            elif str(m) in setups:
                special_setup_index = str(m)  # type: ignore[assignment]
            else:
                raise Exception("Having trouble with special setup index {}."
                                " Please use an int.".format(m))
            potcar = join(pp_folder, setups[special_setup_index], 'POTCAR')
            for path in pppaths:
                filename = join(path, potcar)

                if isfile(filename) or islink(filename):
                    ppp_list.append(filename)
                    break
                elif isfile(filename + '.Z') or islink(filename + '.Z'):
                    ppp_list.append(filename + '.Z')
                    break
            else:
                symbol = atoms.symbols[m]
                msg = """Looking for {}.
                No pseudopotential for symbol{} with setup {} """.format(
                    potcar, symbol, setups[special_setup_index])
                raise RuntimeError(msg)

        for symbol in symbols:
            try:
                potcar = join(pp_folder, symbol + setups[symbol], 'POTCAR')
            except (TypeError, KeyError):
                potcar = join(pp_folder, symbol, 'POTCAR')
            for path in pppaths:
                filename = join(path, potcar)

                if isfile(filename) or islink(filename):
                    ppp_list.append(filename)
                    break
                elif isfile(filename + '.Z') or islink(filename + '.Z'):
                    ppp_list.append(filename + '.Z')
                    break
            else:
                msg = ("""Looking for PP for {}
                        The pseudopotentials are expected to be in:
                        LDA:  $VASP_PP_PATH/potpaw/
                        PBE:  $VASP_PP_PATH/potpaw_PBE/
                        PW91: $VASP_PP_PATH/potpaw_GGA/

                        No pseudopotential for {}!""".format(potcar, symbol))
                raise RuntimeError(msg)
        return ppp_list

    def initialize(self, atoms):
        """Initialize a VASP calculation

        Constructs the POTCAR file (does not actually write it).
        User should specify the PATH
        to the pseudopotentials in VASP_PP_PATH environment variable

        The pseudopotentials are expected to be in:
        LDA:  $VASP_PP_PATH/potpaw/
        PBE:  $VASP_PP_PATH/potpaw_PBE/
        PW91: $VASP_PP_PATH/potpaw_GGA/

        if your pseudopotentials are somewhere else, or named
        differently you may make symlinks at the paths above that
        point to the right place. Alternatively, you may pass the full
        name of a folder on the VASP_PP_PATH to the 'pp' parameter.
        """

        self.check_xc()
        self.atoms = atoms
        self.all_symbols = atoms.get_chemical_symbols()
        self.natoms = len(atoms)

        self._set_spinpol(atoms)

        setups, special_setups = get_pp_setup(self.input_params['setups'])

        # Determine the number of atoms of each atomic species
        # sorted after atomic species
        symbols, symbolcount = count_symbols(atoms, exclude=special_setups)
        self.sort, self.resort = self._make_sort(atoms,
                                                 special_setups=special_setups)

        self.atoms_sorted = atoms[self.sort]

        # Check if the necessary POTCAR files exists and
        # create a list of their paths.
        atomtypes = atoms.get_chemical_symbols()
        self.symbol_count = []
        for m in special_setups:
            self.symbol_count.append([atomtypes[m], 1])
        for m in symbols:
            self.symbol_count.append([m, symbolcount[m]])

        # create pseudopotential list
        self.ppp_list = self._build_pp_list(atoms,
                                            setups=setups,
                                            special_setups=special_setups)

        self.converged = None
        self.setups_changed = None

    def default_nelect_from_ppp(self):
        """ Get default number of electrons from ppp_list and symbol_count

        "Default" here means that the resulting cell would be neutral.
        """
        symbol_valences = []
        for filename in self.ppp_list:
            with open_potcar(filename=filename) as ppp_file:
                r = read_potcar_numbers_of_electrons(ppp_file)
                symbol_valences.extend(r)
        assert len(self.symbol_count) == len(symbol_valences)
        default_nelect = 0
        for ((symbol1, count),
             (symbol2, valence)) in zip(self.symbol_count, symbol_valences):
            assert symbol1 == symbol2
            default_nelect += count * valence
        return default_nelect

    def write_input(self, atoms, directory='./'):
        from ase.io.vasp import write_vasp
        write_vasp(join(directory, 'POSCAR'),
                   self.atoms_sorted,
                   symbol_count=self.symbol_count,
                   ignore_constraints=self.input_params['ignore_constraints'])
        self.write_incar(atoms, directory=directory)
        self.write_potcar(directory=directory)
        self.write_kpoints(atoms=atoms, directory=directory)
        self.write_sort_file(directory=directory)
        self.copy_vdw_kernel(directory=directory)

    def copy_vdw_kernel(self, directory='./'):
        """Method to copy the vdw_kernel.bindat file.
        Set ASE_VASP_VDW environment variable to the vdw_kernel.bindat
        folder location. Checks if LUSE_VDW is enabled, and if no location
        for the vdW kernel is specified, a warning is issued."""

        vdw_env = 'ASE_VASP_VDW'
        kernel = 'vdw_kernel.bindat'
        dst = os.path.join(directory, kernel)

        # No need to copy the file again
        if isfile(dst):
            return

        if self.bool_params['luse_vdw']:
            src = None
            if vdw_env in cfg:
                src = os.path.join(cfg[vdw_env], kernel)

            if not src or not isfile(src):
                warnings.warn(
                    ('vdW has been enabled, however no'
                     ' location for the {} file'
                     ' has been specified.'
                     ' Set {} environment variable to'
                     ' copy the vdW kernel.').format(kernel, vdw_env))
            else:
                shutil.copyfile(src, dst)

    def clean(self):
        """Method which cleans up after a calculation.

        The default files generated by Vasp will be deleted IF this
        method is called.

        """
        files = [
            'CHG', 'CHGCAR', 'POSCAR', 'INCAR', 'CONTCAR', 'DOSCAR',
            'EIGENVAL', 'IBZKPT', 'KPOINTS', 'OSZICAR', 'OUTCAR', 'PCDAT',
            'POTCAR', 'vasprun.xml', 'WAVECAR', 'XDATCAR', 'PROCAR',
            'ase-sort.dat', 'LOCPOT', 'AECCAR0', 'AECCAR1', 'AECCAR2'
        ]
        for f in files:
            try:
                os.remove(f)
            except OSError:
                pass

    def write_incar(self, atoms, directory='./', **kwargs):
        """Writes the INCAR file."""
        incar_params = {}

        # float params
        float_dct = {
            key: f'{val:{FLOAT_FORMAT}}'
            for key, val in self.float_params.items()
            if val is not None
        }

        if 'charge' in self.input_params and self.input_params[
                'charge'] is not None:
            nelect_val = test_nelect_charge_compitability(
                self.float_params['nelect'],
                self.input_params['charge'],
                self.default_nelect_from_ppp())
            if nelect_val:
                float_dct['nelect'] = f'{nelect_val:{FLOAT_FORMAT}}'
        incar_params.update(float_dct)

        # exp params
        exp_dct = {
            key: f'{val:{EXP_FORMAT}}'
            for key, val in self.exp_params.items()
            if val is not None
        }
        incar_params.update(exp_dct)

        # string_params
        string_dct = {
            key: val for key, val in self.string_params.items() if val is not
            None
        }
        incar_params.update(string_dct)

        # int params
        int_dct = {
            key: val for key, val in self.int_params.items() if val is not None
        }
        if 'ichain' in int_dct.keys():
            ichain_dict = check_ichain(
                ichain=int_dct['ichain'],
                ediffg=self.exp_params.get('ediffg', None),
                iopt=int_dct.get('iopt', None),
            )
            int_dct.update(ichain_dict)
        incar_params.update(int_dct)

        # list_bool_params
        bool_dct = {
            key: val
            for key, val in self.list_bool_params.items()
            if val is not None
        }
        for key, val in bool_dct.items():
            bool_dct[key] = [_to_vasp_bool(x) for x in val]
        incar_params.update(bool_dct)

        # list_int_params
        int_dct = {
            key: val
            for key, val in self.list_int_params.items()
            if val is not None
        }
        if 'ldaul' in int_dct.keys() and self.dict_params[
                'ldau_luj'] is not None:
            del int_dct['ldaul']
        incar_params.update(int_dct)

        # list_float_params
        float_dct = {
            key: val
            for key, val in self.list_float_params.items()
            if val is not None
        }
        if 'ldauu' in float_dct.keys() and self.dict_params[
                'ldau_luj'] is not None:
            del float_dct['ldauu']
        if 'ldauj' in float_dct.keys() and self.dict_params[
                'ldau_luj'] is not None:
            del float_dct['ldauj']
        incar_params.update(float_dct)

        # bool params
        bool_dct = {
            key: _to_vasp_bool(val)
            for key, val in self.bool_params.items()
            if val is not None
        }
        incar_params.update(bool_dct)

        # special params
        special_dct = {
            key: val for key, val in self.special_params.items() if val is not
            None
        }
        if 'lreal' in special_dct.keys():
            if isinstance(special_dct['lreal'], bool):
                special_dct['lreal'] = _to_vasp_bool(special_dct['lreal'])
        incar_params.update(special_dct)

        # dict params
        dict_dct = {
            key: val for key, val in self.dict_params.items() if val is not None
        }
        if 'ldau_luj' in dict_dct.keys():
            ldau_dict = set_ldau(
                ldau_param=self.bool_params['ldau'],
                luj_params=dict_dct['ldau_luj'],
                symbol_count=self.symbol_count)
            dict_dct.update(ldau_dict)
            del dict_dct['ldau_luj']
        incar_params.update(dict_dct)

        # set magmom based on input or initial atoms object
        spinpol, magmom_dct = set_magmom(
            atoms=atoms,
            ispin=self.int_params['ispin'],
            spinpol=self.spinpol,
            magmom_input=float_dct.get('magmom', None),
            sorting=self.sort,
        )
        self.spinpol = spinpol
        incar_params.update(magmom_dct)

        # Custom key-value pairs, which receive no formatting
        # Use the comment "# <Custom ASE key>" to denote such
        # a custom key-value pair, as we cannot otherwise
        # reliably and easily identify such non-standard entries

        cust_dict = {
            key: str(val) + '  # <Custom ASE key>'
            for key, val in self.input_params['custom'].items()
            if val is not None
        }
        incar_params.update(cust_dict)

        write_incar(directory=directory, parameters=incar_params)

    def write_kpoints(self, atoms=None, directory='./', **kwargs):
        """Writes the KPOINTS file."""

        if atoms is None:
            atoms = self.atoms

        # Don't write anything if KSPACING is being used
        if self.float_params['kspacing'] is not None:
            if self.float_params['kspacing'] > 0:
                return
            else:
                raise ValueError("KSPACING value {} is not allowable. "
                                 "Please use None or a positive number."
                                 "".format(self.float_params['kspacing']))

        kpointstring = format_kpoints(
            kpts=self.input_params['kpts'],
            atoms=atoms,
            reciprocal=self.input_params['reciprocal'],
            gamma=self.input_params['gamma'])
        with open(join(directory, 'KPOINTS'), 'w') as kpoints:
            kpoints.write(kpointstring)

    def write_potcar(self, suffix="", directory='./'):
        """Writes the POTCAR file."""

        with open(join(directory, 'POTCAR' + suffix), 'w') as potfile:
            for filename in self.ppp_list:
                with open_potcar(filename=filename) as ppp_file:
                    for line in ppp_file:
                        potfile.write(line)

    def write_sort_file(self, directory='./'):
        """Writes a sortings file.

        This file contains information about how the atoms are sorted in
        the first column and how they should be resorted in the second
        column. It is used for restart purposes to get sorting right
        when reading in an old calculation to ASE."""

        with open(join(directory, 'ase-sort.dat'), 'w') as fd:
            for n in range(len(self.sort)):
                fd.write('%5i %5i \n' % (self.sort[n], self.resort[n]))

    # The below functions are used to restart a calculation

    def read_incar(self, filename):
        """Method that imports settings from INCAR file.

        Typically named INCAR."""

        self.spinpol = False
        with open(filename) as fd:
            lines = fd.readlines()

        for line in lines:
            try:
                # Make multiplication, comments, and parameters easier to spot
                line = line.replace("*", " * ")
                line = line.replace("=", " = ")
                line = line.replace("#", "# ")
                data = line.split()
                # Skip empty and commented lines.
                if len(data) == 0:
                    continue
                elif data[0][0] in ['#', '!']:
                    continue
                key = data[0].lower()
                if '<Custom ASE key>' in line:
                    # This key was added with custom key-value pair formatting.
                    # Unconditionally add it, no type checking
                    # Get value between "=" and the comment, e.g.
                    # key = 1 2 3  # <Custom ASE key>
                    # value should be '1 2 3'

                    # Split at first occurence of "="
                    value = line.split('=', 1)[1]
                    # First "#" denotes beginning of comment
                    # Add everything before comment as a string to custom dict
                    value = value.split('#', 1)[0].strip()
                    self.input_params['custom'][key] = value
                elif key in float_keys:
                    self.float_params[key] = float(data[2])
                elif key in exp_keys:
                    self.exp_params[key] = float(data[2])
                elif key in string_keys:
                    self.string_params[key] = str(data[2])
                elif key in int_keys:
                    if key == 'ispin':
                        # JRK added. not sure why we would want to leave ispin
                        # out
                        self.int_params[key] = int(data[2])
                        if int(data[2]) == 2:
                            self.spinpol = True
                    else:
                        self.int_params[key] = int(data[2])
                elif key in bool_keys:
                    val_char = data[2].lower().replace('.', '', 1)
                    if val_char.startswith('t'):
                        self.bool_params[key] = True
                    elif val_char.startswith('f'):
                        self.bool_params[key] = False
                    else:
                        raise ValueError(f'Invalid value "{data[2]}" for bool '
                                         f'key "{key}"')

                elif key in list_bool_keys:
                    self.list_bool_params[key] = [
                        _from_vasp_bool(x)
                        for x in _args_without_comment(data[2:])
                    ]

                elif key in list_int_keys:
                    self.list_int_params[key] = [
                        int(x) for x in _args_without_comment(data[2:])
                    ]

                elif key in list_float_keys:
                    if key == 'magmom':
                        lst = []
                        i = 2
                        while i < len(data):
                            if data[i] in ["#", "!"]:
                                break
                            if data[i] == "*":
                                b = lst.pop()
                                i += 1
                                for _ in range(int(b)):
                                    lst.append(float(data[i]))
                            else:
                                lst.append(float(data[i]))
                            i += 1
                        self.list_float_params['magmom'] = lst
                        lst = np.array(lst)
                        if self.atoms is not None:
                            self.atoms.set_initial_magnetic_moments(
                                lst[self.resort])
                    else:
                        data = _args_without_comment(data)
                        self.list_float_params[key] = [
                            float(x) for x in data[2:]
                        ]
                elif key in special_keys:
                    if key == 'lreal':
                        val_char = data[2].lower().replace('.', '', 1)
                        if val_char.startswith('t'):
                            self.bool_params[key] = True
                        elif val_char.startswith('f'):
                            self.bool_params[key] = False
                        else:
                            self.special_params[key] = data[2]

                # non-registered keys
                elif data[2].lower() in {'t', 'true', '.true.'}:
                    self.bool_params[key] = True
                elif data[2].lower() in {'f', 'false', '.false.'}:
                    self.bool_params[key] = False
                elif data[2].isdigit():
                    self.int_params[key] = int(data[2])
                else:
                    try:
                        self.float_params[key] = float(data[2])
                    except ValueError:
                        self.string_params[key] = data[2]

            except KeyError as exc:
                raise KeyError(
                    f'Keyword "{key}" in INCAR is not known by calculator.'
                ) from exc
            except IndexError as exc:
                raise IndexError(
                    f'Value missing for keyword "{key}".'
                ) from exc

    def read_kpoints(self, filename):
        """Read kpoints file, typically named KPOINTS."""
        # If we used VASP builtin kspacing,
        if self.float_params['kspacing'] is not None:
            # Don't update kpts array
            return

        with open(filename) as fd:
            lines = fd.readlines()

        ktype = lines[2].split()[0].lower()[0]
        if ktype in ['g', 'm', 'a']:
            if ktype == 'g':
                self.set(gamma=True)
                kpts = np.array([int(lines[3].split()[i]) for i in range(3)])
            elif ktype == 'a':
                kpts = np.array([int(lines[3].split()[i]) for i in range(1)])
            elif ktype == 'm':
                kpts = np.array([int(lines[3].split()[i]) for i in range(3)])
        else:
            if ktype in ['c', 'k']:
                self.set(reciprocal=False)
            else:
                self.set(reciprocal=True)
            kpts = np.array(
                [list(map(float, line.split())) for line in lines[3:]])
        self.set(kpts=kpts)

    def read_potcar(self, filename):
        """ Read the pseudopotential XC functional from POTCAR file.
        """

        # Search for key 'LEXCH' in POTCAR
        xc_flag = None
        with open(filename) as fd:
            for line in fd:
                key = line.split()[0].upper()
                if key == 'LEXCH':
                    xc_flag = line.split()[-1].upper()
                    break

        if xc_flag is None:
            raise ValueError('LEXCH flag not found in POTCAR file.')

        # Values of parameter LEXCH and corresponding XC-functional
        xc_dict = {'PE': 'PBE', '91': 'PW91', 'CA': 'LDA'}

        if xc_flag not in xc_dict.keys():
            raise ValueError('Unknown xc-functional flag found in POTCAR,'
                             ' LEXCH=%s' % xc_flag)

        self.input_params['pp'] = xc_dict[xc_flag]

    def todict(self):
        """Returns a dictionary of all parameters
        that can be used to construct a new calculator object"""
        dict_list = [
            'float_params', 'exp_params', 'string_params', 'int_params',
            'bool_params', 'list_bool_params', 'list_int_params',
            'list_float_params', 'special_params', 'dict_params',
            'input_params'
        ]
        dct = {}
        for item in dict_list:
            dct.update(getattr(self, item))
        dct = {key: value for key, value in dct.items() if value is not None}
        return dct


def _args_without_comment(data, marks=['!', '#']):
    """Check split arguments list for a comment, return data up to marker

    INCAR reader splits list arguments on spaces and leaves comment markers as
    individual items. This function returns only the data portion of the list.

    """
    comment_locs = [data.index(mark) for mark in marks if mark in data]
    if comment_locs == []:
        return data
    else:
        return data[:min(comment_locs)]


def _from_vasp_bool(x):
    """Cast vasp boolean to Python bool

    VASP files sometimes use T or F as shorthand for the preferred Boolean
    notation .TRUE. or .FALSE. As capitalisation is pretty inconsistent in
    practice, we allow all cases to be cast to a Python bool.

    """
    assert isinstance(x, str)
    if x.lower() == '.true.' or x.lower() == 't':
        return True
    elif x.lower() == '.false.' or x.lower() == 'f':
        return False
    else:
        raise ValueError(f'Value "{x}" not recognized as bool')


def _to_vasp_bool(x):
    """Convert Python boolean to string for VASP input

    In case the value was modified to a string already, appropriate strings
    will also be accepted and cast to a standard .TRUE. / .FALSE. format.

    """
    if isinstance(x, str):
        if x.lower() in ('.true.', 't'):
            x = True
        elif x.lower() in ('.false.', 'f'):
            x = False
        else:
            raise ValueError('"%s" not recognised as VASP Boolean')
    assert isinstance(x, bool)
    if x:
        return '.TRUE.'
    else:
        return '.FALSE.'


def open_potcar(filename):
    """ Open POTCAR file with transparent decompression if it's an archive (.Z)
    """
    import gzip
    if filename.endswith('R'):
        return open(filename)
    elif filename.endswith('.Z'):
        return gzip.open(filename)
    else:
        raise ValueError(f'Invalid POTCAR filename: "{filename}"')


def read_potcar_numbers_of_electrons(file_obj):
    """ Read list of tuples (atomic symbol, number of valence electrons)
    for each atomtype from a POTCAR file."""
    nelect = []
    lines = file_obj.readlines()
    for n, line in enumerate(lines):
        if 'TITEL' in line:
            symbol = line.split('=')[1].split()[1].split('_')[0].strip()
            valence = float(
                lines[n + 4].split(';')[1].split('=')[1].split()[0].strip())
            nelect.append((symbol, valence))
    return nelect


def count_symbols(atoms, exclude=()):
    """Count symbols in atoms object, excluding a set of indices

    Parameters:
        atoms: Atoms object to be grouped
        exclude: List of indices to be excluded from the counting

    Returns:
        Tuple of (symbols, symbolcount)
        symbols: The unique symbols in the included list
        symbolscount: Count of symbols in the included list

    Example:

    >>> from ase.build import bulk
    >>> atoms = bulk('NaCl', crystalstructure='rocksalt', a=4.1, cubic=True)
    >>> count_symbols(atoms)
    (['Na', 'Cl'], {'Na': 4, 'Cl': 4})
    >>> count_symbols(atoms, exclude=(1, 2, 3))
    (['Na', 'Cl'], {'Na': 3, 'Cl': 2})
    """
    symbols = []
    symbolcount = {}
    for m, symbol in enumerate(atoms.symbols):
        if m in exclude:
            continue
        if symbol not in symbols:
            symbols.append(symbol)
            symbolcount[symbol] = 1
        else:
            symbolcount[symbol] += 1
    return symbols, symbolcount
