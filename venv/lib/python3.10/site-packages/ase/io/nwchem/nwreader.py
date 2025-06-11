# fmt: off

import re
from collections import OrderedDict

import numpy as np

from ase import Atoms
from ase.calculators.singlepoint import (
    SinglePointDFTCalculator,
    SinglePointKPoint,
)
from ase.units import Bohr, Hartree

from .parser import _define_pattern

# Note to the reader of this code: Here and below we use the function
# _define_pattern from parser.py in this same directory to compile
# regular expressions. These compiled expressions are stored along with
# an example string that the expression should match in a list that
# is used during tests (test/nwchem/nwchem_parser.py) to ensure that
# the regular expressions are still working correctly.

# Matches the beginning of a GTO calculation
_gauss_block = _define_pattern(
    r'^[\s]+NWChem (?:SCF|DFT) Module\n$',
    "                                 NWChem SCF Module\n",
)


# Matches the beginning of a plane wave calculation
_pw_block = _define_pattern(
    r'^[\s]+\*[\s]+NWPW (?:PSPW|BAND|PAW|Band Structure) Calculation'
    r'[\s]+\*[\s]*\n$',
    "          *               NWPW PSPW Calculation              *\n",
)


# Top-level parser
def read_nwchem_out(fobj, index=-1):
    """Splits an NWChem output file into chunks corresponding to
    individual single point calculations."""
    lines = fobj.readlines()

    if index == slice(-1, None, None):
        for line in lines:
            if _gauss_block.match(line):
                return [parse_gto_chunk(''.join(lines))]
            if _pw_block.match(line):
                return [parse_pw_chunk(''.join(lines))]
        raise ValueError('This does not appear to be a valid NWChem '
                         'output file.')

    # First, find each SCF block
    group = []
    atomslist = []
    header = True
    lastgroup = []
    lastparser = None
    parser = None
    for line in lines:
        group.append(line)
        if _gauss_block.match(line):
            next_parser = parse_gto_chunk
        elif _pw_block.match(line):
            next_parser = parse_pw_chunk
        else:
            continue

        if header:
            header = False
        else:
            atoms = parser(''.join(group))
            if atoms is None and parser is lastparser:
                atoms = parser(''.join(lastgroup + group))
                if atoms is not None:
                    atomslist[-1] = atoms
                    lastgroup += group
            else:
                atomslist.append(atoms)
                lastgroup = group
                lastparser = parser
            group = []
        parser = next_parser
    if not header:
        atoms = parser(''.join(group))
        if atoms is not None:
            atomslist.append(atoms)

    return atomslist[index]


# Matches a geometry block and returns the geometry specification lines
_geom = _define_pattern(
    r'\n[ \t]+Geometry \"[ \t\S]+\" -> \"[ \t\S]*\"[ \t]*\n'
    r'^[ \t-]+\n'
    r'(?:^[ \t\S]*\n){3}'
    r'^[ \t]+No\.[ \t]+Tag[ \t]+Charge[ \t]+X[ \t]+Y[ \t]+Z\n'
    r'^[ \t-]+\n'
    r'((?:^(?:[ \t]+[\S]+){6}[ \t]*\n)+)',
    """\

                             Geometry "geometry" -> ""
                             -------------------------

 Output coordinates in angstroms (scale by  1.889725989 to convert to a.u.)

  No.       Tag          Charge          X              Y              Z
 ---- ---------------- ---------- -------------- -------------- --------------
    1 C                    6.0000     0.00000000     0.00000000     0.00000000
    2 H                    1.0000     0.62911800     0.62911800     0.62911800
    3 H                    1.0000    -0.62911800    -0.62911800     0.62911800
    4 H                    1.0000     0.62911800    -0.62911800    -0.62911800
""", re.M)

# Unit cell parser
_cell_block = _define_pattern(r'^[ \t]+Lattice Parameters[ \t]*\n'
                              r'^(?:[ \t\S]*\n){4}'
                              r'((?:^(?:[ \t]+[\S]+){5}\n){3})',
                              """\
      Lattice Parameters
      ------------------

      lattice vectors in angstroms (scale by  1.889725989 to convert to a.u.)

      a1=<   4.000   0.000   0.000 >
      a2=<   0.000   5.526   0.000 >
      a3=<   0.000   0.000   4.596 >
      a=       4.000 b=      5.526 c=       4.596
      alpha=  90.000 beta=  90.000 gamma=  90.000
      omega=   101.6
""", re.M)


# Parses the geometry and returns the corresponding Atoms object
def _parse_geomblock(chunk):
    geomblocks = _geom.findall(chunk)
    if not geomblocks:
        return None
    geomblock = geomblocks[-1].strip().split('\n')
    natoms = len(geomblock)
    symbols = []
    pos = np.zeros((natoms, 3))
    for i, line in enumerate(geomblock):
        line = line.strip().split()
        symbols.append(line[1])
        pos[i] = [float(x) for x in line[3:6]]

    cellblocks = _cell_block.findall(chunk)
    if cellblocks:
        cellblock = cellblocks[-1].strip().split('\n')
        cell = np.zeros((3, 3))
        for i, line in enumerate(cellblock):
            line = line.strip().split()
            cell[i] = [float(x) for x in line[1:4]]
    else:
        cell = None
    return Atoms(symbols, positions=pos, cell=cell)


# GTO-specific parser stuff

# Matches gradient block from a GTO calculation
_gto_grad = _define_pattern(
    r'^[ \t]+[\S]+[ \t]+ENERGY GRADIENTS[ \t]*[\n]+'
    r'^[ \t]+atom[ \t]+coordinates[ \t]+gradient[ \t]*\n'
    r'^(?:[ \t]+x[ \t]+y[ \t]+z){2}[ \t]*\n'
    r'((?:^(?:[ \t]+[\S]+){8}\n)+)[ \t]*\n',
    """\
                         UHF ENERGY GRADIENTS

    atom               coordinates                        gradient
                 x          y          z           x          y          z
   1 C       0.293457  -0.293457   0.293457   -0.000083   0.000083  -0.000083
   2 H       1.125380   1.355351   1.125380    0.000086   0.000089   0.000086
   3 H      -1.355351  -1.125380   1.125380   -0.000089  -0.000086   0.000086
   4 H       1.125380  -1.125380  -1.355351    0.000086  -0.000086  -0.000089

""", re.M)

# Energy parsers for a variety of different GTO calculations
_e_gto = OrderedDict()
_e_gto['tce'] = _define_pattern(
    r'^[\s]+[\S]+[\s]+total energy \/ hartree[\s]+'
    r'=[\s]+([\S]+)[\s]*\n',
    " CCD total energy / hartree       "
    "=       -75.715332545665888\n", re.M,
)
_e_gto['ccsd'] = _define_pattern(
    r'^[\s]+Total CCSD energy:[\s]+([\S]+)[\s]*\n',
    " Total CCSD energy:            -75.716168566598569\n",
    re.M,
)
_e_gto['tddft'] = _define_pattern(
    r'^[\s]+Excited state energy =[\s]+([\S]+)[\s]*\n',
    "     Excited state energy =    -75.130134499965\n",
    re.M,
)
_e_gto['mp2'] = _define_pattern(
    r'^[\s]+Total MP2 energy[\s]+([\S]+)[\s]*\n',
    "          Total MP2 energy           -75.708800087578\n",
    re.M,
)
_e_gto['mf'] = _define_pattern(
    r'^[\s]+Total (?:DFT|SCF) energy =[\s]+([\S]+)[\s]*\n',
    "         Total SCF energy =    -75.585555997789\n",
    re.M,
)


# GTO parser
def parse_gto_chunk(chunk):
    atoms = None
    forces = None
    energy = None
    dipole = None
    for theory, pattern in _e_gto.items():
        matches = pattern.findall(chunk)
        if matches:
            energy = float(matches[-1].replace('D', 'E')) * Hartree
            break

    gradblocks = _gto_grad.findall(chunk)
    if gradblocks:
        gradblock = gradblocks[-1].strip().split('\n')
        natoms = len(gradblock)
        symbols = []
        pos = np.zeros((natoms, 3))
        forces = np.zeros((natoms, 3))
        for i, line in enumerate(gradblock):
            line = line.strip().split()
            symbols.append(line[1])
            pos[i] = [float(x) for x in line[2:5]]
            forces[i] = [-float(x) for x in line[5:8]]
        pos *= Bohr
        forces *= Hartree / Bohr
        atoms = Atoms(symbols, positions=pos)

    dipole, _quadrupole = _get_multipole(chunk)

    kpts = _get_gto_kpts(chunk)

    if atoms is None:
        atoms = _parse_geomblock(chunk)

    if atoms is None:
        return None

    # SinglePointDFTCalculator doesn't support quadrupole moment currently
    calc = SinglePointDFTCalculator(atoms=atoms,
                                    energy=energy,
                                    free_energy=energy,  # XXX Is this right?
                                    forces=forces,
                                    dipole=dipole,
                                    # quadrupole=quadrupole,
                                    )
    calc.kpts = kpts
    atoms.calc = calc
    return atoms


# Extracts dipole and quadrupole moment for a GTO calculation
# Note on the regex: Some, but not all, versions of NWChem
# insert extra spaces in the blank lines. Do not remove the \s*
# in between \n and \n
_multipole = _define_pattern(
    r'^[ \t]+Multipole analysis of the density[ \t\S]*\n'
    r'^[ \t-]+\n\s*\n^[ \t\S]+\n^[ \t-]+\n'
    r'((?:(?:(?:[ \t]+[\S]+){7,8}\n)|[ \t]*\n){12})',
    """\
     Multipole analysis of the density
     ---------------------------------

     L   x y z        total         alpha         beta         nuclear
     -   - - -        -----         -----         ----         -------
     0   0 0 0     -0.000000     -5.000000     -5.000000     10.000000

     1   1 0 0      0.000000      0.000000      0.000000      0.000000
     1   0 1 0     -0.000001     -0.000017     -0.000017      0.000034
     1   0 0 1     -0.902084     -0.559881     -0.559881      0.217679

     2   2 0 0     -5.142958     -2.571479     -2.571479      0.000000
     2   1 1 0     -0.000000     -0.000000     -0.000000      0.000000
     2   1 0 1      0.000000      0.000000      0.000000      0.000000
     2   0 2 0     -3.153324     -3.807308     -3.807308      4.461291
     2   0 1 1      0.000001     -0.000009     -0.000009      0.000020
     2   0 0 2     -4.384288     -3.296205     -3.296205      2.208122
""", re.M)


# Parses the dipole and quadrupole moment from a GTO calculation
def _get_multipole(chunk):
    matches = _multipole.findall(chunk)
    if not matches:
        return None, None
    # This pulls the 5th column out of the multipole moments block;
    # this column contains the actual moments.
    moments = [float(x.split()[4]) for x in matches[-1].split('\n')
               if x and not x.isspace()]
    dipole = np.array(moments[1:4]) * Bohr
    quadrupole = np.zeros(9)
    quadrupole[[0, 1, 2, 4, 5, 8]] = [moments[4:]]
    quadrupole[[3, 6, 7]] = quadrupole[[1, 2, 5]]
    return dipole, quadrupole.reshape((3, 3)) * Bohr**2


# MO eigenvalue and occupancy parser for GTO calculations
_eval_block = _define_pattern(
    r'^[ \t]+[\S]+ Final (?:Alpha |Beta )?Molecular Orbital Analysis[ \t]*'
    r'\n^[ \t-]+\n\n'
    r'(?:^[ \t]+Vector [ \t\S]+\n(?:^[ \t\S]+\n){3}'
    r'(?:^(?:(?:[ \t]+[\S]+){5}){1,2}[ \t]*\n)+\n)+',
    """\
                       ROHF Final Molecular Orbital Analysis
                       -------------------------------------

 Vector    1  Occ=2.000000D+00  E=-2.043101D+01
              MO Center=  1.1D-20,  1.5D-18,  1.2D-01, r^2= 1.5D-02
   Bfn.  Coefficient  Atom+Function         Bfn.  Coefficient  Atom+Function  
  ----- ------------  ---------------      ----- ------------  ---------------
     1      0.983233  1 O  s          

 Vector    2  Occ=2.000000D+00  E=-1.324439D+00
              MO Center= -2.1D-18, -8.6D-17, -7.1D-02, r^2= 5.1D-01
   Bfn.  Coefficient  Atom+Function         Bfn.  Coefficient  Atom+Function  
  ----- ------------  ---------------      ----- ------------  ---------------
     6      0.708998  1 O  s                  1     -0.229426  1 O  s          
     2      0.217752  1 O  s          
     """, re.M)  # noqa: W291


# Parses the eigenvalues and occupations from a GTO calculation
def _get_gto_kpts(chunk):
    eval_blocks = _eval_block.findall(chunk)
    if not eval_blocks:
        return []
    kpts = []
    kpt = _get_gto_evals(eval_blocks[-1])
    if kpt.s == 1:
        kpts.append(_get_gto_evals(eval_blocks[-2]))
    kpts.append(kpt)
    return kpts


# Extracts MO eigenvalue and occupancy for a GTO calculation
_extract_vector = _define_pattern(
    r'^[ \t]+Vector[ \t]+([\S])+[ \t]+Occ=([\S]+)[ \t]+E=[ \t]*([\S]+)[ \t]*\n',
    " Vector    1  Occ=2.000000D+00  E=-2.043101D+01\n", re.M,
)


# Extracts the eigenvalues and occupations from a GTO calculation
def _get_gto_evals(chunk):
    spin = 1 if re.match(r'[ \t\S]+Beta', chunk) else 0
    data = []
    for vector in _extract_vector.finditer(chunk):
        data.append([float(x.replace('D', 'E')) for x in vector.groups()[1:]])
    data = np.array(data)
    occ = data[:, 0]
    energies = data[:, 1] * Hartree

    return SinglePointKPoint(1., spin, 0, energies, occ)


# Plane wave specific parsing stuff

# Matches the gradient block from a plane wave calculation
_nwpw_grad = _define_pattern(
    r'^[ \t]+[=]+[ \t]+Ion Gradients[ \t]+[=]+[ \t]*\n'
    r'^[ \t]+Ion Forces:[ \t]*\n'
    r'((?:^(?:[ \t]+[\S]+){7}\n)+)',
    """\
          =============  Ion Gradients =================
 Ion Forces:
        1 O    (   -0.000012    0.000027   -0.005199 )
        2 H    (    0.000047   -0.013082    0.020790 )
        3 H    (    0.000047    0.012863    0.020786 )
        C.O.M. (   -0.000000   -0.000000   -0.000000 )
          ===============================================
""", re.M)

# Matches the gradient block from a PAW calculation
_paw_grad = _define_pattern(
    r'^[ \t]+[=]+[ \t]+Ion Gradients[ \t]+[=]+[ \t]*\n'
    r'^[ \t]+Ion Positions:[ \t]*\n'
    r'((?:^(?:[ \t]+[\S]+){7}\n)+)'
    r'^[ \t]+Ion Forces:[ \t]*\n'
    r'((?:^(?:[ \t]+[\S]+){7}\n)+)',
    """\
          =============  Ion Gradients =================
 Ion Positions:
        1 O    (   -3.77945   -5.22176   -3.77945 )
        2 H    (   -3.77945   -3.77945    3.77945 )
        3 H    (   -3.77945    3.77945    3.77945 )
 Ion Forces:
        1 O    (   -0.00001   -0.00000    0.00081 )
        2 H    (    0.00005   -0.00026   -0.00322 )
        3 H    (    0.00005    0.00030   -0.00322 )
        C.O.M. (   -0.00000   -0.00000   -0.00000 )
          ===============================================
""", re.M)

# Energy parser for plane wave calculations
_nwpw_energy = _define_pattern(
    r'^[\s]+Total (?:PSPW|BAND|PAW) energy'
    r'[\s]+:[\s]+([\S]+)[\s]*\n',
    " Total PSPW energy     :  -0.1709317826E+02\n",
    re.M,
)

# Parser for the fermi energy in a plane wave calculation
_fermi_energy = _define_pattern(
    r'^[ \t]+Fermi energy =[ \t]+([\S]+) \([ \t]+[\S]+[ \t]*\n',
    "  Fermi energy =    -0.5585062E-01 (  -1.520eV)\n", re.M,
)


# Plane wave parser
def parse_pw_chunk(chunk):
    atoms = _parse_geomblock(chunk)
    if atoms is None:
        return None

    energy = None
    efermi = None
    forces = None
    stress = None

    matches = _nwpw_energy.findall(chunk)
    if matches:
        energy = float(matches[-1].replace('D', 'E')) * Hartree

    matches = _fermi_energy.findall(chunk)
    if matches:
        efermi = float(matches[-1].replace('D', 'E')) * Hartree

    gradblocks = _nwpw_grad.findall(chunk)
    if not gradblocks:
        gradblocks = _paw_grad.findall(chunk)
    if gradblocks:
        gradblock = gradblocks[-1].strip().split('\n')
        natoms = len(gradblock)
        symbols = []
        forces = np.zeros((natoms, 3))
        for i, line in enumerate(gradblock):
            line = line.strip().split()
            symbols.append(line[1])
            forces[i] = [float(x) for x in line[3:6]]
        forces *= Hartree / Bohr

    if atoms.cell:
        stress = _get_stress(chunk, atoms.cell)

    ibz_kpts, kpts = _get_pw_kpts(chunk)

    # NWChem does not calculate an energy extrapolated to the 0K limit,
    # so right now, energy and free_energy will be the same.
    calc = SinglePointDFTCalculator(atoms=atoms,
                                    energy=energy,
                                    efermi=efermi,
                                    free_energy=energy,
                                    forces=forces,
                                    stress=stress,
                                    ibzkpts=ibz_kpts)
    calc.kpts = kpts
    atoms.calc = calc
    return atoms


# Extracts stress tensor from a plane wave calculation
_stress = _define_pattern(
    r'[ \t]+[=]+[ \t]+(?:total gradient|E all FD)[ \t]+[=]+[ \t]*\n'
    r'^[ \t]+S =((?:(?:[ \t]+[\S]+){5}\n){3})[ \t=]+\n',
    """\
          ============= total gradient ==============
      S =  (   -0.22668    0.27174    0.19134 )
           (    0.23150   -0.26760    0.23226 )
           (    0.19090    0.27206   -0.22700 )
          ===================================================
""", re.M)


# Extract stress tensor from a plane wave calculation
def _get_stress(chunk, cell):
    stress_blocks = _stress.findall(chunk)
    if not stress_blocks:
        return None
    stress_block = stress_blocks[-1]
    stress = np.zeros((3, 3))
    for i, row in enumerate(stress_block.strip().split('\n')):
        stress[i] = [float(x) for x in row.split()[1:4]]
    stress = (stress @ cell) * Hartree / Bohr / cell.volume
    stress = 0.5 * (stress + stress.T)
    # convert from 3x3 array to Voigt form
    return stress.ravel()[[0, 4, 8, 5, 2, 1]]


# MO/band eigenvalue and occupancy parser for plane wave calculations
_nwpw_eval_block = _define_pattern(
    r'(?:(?:^[ \t]+Brillouin zone point:[ \t]+[\S]+[ \t]*\n'
    r'(?:[ \t\S]*\n){3,4})?'
    r'^[ \t]+(?:virtual )?orbital energies:\n'
    r'(?:^(?:(?:[ \t]+[\S]+){3,4}){1,2}[ \t]*\n)+\n{,3})+',
    """\
 Brillouin zone point:      1
    weight=  0.074074
    k     =<   0.333   0.333   0.333> . <b1,b2,b3> 
          =<   0.307   0.307   0.307>

 orbital energies:
     0.3919370E+00 (  10.665eV) occ=1.000
     0.3908827E+00 (  10.637eV) occ=1.000     0.4155535E+00 (  11.308eV) occ=1.000
     0.3607689E+00 (   9.817eV) occ=1.000     0.3827820E+00 (  10.416eV) occ=1.000
     0.3544000E+00 (   9.644eV) occ=1.000     0.3782641E+00 (  10.293eV) occ=1.000
     0.3531137E+00 (   9.609eV) occ=1.000     0.3778819E+00 (  10.283eV) occ=1.000
     0.2596367E+00 (   7.065eV) occ=1.000     0.2820723E+00 (   7.676eV) occ=1.000

 Brillouin zone point:      2
    weight=  0.074074
    k     =<  -0.000   0.333   0.333> . <b1,b2,b3> 
          =<   0.614   0.000   0.000>

 orbital energies:
     0.3967132E+00 (  10.795eV) occ=1.000
     0.3920006E+00 (  10.667eV) occ=1.000     0.4197952E+00 (  11.423eV) occ=1.000
     0.3912442E+00 (  10.646eV) occ=1.000     0.4125086E+00 (  11.225eV) occ=1.000
     0.3910472E+00 (  10.641eV) occ=1.000     0.4124238E+00 (  11.223eV) occ=1.000
     0.3153977E+00 (   8.582eV) occ=1.000     0.3379797E+00 (   9.197eV) occ=1.000
     0.2801606E+00 (   7.624eV) occ=1.000     0.3052478E+00 (   8.306eV) occ=1.000
""", re.M)  # noqa: E501, W291

# Parser for kpoint weights for a plane wave calculation
_kpt_weight = _define_pattern(
    r'^[ \t]+Brillouin zone point:[ \t]+([\S]+)[ \t]*\n'
    r'^[ \t]+weight=[ \t]+([\S]+)[ \t]*\n',
    """\
 Brillouin zone point:      1
    weight=  0.074074  
""", re.M)  # noqa: W291


# Parse eigenvalues and occupancies from a plane wave calculation
def _get_pw_kpts(chunk):
    eval_blocks = []
    for block in _nwpw_eval_block.findall(chunk):
        if 'pathlength' not in block:
            eval_blocks.append(block)
    if not eval_blocks:
        return []
    if 'virtual' in eval_blocks[-1]:
        occ_block = eval_blocks[-2]
        virt_block = eval_blocks[-1]
    else:
        occ_block = eval_blocks[-1]
        virt_block = ''
    kpts = NWChemKpts()
    _extract_pw_kpts(occ_block, kpts, 1.)
    _extract_pw_kpts(virt_block, kpts, 0.)
    for match in _kpt_weight.finditer(occ_block):
        index, weight = match.groups()
        kpts.set_weight(index, float(weight))
    return kpts.to_ibz_kpts(), kpts.to_singlepointkpts()


# Helper class for keeping track of kpoints and converting to
# SinglePointKPoint objects.
class NWChemKpts:
    def __init__(self):
        self.data = {}
        self.ibz_kpts = {}
        self.weights = {}

    def add_ibz_kpt(self, index, raw_kpt):
        kpt = np.array([float(x.strip('>')) for x in raw_kpt.split()[1:4]])
        self.ibz_kpts[index] = kpt

    def add_eval(self, index, spin, energy, occ):
        if index not in self.data:
            self.data[index] = {}
        if spin not in self.data[index]:
            self.data[index][spin] = []
        self.data[index][spin].append((energy, occ))

    def set_weight(self, index, weight):
        self.weights[index] = weight

    def to_ibz_kpts(self):
        if not self.ibz_kpts:
            return np.array([[0., 0., 0.]])
        sorted_kpts = sorted(list(self.ibz_kpts.items()), key=lambda x: x[0])
        return np.array(list(zip(*sorted_kpts))[1])

    def to_singlepointkpts(self):
        kpts = []
        for i, (index, spins) in enumerate(self.data.items()):
            weight = self.weights[index]
            for spin, (_, data) in enumerate(spins.items()):
                energies, occs = np.array(sorted(data, key=lambda x: x[0])).T
                kpts.append(SinglePointKPoint(weight, spin, i, energies, occs))
        return kpts


# Extracts MO/band data from a pattern matched by _nwpw_eval_block above
_kpt = _define_pattern(
    r'^[ \t]+Brillouin zone point:[ \t]+([\S]+)[ \t]*\n'
    r'^[ \t]+weight=[ \t]+([\S])+[ \t]*\n'
    r'^[ \t]+k[ \t]+([ \t\S]+)\n'
    r'(?:^[ \t\S]*\n){1,2}'
    r'^[ \t]+(?:virtual )?orbital energies:\n'
    r'((?:^(?:(?:[ \t]+[\S]+){3,4}){1,2}[ \t]*\n)+)',
    """\
 Brillouin zone point:      1
    weight=  0.074074
    k     =<   0.333   0.333   0.333> . <b1,b2,b3> 
          =<   0.307   0.307   0.307>

 orbital energies:
     0.3919370E+00 (  10.665eV) occ=1.000
     0.3908827E+00 (  10.637eV) occ=1.000     0.4155535E+00 (  11.308eV) occ=1.000
     0.3607689E+00 (   9.817eV) occ=1.000     0.3827820E+00 (  10.416eV) occ=1.000
     0.3544000E+00 (   9.644eV) occ=1.000     0.3782641E+00 (  10.293eV) occ=1.000
     0.3531137E+00 (   9.609eV) occ=1.000     0.3778819E+00 (  10.283eV) occ=1.000
     0.2596367E+00 (   7.065eV) occ=1.000     0.2820723E+00 (   7.676eV) occ=1.000
""", re.M)  # noqa: E501, W291


# Extracts kpoints from a plane wave calculation
def _extract_pw_kpts(chunk, kpts, default_occ):
    for match in _kpt.finditer(chunk):
        point, weight, raw_kpt, orbitals = match.groups()
        index = int(point) - 1
        for line in orbitals.split('\n'):
            tokens = line.strip().split()
            if not tokens:
                continue
            ntokens = len(tokens)
            a_e = float(tokens[0]) * Hartree
            if ntokens % 3 == 0:
                a_o = default_occ
            else:
                a_o = float(tokens[3].split('=')[1])
            kpts.add_eval(index, 0, a_e, a_o)

            if ntokens <= 4:
                continue
            if ntokens == 6:
                b_e = float(tokens[3]) * Hartree
                b_o = default_occ
            elif ntokens == 8:
                b_e = float(tokens[4]) * Hartree
                b_o = float(tokens[7].split('=')[1])
            kpts.add_eval(index, 1, b_e, b_o)
        kpts.set_weight(index, float(weight))
        kpts.add_ibz_kpt(index, raw_kpt)
