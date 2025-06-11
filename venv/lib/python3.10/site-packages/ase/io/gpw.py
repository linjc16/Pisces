# fmt: off

"""Read gpw-file from GPAW."""
import ase.io.ulm as ulm
from ase import Atoms
from ase.calculators.singlepoint import (
    SinglePointDFTCalculator,
    SinglePointKPoint,
    all_properties,
)
from ase.io.trajectory import read_atoms
from ase.units import Bohr, Hartree


def read_gpw(filename):
    try:
        reader = ulm.open(filename)
    except ulm.InvalidULMFileError:
        return read_old_gpw(filename)

    atoms = read_atoms(reader.atoms, _try_except=False)

    wfs = reader.wave_functions
    kpts = wfs.get('kpts')
    if kpts is None:
        ibzkpts = None
        bzkpts = None
        bz2ibz = None
    else:
        ibzkpts = kpts.ibzkpts
        bzkpts = kpts.get('bzkpts')
        bz2ibz = kpts.get('bz2ibz')

    if reader.version >= 3:
        efermi = reader.wave_functions.fermi_levels.mean()
    else:
        efermi = reader.occupations.fermilevel

    atoms.calc = SinglePointDFTCalculator(
        atoms,
        efermi=efermi,
        ibzkpts=ibzkpts,
        bzkpts=bzkpts,
        bz2ibz=bz2ibz,
        # New gpw-files may have "non_collinear_magmom(s)" which ASE
        # doesn't know:
        **{property: value
           for property, value in reader.results.asdict().items()
           if property in all_properties})

    if kpts is not None:
        atoms.calc.kpts = []
        for spin, (eps_kn, f_kn) in enumerate(zip(wfs.eigenvalues,
                                                  wfs.occupations)):
            for kpt, (weight, eps_n, f_n) in enumerate(zip(kpts.weights,
                                                           eps_kn, f_kn)):
                atoms.calc.kpts.append(
                    SinglePointKPoint(weight, spin, kpt, eps_n, f_n))
    reader.close()

    return atoms


def read_old_gpw(filename):
    from gpaw.io.tar import Reader
    r = Reader(filename)
    positions = r.get('CartesianPositions') * Bohr
    numbers = r.get('AtomicNumbers')
    cell = r.get('UnitCell') * Bohr
    pbc = r.get('BoundaryConditions')
    tags = r.get('Tags')
    magmoms = r.get('MagneticMoments')
    energy = r.get('PotentialEnergy') * Hartree

    if r.has_array('CartesianForces'):
        forces = r.get('CartesianForces') * Hartree / Bohr
    else:
        forces = None

    atoms = Atoms(positions=positions,
                  numbers=numbers,
                  cell=cell,
                  pbc=pbc)
    if tags.any():
        atoms.set_tags(tags)

    if magmoms.any():
        atoms.set_initial_magnetic_moments(magmoms)
        magmom = magmoms.sum()
    else:
        magmoms = None
        magmom = None

    atoms.calc = SinglePointDFTCalculator(atoms, energy=energy,
                                          forces=forces,
                                          magmoms=magmoms,
                                          magmom=magmom)
    kpts = []
    if r.has_array('IBZKPoints'):
        for w, kpt, eps_n, f_n in zip(r.get('IBZKPointWeights'),
                                      r.get('IBZKPoints'),
                                      r.get('Eigenvalues'),
                                      r.get('OccupationNumbers')):
            kpts.append(SinglePointKPoint(w, kpt[0], kpt[1],
                                          eps_n[0], f_n[0]))
    atoms.calc.kpts = kpts

    return atoms
