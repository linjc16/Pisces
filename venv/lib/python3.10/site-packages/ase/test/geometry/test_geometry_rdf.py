import numpy as np
import pytest

from ase.build.bulk import bulk
from ase.build.molecule import molecule
from ase.calculators.emt import EMT
from ase.cluster import Icosahedron
from ase.geometry.rdf import (
    CellTooSmall,
    VolumeNotDefined,
    get_rdf,
    get_volume_estimate,
)
from ase.lattice.compounds import L1_2
from ase.optimize.fire import FIRE


@pytest.fixture()
def atoms_h2():
    return molecule('H2')


def test_rdf_providing_volume_argument(atoms_h2):
    volume_estimate = get_volume_estimate(atoms_h2)
    rdf, dists = get_rdf(atoms_h2, 2.0, 5, volume=volume_estimate)

    rdf_ref = (0.0, 2.91718861, 0.0, 0.0, 0.0)
    dists_ref = (0.2, 0.6, 1.0, 1.4, 1.8)
    assert rdf == pytest.approx(rdf_ref)
    assert dists == pytest.approx(dists_ref)


def test_rdf_volume_not_defined_exception(atoms_h2):
    with pytest.raises(VolumeNotDefined):
        get_rdf(atoms_h2, 2.0, 5)


def test_rdf_cell_too_small_exception():
    with pytest.raises(CellTooSmall):
        get_rdf(bulk('Ag'), 2.0, 5)


def test_rdf_compute():
    eps = 1e-5

    atoms = Icosahedron('Cu', 3)
    atoms.center(vacuum=0.0)
    atoms.numbers[[0, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28, 30]] = 79
    atoms.calc = EMT()
    with FIRE(atoms, logfile=None) as opt:
        opt.run(fmax=0.05)

    rmax = 8.
    nbins = 5
    rdf, dists = get_rdf(atoms, rmax, nbins)
    calc_dists = np.arange(rmax / (2 * nbins), rmax, rmax / nbins)
    assert all(abs(dists - calc_dists) < eps)
    reference_rdf1 = [0., 0.84408157, 0.398689, 0.23748934, 0.15398546]
    assert all(abs(rdf - reference_rdf1) < eps)

    dm = atoms.get_all_distances()
    s = np.zeros(5)
    for c in [(29, 29), (29, 79), (79, 29), (79, 79)]:
        inv_norm = len(np.where(atoms.numbers == c[0])[0]) / len(atoms)
        s += get_rdf(atoms, rmax, nbins, elements=c,
                     distance_matrix=dm, no_dists=True) * inv_norm
    assert all(abs(s - reference_rdf1) < eps)

    AuAu = get_rdf(atoms, rmax, nbins, elements=(79, 79),
                   distance_matrix=dm, no_dists=True)
    assert all(abs(AuAu[-2:] - [0.12126445, 0.]) < eps)

    bulk = L1_2(['Au', 'Cu'], size=(3, 3, 3), latticeconstant=2 * np.sqrt(2))
    rdf = get_rdf(bulk, 4.2, 5)[0]
    reference_rdf2 = [0., 0., 1.43905094, 0.36948605, 1.34468694]
    assert all(abs(rdf - reference_rdf2) < eps)
