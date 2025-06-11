import numpy as np
import pytest

from ase.build import molecule


@pytest.fixture()
def atoms():
    atoms = molecule('H2')
    atoms.positions -= atoms.positions[0]
    assert atoms.positions[0] == pytest.approx([0, 0, 0])
    atoms.pbc = 1
    atoms.cell = [5, 5, 6]
    return atoms


k_ref_0 = 40.0  # Arbitrary "default" reference value


# Spring constant of H-H bond
k_refs = dict(
    abinit=46,
    cp2k=44,
    espresso=43,
    gpaw=39,
    mopac=66,
    nwchem=42,
    siesta=45,
)


calc = pytest.mark.calculator


@calc('abinit', chksymtnons=0)
@calc('cp2k')
@calc('espresso', input_data={"control": {"tprnfor": True}})
@calc('gpaw', mode='pw', symmetry='off', txt=None)
@calc('mopac', method='PM7', task='1SCF UHF GRADIENTS')
@calc('nwchem')
@calc('siesta')
def test_h2_bond(factory, atoms):
    d0 = atoms.get_distance(0, 1)
    atoms.calc = factory.calc()

    X = d0 + np.linspace(-0.08, 0.08, 5)
    E = []
    F = []

    for x in X:
        atoms.positions[1, 2] = x
        e = atoms.get_potential_energy(force_consistent=True)
        f = atoms.get_forces()
        E.append(e)
        F.append(f[1, 2])
    E = np.array(E)
    F = np.array(F)

    a, b, _c = np.polyfit(X, E, 2)
    xmin = -b / (2.0 * a)
    fa, _fb = np.polyfit(X, F, 1)

    k_from_energy = 2 * a
    k_from_forces = -fa

    # Not very strict for a bond length, but parameters are not consistent:
    assert xmin == pytest.approx(0.77, rel=0.05)
    assert k_from_energy == pytest.approx(k_from_forces, rel=0.05)
    assert k_from_energy == pytest.approx(k_refs.get(factory.name, k_ref_0),
                                          rel=0.05)
