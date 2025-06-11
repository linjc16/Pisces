import numpy as np
import pytest

from ase.build import bulk
from ase.calculators.abc import GetOutputsMixin
from ase.calculators.singlepoint import (
    SinglePointDFTCalculator,
    arrays_to_kpoints,
)
from ase.outputs import Properties, all_outputs


@pytest.fixture()
def rng():
    return np.random.RandomState(17)


@pytest.fixture()
def props(rng):
    nspins, nkpts, nbands = 2, 3, 5
    natoms = 4

    results = dict(
        natoms=natoms,
        energy=rng.random(),
        free_energy=rng.random(),
        energies=rng.random(natoms),
        forces=rng.random((natoms, 3)),
        stress=rng.random(6),
        stresses=rng.random((natoms, 6)),
        nspins=nspins,
        nkpts=nkpts,
        nbands=nbands,
        eigenvalues=rng.random((nspins, nkpts, nbands)),
        occupations=rng.random((nspins, nkpts, nbands)),
        fermi_level=rng.random(),
        ibz_kpoints=rng.random((nkpts, 3)),
        kpoint_weights=rng.random(nkpts),
        dipole=rng.random(3),
        charges=rng.random(natoms),
        magmom=rng.random(),
        magmoms=rng.random(natoms),
        polarization=rng.random(3),
        dielectric_tensor=rng.random((3, 3)),
        born_effective_charges=rng.random((natoms, 3, 3)),
    )
    return Properties(results)


def test_properties_big(props):
    for name in all_outputs:
        assert name in props, name
        obj = props[name]
        print(name, obj)


def test_singlepoint_roundtrip(props):

    atoms = bulk('Au') * (1, 1, props['natoms'])

    kpts = arrays_to_kpoints(props['eigenvalues'], props['occupations'],
                             props['kpoint_weights'])
    calc = SinglePointDFTCalculator(atoms=atoms, kpts=kpts,
                                    efermi=props['fermi_level'],
                                    forces=props['forces'])

    props1 = calc.properties()
    print(props1)

    assert set(props1) >= {
        'eigenvalues', 'occupations', 'kpoint_weights', 'fermi_level'}

    for prop in props1:
        assert props[prop] == pytest.approx(props1[prop])


def test_output_mixin(props, rng):

    class OutputsMixinTester(GetOutputsMixin):
        def __init__(self, props):
            self.results = props

        def _outputmixin_get_results(self):
            return self.results

    tester = OutputsMixinTester(props)

    for getter, key in [('get_fermi_level', 'fermi_level'),
                        ('get_ibz_k_points', 'ibz_kpoints'),
                        ('get_k_point_weights', 'kpoint_weights'),
                        ('get_number_of_bands', 'nbands'),
                        ('get_number_of_spins', 'nspins')]:

        assert getattr(tester, getter)() == pytest.approx(props[key])

        assert tester.get_spin_polarized() is True

    for spin in range(props['nspins']):
        for kpt_index in rng.choice(range(props['nkpts']), size=2):
            assert (tester.get_eigenvalues(kpt=kpt_index, spin=spin)
                    ==
                    pytest.approx(props['eigenvalues'][spin][kpt_index]))
            assert (tester.get_occupation_numbers(kpt=kpt_index, spin=spin)
                    ==
                    pytest.approx(props['occupations'][spin][kpt_index]))
