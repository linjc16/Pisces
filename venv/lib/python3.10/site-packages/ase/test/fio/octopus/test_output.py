"""Tests for Octopus outputs."""
from typing import Any, Dict

import numpy as np
import pytest

from ase.io.octopus.output import read_eigenvalues_file, read_static_info
from ase.units import Bohr, Debye, Hartree


@pytest.fixture(name='info_iron')
def fixture_info_iron(datadir) -> Dict[str, Any]:
    """`info` of 'periodic_systems/25-Fe_polarized.01-gs'"""
    file = datadir / 'octopus/periodic_systems_25-Fe_polarized.01-gs_info'
    with file.open(encoding='utf-8') as fd:
        return read_static_info(fd)


def test_eigenvalues(datadir):
    """Test if the eigenvalues are parsed correctly from `eigenvalues`."""
    file = (
        datadir
        / 'octopus/periodic_systems_03-sodium_chain.02-unocc_eigenvalues'
    )
    with file.open(encoding='utf-8') as fd:
        kptsarr, eigsarr, occsarr = read_eigenvalues_file(fd)
    kptsarr_ref = [
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.2, 0.0, 0.0],
        [0.3, 0.0, 0.0],
        [0.4, 0.0, 0.0],
        [0.5, 0.0, 0.0],
    ]
    eigsarr_ref = [
        [[-3.647610, -1.236399]],
        [[-3.551626, -1.144852]],
        [[-3.264556, -0.870333]],
        [[-2.789110, -0.413463]],
        [[-2.130396, -0.280652]],
        [[-1.367358, -1.210151]],
    ]
    occsarr_ref = [
        [[2.0, 0.0]],
        [[2.0, 0.0]],
        [[2.0, 0.0]],
        [[0.0, 0.0]],
        [[0.0, 0.0]],
        [[0.0, 0.0]],
    ]
    np.testing.assert_allclose(kptsarr, kptsarr_ref)
    np.testing.assert_allclose(eigsarr, eigsarr_ref)
    np.testing.assert_allclose(occsarr, occsarr_ref)


def test_fermi_level(info_iron: Dict[str, Any]):
    """Test if the Fermi level is parsed correctly."""
    efermi_ref = 0.153766 * Hartree
    np.testing.assert_allclose(info_iron['fermi_level'], efermi_ref)


def test_dipole_moment(datadir):
    """Test if the dipole moment is parsed correctly."""
    file = datadir / 'octopus/linear_response_02-h2o_pol_lr.01_h2o_gs_info'
    with file.open(encoding='utf-8') as fd:
        results = read_static_info(fd)
    dipole_ref = np.array((7.45151E-16, 9.30594E-01, 3.24621E-15)) * Debye
    np.testing.assert_allclose(results['dipole'], dipole_ref)


def test_magnetic_moment(info_iron: Dict[str, Any]):
    """Test if the magnetic moment is parsed correctly."""
    magmom_ref = 7.409638
    magmoms_ref = [3.385730, 3.385730]
    np.testing.assert_allclose(info_iron['magmom'], magmom_ref)
    np.testing.assert_allclose(info_iron['magmoms'], magmoms_ref)


def test_stress(datadir):
    """Test if the stress tensor is parsed correctly."""
    file = datadir / 'octopus/periodic_systems_30-stress.05-output_scf_info'
    with file.open(encoding='utf-8') as fd:
        results = read_static_info(fd)
    stress_ref = (
        (-5.100825936E-04, -9.121282047E-16, +4.495864617E-16),
        (-9.121277710E-16, -5.100825936E-04, -1.086184124E-15),
        (+4.496421897E-16, -1.086295146E-15, -5.100825937E-04),
    )
    stress_ref = np.array(stress_ref) * Hartree / Bohr**3
    np.testing.assert_allclose(results['stress'], stress_ref)


def test_kpoints(info_iron: Dict[str, Any]):
    """Test if the kpoints are parsed correctly."""
    nkpts_ref = 4
    kpoint_weights_ref = [0.25, 0.25, 0.25, 0.25]
    ibz_kpoints_ref = [
        [0.0000, 0.0000, 0.0000],
        [0.5000, 0.0000, 0.0000],
        [0.0000, 0.5000, 0.0000],
        [0.5000, 0.5000, 0.0000],
    ]
    assert info_iron['nkpts'] == nkpts_ref
    np.testing.assert_allclose(info_iron['kpoint_weights'], kpoint_weights_ref)
    np.testing.assert_allclose(info_iron['ibz_kpoints'], ibz_kpoints_ref)
