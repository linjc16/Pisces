import os

import numpy as np

from ase import Atoms, units
from ase.build import bulk, molecule
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
from ase.io.trajectory import Trajectory
from ase.phonons import Phonons


def check_set_atoms(atoms, set_atoms, expected_atoms):
    """ Perform a test that .set_atoms() only displaces the expected atoms. """
    atoms.calc = EMT()
    phonons = Phonons(atoms, EMT())
    phonons.set_atoms(set_atoms)

    # TODO: For now, because there is no public API to iterate over/inspect
    #       displacements, we run and check the number of files in the cache.
    #       Later when the requisite API exists, we should use it both to
    #       check the actual atom indices and to avoid computation.
    phonons.run()
    assert len(phonons.cache) == 6 * len(expected_atoms) + 1


def test_set_atoms_indices(testdir):
    check_set_atoms(molecule('CO2'), set_atoms=[0, 1], expected_atoms=[0, 1])


def test_set_atoms_symbol(testdir):
    check_set_atoms(molecule('CO2'), set_atoms=['O'], expected_atoms=[1, 2])


def test_check_eq_forces(testdir):
    atoms = bulk('C')
    atoms.calc = EMT()

    phonons = Phonons(atoms, EMT(), supercell=(1, 2, 1))
    phonons.run()
    fmin, fmax, _i_min, _i_max = phonons.check_eq_forces()
    assert fmin < fmax


# Regression test for #953;  data stored for eq should resemble data for
# displacements
def test_check_consistent_format(testdir):
    atoms = molecule('H2')
    atoms.calc = EMT()

    phonons = Phonons(atoms, EMT())
    phonons.run()

    # Check that the data stored for `eq` is shaped like the data stored for
    # displacements.
    eq_data = phonons.cache['eq']
    disp_data = phonons.cache['0x-']
    assert isinstance(eq_data, dict) and isinstance(disp_data, dict)
    assert set(eq_data) == set(disp_data), "dict keys mismatch"
    for array_key in eq_data:
        assert eq_data[array_key].shape == disp_data[array_key].shape, array_key


def test_get_band_structure_with_modes(testdir):

    atoms = bulk('Al', 'fcc', a=4.05)
    N = 7
    npoints = 100   # k-points in band path
    natoms = len(atoms)

    ph = Phonons(atoms, EMT(), supercell=(N, N, N), delta=0.05)
    ph.run()
    ph.read(acoustic=True)
    ph.clean()

    path = atoms.cell.bandpath('GXULGK', npoints=npoints)
    band_structure, modes = ph.get_band_structure(path,
                                                  modes=True,
                                                  verbose=False)

    # Assertions
    assert band_structure is not None, "Band structure should not be None"
    assert modes is not None, "Modes should not be None"
    assert modes.ndim == 4, "Modes should be a 4-dimensional numpy array"
    assert modes.shape == (npoints, 3 * natoms, natoms, 3), \
        "Modes should have shape (k-points, nbands, natoms, 3)"


def test_frequencies_amplitudes(testdir):
    """Test frequencies and amplitudes of the phonon modes.

    The calculation is done for a diatomic chain of atoms
    with different masses and known spring constants, so
    frequencies and mode amplitudes can be compared with
    the theoretical values.  We use Cu and O with a
    Lennard-Jones potential.
    """

    # Reasonable Lennard-Jones parameters taken from the
    # OpenKIM project, DOI: 10.25950/962b4967
    LJ_epsilon_O = 5.1264700
    LJ_sigma_O = 1.1759900
    LJ_epsilon_Cu = 2.0446300
    LJ_sigma_Cu = 2.3519700

    LJ_epsilon = np.sqrt(LJ_epsilon_O * LJ_epsilon_Cu)
    LJ_sigma = (LJ_sigma_O + LJ_sigma_Cu) / 2

    # Equilibrium distance
    d0 = 2**(1 / 6) * LJ_sigma
    # Spring constant
    C = 36 * 2**(2 / 3) * LJ_epsilon / LJ_sigma**2
    print(f'LJ epsilon: {LJ_epsilon:.5f} eV    sigma: {LJ_sigma:.5f} Å')
    print(f'Bond length d0 = {d0:.5f} Å.     C = {C:.5f} eV / Å^2')

    # Make chain of Cu-O atoms
    pos = np.array(
        [[0, 0, 0],
         [d0, 0, 0]]
    )
    atoms = Atoms(
        symbols='CuO',
        positions=pos,
        cell=[2 * d0, 10., 10.],
        pbc=(True, False, False)
    )
    atoms.center()
    calc = LennardJones(sigma=LJ_sigma, epsilon=LJ_epsilon, rc=1.5 * d0)

    print('Calculating phonons')
    N = 7
    ph = Phonons(atoms, calc, supercell=(N, 1, 1),
                 delta=0.005, name='phonon_CuO')
    ph.run()
    assert ph.check_eq_forces()[1] < 1e-9, "System is not at equilibrium"
    # Read forces and assemble the dynamical matrix
    ph.read(acoustic=True)
    ph.clean()

    path = atoms.cell.bandpath(npoints=50, pbc=[1, 0, 0])
    kpoints = path.kpts
    gamma_point = path.special_points['G']
    x_point = path.special_points['X']
    omegas, us = ph.band_structure(kpoints, modes=True)

    e_gamma_O = omegas[0, 5]
    e_X_O = omegas[-1, 5]
    e_X_A = omegas[-1, 4]

    (m1, m2) = atoms.get_masses()
    hbar = units._hbar * units.J * units.second

    e_gamma_O_th = hbar * np.sqrt(2 * C * (1 / m1 + 1 / m2))
    e_X_O_th = hbar * np.sqrt(2 * C / m2)
    e_X_A_th = hbar * np.sqrt(2 * C / m1)

    print()
    print('ENERGIES      Numerical  Analytical')
    print(f'Gamma, O:     {e_gamma_O:<9.5f}  {e_gamma_O_th:<9.5f}    eV')
    print(f'Zone edge, O: {e_X_O:<9.5f}  {e_X_O_th:<9.5f}    eV')
    print(f'Zone edge, A: {e_X_A:<9.5f}  {e_X_A_th:<9.5f}    eV')

    assert np.isclose(e_gamma_O, e_gamma_O_th, atol=1e-4)
    assert np.isclose(e_X_O, e_X_O_th, atol=1e-4)
    assert np.isclose(e_X_A, e_X_A_th, atol=1e-4)

    print()
    mr = m1 / m2
    print(f'Mass ratios:  {mr:.5f}   {m2 / m1:.5f}')

    print('Amplitude ratios, Optical mode at gamma point:')
    amp_gamma_O = us[0, 5, 1, 0] / us[0, 5, 0, 0]
    print(amp_gamma_O)
    assert np.isclose(amp_gamma_O, -mr, atol=1e-5)

    print('Amplitude ratios, Optical mode NEAR gamma point:')
    amp_nearG_O = us[1, 5, 1, 0] / us[1, 5, 0, 0]
    print(amp_nearG_O, np.abs(amp_nearG_O))
    assert np.isclose(np.abs(amp_nearG_O), mr, atol=1e-2)

    print('Amplitude ratios, Acoustic mode NEAR gamma point:')
    amp_nearG_A = us[1, 4, 1, 0] / us[1, 4, 0, 0]
    print(amp_nearG_A, np.abs(amp_nearG_A))
    assert np.isclose(np.abs(amp_nearG_A), 1.0, atol=1e-3)

    # Test of absolute amplitudes
    # This is done by testing that the kinetic energy of each
    # mode is 1/2 k T

    repeat = 8
    T = 300

    checkmodes = (
        (gamma_point, 5, e_gamma_O),
        (x_point, 5, e_X_O),
        (x_point, 4, e_X_A),
    )

    for kk, br, hbaromega in checkmodes:
        print(f'Ckecking k-point {kk} branch {br}')
        ph.write_modes(kk, branches=(br,),
                       repeat=(repeat, 1, 1), kT=T * units.kB)
        filename = f'{ph.name}.mode.{br}.traj'
        pos = []
        with Trajectory(filename) as traj:
            for a in traj:
                pos.append(a.get_positions())
                masses = a.get_masses()
        os.unlink(filename)

        # Now calculate the velocities from the differences in the positions.
        # The first frame should be compared to the last.
        delta_t = 2 * np.pi * hbar / (hbaromega * len(pos))

        vel = []
        ekin = []
        for i in range(len(pos)):
            v = (pos[i] - pos[i - 1]) / delta_t
            vel.append(v)
            ek = (0.5 * masses * (v * v).sum(axis=1)).sum() / repeat
            ekin.append(ek)

        ekin_avg = np.mean(ekin)
        ekin_exp = units.kB * T / 2
        print(f"Avg. kinetic energy: {ekin_avg} eV - expected {ekin_exp} eV")
        assert np.isclose(ekin_avg, ekin_exp, rtol=0.01)


def test_partial_dos(testdir):
    """Test partial phonon DOS.

    Tests that the partial phonon densities of states sum
    up to the total density of states.
    """

    # Reasonable Lennard-Jones parameters taken from the
    # OpenKIM project, DOI: 10.25950/962b4967
    Epsilon_Zn, Sigma_Zn = 0.1915460, 2.1737900
    Epsilon_S, Sigma_S = 4.3692700, 1.8708900

    LJ_epsilon = np.sqrt(Epsilon_Zn * Epsilon_S)
    LJ_sigma = (Sigma_Zn + Sigma_S) / 2

    # Equilibrium distance
    d0 = 2**(1 / 6) * LJ_sigma
    latconst = d0 * np.sqrt(2)
    cutoff = d0 * (1 + np.sqrt(2)) / 2  # Between first and second neighbor

    # Set up a zincblende structure with a reasonable lattice constant
    atoms = bulk('ZnS', 'zincblende', a=latconst)
    calc = LennardJones(sigma=LJ_sigma, epsilon=LJ_epsilon, rc=cutoff)

    # Calculating phonons
    N = 3
    ph = Phonons(atoms, calc, supercell=(N, N, N),
                 delta=0.005, name='phonon_ZnS')
    ph.run()
    assert ph.check_eq_forces()[1] < 1e-9, "System is not at equilibrium"
    # Read forces and assemble the dynamical matrix
    ph.read(acoustic=True)
    ph.clean()

    # Analyzing phonons
    kpts = (10, 10, 10)
    w = 3e-2
    dos = ph.get_dos(kpts=kpts).sample_grid(npts=500, width=w)
    dosZn = ph.get_dos(kpts=kpts, indices=(0,)).sample_grid(npts=500, width=w)
    dosS = ph.get_dos(kpts=kpts, indices=(1,)).sample_grid(npts=500, width=w)

    dosZn_array = dosZn.get_weights()
    dosS_array = dosS.get_weights()
    dosTotal_array = dos.get_weights()

    assert np.allclose(dosTotal_array, dosS_array + dosZn_array)
