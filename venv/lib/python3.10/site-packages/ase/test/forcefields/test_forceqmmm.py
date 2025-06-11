import numpy as np
import pytest

from ase.build import bulk
from ase.calculators.eam import EAM
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
from ase.calculators.qmmm import ForceQMMM, RescaledCalculator
from ase.eos import EquationOfState
from ase.geometry import get_distances
from ase.neighborlist import neighbor_list
from ase.optimize import FIRE
from ase.units import GPa


@pytest.fixture()
def mm_calc():
    bulk_at = bulk("Cu", cubic=True)
    sigma = (bulk_at * 2).get_distance(0, 1) * (2. ** (-1. / 6))

    return LennardJones(sigma=sigma, epsilon=0.05)


@pytest.fixture()
def qm_calc():
    return EMT()


@pytest.fixture()
def bulk_at():
    bulk_at = bulk("Cu", cubic=True)

    return bulk_at


@pytest.mark.slow()
def test_qm_buffer_mask(qm_calc, mm_calc, bulk_at):
    """
    test number of atoms in qm_buffer_mask for
    spherical region in a fully periodic cell
    also tests that "region" array returns the same mapping
    """

    alat = bulk_at.cell[0, 0]
    N_cell_geom = 10
    at0 = bulk_at * N_cell_geom
    r = at0.get_distances(0, np.arange(len(at0)), mic=True)
    print("N_cell", N_cell_geom, 'N_MM', len(at0), "Size", N_cell_geom * alat)
    qm_rc = 5.37  # cutoff for EMC()

    for R_QM in [1.0e-3,  # one atom in the center
                 alat / np.sqrt(2.0) + 1.0e-3,  # should give 12 nearest
                                                # neighbours + central atom
                 alat + 1.0e-3]:  # should give 18 neighbours + central atom

        at = at0.copy()
        qm_mask = r < R_QM
        qm_buffer_mask_ref = r < 2 * qm_rc + R_QM
        # exclude atoms that are too far (in case of non spherical region)
        # this is the old way to do it
        _, r_qm_buffer = get_distances(at.positions[qm_buffer_mask_ref],
                                       at.positions[qm_mask], at.cell, at.pbc)
        updated_qm_buffer_mask = np.ones_like(at[qm_buffer_mask_ref])
        for i, r_qm in enumerate(r_qm_buffer):
            if r_qm.min() > 2 * qm_rc:
                updated_qm_buffer_mask[i] = False

        qm_buffer_mask_ref[qm_buffer_mask_ref] = updated_qm_buffer_mask
        '''
        print(f'R_QM             {R_QM}   N_QM        {qm_mask.sum()}')
        print(f'R_QM + buffer: {2 * qm_rc + R_QM:.2f}'
              f' N_QM_buffer {qm_buffer_mask_ref.sum()}')
        print(f'                     N_total:    {len(at)}')
        '''
        qmmm = ForceQMMM(at, qm_mask, qm_calc, mm_calc, buffer_width=2 * qm_rc)
        # build qm_buffer_mask and test it
        qmmm.initialize_qm_buffer_mask(at)
        # print(f'      Calculator N_QM_buffer:'
        #       f'    {qmmm.qm_buffer_mask.sum().sum()}')
        assert qmmm.qm_buffer_mask.sum() == qm_buffer_mask_ref.sum()
        # same test for qmmm.get_cluster()
        qm_cluster = qmmm.get_qm_cluster(at)
        assert len(qm_cluster) == qm_buffer_mask_ref.sum()
        # test region mappings
        region = qmmm.get_region_from_masks(at)
        qm_mask_region = region == "QM"
        assert qm_mask_region.sum() == qm_mask.sum()
        buffer_mask_region = region == "buffer"
        assert qm_mask_region.sum() + \
            buffer_mask_region.sum() == qm_buffer_mask_ref.sum()


def compare_qm_cell_and_pbc(qm_calc, mm_calc, bulk_at,
                            test_size=4,
                            expected_pbc=np.array([True, True, True]),
                            buffer_width=5 * 3.61):
    """
    test qm cell shape and choice of pbc:
    make a non-periodic pdc in a direction
    if qm_radius + buffer is larger than the original cell
    keep the periodic cell otherwise i. e. if cell[i, i] > qm_radius + buffer
    the scenario is controlled by the test_size used to create at0
    as well as buffer_width.
    If the size of the at0 is larger than the r_qm + buffer + vacuum
    the cell stays periodic and the size is the same is original
    otherwise cell is non-periodic and size is different.
    """

    alat = bulk_at.cell[0, 0]
    at0 = bulk_at * test_size
    r = at0.get_distances(0, np.arange(len(at0)), mic=True)
    # should give 12 nearest neighbours + atom in the center
    R_QM = alat / np.sqrt(2.0) + 1.0e-3
    qm_mask = r < R_QM

    qmmm = ForceQMMM(at0, qm_mask, qm_calc, mm_calc,
                     buffer_width=buffer_width)
    # equal to 1 alat
    # build qm_buffer_mask to build the cell
    qmmm.initialize_qm_buffer_mask(at0)
    qm_cluster = qmmm.get_qm_cluster(at0)

    # test if qm pbc match expected in qmmm.get_cluster()
    assert all(qm_cluster.pbc == expected_pbc)

    # test the cell size for qmmm.get_qm_cluster()
    if not all(expected_pbc):  # at least one F. avoid comparing empty arrays
        assert not all(qm_cluster.cell.lengths()[~expected_pbc] ==
                       at0.cell.lengths()[~expected_pbc])
    if any(expected_pbc):  # at least one T. avoid comparing empty arrays
        np.testing.assert_allclose(qm_cluster.cell.lengths()[expected_pbc],
                                   at0.cell.lengths()[expected_pbc])


@pytest.mark.parametrize("kwargs",
                         [  # test the case of a cluster in
                            # a fully periodic cell:
                            # fist qm_radius + buffer > cell,
                            # thus should give a cluster with pbc=[T, T, T]
                            # (qm cluster is the same as the original cell)'''
                            {"test_size": 4,
                             "expected_pbc": np.array([True, True, True]),
                             "buffer_width": 5 * 3.61},
                            # test the case of a spherical
                            # cluster in a fully periodic cell:
                            # fist qm_radius + buffer < cell,
                            # thus should give a cluster with pbc=[F, F, F]
                            # (qm cluster cell must be DIFFERENT
                            # form the original cell)
                            {"test_size": 4,
                             "expected_pbc": np.array([False, False, False]),
                             "buffer_width": 3.61},
                            # testing the mixed scenario when the qm_cluster
                            # is periodic in one direction
                            # (relevant for dislocation or crack cells)
                            # (qm cluster cell must be the same as
                            # the original cell in periodic direction
                            # and DIFFERENT form the original cell
                            # in non periodic directions
                            # three tests for three different directions
                            {"test_size": [4, 4, 1],
                             "expected_pbc": np.array([False, False, True]),
                             "buffer_width": 3.61},
                            {"test_size": [4, 1, 4],
                             "expected_pbc": np.array([False, True, False]),
                             "buffer_width": 3.61},
                            {"test_size": [1, 4, 4],
                             "expected_pbc": np.array([True, False, False]),
                             "buffer_width": 3.61},
                             # testing scenario periodic in one direction
                             # and non periodic in the other two
                             # relevant for surfaces.
                             # testing three different scenarios
                             {"test_size": [1, 1, 4],
                              "expected_pbc": np.array([True, True, False]),
                              "buffer_width": 3.61},
                             {"test_size": [4, 1, 1],
                              "expected_pbc": np.array([False, True, True]),
                              "buffer_width": 3.61},
                             {"test_size": [1, 4, 1],
                              "expected_pbc": np.array([True, False, True]),
                              "buffer_width": 3.61}
                         ])
def test_qm_pbc(kwargs, qm_calc, mm_calc, bulk_at):
    kwargs1 = {}
    kwargs1.update(kwargs)
    compare_qm_cell_and_pbc(qm_calc, mm_calc, bulk_at, **kwargs1)


def test_rescaled_calculator():
    """
    Test rescaled RescaledCalculator() by computing lattice constant
    and bulk modulus using fit to equation of state
    and comparing it to the desired values
    """

    # A simple empirical N-body potential for
    # transition metals by M. W. Finnis & J.E. Sinclair
    # https://www.tandfonline.com/doi/abs/10.1080/01418618408244210
    # using analytical formulation in order to avoid extra file dependence
    # All the constants are taken from the paper.
    # Please refer to the paper for more details

    def pair_potential(r):
        """
        returns the pair potential as a equation 27 in pair_potential
        r - numpy array with the values of distance to compute the pair function
        """
        # parameters for W
        c = 3.25
        c0 = 47.1346499
        c1 = -33.7665655
        c2 = 6.2541999

        energy = (c0 + c1 * r + c2 * r ** 2.0) * (r - c) ** 2.0
        energy[r > c] = 0.0

        return energy

    def cohesive_potential(r):
        """
        returns the cohesive potential as a equation 28 in pair_potential
        r - numpy array with the values of distance to compute the pair function
        """
        # parameters for W
        d = 4.400224

        rho = (r - d) ** 2.0
        rho[r > d] = 0.0

        return rho

    def embedding_function(rho):
        """
        returns energy as a function of electronic density from eq 3
        """

        A = 1.896373
        energy = - A * np.sqrt(rho)

        return energy

    cutoff = 4.400224
    W_FS = EAM(elements=['W'], embedded_energy=np.array([embedding_function]),
               electron_density=np.array([[cohesive_potential]]),
               phi=np.array([[pair_potential]]), cutoff=cutoff, form='fs')

    # compute MM and QM equations of state
    def strain(at, e, calc):
        at = at.copy()
        at.set_cell((1.0 + e) * at.cell, scale_atoms=True)
        at.calc = calc
        v = at.get_volume()
        e = at.get_potential_energy()
        return v, e

    # desired DFT values
    a0_qm = 3.18556
    C11_qm = 522  # pm 15 GPa
    C12_qm = 193  # pm 5 GPa
    B_qm = (C11_qm + 2.0 * C12_qm) / 3.0

    bulk_at = bulk("W", cubic=True)

    mm_calc = W_FS
    eps = np.linspace(-0.01, 0.01, 13)
    v_mm, E_mm = zip(*[strain(bulk_at, e, mm_calc) for e in eps])

    eos_mm = EquationOfState(v_mm, E_mm)
    v0_mm, _E0_mm, B_mm = eos_mm.fit()
    B_mm /= GPa
    a0_mm = v0_mm ** (1.0 / 3.0)

    mm_r = RescaledCalculator(mm_calc, a0_qm, B_qm, a0_mm, B_mm)
    bulk_at = bulk("W", cubic=True, a=a0_qm)
    v_mm_r, E_mm_r = zip(*[strain(bulk_at, e, mm_r) for e in eps])

    eos_mm_r = EquationOfState(v_mm_r, E_mm_r)
    v0_mm_r, _E0_mm_r, B_mm_r = eos_mm_r.fit()
    B_mm_r /= GPa
    a0_mm_r = v0_mm_r ** (1.0 / 3)

    # check match of a0 and B after rescaling is adequate
    # 0.1% error in lattice constant and bulk modulus
    assert abs((a0_mm_r - a0_qm) / a0_qm) < 1e-3
    assert abs((B_mm_r - B_qm) / B_qm) < 1e-3


@pytest.mark.slow()
def test_forceqmmm(qm_calc, mm_calc, bulk_at):

    # parameters
    N_cell = 2
    R_QMs = np.array([3, 7])

    sigma = (bulk_at * 2).get_distance(0, 1) * (2. ** (-1. / 6))

    at0 = bulk_at * N_cell
    r = at0.get_distances(0, np.arange(1, len(at0)), mic=True)
    print(len(r))
    del at0[0]  # introduce a vacancy
    print("N_cell", N_cell, 'N_MM', len(at0),
          "Size", N_cell * bulk_at.cell[0, 0])

    ref_at = at0.copy()
    ref_at.calc = qm_calc
    opt = FIRE(ref_at)
    opt.run(fmax=1e-3)
    u_ref = ref_at.positions - at0.positions

    us = []
    for R_QM in R_QMs:
        at = at0.copy()
        qm_mask = r < R_QM
        qm_buffer_mask_ref = r < 2 * qm_calc.rc + R_QM
        print(f'R_QM             {R_QM}   N_QM        {qm_mask.sum()}')
        print(f'R_QM + buffer: {2 * qm_calc.rc + R_QM:.2f}'
              f' N_QM_buffer {qm_buffer_mask_ref.sum()}')
        print(f'                     N_total:    {len(at)}')
        # Warning: Small size of the cell and large size of the buffer
        # lead to the qm calculation performed on the whole cell.
        qmmm = ForceQMMM(at, qm_mask, qm_calc, mm_calc,
                         buffer_width=2 * qm_calc.rc)
        qmmm.initialize_qm_buffer_mask(at)
        at.calc = qmmm
        opt = FIRE(at)
        opt.run(fmax=1e-3)
        us.append(at.positions - at0.positions)

    # compute error in energy norm |\nabla u - \nabla u_ref|
    def strain_error(at0, u_ref, u, cutoff, mask):
        I, J = neighbor_list('ij', at0, cutoff)
        I, J = np.array([(i, j) for i, j in zip(I, J) if mask[i]]).T
        v = u_ref - u
        dv = np.linalg.norm(v[I, :] - v[J, :], axis=1)
        return np.linalg.norm(dv)

    du_global = [strain_error(at0, u_ref, u, 1.5 * sigma,
                              np.ones(len(r))) for u in us]
    du_local = [strain_error(at0, u_ref, u, 1.5 * sigma, r < 3.0) for u in us]

    print('du_local', du_local)
    print('du_global', du_global)

    # check local errors are monotonically decreasing
    assert np.all(np.diff(du_local) < 0)

    # check global errors are monotonically converging
    assert np.all(np.diff(du_global) < 0)

    # biggest QM/MM should match QM result
    assert du_local[-1] < 1e-10
    assert du_global[-1] < 1e-10


@pytest.fixture()
def at0(qm_calc, mm_calc, bulk_at):
    alat = bulk_at.cell[0, 0]
    at0 = bulk_at * 5
    r = at0.get_distances(0, np.arange(len(at0)), mic=True)
    # should give 12 nearest neighbours + atom in the center
    R_QM = alat / np.sqrt(2.0) + 1.0e-3
    qm_mask = r < R_QM

    qmmm = ForceQMMM(at0, qm_mask, qm_calc, mm_calc,
                     buffer_width=3.61)

    qmmm.initialize_qm_buffer_mask(at0)
    at0.calc = qmmm

    return at0


def test_export_xyz(at0, testdir):
    """
    test the export_extxyz function and checks the region adn forces arrays
    """

    # evaluating forces to test exporting of forces
    forces = at0.get_forces()
    filename = "qmmm_export_test.xyz"

    qmmm = at0.calc
    qmmm.export_extxyz(filename=filename)

    from ase.io import read
    read_atoms = read(filename)

    assert "region" in read_atoms.arrays
    original_region = qmmm.get_region_from_masks()
    assert all(original_region == read_atoms.get_array("region"))

    assert "forces" in read_atoms.calc.results
    # absolute tolerance for comparing forces close to zero
    np.testing.assert_allclose(forces, read_atoms.get_forces(), atol=1.0e-6)


def test_set_masks_from_region(at0, qm_calc, mm_calc):
    """
    Test setting masks from region array
    """

    qmmm = at0.calc
    region = qmmm.get_region_from_masks(at0)

    # initialise another qmmm with different masks
    r = at0.get_distances(0, np.arange(len(at0)), mic=True)
    R_QM = 1.0e-3
    qm_mask = r < R_QM

    test_qmmm = ForceQMMM(at0, qm_mask, qm_calc, mm_calc,
                          buffer_width=3.61)

    # assert that number of qm atoms is different
    assert np.count_nonzero(qmmm.qm_selection_mask) != np.count_nonzero(
        test_qmmm.qm_selection_mask
    )

    test_qmmm.set_masks_from_region(region)

    assert all(test_qmmm.qm_selection_mask == qmmm.qm_selection_mask)
    assert all(test_qmmm.qm_buffer_mask == qmmm.qm_buffer_mask)

    test_region = test_qmmm.get_region_from_masks(at0)
    assert all(region == test_region)


def test_import_xyz(at0, qm_calc, mm_calc, testdir):
    """
    test the import_extxyz function and checks the mapping
    """

    filename = "qmmm_export_test.xyz"

    qmmm = at0.calc
    qmmm.export_extxyz(filename=filename, atoms=at0)

    imported_qmmm = ForceQMMM.import_extxyz(filename, qm_calc, mm_calc)

    assert all(imported_qmmm.qm_selection_mask == qmmm.qm_selection_mask)
    assert all(imported_qmmm.qm_buffer_mask == qmmm.qm_buffer_mask)
