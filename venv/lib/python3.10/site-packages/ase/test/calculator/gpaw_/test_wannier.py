from functools import partial

import numpy as np
import pytest

from ase import Atoms
from ase.build import bulk, molecule
from ase.dft.kpoints import monkhorst_pack
from ase.dft.wannier import (
    Wannier,
    arbitrary_s_orbitals,
    calculate_weights,
    gram_schmidt,
    init_orbitals,
    lowdin,
    md_min,
    neighbor_k_search,
    rotation_from_projection,
    scdm,
    search_for_gamma_point,
    steepest_descent,
)
from ase.dft.wannierstate import random_orthogonal_matrix
from ase.io.cube import read_cube
from ase.lattice import (
    BCC,
    BCT,
    CRECT,
    CUB,
    FCC,
    HEX,
    HEX2D,
    LINE,
    MCL,
    MCLC,
    OBL,
    ORC,
    ORCC,
    ORCF,
    ORCI,
    RECT,
    RHL,
    SQR,
    TET,
    TRI,
)
from ase.transport.tools import dagger, normalize

calc = pytest.mark.calculator
Nk = 2


@pytest.fixture()
def rng():
    return np.random.RandomState(0)


@pytest.fixture(scope='module')
def _base_calculator_gpwfile(tmp_path_factory, factories):
    """
    Generic method to cache calculator in a file on disk.
    """
    def __base_calculator_gpwfile(atoms, filename,
                                  nbands, gpts=(8, 8, 8),
                                  kpts=(Nk, Nk, Nk)):
        factories.require('gpaw')
        import gpaw
        gpw_path = tmp_path_factory.mktemp('sub') / filename
        calc = gpaw.GPAW(
            mode='fd',
            gpts=gpts,
            nbands=nbands,
            kpts={'size': kpts, 'gamma': True},
            symmetry='off',
            txt=None)
        atoms.calc = calc
        atoms.get_potential_energy()
        calc.write(gpw_path, mode='all')
        return gpw_path
    return __base_calculator_gpwfile


@pytest.fixture(scope='module')
def _h2_calculator_gpwfile(_base_calculator_gpwfile):
    atoms = molecule('H2', pbc=True)
    atoms.center(vacuum=3.)
    gpw_path = _base_calculator_gpwfile(
        atoms=atoms,
        filename='wan_h2.gpw',
        nbands=4
    )
    return gpw_path


@pytest.fixture(scope='module')
def h2_calculator(_h2_calculator_gpwfile):
    import gpaw
    return gpaw.GPAW(_h2_calculator_gpwfile, txt=None)


@pytest.fixture(scope='module')
def _si_calculator_gpwfile(_base_calculator_gpwfile):
    atoms = bulk('Si')
    gpw_path = _base_calculator_gpwfile(
        atoms=atoms,
        filename='wan_si.gpw',
        nbands=8
    )
    return gpw_path


@pytest.fixture(scope='module')
def si_calculator(_si_calculator_gpwfile):
    import gpaw
    return gpaw.GPAW(_si_calculator_gpwfile, txt=None)


@pytest.fixture(scope='module')
def _ti_calculator_gpwfile(_base_calculator_gpwfile):
    atoms = bulk('Ti', crystalstructure='hcp')
    gpw_path = _base_calculator_gpwfile(
        atoms=atoms,
        filename='wan_ti.gpw',
        nbands=None
    )
    return gpw_path


@pytest.fixture(scope='module')
def ti_calculator(_ti_calculator_gpwfile):
    import gpaw
    return gpaw.GPAW(_ti_calculator_gpwfile, txt=None)


@pytest.fixture()
def wan(rng, h2_calculator):
    def _wan(
        atoms=None,
        calc=None,
        nwannier=2,
        fixedstates=None,
        fixedenergy=None,
        initialwannier='bloch',
        functional='std',
        gpts=(8, 8, 8),
        kpts=None,
        file=None,
        rng=rng,
        full_calc=False,
        std_calc=True,
    ):
        """
        Generate a Wannier object.

        full_calc: the provided calculator has a converged calculation
        std_calc: the default H2 calculator object is used
        """
        # If the calculator, or some fundamental parameters, are provided
        # we clearly do not want a default calculator
        if calc is not None or kpts is not None or atoms is not None:
            std_calc = False
            # Default value for kpts, if we need to generate atoms/calc
            if kpts is None:
                kpts = (Nk, Nk, Nk)

        if std_calc:
            calc = h2_calculator
            full_calc = True
        elif atoms is None and not full_calc:
            pbc = (np.array(kpts) > 1).any()
            atoms = molecule('H2', pbc=pbc)
            atoms.center(vacuum=3.)

        if calc is None:
            gpaw = pytest.importorskip('gpaw')
            calc = gpaw.GPAW(
                mode='fd',
                gpts=gpts,
                nbands=nwannier,
                kpts=kpts,
                symmetry='off',
                txt=None
            )

        if not full_calc:
            atoms.calc = calc
            atoms.get_potential_energy()

        return Wannier(
            nwannier=nwannier,
            fixedstates=fixedstates,
            fixedenergy=fixedenergy,
            calc=calc,
            initialwannier=initialwannier,
            file=None,
            functional=functional,
            rng=rng,
        )
    return _wan


def bravais_lattices():
    return [CUB(1), FCC(1), BCC(1), TET(1, 2), BCT(1, 2),
            ORC(1, 2, 3), ORCF(1, 2, 3), ORCI(1, 2, 3),
            ORCC(1, 2, 3), HEX(1, 2), RHL(1, 110),
            MCL(1, 2, 3, 70), MCLC(1, 2, 3, 70),
            TRI(1, 2, 3, 60, 70, 80), OBL(1, 2, 70),
            HEX2D(1), RECT(1, 2), CRECT(1, 70), SQR(1),
            LINE(1)]


class Paraboloid:

    def __init__(self, pos=(10., 10., 10.), shift=1.):
        self.pos = np.array(pos, dtype=complex)
        self.shift = shift

    def get_gradients(self):
        return 2 * self.pos

    def step(self, dF, updaterot=True, updatecoeff=True):
        self.pos -= dF

    def get_functional_value(self):
        return np.sum(self.pos**2) + self.shift


def unitarity_error(matrix):
    return np.abs(dagger(matrix) @ matrix - np.eye(len(matrix))).max()


def orthogonality_error(matrix):
    errors = []
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            errors.append(np.abs(dagger(matrix[i]) @ matrix[j]))
    return np.max(errors)


def normalization_error(matrix):
    old_matrix = matrix.copy()
    normalize(matrix)
    return np.abs(matrix - old_matrix).max()


def test_gram_schmidt(rng):
    matrix = rng.random((4, 4))
    assert unitarity_error(matrix) > 1
    gram_schmidt(matrix)
    assert unitarity_error(matrix) < 1e-12


def test_lowdin(rng):
    matrix = rng.random((4, 4))
    assert unitarity_error(matrix) > 1
    lowdin(matrix)
    assert unitarity_error(matrix) < 1e-12


def test_random_orthogonal_matrix(rng):
    dim = 4
    matrix = random_orthogonal_matrix(dim, rng=rng, real=True)
    assert matrix.shape[0] == matrix.shape[1]
    assert unitarity_error(matrix) < 1e-12
    matrix = random_orthogonal_matrix(dim, rng=rng, real=False)
    assert matrix.shape[0] == matrix.shape[1]
    assert unitarity_error(matrix) < 1e-12


def test_neighbor_k_search():
    kpt_kc = monkhorst_pack((4, 4, 4))
    Gdir_dc = [[1, 0, 0], [0, 1, 0], [0, 0, 1],
               [1, 1, 0], [1, 0, 1], [0, 1, 1]]
    tol = 1e-4
    for d, Gdir_c in enumerate(Gdir_dc):
        for k, k_c in enumerate(kpt_kc):
            kk, k0 = neighbor_k_search(k_c, Gdir_c, kpt_kc, tol=tol)
            assert np.linalg.norm(kpt_kc[kk] - k_c - Gdir_c + k0) < tol


@pytest.mark.parametrize('lat', bravais_lattices())
def test_calculate_weights(lat):
    # Equation from Berghold et al. PRB v61 n15 (2000)
    tol = 1e-5
    cell = lat.tocell()
    g = cell @ cell.T
    w, G = calculate_weights(cell, normalize=False)

    errors = []
    for i in range(3):
        for j in range(3):
            errors.append(np.abs((w * G[:, i] @ G[:, j]) - g[i, j]))

    assert np.max(errors) < tol


def test_steepest_descent():
    tol = 1e-6
    step = 0.1
    func = Paraboloid(pos=np.array([10, 10, 10], dtype=float), shift=1.)
    steepest_descent(func=func, step=step, tolerance=tol)
    assert func.get_functional_value() == pytest.approx(1, abs=1e-5)


def test_md_min():
    tol = 1e-8
    step = 0.1
    func = Paraboloid(pos=np.array([10, 10, 10], dtype=complex), shift=1.)
    md_min(func=func, step=step, tolerance=tol,
           max_iter=1e6)
    assert func.get_functional_value() == pytest.approx(1, abs=1e-5)


def test_rotation_from_projection(rng):
    proj_nw = rng.random((6, 4))
    assert unitarity_error(proj_nw[:int(min(proj_nw.shape))]) > 1
    U_ww, C_ul = rotation_from_projection(proj_nw, fixed=2, ortho=True)
    assert unitarity_error(U_ww) < 1e-10, 'U_ww not unitary'
    assert orthogonality_error(C_ul.T) < 1e-10, 'C_ul columns not orthogonal'
    assert normalization_error(C_ul) < 1e-10, 'C_ul not normalized'
    U_ww, C_ul = rotation_from_projection(proj_nw, fixed=2, ortho=False)
    assert normalization_error(U_ww) < 1e-10, 'U_ww not normalized'


@pytest.mark.calculator_lite()
def test_save(tmpdir, wan):
    wanf = wan(nwannier=4, fixedstates=2, initialwannier='bloch')
    jsonfile = tmpdir.join('wanf.json')
    f1 = wanf.get_functional_value()
    wanf.save(jsonfile)
    wanf.initialize(file=jsonfile, initialwannier='bloch')
    assert pytest.approx(f1) == wanf.get_functional_value()


@pytest.mark.parametrize('lat', bravais_lattices())
def test_get_radii(lat, wan):
    # Sanity check, the Wannier functions' spread should always be positive.
    # Also, make sure that the method does not fail for any lattice.
    if ((lat.tocell() == FCC(a=1).tocell()).all() or
            (lat.tocell() == ORCF(a=1, b=2, c=3).tocell()).all()):
        pytest.skip("Lattices not supported by this function,"
                    " use get_spreads() instead.")
    atoms = molecule('H2', pbc=True)
    atoms.cell = lat.tocell()
    atoms.center(vacuum=3.)
    wanf = wan(nwannier=4, fixedstates=2, atoms=atoms, initialwannier='bloch')
    assert all(wanf.get_radii() > 0)


@pytest.mark.parametrize('lat', bravais_lattices())
def test_get_spreads(lat, wan):
    # Sanity check, the Wannier functions' spread should always be positive.
    # Also, make sure that the method does not fail for any lattice.
    atoms = molecule('H2', pbc=True)
    atoms.cell = lat.tocell()
    atoms.center(vacuum=3.)
    wanf = wan(nwannier=4, fixedstates=2, atoms=atoms, initialwannier='bloch')
    assert all(wanf.get_spreads() > 0)


@pytest.mark.parametrize('fun', ['std', 'var'])
def test_get_functional_value(fun, wan):
    # Only testing if the functional scales with the number of functions
    wan1 = wan(nwannier=3, functional=fun)
    f1 = wan1.get_functional_value()
    wan2 = wan(nwannier=4)
    f2 = wan2.get_functional_value()
    assert f1 < f2


@pytest.mark.calculator_lite()
@calc('gpaw')
def test_get_centers(factory):
    # Rough test on the position of the Wannier functions' centers
    gpaw = pytest.importorskip('gpaw')
    calc = gpaw.GPAW(mode='fd', gpts=(32, 32, 32), nbands=4, txt=None)
    atoms = molecule('H2', calculator=calc)
    atoms.center(vacuum=3.)
    atoms.get_potential_energy()
    wanf = Wannier(nwannier=2, calc=calc, initialwannier='bloch')
    centers = wanf.get_centers()
    com = atoms.get_center_of_mass()
    assert np.abs(centers - [com, com]).max() < 1e-4


def test_write_cube_default(wan, h2_calculator, testdir):
    # Chek the value saved in the CUBE file and the atoms object.
    # The default saved value is the absolute value of the Wannier function,
    # and the supercell is repeated per the number of k-points in each
    # direction.
    atoms = h2_calculator.atoms
    wanf = wan(calc=h2_calculator, full_calc=True)
    index = 0

    # It returns some errors when using file objects, so we use a string
    cubefilename = 'wanf.cube'
    wanf.write_cube(index, cubefilename)
    with open(cubefilename) as inputfile:
        content = read_cube(inputfile)
    assert pytest.approx(content['atoms'].cell.array) == atoms.cell.array * 2
    assert pytest.approx(content['data']) == abs(wanf.get_function(index))


def test_write_cube_angle(wan, testdir):
    # Check that the complex phase is correctly saved to the CUBE file, together
    # with the right atoms object.
    atoms = molecule('H2')
    atoms.center(vacuum=3.)
    wanf = wan(atoms=atoms, kpts=(1, 1, 1))
    index = 0

    # It returns some errors when using file objects, so we use a string
    cubefilename = 'wanf.cube'
    wanf.write_cube(index, cubefilename, angle=True)
    with open(cubefilename) as inputfile:
        content = read_cube(inputfile)
    assert pytest.approx(content['atoms'].cell.array) == atoms.cell.array
    assert pytest.approx(content['data']) == np.angle(wanf.get_function(index))


def test_write_cube_repeat(wan, testdir):
    # Check that the repeated supercell and Wannier functions are correctly
    # saved to the CUBE file, together with the right atoms object.
    atoms = molecule('H2')
    atoms.center(vacuum=3.)
    wanf = wan(atoms=atoms, kpts=(1, 1, 1))
    index = 0
    repetition = [4, 4, 4]

    # It returns some errors when using file objects, so we use simple filename
    cubefilename = 'wanf.cube'
    wanf.write_cube(index, cubefilename, repeat=repetition)

    with open(cubefilename) as inputfile:
        content = read_cube(inputfile)
    assert pytest.approx(content['atoms'].cell.array) == \
        (atoms * repetition).cell.array
    assert pytest.approx(content['data']) == \
        abs(wanf.get_function(index, repetition))


def test_localize(wan):
    wanf = wan(initialwannier='random')
    fvalue = wanf.get_functional_value()
    wanf.localize()
    assert wanf.get_functional_value() > fvalue


def test_get_spectral_weight_bloch(wan):
    nwannier = 4
    wanf = wan(initialwannier='bloch', nwannier=nwannier)
    for i in range(nwannier):
        assert wanf.get_spectral_weight(i)[:, i].sum() == pytest.approx(1)


def test_get_spectral_weight_random(wan, rng):
    nwannier = 4
    wanf = wan(initialwannier='random', nwannier=nwannier, rng=rng)
    for i in range(nwannier):
        assert wanf.get_spectral_weight(i).sum() == pytest.approx(1)


@pytest.mark.calculator_lite()
def test_get_pdos(wan):
    nwannier = 4
    gpaw = pytest.importorskip('gpaw')
    calc = gpaw.GPAW(mode='fd', gpts=(16, 16, 16), nbands=nwannier, txt=None)
    atoms = molecule('H2')
    atoms.center(vacuum=3.)
    atoms.calc = calc
    atoms.get_potential_energy()
    wanf = wan(atoms=atoms, calc=calc,
               nwannier=nwannier, initialwannier='bloch')
    eig_n = calc.get_eigenvalues()
    for i in range(nwannier):
        pdos_n = wanf.get_pdos(w=i, energies=eig_n, width=0.001)
        assert pdos_n[i] != pytest.approx(0)


def test_translate(wan, h2_calculator):
    nwannier = 2
    calc = h2_calculator
    atoms = calc.get_atoms()
    wanf = wan(nwannier=nwannier, initialwannier='bloch',
               calc=calc, full_calc=True)
    wanf.translate_all_to_cell(cell=[0, 0, 0])
    c0_w = wanf.get_centers()
    for i in range(nwannier):
        c2_w = np.delete(wanf.get_centers(), i, 0)
        wanf.translate(w=i, R=[1, 1, 1])
        c1_w = wanf.get_centers()
        assert np.linalg.norm(c1_w[i] - c0_w[i]) == \
            pytest.approx(np.linalg.norm(atoms.cell.array.diagonal()))
        c1_w = np.delete(c1_w, i, 0)
        assert c1_w == pytest.approx(c2_w)


def test_translate_to_cell(wan, h2_calculator):
    nwannier = 2
    calc = h2_calculator
    atoms = calc.get_atoms()
    wanf = wan(nwannier=nwannier, initialwannier='bloch',
               calc=calc, full_calc=True)
    for i in range(nwannier):
        wanf.translate_to_cell(w=i, cell=[0, 0, 0])
        c0_w = wanf.get_centers()
        assert (c0_w[i] < atoms.cell.array.diagonal()).all()
        wanf.translate_to_cell(w=i, cell=[1, 1, 1])
        c1_w = wanf.get_centers()
        assert (c1_w[i] > atoms.cell.array.diagonal()).all()
        assert np.linalg.norm(c1_w[i] - c0_w[i]) == \
            pytest.approx(np.linalg.norm(atoms.cell.array.diagonal()))
        c0_w = np.delete(c0_w, i, 0)
        c1_w = np.delete(c1_w, i, 0)
        assert c0_w == pytest.approx(c1_w)


def test_translate_all_to_cell(wan, h2_calculator):
    nwannier = 2
    calc = h2_calculator
    atoms = calc.get_atoms()
    wanf = wan(nwannier=nwannier, initialwannier='bloch',
               calc=calc, full_calc=True)
    wanf.translate_all_to_cell(cell=[0, 0, 0])
    c0_w = wanf.get_centers()
    assert (c0_w < atoms.cell.array.diagonal()).all()
    wanf.translate_all_to_cell(cell=[1, 1, 1])
    c1_w = wanf.get_centers()
    assert (c1_w > atoms.cell.array.diagonal()).all()
    for i in range(nwannier):
        assert np.linalg.norm(c1_w[i] - c0_w[i]) == \
            pytest.approx(np.linalg.norm(atoms.cell.array.diagonal()))


def test_distances(wan, h2_calculator):
    nwannier = 2
    calc = h2_calculator
    atoms = calc.get_atoms()
    wanf = wan(nwannier=nwannier, initialwannier='bloch')
    cent_w = wanf.get_centers()
    dist_ww = wanf.distances([0, 0, 0])
    dist1_ww = wanf.distances([1, 1, 1])
    for i in range(nwannier):
        assert dist_ww[i, i] == pytest.approx(0)
        assert dist1_ww[i, i] == pytest.approx(np.linalg.norm(atoms.cell.array))
        for j in range(i + 1, nwannier):
            assert dist_ww[i, j] == dist_ww[j, i]
            assert dist_ww[i, j] == \
                pytest.approx(np.linalg.norm(cent_w[i] - cent_w[j]))


def test_get_hopping_bloch(wan):
    nwannier = 4
    wanf = wan(nwannier=nwannier, initialwannier='bloch')
    hop0_ww = wanf.get_hopping([0, 0, 0])
    hop1_ww = wanf.get_hopping([1, 1, 1])
    for i in range(nwannier):
        assert hop0_ww[i, i] != 0
        assert hop1_ww[i, i] != 0
        for j in range(i + 1, nwannier):
            assert hop0_ww[i, j] == 0
            assert hop1_ww[i, j] == 0
            assert hop0_ww[i, j] == hop0_ww[j, i]
            assert hop1_ww[i, j] == hop1_ww[j, i]


def test_get_hopping_random(wan, rng):
    nwannier = 4
    wanf = wan(nwannier=nwannier, initialwannier='random')
    hop0_ww = wanf.get_hopping([0, 0, 0])
    hop1_ww = wanf.get_hopping([1, 1, 1])
    for i in range(nwannier):
        for j in range(i + 1, nwannier):
            assert np.abs(hop0_ww[i, j]) == pytest.approx(np.abs(hop0_ww[j, i]))
            assert np.abs(hop1_ww[i, j]) == pytest.approx(np.abs(hop1_ww[j, i]))


def test_get_hamiltonian_bloch(wan):
    nwannier = 4
    atoms = molecule('H2', pbc=True)
    atoms.center(vacuum=3.)
    kpts = (2, 2, 2)
    number_kpts = kpts[0] * kpts[1] * kpts[2]
    wanf = wan(atoms=atoms, kpts=kpts,
               nwannier=nwannier, initialwannier='bloch')
    for k in range(number_kpts):
        H_ww = wanf.get_hamiltonian(k=k)
        for i in range(nwannier):
            assert H_ww[i, i] != 0
            for j in range(i + 1, nwannier):
                assert H_ww[i, j] == 0
                assert H_ww[i, j] == pytest.approx(H_ww[j, i])


def test_get_hamiltonian_random(wan, rng):
    nwannier = 4
    atoms = molecule('H2', pbc=True)
    atoms.center(vacuum=3.)
    kpts = (2, 2, 2)
    number_kpts = kpts[0] * kpts[1] * kpts[2]
    wanf = wan(atoms=atoms, kpts=kpts, rng=rng,
               nwannier=nwannier, initialwannier='random')
    for k in range(number_kpts):
        H_ww = wanf.get_hamiltonian(k=k)
        for i in range(nwannier):
            for j in range(i + 1, nwannier):
                assert np.abs(H_ww[i, j]) == pytest.approx(np.abs(H_ww[j, i]))


def test_get_hamiltonian_kpoint(wan, rng, h2_calculator):
    nwannier = 4
    calc = h2_calculator
    atoms = calc.get_atoms()
    wanf = wan(nwannier=nwannier, initialwannier='random')
    kpts = atoms.cell.bandpath(density=50).cartesian_kpts()
    for kpt_c in kpts:
        H_ww = wanf.get_hamiltonian_kpoint(kpt_c=kpt_c)
        for i in range(nwannier):
            for j in range(i + 1, nwannier):
                assert np.abs(H_ww[i, j]) == pytest.approx(np.abs(H_ww[j, i]))


def test_get_function(wan):
    nwannier = 2
    gpts_np = np.array((8, 8, 8))
    wanf = wan(nwannier=nwannier, initialwannier='bloch')
    assert (wanf.get_function(index=[0, 0]) == 0).all()
    assert wanf.get_function(index=[0, 1]) + wanf.get_function(index=[1, 0]) \
        == pytest.approx(wanf.get_function(index=[1, 1]))
    for i in range(nwannier):
        assert (gpts_np * Nk == wanf.get_function(index=i).shape).all()
        assert (gpts_np * [1, 2, 3] ==
                wanf.get_function(index=i, repeat=[1, 2, 3]).shape).all()


# Make sure this can run in CI: https://gitlab.com/ase/ase/-/issues/1100
# @pytest.mark.calculator_lite
@pytest.mark.parametrize('fun', ['std', 'var'])
def test_get_gradients(fun, wan, rng):
    wanf = wan(nwannier=4, fixedstates=2, gpts=(12, 12, 12), kpts=(1, 1, 1),
               initialwannier='bloch', functional=fun)
    # create an anti-hermitian array/matrix
    step = rng.random(wanf.get_gradients().size) + \
        1.j * rng.random(wanf.get_gradients().size)
    step *= 1e-8
    step -= dagger(step)
    f1 = wanf.get_functional_value()
    wanf.step(step)
    f2 = wanf.get_functional_value()
    assert (np.abs((f2 - f1) / step).ravel() -
            np.abs(wanf.get_gradients())).max() < 1e-4


@pytest.mark.parametrize('init', ['bloch', 'random', 'orbitals', 'scdm'])
def test_initialwannier(init, wan, ti_calculator):
    # dummy check to run the module with different initialwannier methods
    wanf = wan(calc=ti_calculator, full_calc=True,
               initialwannier=init, nwannier=14, fixedstates=12)
    assert wanf.get_functional_value() > 0


def test_nwannier_auto(wan, ti_calculator):
    """ Test 'auto' value for parameter 'nwannier'. """
    partial_wan = partial(
        wan,
        calc=ti_calculator,
        full_calc=True,
        initialwannier='bloch',
        nwannier='auto'
    )

    # Check default value
    wanf = partial_wan()
    assert wanf.nwannier == 15

    # Check value setting fixedenergy
    wanf = partial_wan(fixedenergy=0)
    assert wanf.nwannier == 15
    wanf = partial_wan(fixedenergy=5)
    assert wanf.nwannier == 18

    # Check value setting fixedstates
    number_kpts = Nk**3
    list_fixedstates = [14] * number_kpts
    list_fixedstates[Nk] = 18
    wanf = partial_wan(fixedstates=list_fixedstates)
    assert wanf.nwannier == 18


def test_arbitrary_s_orbitals(rng):
    atoms = Atoms('3H', positions=[[0, 0, 0],
                                   [1, 1.5, 1],
                                   [2, 3, 0]])
    orbs = arbitrary_s_orbitals(atoms, 10, rng)

    atoms.append('H')
    s_pos = atoms.get_scaled_positions()
    for orb in orbs:
        # Test if they are actually s-orbitals
        assert orb[1] == 0

        # Read random position
        x, y, z = orb[0]
        s_pos[-1] = [x, y, z]
        atoms.set_scaled_positions(s_pos)

        # Use dummy H atom to measure distance from any other atom
        dists = atoms.get_distances(
            a=-1,
            indices=range(atoms.get_global_number_of_atoms() - 1))

        # Test that the s-orbital is close to at least one atom
        assert (dists < 1.5).any()


def test_init_orbitals_h2(rng):
    # Check that the initial orbitals for H2 are as many as requested and they
    # are all s-orbitals (l=0).
    atoms = molecule('H2')
    atoms.center(vacuum=3.)
    ntot = 2
    orbs = init_orbitals(atoms=atoms, ntot=ntot, rng=rng)
    angular_momenta = [orb[1] for orb in orbs]
    assert sum(l_ * 2 + 1 for l_ in angular_momenta) == ntot
    assert angular_momenta == [0] * ntot


def test_init_orbitals_ti(rng):
    # Check that the initial orbitals for Ti bulk are as many as requested and
    # there are both s-orbitals (l=0) and d-orbitals (l=2).
    atoms = bulk('Ti')
    ntot = 14
    orbs = init_orbitals(atoms=atoms, ntot=ntot, rng=rng)
    angular_momenta = [orb[1] for orb in orbs]
    assert sum(l_ * 2 + 1 for l_ in angular_momenta) == ntot
    assert 0 in angular_momenta
    assert 2 in angular_momenta


def test_search_for_gamma_point():
    list_with_gamma = [[-1.0, -1.0, -1.0],
                       [0.0, 0.0, 0.0],
                       [0.1, 0.0, 0.0],
                       [1.5, 2.5, 0.5]]
    gamma_idx = search_for_gamma_point(list_with_gamma)
    assert gamma_idx == 1

    list_without_gamma = [[-1.0, -1.0, -1.0],
                          [0.1, 0.0, 0.0],
                          [1.5, 2.5, 0.5]]
    gamma_idx = search_for_gamma_point(list_without_gamma)
    assert gamma_idx is None


def test_scdm(ti_calculator):
    calc = ti_calculator
    Nw = 14
    ps = calc.get_pseudo_wave_function(band=Nw, kpt=0, spin=0)
    Ng = ps.size
    kpt_kc = calc.get_bz_k_points()
    number_kpts = len(kpt_kc)
    nbands = calc.get_number_of_bands()
    pseudo_nkG = np.zeros((nbands, number_kpts, Ng), dtype=np.complex128)
    for k in range(number_kpts):
        for n in range(nbands):
            pseudo_nkG[n, k] = calc.get_pseudo_wave_function(
                band=n, kpt=k, spin=0).ravel()
    fixed_k = [Nw - 2] * number_kpts
    C_kul, U_kww = scdm(pseudo_nkG, kpts=kpt_kc,
                        fixed_k=fixed_k, Nw=Nw)
    for k in range(number_kpts):
        assert unitarity_error(U_kww[k]) < 1e-10, 'U_ww not unitary'
        assert orthogonality_error(C_kul[k].T) < 1e-10, \
            'C_ul columns not orthogonal'
        assert normalization_error(C_kul[k]) < 1e-10, 'C_ul not normalized'


@pytest.mark.xfail()
def test_get_optimal_nwannier(wan, si_calculator):
    """ Test method to compute the optimal 'nwannier' value. """

    wanf = wan(calc=si_calculator, full_calc=True,
               initialwannier='bloch', nwannier='auto', fixedenergy=1)

    # Test with default parameters
    opt_nw = wanf.get_optimal_nwannier()
    assert opt_nw == 7

    # Test with non-default parameters.
    # This is mostly to test that is does actually support this parameters,
    # it's not really testing the actual result.
    opt_nw = wanf.get_optimal_nwannier(nwrange=10)
    assert opt_nw == 7
    opt_nw = wanf.get_optimal_nwannier(tolerance=1e-2)
    assert opt_nw == 8

    # This should give same result since the initialwannier does not include
    # randomness.
    opt_nw = wanf.get_optimal_nwannier(random_reps=10)
    assert opt_nw == 7

    # Test with random repetitions, just test if it runs.
    wanf = wan(calc=si_calculator, full_calc=True,
               initialwannier='orbitals', nwannier='auto', fixedenergy=0)
    opt_nw = wanf.get_optimal_nwannier(random_reps=10)
    assert opt_nw >= 0


@pytest.mark.xfail()
def test_spread_contributions(wan):
    # Only a test on a constant value to make sure it does not deviate too much
    wan1 = wan()
    test_values_w = wan1._spread_contributions()
    ref_values_w = [2.28535569, 0.04660427]
    assert test_values_w == pytest.approx(ref_values_w, abs=1e-4)
