import json

import numpy as np
import pytest

from ase.build import bulk
from ase.calculators.morse import MorsePotential
from ase.constraints import FixBondLength
from ase.geometry.geometry import find_mic, get_distances
from ase.mep import NEB, NEBTools
from ase.mep.neb import NEBOptimizer
from ase.optimize import BFGS, ODE12r
from ase.optimize.precon import Exp
from ase.utils.forcecurve import fit_images

pytestmark = pytest.mark.optimize


def calc():
    return MorsePotential(A=4.0, epsilon=1.0, r0=2.55)


@pytest.fixture(scope='module')
def _setup_images_global():
    N_intermediate = 3
    N_cell = 2
    initial = bulk('Cu', cubic=True)
    initial *= N_cell

    # place vacancy near centre of cell
    D, D_len = get_distances(np.diag(initial.cell) / 2,
                             initial.positions,
                             initial.cell, initial.pbc)
    vac_index = D_len.argmin()
    vac_pos = initial.positions[vac_index]
    del initial[vac_index]

    # identify two opposing nearest neighbours of the vacancy
    D, D_len = get_distances(vac_pos,
                             initial.positions,
                             initial.cell, initial.pbc)
    D = D[0, :]
    D_len = D_len[0, :]

    nn_mask = np.abs(D_len - D_len.min()) < 1e-8
    i1 = nn_mask.nonzero()[0][0]
    i2 = ((D + D[i1])**2).sum(axis=1).argmin()

    print(f'vac_index={vac_index} i1={i1} i2={i2} '
          f'distance={initial.get_distance(i1, i2, mic=True)}')

    final = initial.copy()
    final.positions[i1] = vac_pos

    initial.calc = calc()
    final.calc = calc()

    qn = ODE12r(initial)
    qn.run(fmax=1e-3)
    qn = ODE12r(final)
    qn.run(fmax=1e-3)

    images = [initial]
    for image in range(N_intermediate):
        image = initial.copy()
        image.calc = calc()
        images.append(image)
    images.append(final)

    neb = NEB(images)
    neb.interpolate()

    return neb.images, i1, i2


@pytest.fixture()
def setup_images(_setup_images_global):
    images, i1, i2 = _setup_images_global
    new_images = [img.copy() for img in images]
    for img in new_images:
        img.calc = calc()
    return new_images, i1, i2


@pytest.fixture(scope='module')
def _ref_vacancy_global(_setup_images_global):
    # use distance from moving atom to one of its neighbours as reaction coord
    # relax intermediate image to the saddle point using a bondlength constraint
    images, i1, i2 = _setup_images_global
    initial, saddle, final = (images[0].copy(),
                              images[2].copy(),
                              images[4].copy())
    initial.calc = calc()
    saddle.calc = calc()
    final.calc = calc()
    saddle.set_constraint(FixBondLength(i1, i2))
    opt = ODE12r(saddle)
    opt.run(fmax=1e-2)
    nebtools = NEBTools([initial, saddle, final])
    Ef_ref, dE_ref = nebtools.get_barrier(fit=False)
    print('REF:', Ef_ref, dE_ref)
    return Ef_ref, dE_ref, saddle


@pytest.fixture()
def ref_vacancy(_ref_vacancy_global):
    Ef_ref, dE_ref, saddle = _ref_vacancy_global
    return Ef_ref, dE_ref, saddle.copy()


@pytest.mark.slow()
@pytest.mark.filterwarnings('ignore:estimate_mu')
@pytest.mark.parametrize('method, optimizer, precon, optmethod',
                         [('aseneb', BFGS, None, None),
                          ('improvedtangent', BFGS, None, None),
                          ('spline', NEBOptimizer, None, 'ODE'),
                          ('string', NEBOptimizer, 'Exp', 'ODE')])
def test_neb_methods(testdir, method, optimizer, precon,
                     optmethod, ref_vacancy, setup_images):
    # unpack the reference result
    Ef_ref, dE_ref, saddle_ref = ref_vacancy

    # now relax the MEP for comparison
    images, _, _ = setup_images

    fmax_history = []

    def save_fmax_history(mep):
        fmax_history.append(mep.get_residual())

    k = 0.1
    if precon == 'Exp':
        k = 0.01
    mep = NEB(images, k=k, method=method, precon=precon)

    if optmethod is not None:
        opt = optimizer(mep, method=optmethod)
    else:
        opt = optimizer(mep)
    opt.attach(save_fmax_history, 1, mep)
    opt.run(fmax=1e-2)

    nebtools = NEBTools(images)
    Ef, dE = nebtools.get_barrier(fit=False)
    print(f'{method},{optimizer.__name__},{precon} '
          f'=> Ef = {Ef:.3f}, dE = {dE:.3f}')

    forcefit = fit_images(images)

    with open(f'MEP_{method}_{optimizer.__name__}_{optmethod}'
              f'_{precon}.json', 'w') as fd:
        json.dump({'fmax_history': fmax_history,
                   'method': method,
                   'optmethod': optmethod,
                   'precon': precon,
                   'optimizer': optimizer.__name__,
                   'path': forcefit.path,
                   'energies': forcefit.energies.tolist(),
                   'fit_path': forcefit.fit_path.tolist(),
                   'fit_energies': forcefit.fit_energies.tolist(),
                   'lines': np.array(forcefit.lines).tolist(),
                   'Ef': Ef,
                   'dE': dE}, fd)

    centre = 2  # we have 5 images total, so central image has index 2
    vdiff, _ = find_mic(images[centre].positions - saddle_ref.positions,
                        images[centre].cell)
    print(f'Ef error {Ef - Ef_ref} dE error {dE - dE_ref} '
          f'position error at saddle {abs(vdiff).max()}')
    assert abs(Ef - Ef_ref) < 1e-2
    assert abs(dE - dE_ref) < 1e-2
    assert abs(vdiff).max() < 1e-2


@pytest.mark.parametrize('method', ['ODE', 'static'])
@pytest.mark.filterwarnings('ignore:NEBOptimizer did not converge')
def test_neb_optimizers(setup_images, method):
    images, _, _ = setup_images
    mep = NEB(images, method='spline', precon='Exp')
    mep.get_forces()  # needed so residuals are available
    R0 = mep.get_residual()
    opt = NEBOptimizer(mep, method=method)
    opt.run(steps=2)  # take two steps
    R1 = mep.get_residual()
    # check residual has got smaller
    assert R1 < R0


def test_precon_initialisation(setup_images):
    images, _, _ = setup_images
    mep = NEB(images, method='spline', precon='Exp')
    mep.get_forces()
    assert len(mep.precon) == len(mep.images)
    assert mep.precon[0].mu == mep.precon[1].mu


def test_single_precon_initialisation(setup_images):
    images, _, _ = setup_images
    precon = Exp()
    mep = NEB(images, method='spline', precon=precon)
    mep.get_forces()
    assert len(mep.precon) == len(mep.images)
    assert mep.precon[0].mu == mep.precon[1].mu


def test_list_precon_initialisation(setup_images):
    images, _, _ = setup_images

    precon = Exp()
    mep = NEB(images, method='spline', precon=precon)
    mep.get_forces()

    # the tested scenario is saving computed precon
    # for restarting of a calculation

    # saving as PreconImages object
    mep_restart = NEB(images, method='spline', precon=mep.precon)
    mep_restart.get_forces()
    assert len(mep_restart.precon) == len(mep_restart.images)
    assert mep_restart.precon[0].mu == mep_restart.precon[1].mu
    # saving as a list of precon objects
    mep_restart = NEB(images, method='spline', precon=mep.precon.precon)
    mep_restart.get_forces()
    assert len(mep_restart.precon) == len(mep_restart.images)
    assert mep_restart.precon[0].mu == mep_restart.precon[1].mu


def test_precon_assembly(setup_images):
    images, _, _ = setup_images
    neb = NEB(images, method='spline', precon='Exp')
    neb.get_forces()  # trigger precon assembly

    # check precon for each image is symmetric positive definite
    for image, precon in zip(neb.images, neb.precon):
        assert isinstance(precon, Exp)
        P = precon.asarray()
        N = 3 * len(image)
        assert P.shape == (N, N)
        assert np.abs(P - P.T).max() < 1e-6
        assert np.all(np.linalg.eigvalsh(P)) > 0


def test_spline_fit(setup_images):
    images, _, _ = setup_images
    neb = NEB(images)
    fit = neb.spline_fit()

    # check spline points are equally spaced
    assert np.allclose(fit.s, np.linspace(0, 1, len(images)))

    # check spline matches target at fit points
    assert np.allclose(fit.x(fit.s), fit.x_data)

    # ensure derivative is smooth across central fit point
    eps = 1e-4
    assert np.allclose(fit.dx_ds(fit.s[2] + eps), fit.dx_ds(fit.s[2] + eps))


def test_integrate_forces(setup_images):
    images, _, _ = setup_images
    forcefit = fit_images(images)

    neb = NEB(images)
    spline_points = 1000  # it is the default value
    _s, E, _F = neb.integrate_forces(spline_points=spline_points)
    # check the difference between initial and final images
    np.testing.assert_allclose(E[0] - E[-1],
                               forcefit.energies[0] - forcefit.energies[-1],
                               atol=1.0e-10)
    # assert the maximum Energy value is in the middle
    assert np.argmax(E) == spline_points // 2 - 1
    # check the maximum values (barrier value)
    # tolerance value is rather high since the images are not relaxed
    np.testing.assert_allclose(E.max(),
                               forcefit.energies.max(), rtol=2.5e-2)
