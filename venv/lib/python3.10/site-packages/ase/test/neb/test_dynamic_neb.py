import numpy as np
import pytest

from ase import Atoms
from ase.build import fcc111
from ase.calculators.emt import EMT as OrigEMT
from ase.mep import DyNEB
from ase.optimize import BFGS


@pytest.mark.optimize()
@pytest.mark.slow()
def test_dynamic_neb():
    # Global counter of force evaluations:
    force_evaluations = [0]

    class EMT(OrigEMT):
        def calculate(self, *args, **kwargs):
            force_evaluations[0] += 1
            OrigEMT.calculate(self, *args, **kwargs)

    # Build Pt(111) slab with six surface atoms and add oxygen adsorbate
    initial = fcc111('Pt', size=(3, 2, 3), orthogonal=True)
    initial.center(axis=2, vacuum=10)
    oxygen = Atoms('O')
    oxygen.translate(initial[7].position + (0., 0., 3.5))
    initial.extend(oxygen)

    # EMT potential
    initial.calc = EMT()

    # Optimize initial state
    opt = BFGS(initial)
    opt.run(fmax=0.03)

    # Move oxygen adsorbate to neighboring hollow site
    final = initial.copy()
    final[18].x += 2.8
    final[18].y += 1.8

    final.calc = EMT()

    opt = BFGS(final)
    opt.run(fmax=0.03)

    # NEB with seven interior images
    images = [initial]
    for i in range(7):
        images.append(initial.copy())
    images.append(final)

    fmax = 0.03  # Same for NEB and optimizer

    for i in range(1, len(images) - 1):
        calc = EMT()
        images[i].calc = calc

    def run_NEB():
        if method == 'dyn':
            neb = DyNEB(images, fmax=fmax, dynamic_relaxation=True)
            neb.interpolate()
        elif method == 'dyn_scale':
            neb = DyNEB(images, fmax=fmax, dynamic_relaxation=True,
                        scale_fmax=6.)
            neb.interpolate()
        else:
            # Default NEB
            neb = DyNEB(images, dynamic_relaxation=False)
            neb.interpolate()

        # Optimize and check number of calculations.
        # We use a hack with a global counter to count the force evaluations:
        force_evaluations[0] = 0
        opt = BFGS(neb)
        opt.run(fmax=fmax)
        force_calls.append(force_evaluations[0])

        # Get potential energy of transition state.
        Emax.append(np.sort([image.get_potential_energy()
                             for image in images[1:-1]])[-1])

    force_calls, Emax = [], []
    for method in ['def', 'dyn', 'dyn_scale']:
        run_NEB()

    # Check force calculation count for default and dynamic NEB implementations
    print(f'\n# Force calls with default NEB: {force_calls[0]}')
    print(f'# Force calls with dynamic NEB: {force_calls[1]}')
    print('# Force calls with dynamic and scaled NEB: {}\n'.format(
        force_calls[2]))
    assert force_calls[2] < force_calls[1] < force_calls[0]

    # Assert reaction barriers are within 1 meV of default NEB
    assert (abs(Emax[1] - Emax[0]) < 1e-3)
    assert (abs(Emax[2] - Emax[0]) < 1e-3)
