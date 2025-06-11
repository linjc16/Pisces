import numpy as np

from ase.dft.dos import linear_tetrahedron_integration as lti
from ase.dft.kpoints import monkhorst_pack


def test_dos():
    """Check density of states tetrahedron code."""

    cell = np.eye(3)
    shape = (11, 13, 9)
    kpts = np.dot(monkhorst_pack(shape),
                  np.linalg.inv(cell).T).reshape(shape + (3,))

    # Free electron eigenvalues:
    eigs = 0.5 * (kpts**2).sum(3)[..., np.newaxis]  # new axis for 1 band

    energies = np.linspace(0.0001, eigs.max() + 0.0001, 500)

    # Do 3-d, 2-d and 1-d:
    dos3 = lti(cell, eigs, energies)
    eigs = eigs[:, :, 4:5]
    dos2 = lti(cell, eigs, energies)
    eigs = eigs[5:6]
    dos1 = lti(cell, eigs, energies)

    # With weights:
    dos1w = lti(cell, eigs, energies, np.ones_like(eigs))
    assert abs(dos1 - dos1w).max() < 2e-14

    # Analytic results:
    ref3 = 4 * np.pi * (2 * energies)**0.5
    ref2 = 2 * np.pi * np.ones_like(energies)
    ref1 = 2 * (2 * energies)**-0.5

    mask = np.bitwise_and(energies > 0.02, energies < 0.1)
    for dims, (dos, ref) in enumerate([(dos1, ref1), (dos2, ref2),
                                       (dos3, ref3)],
                                      start=1):
        error = abs(1 - dos / ref)[mask].max()
        norm = dos.sum() * (energies[1] - energies[0])
        print(dims, norm, error)
        assert error < 0.2, error
        assert abs(norm - 1) < 0.11**dims, norm
