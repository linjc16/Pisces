from ase import Atoms
from ase.calculators.acn import ACN, m_me, r_cn, r_mec
from ase.calculators.fd import calculate_numerical_forces


def test_acn():
    """Test ACN forces."""

    dimer = Atoms('CCNCCN',
                  [(-r_mec, 0, 0),
                   (0, 0, 0),
                   (r_cn, 0, 0),
                   (r_mec, 3.7, 0),
                   (0, 3.7, 0),
                   (-r_cn, 3.7, 0)])

    # Set mass of methyls
    masses = dimer.get_masses()
    masses[::3] = m_me
    dimer.set_masses(masses)

    dimer.calc = ACN(rc=5.0, width=2.0)  # Put C-C distance in the cutoff range
    F = dimer.get_forces()
    print(F)
    Fnum = calculate_numerical_forces(dimer)
    dF = Fnum - F
    print(dF)
    assert abs(dF).max() < 2e-6
