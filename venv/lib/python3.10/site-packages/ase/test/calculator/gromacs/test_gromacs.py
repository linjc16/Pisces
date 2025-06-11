""" test run for gromacs calculator """

import pytest

from ase.calculators.gromacs import get_gromacs_version, parse_gromacs_version

sample_header = """\
blahblah...
Command line:
  gmx --version

GROMACS version:    2020.1-Ubuntu-2020.1-1
Precision:          single
blahblah...
"""


def test_parse_gromacs_version():
    assert parse_gromacs_version(sample_header) == '2020.1-Ubuntu-2020.1-1'


@pytest.mark.calculator('gromacs')
def test_get_gromacs_version(factory):
    exe = factory.factory.executable
    # XXX Can we do this in a different way?
    version = get_gromacs_version(exe)
    # Hmm.  Version could be any string.
    assert isinstance(version, str) and len(version) > 0


data = """HISE for testing
   20
    3HISE     N    1   1.966   1.938   1.722
    3HISE    H1    2   2.053   1.892   1.711
    3HISE    H2    3   1.893   1.882   1.683
    3HISE    H3    4   1.969   2.026   1.675
    3HISE    CA    5   1.939   1.960   1.866
    3HISE    HA    6   1.934   1.869   1.907
    3HISE    CB    7   2.055   2.041   1.927
    3HISE   HB1    8   2.141   2.007   1.890
    3HISE   HB2    9   2.043   2.137   1.903
    3HISE   ND1   10   1.962   2.069   2.161
    3HISE    CG   11   2.065   2.032   2.077
    3HISE   CE1   12   2.000   2.050   2.287
    3HISE   HE1   13   1.944   2.069   2.368
    3HISE   NE2   14   2.123   2.004   2.287
    3HISE   HE2   15   2.177   1.981   2.369
    3HISE   CD2   16   2.166   1.991   2.157
    3HISE   HD2   17   2.256   1.958   2.128
    3HISE     C   18   1.806   2.032   1.888
    3HISE   OT1   19   1.736   2.000   1.987
    3HISE   OT2   20   1.770   2.057   2.016
   4.00000   4.00000   4.00000"""


@pytest.mark.calculator_lite()
@pytest.mark.calculator('gromacs')
def test_gromacs(factory):
    GRO_INIT_FILE = 'hise_box.gro'

    # write structure file
    with open(GRO_INIT_FILE, 'w') as outfile:
        outfile.write(data)

    calc = factory.calc(
        force_field='charmm27',
        define='-DFLEXIBLE',
        integrator='cg',
        nsteps='10000',
        nstfout='10',
        nstlog='10',
        nstenergy='10',
        nstlist='10',
        ns_type='grid',
        pbc='xyz',
        rlist='0.7',
        coulombtype='PME-Switch',
        rcoulomb='0.6',
        vdwtype='shift',
        rvdw='0.6',
        rvdw_switch='0.55',
        DispCorr='Ener')
    calc.set_own_params_runs(
        'init_structure', GRO_INIT_FILE)
    calc.generate_topology_and_g96file()
    calc.write_input()
    calc.generate_gromacs_run_file()
    calc.run()
    atoms = calc.get_atoms()
    final_energy = calc.get_potential_energy(atoms)

    # e.g., -4.17570101 eV = -402.893902 kJ / mol by Gromacs 2019.1 double
    # precision
    final_energy_ref = -4.175
    tolerance = 0.010
    assert abs(final_energy - final_energy_ref) < tolerance
