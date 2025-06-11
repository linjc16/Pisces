from ase.calculators.vasp import get_vasp_version

vasp_sample_header = """\
 running on    1 total cores
 distrk:  each k-point on    1 cores,    1 groups
 distr:  one band on    1 cores,    1 groups
 using from now: INCAR
 vasp.6.1.2 22Jul20 (build Jan 19 2021 13:49:35) complex
"""


def test_vasp_version():
    assert get_vasp_version(vasp_sample_header) == '6.1.2'
