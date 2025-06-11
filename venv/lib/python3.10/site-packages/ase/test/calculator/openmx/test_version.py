from ase.calculators.openmx.openmx import parse_omx_version

sample_output = """\

The number of threads in each node for OpenMP parallelization is 1.


*******************************************************
*******************************************************
 Welcome to OpenMX   Ver. 3.8.5
 Copyright (C), 2002-2014, T. Ozaki
"""


def test_parse_omx_version():
    assert parse_omx_version(sample_output) == '3.8.5'
