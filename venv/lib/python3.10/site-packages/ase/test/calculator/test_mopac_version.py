import pytest

snippets = {
    '22.0.2': """\
*******************************************************************************
**                                                                           **
**                              MOPAC v22.0.2                                **
**                                                                           **
*******************************************************************************
""",
    '21.298M': """\
*******************************************************************************
** Cite this program as: MOPAC2016, Version: 21.298M, James J. P. Stewart,   **
**                           web-site: HTTP://OpenMOPAC.net.                 **
*******************************************************************************
""",
    '20.173L': """\
*******************************************************************************
** Site#:    0         For non-commercial use only    Version 20.173L 64BITS **
*******************************************************************************
** Cite this program as: MOPAC2016, Version: 20.173L, James J. P. Stewart,   **
** Stewart Computational Chemistry, web-site: HTTP://OpenMOPAC.net.          **
*******************************************************************************
""",
}


@pytest.mark.parametrize('version', [*snippets])
def test_version(version):
    from ase.calculators.mopac import get_version_number
    lines = snippets[version].splitlines()
    assert get_version_number(lines) == version
