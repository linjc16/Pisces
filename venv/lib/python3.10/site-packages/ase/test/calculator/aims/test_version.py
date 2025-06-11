from ase.calculators.aims import get_aims_version

version_string = """\
          Invoking FHI-aims ...

blah blah blah

  FHI-aims version      : 200112.2
  Commit number         : GITDIR-NOTFOUND
"""


def test_get_aims_version():
    assert get_aims_version(version_string) == '200112.2'
