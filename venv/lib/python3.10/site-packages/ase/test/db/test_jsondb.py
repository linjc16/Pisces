from io import StringIO

from ase.io import read, write


def test_jsondb():
    """Read and write json from/to file descriptor."""

    s = """
    {"1":
         {"numbers": [1, 1],
          "positions": [[0.0, 0.0, 0.35],
                        [0.0, 0.0, -0.35]]}}
    """

    fd = StringIO(s)
    a = read(fd, format='json')
    assert a.get_chemical_formula() == 'H2'

    fd = StringIO()
    write(fd, a, format='json')

    fd.seek(0)
    a = read(fd, format='json')
    assert a.get_chemical_formula() == 'H2'
