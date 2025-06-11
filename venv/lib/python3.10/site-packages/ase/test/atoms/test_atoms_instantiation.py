from ase import Atom, Atoms

"""The documentation says:

    These three are equivalent:

    >>> d = 1.104  # N2 bondlength
    >>> a = Atoms('N2', [(0, 0, 0), (0, 0, d)])
    >>> a = Atoms(numbers=[7, 7], positions=[(0, 0, 0), (0, 0, d)])
    >>> a = Atoms([Atom('N', (0, 0, 0)), Atom('N', (0, 0, d))])

so let's check"""


numbers = [7, 7]
symbols = ["N", "N"]
dummy_array = 2 * [3 * [0.0]]

d = 1.104  # N2 bondlength
a1 = Atoms("N2", [(0, 0, 0), (0, 0, d)])
a2 = Atoms(numbers=[7, 7], positions=[(0, 0, 0), (0, 0, d)])
a3 = Atoms([Atom("N", (0, 0, 0)), Atom("N", (0, 0, d))])


def test_atoms(atoms1=a1, atoms2=a2, atoms3=a3):
    assert atoms1 == atoms2
    assert atoms2 == atoms3


# test redundant keywords
def test_symbols(numbers=numbers, symbols=symbols):
    kw = {"numbers": numbers, "symbols": symbols}
    _test_keywords(**kw)


def test_momenta(numbers=numbers, momenta=dummy_array):
    kw = {"momenta": momenta, "velocities": momenta}
    _test_keywords(numbers=numbers, **kw)


def test_positions(numbers=numbers, positions=dummy_array):
    kw = {"positions": positions, "scaled_positions": positions}
    _test_keywords(numbers=numbers, **kw)


def _test_keywords(**kw):
    was_raised = False
    try:
        Atoms(**kw)
    except Exception as inst:
        assert isinstance(inst, TypeError), inst
        was_raised = True

    assert was_raised
