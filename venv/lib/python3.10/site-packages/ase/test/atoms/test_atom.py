from ase import Atom, Atoms


def test_atom():
    m = Atoms('H2')
    a = m[0]
    b = Atom('H')
    for c in [a, b]:
        assert c.x == 0
        c.z = 24.0
        assert c.position[2] == 24.0
        assert c.symbol == 'H'
        c.number = 92
        assert c.symbol == 'U'
        c.symbol = 'Fe'
        assert c.number == 26
        c.tag = 42
        assert c.tag == 42
        c.momentum = (1, 2, 3)
    assert m[0].tag == 42
    momenta = m.get_momenta()
    assert abs(momenta).sum() > 0
    m = Atoms('LiH')
    for a in m:
        print(a.symbol)
    for a in m:
        if a.symbol == 'H':
            a.z = 0.75
    assert m.get_distance(0, 1) == 0.75
    a = m.pop()
    m += a
    del m[:1]
    print(m)


def test_repr():
    txt = repr(Atom('Si', charge=7))
    print(txt)
    assert 'Si' in txt
    assert 'charge' in txt
