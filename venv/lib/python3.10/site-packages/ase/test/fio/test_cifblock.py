import pytest

from ase.io.cif import CIFBlock, CIFLoop, parse_loop


def test_parse_cifloop_simple():
    dct = parse_loop(['_apples',
                      '_verysmallrocks',
                      '2 200',
                      '3 300',
                      '4 400'][::-1])
    assert dct['_apples'] == [2, 3, 4]
    assert dct['_verysmallrocks'] == [200, 300, 400]


def test_parse_cifloop_warn_duplicate_header():
    with pytest.warns(UserWarning):
        parse_loop(['_hello', '_hello'])


def test_parse_cifloop_incomplete():
    with pytest.raises(RuntimeError):
        parse_loop(['_spam', '_eggs', '1 2', '1'][::-1])


def test_cifloop_roundtrip():
    loop = CIFLoop()
    loop.add('_potatoes', [2.5, 3.0, -1.0], '{:8.5f}')
    loop.add('_eggs', [1, 2, 3], '{:2d}')
    string = loop.tostring() + '\n'
    print('hmm', string)
    lines = string.splitlines()[::-1]
    assert lines.pop() == 'loop_'

    for line in lines:
        print(repr(line))
    dct = parse_loop(lines)
    assert dct['_potatoes'] == pytest.approx([2.5, 3.0, -1.0])
    assert dct['_eggs'] == [1, 2, 3]


@pytest.fixture()
def cifblock():
    return CIFBlock('hello', {'_cifkey': 42})


def test_repr(cifblock):
    text = repr(cifblock)
    assert 'hello' in text
    assert '_cifkey' in text


def test_mapping(cifblock):
    assert len(cifblock) == 1
    assert len(list(cifblock)) == 1


def test_various(cifblock):
    assert cifblock.get_cellpar() is None
    assert cifblock.get_cell().rank == 0


def test_deuterium():
    # Verify that the symbol 'D' becomes hydrogen ('H') with mass 2(-ish).
    symbols = ['H', 'D', 'D', 'He']
    block = CIFBlock('deuterium', dict(_atom_site_type_symbol=symbols))
    assert block.get_symbols() == ['H', 'H', 'H', 'He']
    masses = block._get_masses()
    assert all(masses.round().astype(int) == [1, 2, 2, 4])
