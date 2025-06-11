from ase.io.nwchem.parser import _pattern_test_data


def test_parser():
    for regex, pattern in _pattern_test_data:
        assert regex.match(pattern) is not None, pattern
