import re

_pattern_test_data = []


def _define_pattern(pattern, example, *args):
    """Compile a regex and store an example pattern for testing."""
    regex = re.compile(pattern, *args)
    _pattern_test_data.append((regex, example))
    return regex
