"""Tests for the .py format."""
import io

from ase import Atoms
from ase.io.py import write_py


def test_full_results():
    """Test if the .py format has full results without summarization."""
    with io.StringIO() as buf:
        write_py(buf, Atoms('H1000'))
        assert '...' not in buf.getvalue()
