import numpy as np
import pytest


def compare_with_pytest_approx(quantity, expected_values, rel_tol):
    """
    Cast inputted objects to numpy arrays and assert equivalence via
    pytest.approx with the requested relative tolerance
    """
    quantity, expected_values = map(np.asarray, [quantity, expected_values])
    assert quantity == pytest.approx(expected_values, rel=rel_tol)
