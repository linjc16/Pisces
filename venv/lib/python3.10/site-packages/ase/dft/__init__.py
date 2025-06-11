import numpy as np
from scipy.integrate import trapezoid

from ase.dft.dos import DOS
from ase.dft.kpoints import monkhorst_pack

__all__ = ['DOS', 'monkhorst_pack']


def get_distribution_moment(x, y, order=0):
    """Return the moment of nth order of distribution.

    1st and 2nd order moments of a band correspond to the band's
    center and width respectively.

    For integration, the trapezoid rule is used.
    """

    x = np.asarray(x)
    y = np.asarray(y)

    if order == 0:
        return trapezoid(y, x)
    elif isinstance(order, int):
        return trapezoid(x**order * y, x) / trapezoid(y, x)
    elif hasattr(order, '__iter__'):
        return [get_distribution_moment(x, y, n) for n in order]
    else:
        raise ValueError(f'Illegal order: {order}')
