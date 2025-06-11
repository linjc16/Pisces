# fmt: off

import warnings

try:
    from numpy.exceptions import VisibleDeprecationWarning  # NumPy 2.0.0
except ImportError:
    from numpy import (  # type: ignore[attr-defined,no-redef]
        VisibleDeprecationWarning,
    )

from ase.spectrum.band_structure import *  # noqa: F401,F403

warnings.warn("ase.dft.band_structure has been moved to "
              "ase.spectrum.band_structure. Please update your "
              "scripts; this alias will be removed in a future "
              "version of ASE.",
              VisibleDeprecationWarning)
