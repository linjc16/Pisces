import os
import sys

try:
    from .version import __version__  # noqa
except ImportError:
    version_txt = os.path.join(os.path.dirname(__file__), "version.txt")
    with open(version_txt) as f:
        __version__ = f.read().strip()

__all__ = ["pdb"]

from fairseq.distributed import utils as distributed_utils

from fairseq.dataclass.initialize import hydra_init
hydra_init()

import fairseq.metrics
