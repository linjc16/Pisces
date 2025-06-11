"""This module defines an ASE interface to VASP.

The path of the directory containing the pseudopotential
directories (potpaw,potpaw_GGA, potpaw_PBE, ...) should be set
by the environmental flag $VASP_PP_PATH.

The user should also set one of the following environmental flags, which
instructs ASE on how to execute VASP: $ASE_VASP_COMMAND, $VASP_COMMAND, or
$VASP_SCRIPT.

The user can set the environmental flag $VASP_COMMAND pointing
to the command use the launch vasp e.g. 'vasp_std' or 'mpirun -n 16 vasp_std'

Alternatively, the user can also set the environmental flag
$VASP_SCRIPT pointing to a python script looking something like::

   import os
   exitcode = os.system('vasp_std')

www.vasp.at
"""

from ase.utils import deprecated

from .vasp import Vasp


class Vasp2(Vasp):
    @deprecated(
        'Vasp2 has been deprecated. Use the ase.calculators.vasp.Vasp '
        'class instead.'
    )
    def __init__(self, **kwargs):
        """
        .. deprecated:: 3.21.0
            Use :class:`~ase.calculators.vasp.Vasp` instead.
        """
        super().__init__(**kwargs)
