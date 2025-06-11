# fmt: off

import warnings

from ase.calculators.calculator import FileIOCalculator
from ase.io import read, write
from ase.io.gamess_us import clean_userscr, get_userscr


class GAMESSUS(FileIOCalculator):
    implemented_properties = ['energy', 'forces', 'dipole']
    _legacy_default_command = 'rungms PREFIX.inp > PREFIX.log 2> PREFIX.err'
    discard_results_on_any_change = True

    def __init__(self, restart=None,
                 ignore_bad_restart_file=FileIOCalculator._deprecated,
                 label='gamess_us', atoms=None, command=None, userscr=None,
                 **kwargs):
        """
        GAMESS-US keywords are specified using dictionaries of keywords.
        For example, to run a CCSD(T)/cc-pVDZ calculation, you might use the
        following arguments:

            >>> calc = GAMESSUS(contrl={'scftyp': 'rhf', 'cctyp': 'ccsd(t)',
            >>>                         'ispher': 1, 'runtyp': 'energy'},
            >>>                 basis={'gbasis': 'CCD'})

        This will create an input file that looks like the following:

            >>>  $CONTRL
            >>>   SCFTYP=RHF
            >>>   CCTYP=CCSD(T)
            >>>   ISPHER=1
            >>>   RUNTYP=ENERGY
            >>>  $END
            >>>
            >>>  $BASIS
            >>>   GBASIS=CCSD
            >>>  $END

        See the INPUT.DOC file provided by GAMESS-US for more information.

        If `runtyp` is not specified, it will be set automatically.

        If no basis is specified, 3-21G will be used by default.
        A dictionary containing literal per-index or per-element basis sets
        can be passed to the `basis` keyword. This will result in the basis
        set being printed in the $DATA block, alongside the geometry
        specification.
        Otherwise, `basis` is assumed to contain keywords for the $BASIS
        block, such as GBASIS and NGAUSS.

        If a multiplicity is not set in contrl['mult'], the multiplicity
        will be guessed based on the Atoms object's initial magnetic moments.

        The GAMESSUS calculator has some special keyword:

        xc: str
            The exchange-correlation functional to use for DFT calculations.
            In most circumstances, setting xc is equivalent to setting
            contrl['dfttyp']. xc will be ignored if a value has also
            been provided to contrl['dfttyp'].

        userscr: str
            The location of the USERSCR directory specified in your
            `rungms` script. If not provided, an attempt will be made
            to automatically determine the location of this directory.
            If this fails, you will need to manually specify the path
            to this directory to the calculator using this keyword.

        ecp: dict
            A per-index or per-element dictionary of ECP specifications.
        """

        # Backward compatibility fix:
        if command is None and 'ASE_GAMESSUS_COMMAND' in self.cfg:
            command = self.cfg['ASE_GAMESSUS_COMMAND']

        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                  label, atoms, command, **kwargs)
        self.userscr = userscr

    def _get_name(self):
        return 'gamess_us'

    def calculate(self, *args, **kwargs):
        if self.userscr is None:
            if 'rungms' in self.command:
                self.userscr = get_userscr(self.prefix, self.command)

        if self.userscr is None:
            warnings.warn("Could not determine USERSCR! "
                          "GAMESS may refuse to run more than once for "
                          "this job. Please pass userscr to the GAMESSUS "
                          "Calculator if you run into problems!")
        else:
            clean_userscr(self.userscr, self.prefix)

        FileIOCalculator.calculate(self, *args, **kwargs)

    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, atoms, properties, system_changes)
        write(self.label + '.inp', atoms, properties=properties,
              format='gamess-us-in', **self.parameters)

    def read_results(self):
        output = read(self.label + '.log')
        self.calc = output.calc
        self.results = output.calc.results
