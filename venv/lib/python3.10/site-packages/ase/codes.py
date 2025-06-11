# fmt: off

from dataclasses import dataclass

# Note: There could be more than one "calculator" for any given code;
# for example Espresso can work both as GenericFileIOCalculator and
# SocketIOCalculator, or as part of some DFTD3 combination.
#
# Also, DFTD3 is one external code but can be invoked alone (as PureDFTD3)
# as well as together with a DFT code (the main DFTD3 calculator).
#
# The current CodeMetadata object only specifies a single calculator class.
# We should be wary of these invisible "one-to-one" restrictions.


@dataclass
class CodeMetadata:
    name: str
    longname: str
    modulename: str
    classname: str

    def calculator_class(self):
        from importlib import import_module
        module = import_module(self.modulename)
        cls = getattr(module, self.classname)
        return cls

    @classmethod
    def define_code(cls, name, longname, importpath):
        modulename, classname = importpath.rsplit('.', 1)
        return cls(name, longname, modulename, classname)

    def _description(self):
        yield f'Name:     {self.longname}'
        yield f'Import:   {self.modulename}.{self.classname}'
        yield f'Type:     {self.calculator_type()}'
        yield ''
        yield from self._config_description()

    def description(self, indent=''):
        return '\n'.join(indent + line for line in self._description())

    def is_legacy_fileio(self):
        from ase.calculators.calculator import FileIOCalculator
        return issubclass(self.calculator_class(), FileIOCalculator)

    def is_generic_fileio(self):
        from ase.calculators.genericfileio import CalculatorTemplate

        # It is nicer to check for the template class, since it has the name,
        # but then calculator_class() should be renamed.
        return issubclass(self.calculator_class(), CalculatorTemplate)

    def is_calculator_oldbase(self):
        from ase.calculators.calculator import Calculator
        return issubclass(self.calculator_class(), Calculator)

    def is_base_calculator(self):
        from ase.calculators.calculator import BaseCalculator
        return issubclass(self.calculator_class(), BaseCalculator)

    def calculator_type(self):
        cls = self.calculator_class()

        if self.is_generic_fileio():
            return 'GenericFileIOCalculator'

        if self.is_legacy_fileio():
            return 'FileIOCalculator (legacy)'

        if self.is_calculator_oldbase():
            return 'Calculator (legacy base class)'

        if self.is_base_calculator():
            return 'Base calculator'

        return f'BAD: Not a proper calculator (superclasses: {cls.__mro__})'

    def profile(self):
        from ase.calculators.calculator import FileIOCalculator
        from ase.calculators.genericfileio import CalculatorTemplate
        from ase.config import cfg
        cls = self.calculator_class()
        if issubclass(cls, CalculatorTemplate):
            return cls().load_profile(cfg)
        elif hasattr(cls, 'fileio_rules'):
            assert issubclass(cls, FileIOCalculator)
            return cls.load_argv_profile(cfg, self.name)
        else:
            raise NotImplementedError('profile() not implemented')

    def _config_description(self):
        from ase.calculators.genericfileio import BadConfiguration
        from ase.config import cfg

        parser = cfg.parser
        if self.name not in parser:
            yield f'Not configured: No [{self.name}] section in configuration'
            return

        try:
            profile = self.profile()
        except BadConfiguration as ex:
            yield f'Error in configuration section [{self.name}]'
            yield 'Missing or bad parameters:'
            yield f'  {ex}'
            return
        except NotImplementedError as ex:
            yield f'N/A: {ex}'
            return

        yield f'Configured by section [{self.name}]:'
        configvars = vars(profile)
        for name in sorted(configvars):
            yield f'  {name} = {configvars[name]}'

        return


def register_codes():

    codes = {}

    def reg(name, *args):
        code = CodeMetadata.define_code(name, *args)
        codes[name] = code

    reg('abinit', 'Abinit', 'ase.calculators.abinit.AbinitTemplate')
    reg('ace', 'ACE molecule', 'ase.calculators.acemolecule.ACE')
    # internal: reg('acn', 'ACN force field', 'ase.calculators.acn.ACN')
    reg('aims', 'FHI-Aims', 'ase.calculators.aims.AimsTemplate')
    reg('amber', 'Amber', 'ase.calculators.amber.Amber')
    reg('castep', 'Castep', 'ase.calculators.castep.Castep')
    # internal: combine_mm
    # internal: counterions
    reg('cp2k', 'CP2K', 'ase.calculators.cp2k.CP2K')
    reg('crystal', 'CRYSTAL', 'ase.calculators.crystal.CRYSTAL')
    reg('demon', 'deMon', 'ase.calculators.demon.Demon')
    reg('demonnano', 'deMon-nano', 'ase.calculators.demonnano.DemonNano')
    reg('dftb', 'DFTB+', 'ase.calculators.dftb.Dftb')
    reg('dftd3', 'DFT-D3', 'ase.calculators.dftd3.DFTD3')
    # reg('dftd3-pure', 'DFT-D3 (pure)', 'ase.calculators.dftd3.puredftd3')
    reg('dmol', 'DMol3', 'ase.calculators.dmol.DMol3')
    # internal: reg('eam', 'EAM', 'ase.calculators.eam.EAM')
    reg('elk', 'ELK', 'ase.calculators.elk.ELK')
    # internal: reg('emt', 'EMT potential', 'ase.calculators.emt.EMT')
    reg('espresso', 'Quantum Espresso',
        'ase.calculators.espresso.EspressoTemplate')
    reg('exciting', 'Exciting',
        'ase.calculators.exciting.exciting.ExcitingGroundStateTemplate')
    # internal: reg('ff', 'FF', 'ase.calculators.ff.ForceField')
    # fleur <- external nowadays
    reg('gamess_us', 'GAMESS-US', 'ase.calculators.gamess_us.GAMESSUS')
    reg('gaussian', 'Gaussian', 'ase.calculators.gaussian.Gaussian')
    reg('gromacs', 'Gromacs', 'ase.calculators.gromacs.Gromacs')
    reg('gulp', 'GULP', 'ase.calculators.gulp.GULP')
    # h2morse.py do we need a specific H2 morse calculator when we have morse??
    # internal: reg('harmonic', 'Harmonic potential',
    #  'ase.calculators.harmonic.HarmonicCalculator')
    # internal: reg('idealgas', 'Ideal gas (dummy)',
    #             'ase.calculators.idealgas.IdealGas')
    # XXX cannot import without kimpy installed, fixme:
    # reg('kim', 'OpenKIM', 'ase.calculators.kim.kim.KIM')
    reg('lammpslib', 'Lammps (python library)',
        'ase.calculators.lammpslib.LAMMPSlib')
    reg('lammpsrun', 'Lammps (external)', 'ase.calculators.lammpsrun.LAMMPS')
    # internal: reg('lj', 'Lennard–Jones potential',
    #             'ase.calculators.lj.LennardJones')
    # internal: loggingcalc.py
    # internal: mixing.py
    reg('mopac', 'MOPAC', 'ase.calculators.mopac.MOPAC')
    # internal: reg('morse', 'Morse potential',
    # 'ase.calculators.morse.MorsePotential')
    reg('nwchem', 'NWChem', 'ase.calculators.nwchem.NWChem')
    reg('octopus', 'Octopus', 'ase.calculators.octopus.OctopusTemplate')
    reg('onetep', 'Onetep', 'ase.calculators.onetep.OnetepTemplate')
    reg('openmx', 'OpenMX', 'ase.calculators.openmx.OpenMX')
    reg('orca', 'ORCA', 'ase.calculators.orca.OrcaTemplate')
    reg('plumed', 'Plumed', 'ase.calculators.plumed.Plumed')
    reg('psi4', 'Psi4', 'ase.calculators.psi4.Psi4')
    reg('qchem', 'QChem', 'ase.calculators.qchem.QChem')
    # internal: qmmm.py
    reg('siesta', 'SIESTA', 'ase.calculators.siesta.Siesta')
    # internal: test.py
    # internal: reg('tip3p', 'TIP3P', 'ase.calculators.tip3p.TIP3P')
    # internal: reg('tip4p', 'TIP4P', 'ase.calculators.tip4p.TIP4P')
    reg('turbomole', 'Turbomole', 'ase.calculators.turbomole.Turbomole')
    reg('vasp', 'VASP', 'ase.calculators.vasp.Vasp')
    # internal: vdwcorrection
    return codes


codes = register_codes()


def list_codes(names):
    from ase.config import cfg
    cfg.print_header()
    print()

    for name in names:
        code = codes[name]
        print(code.name)
        try:
            print(code.description(indent='  '))
        except Exception as ex:
            print(f'Bad configuration of {name}: {ex!r}')
        print()


if __name__ == '__main__':
    import sys
    names = sys.argv[1:]
    if not names:
        names = [*codes]
    list_codes(names)
