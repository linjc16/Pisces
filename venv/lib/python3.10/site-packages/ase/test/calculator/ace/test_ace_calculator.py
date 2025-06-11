from ase import Atoms
from ase.calculators.acemolecule import ACE


def dict_is_subset(d1, d2):
    """True if all the key-value pairs in dict 1 are in dict 2"""

    return all(key in d2 and d2[key] == d1[key] for key in d1)


def test_acemolecule_calculator():

    ace_cmd = "_ase_dummy_ace_command"

    basis = dict(
        Scaling='0.5',
        Cell=7.0,
        Grid='Basic',
        Centered=0,
        Pseudopotential={
            'Pseudopotential': 1,
            'Format': 'upf',
            'PSFilenames': '/PATH/TO/He.pbe.UPF'})
    guess = dict(InitialGuess=1, InitialFilenames='/PATH/TO/He.pbe.UPF')
    scf = dict(IterateMaxCycle=50, ConvergenceType='Energy',
               ConvergenceTolerance=0.000001, EnergyDecomposition=1,
               ExchangeCorrelation={
                   'XFunctional': 'LDA_X',
                   'CFunctional': 'LDA_C_PW'},
               Diagonalize={'Tolerance': 1e-7},
               Mixing={'MixingType': 'Density', 'MixingParameter': 0.3,
                       'MixingMethod': 1})
    he = Atoms("He", positions=[[0.0, 0.0, 0.0]])
    he.calc = ACE(command=ace_cmd, BasicInformation=basis,
                  Guess=guess, Scf=scf)
    sample_parameters = he.calc.parameters
    assert dict_is_subset(basis, sample_parameters['BasicInformation'][0])
    assert dict_is_subset(guess, sample_parameters['Guess'][0])
    assert dict_is_subset(scf, sample_parameters['Scf'][0])
    he.calc.set(BasicInformation={"Pseudopotential": {"UsingDoubleGrid": 1}})
    sample_parameters = he.calc.parameters
    assert dict_is_subset(
        {"UsingDoubleGrid": 1},
        sample_parameters['BasicInformation'][0]["Pseudopotential"])
