# fmt: off

import numpy as np

import ase.units as un
from ase.calculators.polarizability import StaticPolarizabilityCalculator


class SiestaLRTDDFT:
    """Interface for linear response TDDFT for Siesta via `PyNAO`_

    When using PyNAO please cite the papers indicated in the `documentation \
<https://mbarbrywebsite.ddns.net/pynao/doc/html/references.html>`_
    """

    def __init__(self, initialize=False, **kw):
        """
        Parameters
        ----------
        initialize: bool
            To initialize the tddft calculations before
            calculating the polarizability
            Can be useful to calculate multiple frequency range
            without the need to recalculate the kernel
        kw: dictionary
            keywords for the tddft_iter function from PyNAO
        """

        try:
            from pynao import tddft_iter
        except ModuleNotFoundError as err:
            msg = ("running lrtddft with Siesta calculator "
                   "requires pynao package")
            raise ModuleNotFoundError(msg) from err

        self.initialize = initialize
        self.lrtddft_params = kw
        self.tddft = None

        # convert iter_broadening to Ha
        if "iter_broadening" in self.lrtddft_params:
            self.lrtddft_params["iter_broadening"] /= un.Ha

        if self.initialize:
            self.tddft = tddft_iter(**self.lrtddft_params)

    def get_ground_state(self, atoms, **kw):
        """
        Run siesta calculations in order to get ground state properties.
        Makes sure that the proper parameters are unsed in order to be able
        to run pynao afterward, i.e.,

            COOP.Write = True
            WriteDenchar = True
            XML.Write = True
        """
        from ase.calculators.siesta import Siesta

        if "fdf_arguments" not in kw.keys():
            kw["fdf_arguments"] = {"COOP.Write": True,
                                   "WriteDenchar": True,
                                   "XML.Write": True}
        else:
            for param in ["COOP.Write", "WriteDenchar", "XML.Write"]:
                kw["fdf_arguments"][param] = True

        siesta = Siesta(**kw)
        atoms.calc = siesta
        atoms.get_potential_energy()

    def get_polarizability(self, omega, Eext=np.array(
            [1.0, 1.0, 1.0]), inter=True):
        """
        Calculate the polarizability of a molecule via linear response TDDFT
        calculation.

        Parameters
        ----------
        omega: float or array like
            frequency range for which the polarizability should be
            computed, in eV

        Returns
        -------
        polarizability: array like (complex)
            array of dimension (3, 3, nff) with nff the number of frequency,
            the first and second dimension are the matrix elements of the
            polarizability in atomic units::

                P_xx, P_xy, P_xz, Pyx, .......

        Example
        -------

        from ase.calculators.siesta.siesta_lrtddft import siestaLRTDDFT
        from ase.build import molecule
        import numpy as np
        import matplotlib.pyplot as plt

        # Define the systems
        CH4 = molecule('CH4')

        lr = siestaLRTDDFT(label="siesta", jcutoff=7, iter_broadening=0.15,
                            xc_code='LDA,PZ', tol_loc=1e-6, tol_biloc=1e-7)

        # run DFT calculation with Siesta
        lr.get_ground_state(CH4)

        # run TDDFT calculation with PyNAO
        freq=np.arange(0.0, 25.0, 0.05)
        pmat = lr.get_polarizability(freq)
        """
        from pynao import tddft_iter

        if not self.initialize:
            self.tddft = tddft_iter(**self.lrtddft_params)

        if isinstance(omega, float):
            freq = np.array([omega])
        elif isinstance(omega, list):
            freq = np.array([omega])
        elif isinstance(omega, np.ndarray):
            freq = omega
        else:
            raise ValueError("omega soulf")

        freq_cmplx = freq / un.Ha + 1j * self.tddft.eps
        if inter:
            pmat = -self.tddft.comp_polariz_inter_Edir(freq_cmplx, Eext=Eext)
            self.dn = self.tddft.dn
        else:
            pmat = -self.tddft.comp_polariz_nonin_Edir(freq_cmplx, Eext=Eext)
            self.dn = self.tddft.dn0

        return pmat


class RamanCalculatorInterface(SiestaLRTDDFT, StaticPolarizabilityCalculator):
    """Raman interface for Siesta calculator.
    When using the Raman calculator, please cite

    M. Walter and M. Moseler, Ab Initio Wavelength-Dependent Raman
    Spectra: Placzek Approximation and Beyond, J. Chem. Theory
    Comput. 2020, 16, 1, 576–586"""

    def __init__(self, omega=0.0, **kw):
        """
        Parameters
        ----------
        omega: float
            frequency at which the Raman intensity should be computed, in eV

        kw: dictionary
            The parameter for the siesta_lrtddft object
        """

        self.omega = omega
        super().__init__(**kw)

    def calculate(self, atoms):
        """
        Calculate the polarizability for frequency omega

        Parameters
        ----------
        atoms: atoms class
            The atoms definition of the system. Not used but required by Raman
            calculator
        """
        pmat = self.get_polarizability(
            self.omega, Eext=np.array([1.0, 1.0, 1.0]))

        # Specific for raman calls, it expects just the tensor for a single
        # frequency and need only the real part

        # For static raman, imaginary part is zero??
        # Answer from Michael Walter: Yes, in the case of finite systems you may
        # choose the wavefunctions to be real valued. Then also the density
        # response function and hence the polarizability are real.

        # Convert from atomic units to e**2 Ang**2/eV
        return pmat[:, :, 0].real * (un.Bohr**2) / un.Ha


def pol2cross_sec(p, omg):
    """
    Convert the polarizability in au to cross section in nm**2

    Input parameters:
    -----------------
    p (np array): polarizability from mbpt_lcao calc
    omg (np.array): frequency range in eV

    Output parameters:
    ------------------
    sigma (np array): cross section in nm**2
    """

    c = 1 / un.alpha                      # speed of the light in au
    omg /= un.Ha                          # to convert from eV to Hartree
    sigma = 4 * np.pi * omg * p / (c)     # bohr**2
    return sigma * (0.1 * un.Bohr)**2     # nm**2
