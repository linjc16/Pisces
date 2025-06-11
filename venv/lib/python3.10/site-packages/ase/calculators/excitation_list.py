# fmt: off

import numpy as np

from ase.units import Bohr, Hartree


class Excitation:
    """Base class for a single excitation"""

    def __init__(self, energy, index, mur, muv=None, magn=None):
        """
        Parameters
        ----------
        energy: float
          Energy realtive to the ground state
        index: int
          Excited state index
        mur: list of three floats or complex numbers
          Length form dipole matrix element
        muv: list of three floats or complex numbers or None
          Velocity form dipole matrix element, default None
        magn: list of three floats or complex numbers or None
          Magnetic matrix element, default None
        """
        self.energy = energy
        self.index = index
        self.mur = mur
        self.muv = muv
        self.magn = magn
        self.fij = 1.

    def outstring(self):
        """Format yourself as a string"""
        string = f'{self.energy:g}  {self.index}  '

        def format_me(me):
            string = ''
            if me.dtype == float:
                for m in me:
                    string += f' {m:g}'
            else:
                for m in me:
                    string += ' {0.real:g}{0.imag:+g}j'.format(m)
            return string

        string += '  ' + format_me(self.mur)
        if self.muv is not None:
            string += '  ' + format_me(self.muv)
        if self.magn is not None:
            string += '  ' + format_me(self.magn)
        string += '\n'

        return string

    @classmethod
    def fromstring(cls, string):
        """Initialize yourself from a string"""
        l = string.split()
        energy = float(l.pop(0))
        index = int(l.pop(0))
        mur = np.array([float(l.pop(0)) for _ in range(3)])
        try:
            muv = np.array([float(l.pop(0)) for _ in range(3)])
        except IndexError:
            muv = None
        try:
            magn = np.array([float(l.pop(0)) for _ in range(3)])
        except IndexError:
            magn = None

        return cls(energy, index, mur, muv, magn)

    def get_dipole_me(self, form='r'):
        """Return the excitations dipole matrix element
        including the occupation factor sqrt(fij)"""
        if form == 'r':  # length form
            me = - self.mur
        elif form == 'v':  # velocity form
            me = - self.muv
        else:
            raise RuntimeError('Unknown form >' + form + '<')
        return np.sqrt(self.fij) * me

    def get_dipole_tensor(self, form='r'):
        """Return the oscillator strength tensor"""
        me = self.get_dipole_me(form)
        return 2 * np.outer(me, me.conj()) * self.energy / Hartree

    def get_oscillator_strength(self, form='r'):
        """Return the excitations dipole oscillator strength."""
        me2_c = self.get_dipole_tensor(form).diagonal().real
        return np.array([np.sum(me2_c) / 3.] + me2_c.tolist())


class ExcitationList(list):
    """Base class for excitions from the ground state"""

    def __init__(self):
        # initialise empty list
        super().__init__()

        # set default energy scale to get eV
        self.energy_to_eV_scale = 1.


def polarizability(exlist, omega, form='v',
                   tensor=False, index=0):
    """Evaluate the photon energy dependent polarizability
    from the sum over states

    Parameters
    ----------
    exlist: ExcitationList
    omega:
        Photon energy (eV)
    form: {'v', 'r'}
        Form of the dipole matrix element, default 'v'
    index: {0, 1, 2, 3}
        0: averaged, 1,2,3:alpha_xx, alpha_yy, alpha_zz, default 0
    tensor: boolean
        if True returns alpha_ij, i,j=x,y,z
        index is ignored, default False

    Returns
    -------
    alpha:
        Unit (e^2 Angstrom^2 / eV).
        Multiply with Bohr * Ha to get (Angstrom^3)
        shape = (omega.shape,) if tensor == False
        shape = (omega.shape, 3, 3) else
    """
    omega = np.asarray(omega)
    om2 = 1. * omega**2
    esc = exlist.energy_to_eV_scale

    if tensor:
        if not np.isscalar(om2):
            om2 = om2[:, None, None]
        alpha = np.zeros(omega.shape + (3, 3),
                         dtype=om2.dtype)
        for ex in exlist:
            alpha += ex.get_dipole_tensor(form=form) / (
                (ex.energy * esc)**2 - om2)
    else:
        alpha = np.zeros_like(om2)
        for ex in exlist:
            alpha += ex.get_oscillator_strength(form=form)[index] / (
                (ex.energy * esc)**2 - om2)

    return alpha * Bohr**2 * Hartree
