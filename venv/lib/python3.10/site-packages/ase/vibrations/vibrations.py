# fmt: off

"""A class for computing vibrational modes"""

import sys
from collections import namedtuple
from math import log, pi, sqrt
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np

import ase.io
import ase.units as units
from ase.atoms import Atoms
from ase.constraints import FixAtoms
from ase.parallel import paropen, world
from ase.utils.filecache import get_json_cache

from .data import VibrationsData


class AtomicDisplacements:
    def _disp(self, a, i, step):
        if isinstance(i, str):  # XXX Simplify by removing this.
            i = 'xyz'.index(i)
        return Displacement(a, i, np.sign(step), abs(step), self)

    def _eq_disp(self):
        return self._disp(0, 0, 0)

    @property
    def ndof(self):
        return 3 * len(self.indices)


class Displacement(namedtuple('Displacement', ['a', 'i', 'sign', 'ndisp',
                                               'vib'])):
    @property
    def name(self):
        if self.sign == 0:
            return 'eq'

        axisname = 'xyz'[self.i]
        dispname = self.ndisp * ' +-'[self.sign]
        return f'{self.a}{axisname}{dispname}'

    @property
    def _cached(self):
        return self.vib.cache[self.name]

    def forces(self):
        return self._cached['forces'].copy()

    @property
    def step(self):
        return self.ndisp * self.sign * self.vib.delta

    # XXX dipole only valid for infrared
    def dipole(self):
        return self._cached['dipole'].copy()

    # XXX below stuff only valid for TDDFT excitation stuff
    def save_ov_nn(self, ov_nn):
        np.save(Path(self.vib.exname) / (self.name + '.ov'), ov_nn)

    def load_ov_nn(self):
        return np.load(Path(self.vib.exname) / (self.name + '.ov.npy'))

    @property
    def _exname(self):
        return Path(self.vib.exname) / f'ex.{self.name}{self.vib.exext}'

    def calculate_and_save_static_polarizability(self, atoms):
        exobj = self.vib._new_exobj()
        excitation_data = exobj(atoms)
        np.savetxt(self._exname, excitation_data)

    def load_static_polarizability(self):
        return np.loadtxt(self._exname)

    def read_exobj(self):
        # XXX each exobj should allow for self._exname as Path
        return self.vib.read_exobj(str(self._exname))

    def calculate_and_save_exlist(self, atoms):
        # exo = self.vib._new_exobj()
        excalc = self.vib._new_exobj()
        exlist = excalc.calculate(atoms)
        # XXX each exobj should allow for self._exname as Path
        exlist.write(str(self._exname))


class Vibrations(AtomicDisplacements):
    """Class for calculating vibrational modes using finite difference.

    The vibrational modes are calculated from a finite difference
    approximation of the Hessian matrix.

    The *summary()*, *get_energies()* and *get_frequencies()* methods all take
    an optional *method* keyword.  Use method='Frederiksen' to use the method
    described in:

      T. Frederiksen, M. Paulsson, M. Brandbyge, A. P. Jauho:
      "Inelastic transport theory from first-principles: methodology and
      applications for nanoscale devices", Phys. Rev. B 75, 205413 (2007)

    atoms: Atoms object
        The atoms to work on.
    indices: list of int
        List of indices of atoms to vibrate.  Default behavior is
        to vibrate all atoms.
    name: str
        Name to use for files.
    delta: float
        Magnitude of displacements.
    nfree: int
        Number of displacements per atom and cartesian coordinate, 2 and 4 are
        supported. Default is 2 which will displace each atom +delta and
        -delta for each cartesian coordinate.

    Example:

    >>> from ase import Atoms
    >>> from ase.calculators.emt import EMT
    >>> from ase.optimize import BFGS
    >>> from ase.vibrations import Vibrations
    >>> n2 = Atoms('N2', [(0, 0, 0), (0, 0, 1.1)],
    ...            calculator=EMT())
    >>> BFGS(n2).run(fmax=0.01)
    BFGS:   0  16:01:21        0.440339       3.2518
    BFGS:   1  16:01:21        0.271928       0.8211
    BFGS:   2  16:01:21        0.263278       0.1994
    BFGS:   3  16:01:21        0.262777       0.0088
    >>> vib = Vibrations(n2)
    >>> vib.run()
    >>> vib.summary()
    ---------------------
    #    meV     cm^-1
    ---------------------
    0    0.0       0.0
    1    0.0       0.0
    2    0.0       0.0
    3    1.4      11.5
    4    1.4      11.5
    5  152.7    1231.3
    ---------------------
    Zero-point energy: 0.078 eV
    >>> vib.write_mode(-1)  # write last mode to trajectory file

    """

    def __init__(self, atoms, indices=None, name='vib', delta=0.01, nfree=2):
        assert nfree in [2, 4]
        self.atoms = atoms
        self.calc = atoms.calc
        if indices is None:
            fixed_indices = []
            for constr in atoms.constraints:
                if isinstance(constr, FixAtoms):
                    fixed_indices.extend(constr.get_indices())
            fixed_indices = list(set(fixed_indices))
            indices = [i for i in range(len(atoms)) if i not in fixed_indices]
        if len(indices) != len(set(indices)):
            raise ValueError(
                'one (or more) indices included more than once')
        self.indices = np.asarray(indices)

        self.delta = delta
        self.nfree = nfree
        self.H = None
        self.ir = None
        self._vibrations = None

        self.cache = get_json_cache(name)

    @property
    def name(self):
        return str(self.cache.directory)

    def run(self):
        """Run the vibration calculations.

        This will calculate the forces for 6 displacements per atom +/-x,
        +/-y, +/-z. Only those calculations that are not already done will be
        started. Be aware that an interrupted calculation may produce an empty
        file (ending with .json), which must be deleted before restarting the
        job. Otherwise the forces will not be calculated for that
        displacement.

        Note that the calculations for the different displacements can be done
        simultaneously by several independent processes. This feature relies
        on the existence of files and the subsequent creation of the file in
        case it is not found.

        If the program you want to use does not have a calculator in ASE, use
        ``iterdisplace`` to get all displaced structures and calculate the
        forces on your own.
        """

        if not self.cache.writable:
            raise RuntimeError(
                'Cannot run calculation.  '
                'Cache must be removed or split in order '
                'to have only one sort of data structure at a time.')

        self._check_old_pickles()

        for disp, atoms in self.iterdisplace(inplace=True):
            with self.cache.lock(disp.name) as handle:
                if handle is None:
                    continue

                result = self.calculate(atoms, disp)

                if world.rank == 0:
                    handle.save(result)

    def _check_old_pickles(self):
        from pathlib import Path
        eq_pickle_path = Path(f'{self.name}.eq.pckl')
        pickle2json_instructions = f"""\
Found old pickle files such as {eq_pickle_path}.  \
Please remove them and recalculate or run \
"python -m ase.vibrations.pickle2json --help"."""
        if len(self.cache) == 0 and eq_pickle_path.exists():
            raise RuntimeError(pickle2json_instructions)

    def iterdisplace(self, inplace=False) -> \
            Iterator[Tuple[Displacement, Atoms]]:
        """Iterate over initial and displaced structures.

        Use this to export the structures for each single-point calculation
        to an external program instead of using ``run()``. Then save the
        calculated gradients to <name>.json and continue using this instance.

        Parameters:
        ------------
        inplace: bool
            If True, the atoms object will be modified in-place. Otherwise a
            copy will be made.

        Yields:
        --------
        disp: Displacement
            Displacement object with information about the displacement.
        atoms: Atoms
            Atoms object with the displaced structure.
        """
        # XXX change of type of disp
        atoms = self.atoms if inplace else self.atoms.copy()
        displacements = self.displacements()
        eq_disp = next(displacements)
        assert eq_disp.name == 'eq'
        yield eq_disp, atoms

        for disp in displacements:
            if not inplace:
                atoms = self.atoms.copy()
            pos0 = atoms.positions[disp.a, disp.i]
            atoms.positions[disp.a, disp.i] += disp.step
            yield disp, atoms

            if inplace:
                atoms.positions[disp.a, disp.i] = pos0

    def iterimages(self):
        """Yield initial and displaced structures."""
        for name, atoms in self.iterdisplace():
            yield atoms

    def _iter_ai(self):
        for a in self.indices:
            for i in range(3):
                yield a, i

    def displacements(self):
        yield self._eq_disp()

        for a, i in self._iter_ai():
            for sign in [-1, 1]:
                for ndisp in range(1, self.nfree // 2 + 1):
                    yield self._disp(a, i, sign * ndisp)

    def calculate(self, atoms, disp):
        results = {}
        results['forces'] = self.calc.get_forces(atoms)

        if self.ir:
            results['dipole'] = self.calc.get_dipole_moment(atoms)

        return results

    def clean(self, empty_files=False, combined=True):
        """Remove json-files.

        Use empty_files=True to remove only empty files and
        combined=False to not remove the combined file.

        """

        if world.rank != 0:
            return 0

        if empty_files:
            return self.cache.strip_empties()  # XXX Fails on combined cache

        nfiles = self.cache.filecount()
        self.cache.clear()
        return nfiles

    def combine(self):
        """Combine json-files to one file ending with '.all.json'.

        The other json-files will be removed in order to have only one sort
        of data structure at a time.

        """
        nelements_before = self.cache.filecount()
        self.cache = self.cache.combine()
        return nelements_before

    def split(self):
        """Split combined json-file.

        The combined json-file will be removed in order to have only one
        sort of data structure at a time.

        """
        count = self.cache.filecount()
        self.cache = self.cache.split()
        return count

    def read(self, method='standard', direction='central'):
        self.method = method.lower()
        self.direction = direction.lower()
        assert self.method in ['standard', 'frederiksen']
        assert self.direction in ['central', 'forward', 'backward']

        n = 3 * len(self.indices)
        H = np.empty((n, n))
        r = 0

        eq_disp = self._eq_disp()

        if direction != 'central':
            feq = eq_disp.forces()

        for a, i in self._iter_ai():
            disp_minus = self._disp(a, i, -1)
            disp_plus = self._disp(a, i, 1)

            fminus = disp_minus.forces()
            fplus = disp_plus.forces()
            if self.method == 'frederiksen':
                fminus[a] -= fminus.sum(0)
                fplus[a] -= fplus.sum(0)
            if self.nfree == 4:
                fminusminus = self._disp(a, i, -2).forces()
                fplusplus = self._disp(a, i, 2).forces()
                if self.method == 'frederiksen':
                    fminusminus[a] -= fminusminus.sum(0)
                    fplusplus[a] -= fplusplus.sum(0)
            if self.direction == 'central':
                if self.nfree == 2:
                    H[r] = .5 * (fminus - fplus)[self.indices].ravel()
                else:
                    assert self.nfree == 4
                    H[r] = H[r] = (-fminusminus +
                                   8 * fminus -
                                   8 * fplus +
                                   fplusplus)[self.indices].ravel() / 12.0
            elif self.direction == 'forward':
                H[r] = (feq - fplus)[self.indices].ravel()
            else:
                assert self.direction == 'backward'
                H[r] = (fminus - feq)[self.indices].ravel()
            H[r] /= 2 * self.delta
            r += 1
        H += H.copy().T
        self.H = H
        masses = self.atoms.get_masses()
        if any(masses[self.indices] == 0):
            raise RuntimeError('Zero mass encountered in one or more of '
                               'the vibrated atoms. Use Atoms.set_masses()'
                               ' to set all masses to non-zero values.')

        self.im = np.repeat(masses[self.indices]**-0.5, 3)
        self._vibrations = self.get_vibrations(read_cache=False,
                                               method=self.method,
                                               direction=self.direction)

        omega2, modes = np.linalg.eigh(self.im[:, None] * H * self.im)
        self.modes = modes.T.copy()

        # Conversion factor:
        s = units._hbar * 1e10 / sqrt(units._e * units._amu)
        self.hnu = s * omega2.astype(complex)**0.5

    def get_vibrations(self, method='standard', direction='central',
                       read_cache=True, **kw):
        """Get vibrations as VibrationsData object

        If read() has not yet been called, this will be called to assemble data
        from the outputs of run(). Most of the arguments to this function are
        options to be passed to read() in this case.

        Args:
            method (str): Calculation method passed to read()
            direction (str): Finite-difference scheme passed to read()
            read_cache (bool): The VibrationsData object will be cached for
                quick access. Set False to force regeneration of the cache with
                the current atoms/Hessian/indices data.
            **kw: Any remaining keyword arguments are passed to read()

        Returns:
            VibrationsData

        """
        if read_cache and (self._vibrations is not None):
            return self._vibrations

        else:
            if (self.H is None or method.lower() != self.method or
                    direction.lower() != self.direction):
                self.read(method, direction, **kw)

            return VibrationsData.from_2d(self.atoms, self.H,
                                          indices=self.indices)

    def get_energies(self, method='standard', direction='central', **kw):
        """Get vibration energies in eV."""
        return self.get_vibrations(method=method,
                                   direction=direction, **kw).get_energies()

    def get_frequencies(self, method='standard', direction='central'):
        """Get vibration frequencies in cm^-1."""
        return self.get_vibrations(method=method,
                                   direction=direction).get_frequencies()

    def summary(self, method='standard', direction='central', freq=None,
                log=sys.stdout):
        if freq is not None:
            energies = freq * units.invcm
        else:
            energies = self.get_energies(method=method, direction=direction)

        summary_lines = VibrationsData._tabulate_from_energies(energies)
        log_text = '\n'.join(summary_lines) + '\n'

        if isinstance(log, str):
            with paropen(log, 'a') as log_file:
                log_file.write(log_text)
        else:
            log.write(log_text)

    def get_zero_point_energy(self, freq=None):
        if freq:
            raise NotImplementedError
        return self.get_vibrations().get_zero_point_energy()

    def get_mode(self, n):
        """Get mode number ."""
        return self.get_vibrations().get_modes(all_atoms=True)[n]

    def write_mode(self, n=None, kT=units.kB * 300, nimages=30):
        """Write mode number n to trajectory file. If n is not specified,
        writes all non-zero modes."""
        if n is None:
            for index, energy in enumerate(self.get_energies()):
                if abs(energy) > 1e-5:
                    self.write_mode(n=index, kT=kT, nimages=nimages)
            return

        else:
            n %= len(self.get_energies())

        with ase.io.Trajectory('%s.%d.traj' % (self.name, n), 'w') as traj:
            for image in (self.get_vibrations()
                          .iter_animated_mode(n,
                                              temperature=kT, frames=nimages)):
                traj.write(image)

    def show_as_force(self, n, scale=0.2, show=True):
        return self.get_vibrations().show_as_force(n, scale=scale, show=show)

    def write_jmol(self):
        """Writes file for viewing of the modes with jmol."""

        with open(self.name + '.xyz', 'w') as fd:
            self._write_jmol(fd)

    def _write_jmol(self, fd):
        symbols = self.atoms.get_chemical_symbols()
        freq = self.get_frequencies()
        for n in range(3 * len(self.indices)):
            fd.write('%6d\n' % len(self.atoms))

            if freq[n].imag != 0:
                c = 'i'
                freq[n] = freq[n].imag

            else:
                freq[n] = freq[n].real
                c = ' '

            fd.write('Mode #%d, f = %.1f%s cm^-1'
                     % (n, float(freq[n].real), c))

            if self.ir:
                fd.write(', I = %.4f (D/Å)^2 amu^-1.\n' % self.intensities[n])
            else:
                fd.write('.\n')

            mode = self.get_mode(n)
            for i, pos in enumerate(self.atoms.positions):
                fd.write('%2s %12.5f %12.5f %12.5f %12.5f %12.5f %12.5f\n' %
                         (symbols[i], pos[0], pos[1], pos[2],
                          mode[i, 0], mode[i, 1], mode[i, 2]))

    def fold(self, frequencies, intensities,
             start=800.0, end=4000.0, npts=None, width=4.0,
             type='Gaussian', normalize=False):
        """Fold frequencies and intensities within the given range
        and folding method (Gaussian/Lorentzian).
        The energy unit is cm^-1.
        normalize=True ensures the integral over the peaks to give the
        intensity.
        """

        lctype = type.lower()
        assert lctype in ['gaussian', 'lorentzian']
        if not npts:
            npts = int((end - start) / width * 10 + 1)
        prefactor = 1
        if lctype == 'lorentzian':
            intensities = intensities * width * pi / 2.
            if normalize:
                prefactor = 2. / width / pi
        else:
            sigma = width / 2. / sqrt(2. * log(2.))
            if normalize:
                prefactor = 1. / sigma / sqrt(2 * pi)

        # Make array with spectrum data
        spectrum = np.empty(npts)
        energies = np.linspace(start, end, npts)
        for i, energy in enumerate(energies):
            energies[i] = energy
            if lctype == 'lorentzian':
                spectrum[i] = (intensities * 0.5 * width / pi /
                               ((frequencies - energy)**2 +
                                0.25 * width**2)).sum()
            else:
                spectrum[i] = (intensities *
                               np.exp(-(frequencies - energy)**2 /
                                      2. / sigma**2)).sum()
        return [energies, prefactor * spectrum]

    def write_dos(self, out='vib-dos.dat', start=800, end=4000,
                  npts=None, width=10,
                  type='Gaussian', method='standard', direction='central'):
        """Write out the vibrational density of states to file.

        First column is the wavenumber in cm^-1, the second column the
        folded vibrational density of states.
        Start and end points, and width of the Gaussian/Lorentzian
        should be given in cm^-1."""
        frequencies = self.get_frequencies(method, direction).real
        intensities = np.ones(len(frequencies))
        energies, spectrum = self.fold(frequencies, intensities,
                                       start, end, npts, width, type)

        # Write out spectrum in file.
        outdata = np.empty([len(energies), 2])
        outdata.T[0] = energies
        outdata.T[1] = spectrum

        with open(out, 'w') as fd:
            fd.write(f'# {type.title()} folded, width={width:g} cm^-1\n')
            fd.write('# [cm^-1] arbitrary\n')
            for row in outdata:
                fd.write('%.3f  %15.5e\n' %
                         (row[0], row[1]))
