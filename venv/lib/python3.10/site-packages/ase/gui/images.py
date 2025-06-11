# fmt: off

import warnings
from math import sqrt

import numpy as np

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from ase.data import covalent_radii
from ase.geometry import find_mic
from ase.gui.defaults import read_defaults
from ase.gui.i18n import _
from ase.io import read, string2index, write


class Images:
    def __init__(self, images=None):
        self.covalent_radii = covalent_radii.copy()
        self.config = read_defaults()
        self.atom_scale = self.config['radii_scale']
        if images is None:
            images = [Atoms()]
        self.initialize(images)

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index):
        return self._images[index]

    def __iter__(self):
        return iter(self._images)

    # XXXXXXX hack
    # compatibility hacks while allowing variable number of atoms
    def get_dynamic(self, atoms: Atoms) -> np.ndarray:
        dynamic = np.ones(len(atoms), bool)
        for constraint in atoms.constraints:
            if isinstance(constraint, FixAtoms):
                dynamic[constraint.index] = False
        return dynamic

    def set_dynamic(self, mask, value):
        # Does not make much sense if different images have different
        # atom counts.  Attempts to apply mask to all images,
        # to the extent possible.
        for atoms in self:
            dynamic = self.get_dynamic(atoms)
            dynamic[mask[:len(atoms)]] = value
            atoms.constraints = [c for c in atoms.constraints
                                 if not isinstance(c, FixAtoms)]
            atoms.constraints.append(FixAtoms(mask=~dynamic))

    def scale_radii(self, scaling_factor):
        self.covalent_radii *= scaling_factor

    def get_energy(self, atoms: Atoms) -> np.float64:
        try:
            return atoms.get_potential_energy()
        except RuntimeError:
            return np.nan  # type: ignore[return-value]

    def get_forces(self, atoms: Atoms):
        try:
            return atoms.get_forces(apply_constraint=False)
        except RuntimeError:
            return None

    def initialize(self, images, filenames=None):
        nimages = len(images)
        if filenames is None:
            filenames = [None] * nimages
        self.filenames = filenames

        warning = False

        self._images = []

        # Whether length or chemical composition changes:
        self.have_varying_species = False
        for i, atoms in enumerate(images):
            # copy atoms or not?  Not copying allows back-editing,
            # but copying actually forgets things like the attached
            # calculator (might have forces/energies
            self._images.append(atoms)
            self.have_varying_species |= not np.array_equal(self[0].numbers,
                                                            atoms.numbers)
            if hasattr(self, 'Q'):
                assert False  # XXX askhl fix quaternions
                self.Q[i] = atoms.get_quaternions()
            if (atoms.pbc != self[0].pbc).any():
                warning = True

        if warning:
            import warnings
            warnings.warn('Not all images have the same boundary conditions!')

        self.maxnatoms = max(len(atoms) for atoms in self)
        self.selected = np.zeros(self.maxnatoms, bool)
        self.selected_ordered = []
        self.visible = np.ones(self.maxnatoms, bool)
        self.repeat = np.ones(3, int)

    def get_radii(self, atoms: Atoms) -> np.ndarray:
        radii = np.array([self.covalent_radii[z] for z in atoms.numbers])
        radii *= self.atom_scale
        return radii

    def read(self, filenames, default_index=':', filetype=None):
        if isinstance(default_index, str):
            default_index = string2index(default_index)

        images = []
        names = []
        for filename in filenames:
            from ase.io.formats import parse_filename

            if '@' in filename and 'postgres' not in filename or \
               'postgres' in filename and filename.count('@') == 2:
                actual_filename, index = parse_filename(filename, None)
            else:
                actual_filename, index = parse_filename(filename,
                                                        default_index)

            # Read from stdin:
            if filename == '-':
                import sys
                from io import BytesIO
                buf = BytesIO(sys.stdin.buffer.read())
                buf.seek(0)
                filename = buf
                filetype = 'traj'

            imgs = read(filename, index, filetype)
            if hasattr(imgs, 'iterimages'):
                imgs = list(imgs.iterimages())

            images.extend(imgs)

            # Name each file as filename@index:
            if isinstance(index, slice):
                start = index.start or 0
                step = index.step or 1
            else:
                start = index
                step = 1
            for i, img in enumerate(imgs):
                if isinstance(start, int):
                    names.append('{}@{}'.format(
                        actual_filename, start + i * step))
                else:
                    names.append(f'{actual_filename}@{start}')

        self.initialize(images, names)

    def repeat_results(self, atoms: Atoms, repeat=None, oldprod=None):
        """Return a dictionary which updates the magmoms, energy and forces
        to the repeated amount of atoms.
        """
        def getresult(name, get_quantity):
            # ase/io/trajectory.py line 170 does this by using
            # the get_property(prop, atoms, allow_calculation=False)
            # so that is an alternative option.
            try:
                if (not atoms.calc or
                        atoms.calc.calculation_required(atoms, [name])):
                    quantity = None
                else:
                    quantity = get_quantity()
            except Exception as err:
                quantity = None
                errmsg = ('An error occurred while retrieving {} '
                          'from the calculator: {}'.format(name, err))
                warnings.warn(errmsg)
            return quantity

        if repeat is None:
            repeat = self.repeat.prod()
        if oldprod is None:
            oldprod = self.repeat.prod()

        results = {}

        original_length = len(atoms) // oldprod
        newprod = repeat.prod()

        # Read the old properties
        magmoms = getresult('magmoms', atoms.get_magnetic_moments)
        magmom = getresult('magmom', atoms.get_magnetic_moment)
        energy = getresult('energy', atoms.get_potential_energy)
        forces = getresult('forces', atoms.get_forces)

        # Update old properties to the repeated image
        if magmoms is not None:
            magmoms = np.tile(magmoms[:original_length], newprod)
            results['magmoms'] = magmoms

        if magmom is not None:
            magmom = magmom * newprod / oldprod
            results['magmom'] = magmom

        if forces is not None:
            forces = np.tile(forces[:original_length].T, newprod).T
            results['forces'] = forces

        if energy is not None:
            energy = energy * newprod / oldprod
            results['energy'] = energy

        return results

    def repeat_unit_cell(self):
        for atoms in self:
            # Get quantities taking into account current repeat():'
            results = self.repeat_results(atoms, self.repeat.prod(),
                                          oldprod=self.repeat.prod())

            atoms.cell *= self.repeat.reshape((3, 1))
            atoms.calc = SinglePointCalculator(atoms, **results)
        self.repeat = np.ones(3, int)

    def repeat_images(self, repeat):
        from ase.constraints import FixAtoms
        repeat = np.array(repeat)
        oldprod = self.repeat.prod()
        images = []
        constraints_removed = False

        for i, atoms in enumerate(self):
            refcell = atoms.get_cell()
            fa = []
            for c in atoms._constraints:
                if isinstance(c, FixAtoms):
                    fa.append(c)
                else:
                    constraints_removed = True
            atoms.set_constraint(fa)

            # Update results dictionary to repeated atoms
            results = self.repeat_results(atoms, repeat, oldprod)

            del atoms[len(atoms) // oldprod:]  # Original atoms

            atoms *= repeat
            atoms.cell = refcell

            atoms.calc = SinglePointCalculator(atoms, **results)

            images.append(atoms)

        if constraints_removed:
            from ase.gui.ui import showwarning, tk

            # We must be able to show warning before the main GUI
            # has been created.  So we create a new window,
            # then show the warning, then destroy the window.
            tmpwindow = tk.Tk()
            tmpwindow.withdraw()  # Host window will never be shown
            showwarning(_('Constraints discarded'),
                        _('Constraints other than FixAtoms '
                          'have been discarded.'))
            tmpwindow.destroy()

        self.initialize(images, filenames=self.filenames)
        self.repeat = repeat

    def center(self):
        """Center each image in the existing unit cell, keeping the
        cell constant."""
        for atoms in self:
            atoms.center()

    def graph(self, expr: str) -> np.ndarray:
        """Routine to create the data in graphs, defined by the
        string expr."""
        import ase.units as units
        code = compile(expr + ',', '<input>', 'eval')

        nimages = len(self)

        def d(n1, n2):
            return sqrt(((R[n1] - R[n2])**2).sum())

        def a(n1, n2, n3):
            v1 = R[n1] - R[n2]
            v2 = R[n3] - R[n2]
            arg = np.vdot(v1, v2) / (sqrt((v1**2).sum() * (v2**2).sum()))
            if arg > 1.0:
                arg = 1.0
            if arg < -1.0:
                arg = -1.0
            return 180.0 * np.arccos(arg) / np.pi

        def dih(n1, n2, n3, n4):
            # vector 0->1, 1->2, 2->3 and their normalized cross products:
            a = R[n2] - R[n1]
            b = R[n3] - R[n2]
            c = R[n4] - R[n3]
            bxa = np.cross(b, a)
            bxa /= np.sqrt(np.vdot(bxa, bxa))
            cxb = np.cross(c, b)
            cxb /= np.sqrt(np.vdot(cxb, cxb))
            angle = np.vdot(bxa, cxb)
            # check for numerical trouble due to finite precision:
            if angle < -1:
                angle = -1
            if angle > 1:
                angle = 1
            angle = np.arccos(angle)
            if np.vdot(bxa, c) > 0:
                angle = 2 * np.pi - angle
            return angle * 180.0 / np.pi

        # get number of mobile atoms for temperature calculation
        E = np.array([self.get_energy(atoms) for atoms in self])

        s = 0.0

        # Namespace for eval:
        ns = {'E': E,
              'd': d, 'a': a, 'dih': dih}

        data = []
        for i in range(nimages):
            ns['i'] = i
            ns['s'] = s
            ns['R'] = R = self[i].get_positions()
            ns['V'] = self[i].get_velocities()
            F = self.get_forces(self[i])
            if F is not None:
                ns['F'] = F
            ns['A'] = self[i].get_cell()
            ns['M'] = self[i].get_masses()
            # XXX askhl verify:
            dynamic = self.get_dynamic(self[i])
            if F is not None:
                ns['f'] = f = ((F * dynamic[:, None])**2).sum(1)**.5
                ns['fmax'] = max(f)
                ns['fave'] = f.mean()
            ns['epot'] = epot = E[i]
            ns['ekin'] = ekin = self[i].get_kinetic_energy()
            ns['e'] = epot + ekin
            ndynamic = dynamic.sum()
            if ndynamic > 0:
                ns['T'] = 2.0 * ekin / (3.0 * ndynamic * units.kB)
            data = eval(code, ns)
            if i == 0:
                nvariables = len(data)
                xy = np.empty((nvariables, nimages))
            xy[:, i] = data
            if i + 1 < nimages and not self.have_varying_species:
                dR = find_mic(self[i + 1].positions - R, self[i].get_cell(),
                              self[i].get_pbc())[0]
                s += sqrt((dR**2).sum())
        return xy

    def write(self, filename, rotations='', bbox=None,
              **kwargs):
        # XXX We should show the unit cell whenever there is one
        indices = range(len(self))
        p = filename.rfind('@')
        if p != -1:
            try:
                slice = string2index(filename[p + 1:])
            except ValueError:
                pass
            else:
                indices = indices[slice]
                filename = filename[:p]
                if isinstance(indices, int):
                    indices = [indices]

        images = [self.get_atoms(i) for i in indices]
        if len(filename) > 4 and filename[-4:] in ['.eps', '.png', '.pov']:
            write(filename, images,
                  rotation=rotations,
                  bbox=bbox, **kwargs)
        else:
            write(filename, images, **kwargs)

    def get_atoms(self, frame, remove_hidden=False):
        atoms = self[frame]
        try:
            E = atoms.get_potential_energy()
        except RuntimeError:
            E = None
        try:
            F = atoms.get_forces()
        except RuntimeError:
            F = None

        # Remove hidden atoms if applicable
        if remove_hidden:
            atoms = atoms[self.visible]
            if F is not None:
                F = F[self.visible]
        atoms.calc = SinglePointCalculator(atoms, energy=E, forces=F)
        return atoms

    def delete(self, i):
        self._images.pop(i)
        self.filenames.pop(i)
        self.initialize(self._images, self.filenames)
