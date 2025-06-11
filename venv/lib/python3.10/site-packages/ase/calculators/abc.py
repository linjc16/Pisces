# fmt: off

"""
This module defines abstract helper classes with the objective of reducing
boilerplace method definitions (i.e. duplication) in calculators.
"""

from abc import ABC, abstractmethod
from typing import Any, Mapping


class GetPropertiesMixin(ABC):
    """Mixin class which provides get_forces(), get_stress() and so on.

    Inheriting class must implement get_property()."""

    @abstractmethod
    def get_property(self, name, atoms=None, allow_calculation=True):
        """Get the named property."""

    def get_potential_energy(self, atoms=None, force_consistent=False):
        if force_consistent:
            name = 'free_energy'
        else:
            name = 'energy'
        return self.get_property(name, atoms)

    def get_potential_energies(self, atoms=None):
        return self.get_property('energies', atoms)

    def get_forces(self, atoms=None):
        return self.get_property('forces', atoms)

    def get_stress(self, atoms=None):
        return self.get_property('stress', atoms)

    def get_stresses(self, atoms=None):
        """the calculator should return intensive stresses, i.e., such that
                stresses.sum(axis=0) == stress
        """
        return self.get_property('stresses', atoms)

    def get_dipole_moment(self, atoms=None):
        return self.get_property('dipole', atoms)

    def get_charges(self, atoms=None):
        return self.get_property('charges', atoms)

    def get_magnetic_moment(self, atoms=None):
        return self.get_property('magmom', atoms)

    def get_magnetic_moments(self, atoms=None):
        """Calculate magnetic moments projected onto atoms."""
        return self.get_property('magmoms', atoms)


class GetOutputsMixin(ABC):
    """Mixin class for providing get_fermi_level() and others.

    Effectively this class expresses data in calc.results as
    methods such as get_fermi_level().

    Inheriting class must implement _outputmixin_get_results(),
    typically returning self.results, which must be a mapping
    using the naming defined in ase.outputs.Properties.
    """
    @abstractmethod
    def _outputmixin_get_results(self) -> Mapping[str, Any]:
        """Return Mapping of names to result value.

        This may be called many times and should hence not be
        expensive (except possibly the first time)."""

    def _get(self, name):
        # Cyclic import, should restructure.
        from ase.calculators.calculator import PropertyNotPresent
        dct = self._outputmixin_get_results()
        try:
            return dct[name]
        except KeyError:
            raise PropertyNotPresent(name)

    def get_fermi_level(self):
        return self._get('fermi_level')

    def get_ibz_k_points(self):
        return self._get('ibz_kpoints')

    def get_k_point_weights(self):
        return self._get('kpoint_weights')

    def get_eigenvalues(self, kpt=0, spin=0):
        eigs = self._get('eigenvalues')
        return eigs[spin, kpt]

    def _eigshape(self):
        # We don't need this if we already have a Properties object.
        return self._get('eigenvalues').shape

    def get_occupation_numbers(self, kpt=0, spin=0):
        occs = self._get('occupations')
        return occs[spin, kpt]

    def get_number_of_bands(self):
        return self._eigshape()[2]

    def get_number_of_spins(self):
        nspins = self._eigshape()[0]
        assert nspins in [1, 2]
        return nspins

    def get_spin_polarized(self):
        return self.get_number_of_spins() == 2
