# fmt: off

"""
ASE Calculator for interatomic models compatible with the Knowledgebase
of Interatomic Models (KIM) application programming interface (API).
Written by:

Mingjian Wen
Daniel S. Karls
University of Minnesota
"""
import numpy as np

from ase.calculators.calculator import Calculator, compare_atoms

from . import kimpy_wrappers, neighborlist


class KIMModelData:
    """Initializes and subsequently stores the KIM API Portable Model
    object, KIM API ComputeArguments object, and the neighbor list
    object used by instances of KIMModelCalculator.  Also stores the
    arrays which are registered in the KIM API and which are used to
    communicate with the model.
    """

    def __init__(self, model_name, ase_neigh, neigh_skin_ratio, debug=False):
        self.model_name = model_name
        self.ase_neigh = ase_neigh
        self.debug = debug

        # Initialize KIM API Portable Model object and ComputeArguments object
        self.kim_model, self.compute_args = self._init_kim()

        self.species_map = self._create_species_map()

        # Ask model to provide information relevant for neighbor list
        # construction
        (
            model_influence_dist,
            model_cutoffs,
            padding_not_require_neigh,
        ) = self.get_model_neighbor_list_parameters()

        # Initialize neighbor list object
        self.neigh = self._init_neigh(
            neigh_skin_ratio,
            model_influence_dist,
            model_cutoffs,
            padding_not_require_neigh,
        )

    def _init_kim(self):
        """Create the KIM API Portable Model object and KIM API ComputeArguments
        object
        """
        if self.kim_initialized:
            return None

        kim_model = kimpy_wrappers.PortableModel(self.model_name, self.debug)

        # KIM API model object is what actually creates/destroys the
        # ComputeArguments object, so we must pass it as a parameter
        compute_args = kim_model.compute_arguments_create()

        return kim_model, compute_args

    def _init_neigh(
        self,
        neigh_skin_ratio,
        model_influence_dist,
        model_cutoffs,
        padding_not_require_neigh,
    ):
        """Initialize neighbor list, either an ASE-native neighborlist
        or one created using the neighlist module in kimpy
        """
        neigh_list_object_type = (
            neighborlist.ASENeighborList
            if self.ase_neigh
            else neighborlist.KimpyNeighborList
        )
        return neigh_list_object_type(
            self.compute_args,
            neigh_skin_ratio,
            model_influence_dist,
            model_cutoffs,
            padding_not_require_neigh,
            self.debug,
        )

    def get_model_neighbor_list_parameters(self):
        model_influence_dist = self.kim_model.get_influence_distance()
        (
            model_cutoffs,
            padding_not_require_neigh,
        ) = self.kim_model.get_neighbor_list_cutoffs_and_hints()

        return model_influence_dist, model_cutoffs, padding_not_require_neigh

    def update_compute_args_pointers(self, energy, forces):
        self.compute_args.update(
            self.num_particles,
            self.species_code,
            self._particle_contributing,
            self.coords,
            energy,
            forces,
        )

    def _create_species_map(self):
        """Get all the supported species of the KIM model and the
        corresponding integer codes used by the model

        Returns
        -------
        species_map : dict
            key : str
                chemical symbols (e.g. "Ar")
            value : int
                species integer code (e.g. 1)
        """
        supported_species, codes = self._get_model_supported_species_and_codes()
        species_map = {}
        for i, spec in enumerate(supported_species):
            species_map[spec] = codes[i]
            if self.debug:
                print(
                    "Species {} is supported and its code is: {}".format(
                        spec, codes[i])
                )

        return species_map

    @property
    def padding_image_of(self):
        return self.neigh.padding_image_of

    @property
    def num_particles(self):
        return self.neigh.num_particles

    @property
    def coords(self):
        return self.neigh.coords

    @property
    def _particle_contributing(self):
        return self.neigh.particle_contributing

    @property
    def species_code(self):
        return self.neigh.species_code

    @property
    def kim_initialized(self):
        return hasattr(self, "kim_model")

    @property
    def _neigh_initialized(self):
        return hasattr(self, "neigh")

    @property
    def _get_model_supported_species_and_codes(self):
        return self.kim_model.get_model_supported_species_and_codes


class KIMModelCalculator(Calculator):
    """Calculator that works with KIM Portable Models (PMs).

    Calculator that carries out direct communication between ASE and a
    KIM Portable Model (PM) through the kimpy library (which provides a
    set of python bindings to the KIM API).

    Parameters
    ----------
    model_name : str
      The unique identifier assigned to the interatomic model (for
      details, see https://openkim.org/doc/schema/kim-ids)

    ase_neigh : bool, optional
      False (default): Use kimpy's neighbor list library

      True: Use ASE's internal neighbor list mechanism (usually slower
      than the kimpy neighlist library)

    neigh_skin_ratio : float, optional
      Used to determine the neighbor list cutoff distance, r_neigh,
      through the relation r_neigh = (1 + neigh_skin_ratio) * rcut,
      where rcut is the model's influence distance. (Default: 0.2)

    release_GIL : bool, optional
      Whether to release python GIL.  Releasing the GIL allows a KIM
      model to run with multiple concurrent threads. (Default: False)

    debug : bool, optional
      If True, detailed information is printed to stdout. (Default:
      False)
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    ignored_changes = {"initial_charges", "initial_magmoms"}

    def __init__(
        self,
        model_name,
        ase_neigh=False,
        neigh_skin_ratio=0.2,
        release_GIL=False,
        debug=False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.model_name = model_name
        self.release_GIL = release_GIL
        self.debug = debug

        if neigh_skin_ratio < 0:
            raise ValueError('Argument "neigh_skin_ratio" must be non-negative')
        self.neigh_skin_ratio = neigh_skin_ratio

        # Model output
        self.energy = None
        self.forces = None

        # Create KIMModelData object. This will take care of creating
        # and storing the KIM API Portable Model object, KIM API
        # ComputeArguments object, and the neighbor list object that
        # our calculator needs
        self._kimmodeldata = KIMModelData(
            self.model_name, ase_neigh, self.neigh_skin_ratio, self.debug
        )

        self._parameters_changed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, value, traceback):
        pass

    def __repr__(self):
        return f"KIMModelCalculator(model_name={self.model_name})"

    def calculate(
        self,
        atoms=None,
        properties=["energy", "forces", "stress"],
        system_changes=["positions", "numbers", "cell", "pbc"],
    ):
        """
        Inherited method from the ase Calculator class that is called by
        get_property()

        Parameters
        ----------
        atoms : Atoms
            Atoms object whose properties are desired

        properties : list of str
            List of what needs to be calculated.  Can be any combination
            of 'energy', 'forces' and 'stress'.

        system_changes : list of str
            List of what has changed since last calculation.  Can be any
            combination of these six: 'positions', 'numbers', 'cell',
            and 'pbc'.
        """

        super().calculate(atoms, properties, system_changes)

        if self._parameters_changed:
            self._parameters_changed = False

        if system_changes:

            # Ask model to update all of its parameters and the parameters
            # related to the neighbor list(s). This update is necessary to do
            # here since the user will generally have made changes the model
            # parameters since the last time an update was performed and we
            # need to ensure that any properties calculated here are made using
            # the up-to-date model and neighbor list parameters.
            self._model_refresh_and_update_neighbor_list_parameters()

            if self._need_neigh_update(atoms, system_changes):
                self._update_neigh(atoms, self._species_map)
                self.energy = np.array([0.0], dtype=kimpy_wrappers.c_double)
                self.forces = np.zeros(
                    [self._num_particles[0], 3], dtype=kimpy_wrappers.c_double
                )
                self._update_compute_args_pointers(self.energy, self.forces)
            else:
                self._update_kim_coords(atoms)

            self._kim_model.compute(self._compute_args, self.release_GIL)

        energy = self.energy[0]
        forces = self._assemble_padding_forces()

        try:
            volume = atoms.get_volume()
            stress = self._compute_virial_stress(
                self.forces, self._coords, volume)
        except ValueError:  # Volume cannot be computed
            stress = None

        # Quantities passed back to ASE
        self.results["energy"] = energy
        self.results["free_energy"] = energy
        self.results["forces"] = forces
        self.results["stress"] = stress

    def check_state(self, atoms, tol=1e-15):
        # Check for change in atomic configuration (positions or pbc)
        system_changes = compare_atoms(
            self.atoms, atoms, excluded_properties=self.ignored_changes
        )

        # Check if model parameters were changed
        if self._parameters_changed:
            system_changes.append("calculator")

        return system_changes

    def _assemble_padding_forces(self):
        """
        Assemble forces on padding atoms back to contributing atoms.

        Parameters
        ----------
        forces : 2D array of doubles
            Forces on both contributing and padding atoms

        num_contrib:  int
            Number of contributing atoms

        padding_image_of : 1D array of int
            Atom number, of which the padding atom is an image


        Returns
        -------
            Total forces on contributing atoms.
        """

        total_forces = np.array(self.forces[:self._num_contributing_particles])

        if self._padding_image_of.size != 0:
            pad_forces = self.forces[self._num_contributing_particles:]
            for f, org_index in zip(pad_forces, self._padding_image_of):
                total_forces[org_index] += f

        return total_forces

    @staticmethod
    def _compute_virial_stress(forces, coords, volume):
        """Compute the virial stress in Voigt notation.

        Parameters
        ----------
        forces : 2D array
            Partial forces on all atoms (padding included)

        coords : 2D array
            Coordinates of all atoms (padding included)

        volume : float
            Volume of cell

        Returns
        -------
        stress : 1D array
            stress in Voigt order (xx, yy, zz, yz, xz, xy)
        """
        stress = np.zeros(6)
        stress[0] = -np.dot(forces[:, 0], coords[:, 0]) / volume
        stress[1] = -np.dot(forces[:, 1], coords[:, 1]) / volume
        stress[2] = -np.dot(forces[:, 2], coords[:, 2]) / volume
        stress[3] = -np.dot(forces[:, 1], coords[:, 2]) / volume
        stress[4] = -np.dot(forces[:, 0], coords[:, 2]) / volume
        stress[5] = -np.dot(forces[:, 0], coords[:, 1]) / volume

        return stress

    @property
    def _update_compute_args_pointers(self):
        return self._kimmodeldata.update_compute_args_pointers

    @property
    def _kim_model(self):
        return self._kimmodeldata.kim_model

    @property
    def _compute_args(self):
        return self._kimmodeldata.compute_args

    @property
    def _num_particles(self):
        return self._kimmodeldata.num_particles

    @property
    def _coords(self):
        return self._kimmodeldata.coords

    @property
    def _padding_image_of(self):
        return self._kimmodeldata.padding_image_of

    @property
    def _species_map(self):
        return self._kimmodeldata.species_map

    @property
    def _neigh(self):
        # WARNING: This property is underscored for a reason! The
        # neighborlist(s) itself (themselves) may not be up to date with
        # respect to changes that have been made to the model's parameters, or
        # even since the positions in the Atoms object may have changed.
        # Neighbor lists are only potentially updated inside the ``calculate``
        # method.
        return self._kimmodeldata.neigh

    @property
    def _num_contributing_particles(self):
        return self._neigh.num_contributing_particles

    @property
    def _update_kim_coords(self):
        return self._neigh.update_kim_coords

    @property
    def _need_neigh_update(self):
        return self._neigh.need_neigh_update

    @property
    def _update_neigh(self):
        return self._neigh.update

    @property
    def parameters_metadata(self):
        return self._kim_model.parameters_metadata

    @property
    def parameter_names(self):
        return self._kim_model.parameter_names

    @property
    def get_parameters(self):
        # Ask model to update all of its parameters and the parameters related
        # to the neighbor list(s). This update is necessary to do here since
        # the user will generally have made changes the model parameters since
        # the last time an update was performed and we need to ensure the
        # parameters returned by this method are fully up to date.
        self._model_refresh_and_update_neighbor_list_parameters()

        return self._kim_model.get_parameters

    def set_parameters(self, **kwargs):
        parameters = self._kim_model.set_parameters(**kwargs)
        self._parameters_changed = True

        return parameters

    def _model_refresh_and_update_neighbor_list_parameters(self):
        """
        Call the model's refresh routine and update the neighbor list object
        for any necessary changes arising from changes to the model parameters,
        e.g. a change in one of its cutoffs.  After a model's parameters have
        been changed, this method *must* be called before calling the model's
        compute routine.
        """
        self._kim_model.clear_then_refresh()

        # Update neighbor list parameters
        (
            model_influence_dist,
            model_cutoffs,
            padding_not_require_neigh,
        ) = self._kimmodeldata.get_model_neighbor_list_parameters()

        self._neigh.set_neigh_parameters(
            self.neigh_skin_ratio,
            model_influence_dist,
            model_cutoffs,
            padding_not_require_neigh,
        )
