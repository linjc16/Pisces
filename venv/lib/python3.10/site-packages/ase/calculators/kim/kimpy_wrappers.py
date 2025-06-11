# fmt: off

"""
Wrappers that provide a minimal interface to kimpy methods and objects

Daniel S. Karls
University of Minnesota
"""

import functools
from abc import ABC

import numpy as np

from .exceptions import (
    KIMModelInitializationError,
    KIMModelNotFound,
    KIMModelParameterError,
    KimpyError,
)


class LazyKimpyImport:
    """This class avoids module level import of the optional kimpy module."""

    def __getattr__(self, attr):
        return getattr(self._kimpy, attr)

    @functools.cached_property
    def _kimpy(self):
        import kimpy
        return kimpy


class Wrappers:
    """Shortcuts written in a way that avoids module-level kimpy import."""

    @property
    def collections_create(self):
        return functools.partial(check_call, kimpy.collections.create)

    @property
    def model_create(self):
        return functools.partial(check_call, kimpy.model.create)

    @property
    def simulator_model_create(self):
        return functools.partial(check_call, kimpy.simulator_model.create)

    @property
    def get_species_name(self):
        return functools.partial(
            check_call, kimpy.species_name.get_species_name)

    @property
    def get_number_of_species_names(self):
        return functools.partial(
            check_call, kimpy.species_name.get_number_of_species_names)

    @property
    def collection_item_type_portableModel(self):
        return kimpy.collection_item_type.portableModel


kimpy = LazyKimpyImport()
wrappers = Wrappers()

# Function used for casting parameter/extent indices to C-compatible ints
c_int = np.intc

# Function used for casting floating point parameter values to C-compatible
# doubles
c_double = np.double


def c_int_args(func):
    """
    Decorator for instance methods that will cast all of the args passed,
    excluding the first (which corresponds to 'self'), to C-compatible
    integers.
    """

    @functools.wraps(func)
    def myfunc(*args, **kwargs):
        args_cast = [args[0]]
        args_cast += map(c_int, args[1:])
        return func(*args, **kwargs)

    return myfunc


def check_call(f, *args, **kwargs):
    """Call a kimpy function using its arguments and, if a RuntimeError is
    raised, catch it and raise a KimpyError with the exception's
    message.

    (Starting with kimpy 2.0.0, a RuntimeError is the only exception
    type raised when something goes wrong.)"""

    try:
        return f(*args, **kwargs)
    except RuntimeError as e:
        raise KimpyError(
            f'Calling kimpy function "{f.__name__}" failed:\n  {e!s}')


def check_call_wrapper(func):
    @functools.wraps(func)
    def myfunc(*args, **kwargs):
        return check_call(func, *args, **kwargs)

    return myfunc


class ModelCollections:
    """
    KIM Portable Models and Simulator Models are installed/managed into
    different "collections".  In order to search through the different
    KIM API model collections on the system, a corresponding object must
    be instantiated.  For more on model collections, see the KIM API's
    install file:
    https://github.com/openkim/kim-api/blob/master/INSTALL
    """

    def __init__(self):
        self.collection = wrappers.collections_create()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, value, traceback):
        pass

    def get_item_type(self, model_name):
        try:
            model_type = check_call(self.collection.get_item_type, model_name)
        except KimpyError:
            msg = (
                "Could not find model {} installed in any of the KIM API "
                "model collections on this system.  See "
                "https://openkim.org/doc/usage/obtaining-models/ for "
                "instructions on installing models.".format(model_name)
            )
            raise KIMModelNotFound(msg)

        return model_type

    @property
    def initialized(self):
        return hasattr(self, "collection")


class PortableModel:
    """Creates a KIM API Portable Model object and provides a minimal
    interface to it"""

    def __init__(self, model_name, debug):
        self.model_name = model_name
        self.debug = debug

        # Create KIM API Model object
        units_accepted, self.kim_model = wrappers.model_create(
            kimpy.numbering.zeroBased,
            kimpy.length_unit.A,
            kimpy.energy_unit.eV,
            kimpy.charge_unit.e,
            kimpy.temperature_unit.K,
            kimpy.time_unit.ps,
            self.model_name,
        )

        if not units_accepted:
            raise KIMModelInitializationError(
                "Requested units not accepted in kimpy.model.create"
            )

        if self.debug:
            l_unit, e_unit, c_unit, te_unit, ti_unit = check_call(
                self.kim_model.get_units
            )
            print(f"Length unit is: {l_unit}")
            print(f"Energy unit is: {e_unit}")
            print(f"Charge unit is: {c_unit}")
            print(f"Temperature unit is: {te_unit}")
            print(f"Time unit is: {ti_unit}")
            print()

        self._create_parameters()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, value, traceback):
        pass

    @check_call_wrapper
    def _get_number_of_parameters(self):
        return self.kim_model.get_number_of_parameters()

    def _create_parameters(self):
        def _kim_model_parameter(**kwargs):
            dtype = kwargs["dtype"]

            if dtype == "Integer":
                return KIMModelParameterInteger(**kwargs)
            elif dtype == "Double":
                return KIMModelParameterDouble(**kwargs)
            else:
                raise KIMModelParameterError(
                    f"Invalid model parameter type {dtype}. Supported types "
                    "'Integer' and 'Double'."
                )

        self._parameters = {}
        num_params = self._get_number_of_parameters()
        for index_param in range(num_params):
            parameter_metadata = self._get_one_parameter_metadata(index_param)
            name = parameter_metadata["name"]

            self._parameters[name] = _kim_model_parameter(
                kim_model=self.kim_model,
                dtype=parameter_metadata["dtype"],
                extent=parameter_metadata["extent"],
                name=name,
                description=parameter_metadata["description"],
                parameter_index=index_param,
            )

    def get_model_supported_species_and_codes(self):
        """Get all of the supported species for this model and their
        corresponding integer codes that are defined in the KIM API

        Returns
        -------
        species : list of str
            Abbreviated chemical symbols of all species the mmodel
            supports (e.g. ["Mo", "S"])

        codes : list of int
            Integer codes used by the model for each species (order
            corresponds to the order of ``species``)
        """
        species = []
        codes = []
        num_kim_species = wrappers.get_number_of_species_names()

        for i in range(num_kim_species):
            species_name = wrappers.get_species_name(i)

            species_is_supported, code = self.get_species_support_and_code(
                species_name)

            if species_is_supported:
                species.append(str(species_name))
                codes.append(code)

        return species, codes

    @check_call_wrapper
    def clear_then_refresh(self):
        self.kim_model.clear_then_refresh()

    @c_int_args
    def _get_parameter_metadata(self, index_parameter):
        try:
            dtype, extent, name, description = check_call(
                self.kim_model.get_parameter_metadata, index_parameter
            )
        except KimpyError as e:
            raise KIMModelParameterError(
                "Failed to retrieve metadata for "
                f"parameter at index {index_parameter}"
            ) from e

        return dtype, extent, name, description

    def parameters_metadata(self):
        """Metadata associated with all model parameters.

        Returns
        -------
        dict
            Metadata associated with all model parameters.
        """
        return {
            param_name: param.metadata
            for param_name, param in self._parameters.items()
        }

    def parameter_names(self):
        """Names of model parameters registered in the KIM API.

        Returns
        -------
        tuple
            Names of model parameters registered in the KIM API
        """
        return tuple(self._parameters.keys())

    def get_parameters(self, **kwargs):
        """
        Get the values of one or more model parameter arrays.

        Given the names of one or more model parameters and a set of indices
        for each of them, retrieve the corresponding elements of the relevant
        model parameter arrays.

        Parameters
        ----------
        **kwargs
            Names of the model parameters and the indices whose values should
            be retrieved.

        Returns
        -------
        dict
            The requested indices and the values of the model's parameters.

        Note
        ----
        The output of this method can be used as input of
        ``set_parameters``.

        Example
        -------
        To get `epsilons` and `sigmas` in the LJ universal model for Mo-Mo
        (index 4879), Mo-S (index 2006) and S-S (index 1980) interactions::

            >>> LJ = 'LJ_ElliottAkerson_2015_Universal__MO_959249795837_003'
            >>> calc = KIM(LJ)
            >>> calc.get_parameters(epsilons=[4879, 2006, 1980],
            ...                     sigmas=[4879, 2006, 1980])
            {'epsilons': [[4879, 2006, 1980],
                          [4.47499, 4.421814057295943, 4.36927]],
             'sigmas': [[4879, 2006, 1980],
                        [2.74397, 2.30743, 1.87089]]}
        """
        parameters = {}
        for parameter_name, index_range in kwargs.items():
            parameters.update(
                self._get_one_parameter(
                    parameter_name,
                    index_range))
        return parameters

    def set_parameters(self, **kwargs):
        """
        Set the values of one or more model parameter arrays.

        Given the names of one or more model parameters and a set of indices
        and corresponding values for each of them, mutate the corresponding
        elements of the relevant model parameter arrays.

        Parameters
        ----------
        **kwargs
            Names of the model parameters to mutate and the corresponding
            indices and values to set.

        Returns
        -------
        dict
            The requested indices and the values of the model's parameters
            that were set.

        Example
        -------
        To set `epsilons` in the LJ universal model for Mo-Mo (index 4879),
        Mo-S (index 2006) and S-S (index 1980) interactions to 5.0, 4.5, and
        4.0, respectively::

            >>> LJ = 'LJ_ElliottAkerson_2015_Universal__MO_959249795837_003'
            >>> calc = KIM(LJ)
            >>> calc.set_parameters(epsilons=[[4879, 2006, 1980],
            ...                               [5.0, 4.5, 4.0]])
            {'epsilons': [[4879, 2006, 1980],
                          [5.0, 4.5, 4.0]]}
        """
        parameters = {}
        for parameter_name, parameter_data in kwargs.items():
            index_range, values = parameter_data
            self._set_one_parameter(parameter_name, index_range, values)
            parameters[parameter_name] = parameter_data

        return parameters

    def _get_one_parameter(self, parameter_name, index_range):
        """
        Retrieve value of one or more components of a model parameter array.

        Parameters
        ----------
        parameter_name : str
            Name of model parameter registered in the KIM API.
        index_range : int or list
            Zero-based index (int) or indices (list of int) specifying the
            component(s) of the corresponding model parameter array that are
            to be retrieved.

        Returns
        -------
        dict
            The requested indices and the corresponding values of the model
            parameter array.
        """
        if parameter_name not in self._parameters:
            raise KIMModelParameterError(
                f"Parameter '{parameter_name}' is not "
                "supported by this model. "
                "Please check that the parameter name is spelled correctly."
            )

        return self._parameters[parameter_name].get_values(index_range)

    def _set_one_parameter(self, parameter_name, index_range, values):
        """
        Set the value of one or more components of a model parameter array.

        Parameters
        ----------
        parameter_name : str
            Name of model parameter registered in the KIM API.
        index_range : int or list
            Zero-based index (int) or indices (list of int) specifying the
            component(s) of the corresponding model parameter array that are
            to be mutated.
        values : int/float or list
            Value(s) to assign to the component(s) of the model parameter
            array specified by ``index_range``.
        """
        if parameter_name not in self._parameters:
            raise KIMModelParameterError(
                f"Parameter '{parameter_name}' is not "
                "supported by this model. "
                "Please check that the parameter name is spelled correctly."
            )

        self._parameters[parameter_name].set_values(index_range, values)

    def _get_one_parameter_metadata(self, index_parameter):
        """
        Get metadata associated with a single model parameter.

        Parameters
        ----------
        index_parameter : int
            Zero-based index used by the KIM API to refer to this model
            parameter.

        Returns
        -------
        dict
            Metadata associated with the requested model parameter.
        """
        dtype, extent, name, description = self._get_parameter_metadata(
            index_parameter)
        parameter_metadata = {
            "name": name,
            "dtype": repr(dtype),
            "extent": extent,
            "description": description,
        }
        return parameter_metadata

    @check_call_wrapper
    def compute(self, compute_args_wrapped, release_GIL):
        return self.kim_model.compute(
            compute_args_wrapped.compute_args, release_GIL)

    @check_call_wrapper
    def get_species_support_and_code(self, species_name):
        return self.kim_model.get_species_support_and_code(species_name)

    @check_call_wrapper
    def get_influence_distance(self):
        return self.kim_model.get_influence_distance()

    @check_call_wrapper
    def get_neighbor_list_cutoffs_and_hints(self):
        return self.kim_model.get_neighbor_list_cutoffs_and_hints()

    def compute_arguments_create(self):
        return ComputeArguments(self, self.debug)

    @property
    def initialized(self):
        return hasattr(self, "kim_model")


class KIMModelParameter(ABC):
    def __init__(self, kim_model, dtype, extent,
                 name, description, parameter_index):
        self._kim_model = kim_model
        self._dtype = dtype
        self._extent = extent
        self._name = name
        self._description = description

        # Ensure that parameter_index is cast to a C-compatible integer. This
        # is necessary because this is passed to kimpy.
        self._parameter_index = c_int(parameter_index)

    @property
    def metadata(self):
        return {
            "dtype": self._dtype,
            "extent": self._extent,
            "name": self._name,
            "description": self._description,
        }

    @c_int_args
    def _get_one_value(self, index_extent):
        get_parameter = getattr(self._kim_model, self._dtype_accessor)
        try:
            return check_call(
                get_parameter, self._parameter_index, index_extent)
        except KimpyError as exception:
            raise KIMModelParameterError(
                f"Failed to access component {index_extent} of model "
                f"parameter of type '{self._dtype}' at parameter index "
                f"{self._parameter_index}"
            ) from exception

    def _set_one_value(self, index_extent, value):
        value_typecast = self._dtype_c(value)

        try:
            check_call(
                self._kim_model.set_parameter,
                self._parameter_index,
                c_int(index_extent),
                value_typecast,
            )
        except KimpyError:
            raise KIMModelParameterError(
                f"Failed to set component {index_extent} at parameter index "
                f"{self._parameter_index} to {self._dtype} value "
                f"{value_typecast}"
            )

    def get_values(self, index_range):
        index_range_dim = np.ndim(index_range)
        if index_range_dim == 0:
            values = self._get_one_value(index_range)
        elif index_range_dim == 1:
            values = []
            for idx in index_range:
                values.append(self._get_one_value(idx))
        else:
            raise KIMModelParameterError(
                "Index range must be an integer or a list of integers"
            )
        return {self._name: [index_range, values]}

    def set_values(self, index_range, values):
        index_range_dim = np.ndim(index_range)
        values_dim = np.ndim(values)

        # Check the shape of index_range and values
        msg = "index_range and values must have the same shape"
        assert index_range_dim == values_dim, msg

        if index_range_dim == 0:
            self._set_one_value(index_range, values)
        elif index_range_dim == 1:
            assert len(index_range) == len(values), msg
            for idx, value in zip(index_range, values):
                self._set_one_value(idx, value)
        else:
            raise KIMModelParameterError(
                "Index range must be an integer or a list containing a "
                "single integer"
            )


class KIMModelParameterInteger(KIMModelParameter):
    _dtype_c = c_int
    _dtype_accessor = "get_parameter_int"


class KIMModelParameterDouble(KIMModelParameter):
    _dtype_c = c_double
    _dtype_accessor = "get_parameter_double"


class ComputeArguments:
    """Creates a KIM API ComputeArguments object from a KIM Portable
    Model object and configures it for ASE.  A ComputeArguments object
    is associated with a KIM Portable Model and is used to inform the
    KIM API of what the model can compute.  It is also used to
    register the data arrays that allow the KIM API to pass the atomic
    coordinates to the model and retrieve the corresponding energy and
    forces, etc."""

    def __init__(self, kim_model_wrapped, debug):
        self.kim_model_wrapped = kim_model_wrapped
        self.debug = debug

        # Create KIM API ComputeArguments object
        self.compute_args = check_call(
            self.kim_model_wrapped.kim_model.compute_arguments_create
        )

        # Check compute arguments
        kimpy_arg_name = kimpy.compute_argument_name
        num_arguments = kimpy_arg_name.get_number_of_compute_argument_names()
        if self.debug:
            print(f"Number of compute_args: {num_arguments}")

        for i in range(num_arguments):
            name = check_call(kimpy_arg_name.get_compute_argument_name, i)
            dtype = check_call(
                kimpy_arg_name.get_compute_argument_data_type, name)

            arg_support = self.get_argument_support_status(name)

            if self.debug:
                print(
                    "Compute Argument name {:21} is of type {:7} "
                    "and has support "
                    "status {}".format(*[str(x)
                                       for x in [name, dtype, arg_support]])
                )

            # See if the model demands that we ask it for anything
            # other than energy and forces.  If so, raise an
            # exception.
            if arg_support == kimpy.support_status.required:
                if (
                    name != kimpy.compute_argument_name.partialEnergy
                    and name != kimpy.compute_argument_name.partialForces
                ):
                    raise KIMModelInitializationError(
                        f"Unsupported required ComputeArgument {name}"
                    )

        # Check compute callbacks
        callback_name = kimpy.compute_callback_name
        num_callbacks = callback_name.get_number_of_compute_callback_names()
        if self.debug:
            print()
            print(f"Number of callbacks: {num_callbacks}")

        for i in range(num_callbacks):
            name = check_call(callback_name.get_compute_callback_name, i)

            support_status = self.get_callback_support_status(name)

            if self.debug:
                print(
                    "Compute callback {:17} has support status {}".format(
                        str(name), support_status
                    )
                )

            # Cannot handle any "required" callbacks
            if support_status == kimpy.support_status.required:
                raise KIMModelInitializationError(
                    f"Unsupported required ComputeCallback: {name}"
                )

    @check_call_wrapper
    def set_argument_pointer(self, compute_arg_name, data_object):
        return self.compute_args.set_argument_pointer(
            compute_arg_name, data_object)

    @check_call_wrapper
    def get_argument_support_status(self, name):
        return self.compute_args.get_argument_support_status(name)

    @check_call_wrapper
    def get_callback_support_status(self, name):
        return self.compute_args.get_callback_support_status(name)

    @check_call_wrapper
    def set_callback(self, compute_callback_name,
                     callback_function, data_object):
        return self.compute_args.set_callback(
            compute_callback_name, callback_function, data_object
        )

    @check_call_wrapper
    def set_callback_pointer(
            self, compute_callback_name, callback, data_object):
        return self.compute_args.set_callback_pointer(
            compute_callback_name, callback, data_object
        )

    def update(
        self, num_particles, species_code, particle_contributing,
            coords, energy, forces
    ):
        """Register model input and output in the kim_model object."""
        compute_arg_name = kimpy.compute_argument_name
        set_argument_pointer = self.set_argument_pointer

        set_argument_pointer(compute_arg_name.numberOfParticles, num_particles)
        set_argument_pointer(
            compute_arg_name.particleSpeciesCodes,
            species_code)
        set_argument_pointer(
            compute_arg_name.particleContributing, particle_contributing
        )
        set_argument_pointer(compute_arg_name.coordinates, coords)
        set_argument_pointer(compute_arg_name.partialEnergy, energy)
        set_argument_pointer(compute_arg_name.partialForces, forces)

        if self.debug:
            print("Debug: called update_kim")
            print()


class SimulatorModel:
    """Creates a KIM API Simulator Model object and provides a minimal
    interface to it.  This is only necessary in this package in order to
    extract any information about a given simulator model because it is
    generally embedded in a shared object.
    """

    def __init__(self, model_name):
        # Create a KIM API Simulator Model object for this model
        self.model_name = model_name
        self.simulator_model = wrappers.simulator_model_create(self.model_name)

        # Need to close template map in order to access simulator
        # model metadata
        self.simulator_model.close_template_map()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, value, traceback):
        pass

    @property
    def simulator_name(self):
        simulator_name, _ = self.simulator_model.\
            get_simulator_name_and_version()
        return simulator_name

    @property
    def num_supported_species(self):
        num_supported_species = self.simulator_model.\
            get_number_of_supported_species()
        if num_supported_species == 0:
            raise KIMModelInitializationError(
                "Unable to determine supported species of "
                "simulator model {}.".format(self.model_name)
            )
        return num_supported_species

    @property
    def supported_species(self):
        supported_species = []
        for spec_code in range(self.num_supported_species):
            species = check_call(
                self.simulator_model.get_supported_species, spec_code)
            supported_species.append(species)

        return tuple(supported_species)

    @property
    def num_metadata_fields(self):
        return self.simulator_model.get_number_of_simulator_fields()

    @property
    def metadata(self):
        sm_metadata_fields = {}
        for field in range(self.num_metadata_fields):
            extent, field_name = check_call(
                self.simulator_model.get_simulator_field_metadata, field
            )
            sm_metadata_fields[field_name] = []
            for ln in range(extent):
                field_line = check_call(
                    self.simulator_model.get_simulator_field_line, field, ln
                )
                sm_metadata_fields[field_name].append(field_line)

        return sm_metadata_fields

    @property
    def supported_units(self):
        try:
            supported_units = self.metadata["units"][0]
        except (KeyError, IndexError):
            raise KIMModelInitializationError(
                "Unable to determine supported units of "
                "simulator model {}.".format(self.model_name)
            )

        return supported_units

    @property
    def atom_style(self):
        """
        See if a 'model-init' field exists in the SM metadata and, if
        so, whether it contains any entries including an "atom_style"
        command.  This is specific to LAMMPS SMs and is only required
        for using the LAMMPSrun calculator because it uses
        lammps.inputwriter to create a data file.  All other content in
        'model-init', if it exists, is ignored.
        """
        atom_style = None
        for ln in self.metadata.get("model-init", []):
            if ln.find("atom_style") != -1:
                atom_style = ln.split()[1]

        return atom_style

    @property
    def model_defn(self):
        return self.metadata["model-defn"]

    @property
    def initialized(self):
        return hasattr(self, "simulator_model")
