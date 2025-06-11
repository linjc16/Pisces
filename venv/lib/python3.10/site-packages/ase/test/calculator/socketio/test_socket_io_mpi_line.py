from ase.calculators.abinit import AbinitTemplate
from ase.calculators.espresso import EspressoTemplate
from ase.config import Config


def test_socketio_mpi_generator():
    cfg = Config()
    cfg.parser["espresso"] = {"command": "mpiexec pw.x", "pseudo_dir": "test"}
    cfg.parser["abinit"] = {"command": "mpirun abinit"}

    for temp_class in [EspressoTemplate, AbinitTemplate]:
        template = temp_class()
        profile = template.load_profile(cfg)

        socket_argv = template.socketio_argv(profile, "UNIX:TEST", None)
        profile_command = profile.get_command(
            inputfile=None,
            calc_command=socket_argv
        )

        assert profile_command == [*profile.command.split(), *socket_argv]
