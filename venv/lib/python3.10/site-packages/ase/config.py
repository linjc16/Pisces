# fmt: off

import configparser
import os
import shlex
import warnings
from collections.abc import Mapping
from pathlib import Path

from ase.calculators.names import builtin, names, templates

ASE_CONFIG_FILE = Path.home() / ".config/ase/config.ini"


class ASEEnvDeprecationWarning(DeprecationWarning):
    def __init__(self, message):
        self.message = message


class Config(Mapping):
    def __init__(self):
        def argv_converter(argv):
            return shlex.split(argv)

        self.parser = configparser.ConfigParser(
            converters={"argv": argv_converter},
            interpolation=configparser.ExtendedInterpolation())
        self.paths = []

    def _env(self):
        if self.parser.has_section('environment'):
            return self.parser['environment']
        else:
            return {}

    def __iter__(self):
        yield from self._env()

    def __getitem__(self, item):
        # XXX We should replace the mapping behaviour with individual
        # methods to get from cfg or environment, or only from cfg.
        #
        # We cannot be a mapping very correctly without getting trouble
        # with mutable state needing synchronization with os.environ.

        env = self._env()
        try:
            return env[item]
        except KeyError:
            pass

        value = os.environ[item]
        warnings.warn(f'Loaded {item} from environment. '
                      'Please use configfile.',
                      ASEEnvDeprecationWarning)

        return value

    def __len__(self):
        return len(self._env())

    def check_calculators(self):
        print("Calculators")
        print("===========")
        print()
        print("Configured in ASE")
        print("   |  Installed on machine")
        print("   |   |  Name & version")
        print("   |   |  |")
        for name in names:
            # configured = False
            # installed = False
            template = templates.get(name)
            # if template is None:
            # XXX no template for this calculator.
            # We need templates for all calculators somehow,
            # but we can probably generate those for old FileIOCalculators
            # automatically.
            #    continue

            fullname = name
            try:
                codeconfig = self[name]
            except KeyError:
                codeconfig = None
                version = None
            else:
                if template is None:
                    # XXX we should not be executing this
                    if codeconfig is not None and "builtin" in codeconfig:
                        # builtin calculators
                        version = "builtin"
                    else:
                        version = None
                else:
                    profile = template.load_profile(codeconfig)
                    # XXX should be made robust to failure here:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        version = profile.version()

                fullname = name
                if version is not None:
                    fullname += f"--{version}"

            def tickmark(thing):
                return "[ ]" if thing is None else "[x]"

            msg = "  {configured} {installed} {fullname}".format(
                configured=tickmark(codeconfig),
                installed=tickmark(version),
                fullname=fullname,
            )
            print(msg)

    def print_header(self):
        print("Configuration")
        print("-------------")
        print()
        if not self.paths:
            print("No configuration loaded.")

        for path in self.paths:
            print(f"Loaded: {path}")

    def as_dict(self):
        return {key: dict(val) for key, val in self.parser.items()}

    def _read_paths(self, paths):
        self.paths += self.parser.read(paths)

    @classmethod
    def read(cls):
        envpath = os.environ.get("ASE_CONFIG_PATH")
        if envpath is None:
            paths = [ASE_CONFIG_FILE, ]
        else:
            paths = [Path(p) for p in envpath.split(":")]

        cfg = cls()
        cfg._read_paths(paths)

        # add sections for builtin calculators
        for name in builtin:
            cfg.parser.add_section(name)
            cfg.parser[name]["builtin"] = "True"
        return cfg


cfg = Config.read()
