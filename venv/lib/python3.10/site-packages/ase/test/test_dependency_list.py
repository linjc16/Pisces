from ase.dependencies import format_dependency


def test_format_dependency():
    name, path = format_dependency('ase')
    assert name.startswith('ase-')
    # The path is where the module was installed,
    # *or* maybe it has no path depending on how
    # it was installed.
    assert isinstance(path, str)


def test_format_dependency_builtin():
    # Must work on modules that did not come from files.
    # An example happens to be the built-in module math,
    # but this would typically occur depending on distro.
    #
    # See https://gitlab.com/ase/ase/-/issues/1005
    name, _path = format_dependency('math')
    assert name.startswith('math-')
