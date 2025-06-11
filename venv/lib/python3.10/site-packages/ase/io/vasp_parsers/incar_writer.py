# fmt: off

from collections.abc import Iterable


def write_incar(directory, parameters, header=None):
    incar_string = generate_incar_lines(parameters)
    with open(f"{directory}/INCAR", "w") as incar:
        if header is not None:
            incar.write(header + "\n")
        incar.write(incar_string)


def generate_incar_lines(parameters):
    if isinstance(parameters, str):
        return parameters
    elif parameters is None:
        return ""
    else:
        incar_lines = []
        for item in parameters.items():
            incar_lines += list(generate_line(*item))
        # Adding a newline at the end of the file
        return "\n".join(incar_lines) + "\n"


def generate_line(key, value, num_spaces=0):
    indent = " " * num_spaces
    if isinstance(value, str):
        if value.find("\n") != -1:
            value = '"' + value + '"'
        yield indent + f"{key.upper()} = {value}"
    elif isinstance(value, dict):
        yield indent + f"{key.upper()} {{"
        for item in value.items():
            yield from generate_line(*item, num_spaces + 4)
        yield indent + "}"
    elif isinstance(value, Iterable):
        yield indent + f"{key.upper()} = {' '.join(str(x) for x in value)}"
    else:
        yield indent + f"{key.upper()} = {value}"
