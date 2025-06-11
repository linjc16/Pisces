import math

from ase.utils.parsemath import eval_expression


def test_parsemath():

    param_dct = {
        "param_1": 1,
        "param_2": 4.0,
        "param_3": 478245.7586,
        "param_4": 1.58e-5,
        "param_5": -2.48757,
        "param_6": -2.0,
        "param_7": math.pi / 4.0,
    }

    expressions = [
        "3.0*param_1",
        "param_2**-2.0",
        "(param_6)**2.0",
        "param_1 / param_5",
        "param_1 + param_2 * 30.0 - param_5",
        "(param_1 + 1) / param_2",
        "sqrt(param_2)",
        "fmod(param_4, param_1)",
        "sin(param_7)",
        "cos(param_7)",
        "tan(param_7)",
        "asin(param_7)",
        "acos(param_7)",
        "atan(param_7)",
        "atan2(param_7, 2.0)",
        "hypot(3.0, 4.0)",
        "sinh(param_7)",
        "cosh(param_7)",
        "tanh(param_7)",
        "asinh(param_7)",
        "acosh(param_2)",
        "atanh(param_7)",
        "degrees(radians(param_7))",
        "log(param_3)",
        "log10(param_3)",
        "log2(param_3)",
        "abs(param_5)",
        "ceil(param_5)",
        "floor(param_5)",
        "round(param_5)",
        "exp(param_1)",
        "2.0*e",
        "pi+1",
        "tau / pi",
        "param_5 % param_6",
        "param_3 // param_6",
    ]

    solutions = [
        3 * param_dct["param_1"],
        param_dct["param_2"]**-2.0,
        (param_dct["param_6"])**2.0,
        param_dct["param_1"] / param_dct["param_5"],
        param_dct["param_1"] + param_dct["param_2"] *
        30.0 - param_dct["param_5"],
        (param_dct["param_1"] + 1) / param_dct["param_2"],
        math.sqrt(param_dct["param_2"]),
        math.fmod(param_dct["param_4"], param_dct["param_1"]),
        math.sin(param_dct["param_7"]),
        math.cos(param_dct["param_7"]),
        math.tan(param_dct["param_7"]),
        math.asin(param_dct["param_7"]),
        math.acos(param_dct["param_7"]),
        math.atan(param_dct["param_7"]),
        math.atan2(param_dct["param_7"], 2.0),
        math.hypot(3.0, 4.0),
        math.sinh(param_dct["param_7"]),
        math.cosh(param_dct["param_7"]),
        math.tanh(param_dct["param_7"]),
        math.asinh(param_dct["param_7"]),
        math.acosh(param_dct["param_2"]),
        math.atanh(param_dct["param_7"]),
        math.degrees(math.radians(param_dct["param_7"])),
        math.log(param_dct["param_3"]),
        math.log10(param_dct["param_3"]),
        math.log2(param_dct["param_3"]),
        math.fabs(param_dct["param_5"]),
        math.ceil(param_dct["param_5"]),
        math.floor(param_dct["param_5"]),
        round(param_dct["param_5"]),
        math.exp(param_dct["param_1"]),
        2.0 * math.e,
        math.pi + 1,
        2.0,
        param_dct["param_5"] % param_dct["param_6"],
    ]

    for expr, soln in zip(expressions, solutions):
        assert abs(eval_expression(expr, param_dct) - soln) < 1e-13

    try:
        eval_expression("99**99**99*99**99**99")
        raise RuntimeError(
            "This should not be reached, the parser is now vulnerable "
            "to computational time based DOS attack")
    except ValueError:
        pass

    try:
        eval_expression("e" * 10000000, {})
        raise RuntimeError(
            "This should not be reached, the parser is now vulnerable "
            "to memory based DOS attack")
    except ValueError:
        pass

    try:
        eval_expression("__import__('os').system('echo $HOME')")
        raise RuntimeError(
            "This should not be reached, "
            "the parser can execute malicious code")
    except TypeError:
        pass
