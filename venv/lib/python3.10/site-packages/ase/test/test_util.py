from typing import Any, Dict, List

import pytest

from ase.utils import (
    deprecated,
    devnull,
    get_python_package_path_description,
    string2index,
    tokenize_version,
)


class DummyWarning(UserWarning):
    pass


def _add(
    a: int = 0, b: int = 0, *args, **kwargs
) -> int:
    return a + b + len(args)


DEPRECATION_MESSAGE = "Test"


@pytest.mark.filterwarnings(
    f"ignore:{DEPRECATION_MESSAGE}:ase.test.test_util.DummyWarning"
)
class TestDeprecatedDecorator:
    @staticmethod
    def test_should_raise_future_warning_by_default() -> None:
        deprecated_add = deprecated(DEPRECATION_MESSAGE)(_add)
        with pytest.warns(FutureWarning, match=DEPRECATION_MESSAGE):
            _ = deprecated_add()

    @staticmethod
    def test_should_satisfy_condition_and_raise_warning_by_default() -> None:
        deprecated_add = deprecated(
            DEPRECATION_MESSAGE, category=DummyWarning
        )(_add)
        with pytest.warns(DummyWarning, match=DEPRECATION_MESSAGE):
            _ = deprecated_add()

    @staticmethod
    def test_should_raise_warning_when_callback_returns_true() -> None:
        def callback(_: List, kwargs: Dict) -> bool:
            return "test" in kwargs

        with pytest.warns(DummyWarning, match=DEPRECATION_MESSAGE):
            deprecated_add = deprecated(
                DEPRECATION_MESSAGE,
                category=DummyWarning,
                callback=callback
            )(_add)
            _ = deprecated_add(test=True)

    @staticmethod
    def test_should_not_raise_warning_when_callback_returns_false() -> None:
        deprecated_add = deprecated(
            DEPRECATION_MESSAGE,
            callback=lambda args, kwargs: False
        )(_add)
        _ = deprecated_add()

    @staticmethod
    def test_should_call_function_correctly() -> None:
        deprecated_add = deprecated(
            DEPRECATION_MESSAGE,
            category=DummyWarning
        )(_add)
        assert deprecated_add(2, 2) == 4

    def test_should_call_callback(self) -> None:
        self.callback_called = False

        def callback(_: List, __: Dict) -> bool:
            self.callback_called = True
            return True

        deprecated_add = deprecated(
            DEPRECATION_MESSAGE,
            category=DummyWarning,
            callback=callback
        )(_add)
        _ = deprecated_add()
        assert self.callback_called

    @staticmethod
    def test_should_call_function_with_modified_args() -> None:
        def double_summands(args: List[int], _):
            for i, val in enumerate(args):
                args[i] = 2 * val

        deprecated_add_double = deprecated(
            DEPRECATION_MESSAGE,
            category=DummyWarning,
            callback=double_summands)(_add)
        assert deprecated_add_double(2, 2) == 8

    @staticmethod
    def test_should_call_function_with_modified_kwargs() -> None:
        def double_summands(_: List[int], kwargs: Dict[str, int]):
            for kwarg, val in kwargs.items():
                kwargs[kwarg] = 2 * val

        deprecated_add_double = deprecated(
            DEPRECATION_MESSAGE,
            category=DummyWarning,
            callback=double_summands)(_add)
        assert deprecated_add_double(a=2, b=2) == 8

    @staticmethod
    def test_should_raise_warning_and_modify_args_if_callback_returns_true(
    ) -> None:
        def limit_args_to_two(args: List, _: Dict[str, Any]):
            del args[2:]

        deprecated_add = deprecated(
            DEPRECATION_MESSAGE,
            category=DummyWarning,
            callback=limit_args_to_two)(_add)
        assert deprecated_add(1, 1, 1) == 2

    @staticmethod
    def test_should_work_when_warning_passed_as_message() -> None:
        deprecated_add = deprecated(FutureWarning(DEPRECATION_MESSAGE))(_add)
        with pytest.warns(FutureWarning, match=DEPRECATION_MESSAGE):
            _ = deprecated_add(2, 2)


def test_deprecated_devnull():
    with pytest.warns(DeprecationWarning):
        devnull.tell()


class TestDeprecationFunctional:
    pass


@pytest.mark.parametrize('v1, v2', [
    ('1', '2'),
    ('a', 'b'),
    ('9.0', '10.0'),
    ('3.8.0', '3.8.1'),
    ('3a', '3b'),
    ('3', '3a'),
])
def test_tokenize_version_lessthan(v1, v2):
    v1 = tokenize_version(v1)
    v2 = tokenize_version(v2)
    assert v1 < v2


def test_tokenize_version_equal():
    version = '3.8x.xx'
    assert tokenize_version(version) == tokenize_version(version)


class DummyIterator:
    def __iter__(self):
        yield from ["test", "bla"]


class Dummy:
    @property
    def __path__(self):
        return DummyIterator()


class TestString2Index:
    """Test `string2index`"""

    def test_zero(self):
        """Test 0"""
        assert string2index("0") == 0

    def test_last(self):
        """Test -1"""
        assert string2index("-1") == -1

    def test_last_two(self):
        """Test -2:"""
        assert string2index("-2:") == slice(-2, None)

    def test_all(self):
        """Test :"""
        assert string2index(":") == slice(None)

    def test_even(self):
        """Test ::2"""
        assert string2index("::2") == slice(None, None, 2)

    def test_three_colons(self):
        """Test invalid index string with three colons"""
        with pytest.raises(TypeError):
            string2index(":::")


def test_get_python_package_path_description():
    assert isinstance(get_python_package_path_description(Dummy()), str)
    # test object not containing __path__
    assert isinstance(get_python_package_path_description(object()), str)
