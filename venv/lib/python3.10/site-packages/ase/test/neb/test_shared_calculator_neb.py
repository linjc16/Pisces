"""
This is testing NEB in general, though at the moment focusing on the shared
calculator implementation that is replacing
the SingleCalculatorNEB class.
Intending to be a *true* unittest, by testing small things
"""

from pytest import mark, raises, warns

from ase import Atoms
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator
from ase.mep import neb

pytestmark = mark.optimize


def test_get_neb_method():
    neb_dummy = neb.NEB([])

    assert isinstance(neb.get_neb_method(neb_dummy, "eb"), neb.FullSpringMethod)
    assert isinstance(neb.get_neb_method(neb_dummy, "aseneb"), neb.ASENEBMethod)
    assert isinstance(neb.get_neb_method(neb_dummy, "improvedtangent"),
                      neb.ImprovedTangentMethod)
    assert isinstance(neb.get_neb_method(neb_dummy, "spline"),
                      neb.SplineMethod)
    assert isinstance(neb.get_neb_method(neb_dummy, "string"),
                      neb.StringMethod)

    with raises(ValueError, match=r".*some_random_string.*"):
        _ = neb.get_neb_method(neb_dummy, "some_random_string")


class TestNEB:
    @classmethod
    def setup_class(cls):
        cls.h_atom = Atoms("H", positions=[[0., 0., 0.]], cell=[10., 10., 10.])
        cls.h2_molecule = Atoms("H2", positions=[[0., 0., 0.], [0., 1., 0.]])
        cls.images_dummy = [cls.h_atom.copy(), cls.h_atom.copy(),
                            cls.h_atom.copy()]

    def test_deprecations(self, testdir):
        # future warning on deprecated class
        with warns(FutureWarning, match=r".*Please use.*"):
            deprecated_neb = neb.SingleCalculatorNEB(self.images_dummy)
        assert deprecated_neb.allow_shared_calculator

        neb_dummy = neb.NEB(self.images_dummy)
        with warns(FutureWarning, match=r".*Please use.*idpp_interpolate.*"):
            neb_dummy.idpp_interpolate(steps=1)

    def test_neb_default(self):
        # default should be allow_shared_calculator=False
        neb_dummy = neb.NEB(self.images_dummy)
        assert not neb_dummy.allow_shared_calculator

    def test_raising_parallel_errors(self):
        # error is calculators are shared in parallel run
        with raises(RuntimeError, match=r".*Cannot use shared calculators.*"):
            _ = neb.NEB(self.images_dummy, allow_shared_calculator=True,
                        parallel=True)

    def test_no_shared_calc(self):
        images_shared_calc = [self.h_atom.copy(), self.h_atom.copy(),
                              self.h_atom.copy()]

        shared_calc = EMT()
        for at in images_shared_calc:
            at.calc = shared_calc

        neb_not_allow = neb.NEB(images_shared_calc,
                                allow_shared_calculator=False)

        # error if calculators are shared but not allowed to be
        with raises(ValueError, match=r".*NEB images share the same.*"):
            neb_not_allow.get_forces()

        # raise if calc cannot be shared
        with raises(RuntimeError, match=r".*Cannot set shared calculator.*"):
            neb_not_allow.set_calculators(EMT())

        # same calculators
        new_calculators = [EMT() for _ in range(neb_not_allow.nimages)]
        neb_not_allow.set_calculators(new_calculators)
        for i in range(neb_not_allow.nimages):
            assert new_calculators[i] == neb_not_allow.images[i].calc

        # just for the intermediate images
        neb_not_allow.set_calculators(new_calculators[1:-1])
        for i in range(1, neb_not_allow.nimages - 1):
            assert new_calculators[i] == neb_not_allow.images[i].calc

        # nimages-1 calculator is not allowed
        with raises(RuntimeError, match=r".*does not fit to len.*"):
            neb_not_allow.set_calculators(new_calculators[:-1])

    def test_init_checks(self):
        mismatch_len = [self.h_atom.copy(), self.h2_molecule.copy()]
        with raises(ValueError, match=r".*different numbers of atoms.*"):
            _ = neb.NEB(mismatch_len)

        mismatch_pbc = [self.h_atom.copy(), self.h_atom.copy()]
        mismatch_pbc[-1].set_pbc(True)
        with raises(ValueError, match=r".*different boundary conditions.*"):
            _ = neb.NEB(mismatch_pbc)

        mismatch_numbers = [
            self.h_atom.copy(),
            Atoms("C", positions=[[0., 0., 0.]], cell=[10., 10., 10.])]
        with raises(ValueError, match=r".*atoms in different orders.*"):
            _ = neb.NEB(mismatch_numbers)

        h_atom = self.h_atom.copy()
        h_atom.set_pbc(True)
        mismatch_cell = [h_atom.copy(), h_atom.copy()]
        mismatch_cell[-1].set_cell(mismatch_cell[-1].get_cell() + 0.00001)
        with raises(NotImplementedError, match=r".*Variable cell.*"):
            _ = neb.NEB(mismatch_cell)

    def test_freeze_method(self):
        at = self.h_atom.copy()
        at.calc = EMT()
        at.get_forces()
        results = dict(**at.calc.results)

        neb.NEB.freeze_results_on_image(at, **results)

        assert isinstance(at.calc, SinglePointCalculator)
