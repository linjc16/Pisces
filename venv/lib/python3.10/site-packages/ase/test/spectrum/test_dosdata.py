from collections import OrderedDict
from typing import Any, List, Tuple

import numpy as np
import pytest

from ase.spectrum.dosdata import DOSData, GridDOSData, RawDOSData


class MinimalDOSData(DOSData):
    """Inherit from ABC to test its features"""

    def get_energies(self):
        return NotImplementedError()

    def get_weights(self):
        return NotImplementedError()

    def copy(self):
        return NotImplementedError()


class TestDosData:
    """Test the abstract base class for DOS data"""
    sample_info: List[Tuple[Any, Any]] = [
        (None, {}),
        ({}, {}),
        ({'symbol': 'C', 'index': '2', 'strangekey': 'isallowed'},
         {'symbol': 'C', 'index': '2', 'strangekey': 'isallowed'}),
        ('notadict', TypeError),
        (False, TypeError)]

    @pytest.mark.parametrize('info, expected', sample_info)
    def test_dosdata_init_info(self, info, expected):
        """Check 'info' parameter is handled properly"""
        if isinstance(expected, type) and isinstance(expected(), Exception):
            with pytest.raises(expected):
                dos_data = MinimalDOSData(info=info)
        else:
            dos_data = MinimalDOSData(info=info)
            assert dos_data.info == expected

    @pytest.mark.parametrize('info, expected',
                             [({}, ''),
                              ({'key1': 'value1'}, 'key1: value1'),
                              (OrderedDict([('key1', 'value1'),
                                            ('key2', 'value2')]),
                               'key1: value1; key2: value2'),
                              ({'key1': 'value1', 'label': 'xyz'}, 'xyz'),
                              ({'label': 'xyz'}, 'xyz')])
    def test_label_from_info(self, info, expected):
        assert DOSData.label_from_info(info) == expected


class TestRawDosData:
    """Test the raw DOS data container"""
    @pytest.fixture()
    def sparse_dos(self):
        return RawDOSData([1.2, 3.4, 5.], [3., 2.1, 0.],
                          info={'symbol': 'H', 'number': '1', 'food': 'egg'})

    @pytest.fixture()
    def another_sparse_dos(self):
        return RawDOSData([8., 2., 2., 5.], [1., 1., 1., 1.],
                          info={'symbol': 'H', 'number': '2'})

    def test_init(self):
        with pytest.raises(ValueError):
            RawDOSData([1, 2, 3], [4, 5], info={'symbol': 'H'})

    def test_access(self, sparse_dos):
        assert sparse_dos.info == {'symbol': 'H', 'number': '1', 'food': 'egg'}
        assert np.allclose(sparse_dos.get_energies(), [1.2, 3.4, 5.])
        assert np.allclose(sparse_dos.get_weights(), [3., 2.1, 0.])

    def test_copy(self, sparse_dos):
        copy_dos = sparse_dos.copy()
        assert copy_dos.info == sparse_dos.info
        sparse_dos.info['symbol'] = 'X'
        assert sparse_dos.info['symbol'] == 'X'
        assert copy_dos.info['symbol'] == 'H'

    @pytest.mark.parametrize('other', [True, 1, 0.5, 'string'])
    def test_equality_wrongtype(self, sparse_dos, other):
        assert not sparse_dos._almost_equals(other)

    equality_data = [(((1., 2.), (3., 4.), {'symbol': 'H'}),
                      ((1., 2.), (3., 4.), {'symbol': 'H'}),
                      True),
                     (((1., 3.), (3., 4.), {'symbol': 'H'}),
                      ((1., 2.), (3., 4.), {'symbol': 'H'}),
                      False),
                     (((1., 2.), (3., 5.), {'symbol': 'H'}),
                      ((1., 2.), (3., 4.), {'symbol': 'H'}),
                      False),
                     (((1., 3.), (3., 5.), {'symbol': 'H'}),
                      ((1., 2.), (3., 4.), {'symbol': 'H'}),
                      False),
                     (((1., 2.), (3., 4.), {'symbol': 'H'}),
                      ((1., 2.), (3., 4.), {'symbol': 'He'}),
                      False),
                     (((1., 3.), (3., 4.), {'symbol': 'H'}),
                      ((1., 2.), (3., 4.), {'symbol': 'He'}),
                      False)]

    @pytest.mark.parametrize('data_1, data_2, isequal', equality_data)
    def test_equality(self, data_1, data_2, isequal):
        assert (RawDOSData(*data_1[:2], info=data_1[2])
                ._almost_equals(RawDOSData(*data_2[:2], info=data_2[2]))
                ) == isequal

    def test_addition(self, sparse_dos, another_sparse_dos):
        summed_dos = sparse_dos + another_sparse_dos
        assert summed_dos.info == {'symbol': 'H'}
        assert np.allclose(summed_dos.get_energies(),
                           [1.2, 3.4, 5., 8., 2., 2., 5.])
        assert np.allclose(summed_dos.get_weights(),
                           [3., 2.1, 0., 1., 1., 1., 1.])

    sampling_data_args_results = [
        # Special case: peak max at width 1
        ([[0.], [1.]],
         [[0.], {'width': 1}],
         [1. / (np.sqrt(2. * np.pi))]),
        # Peak max with different width, position
        ([[1.], [2.]],
         [[1.], {'width': 0.5}],
         [2. / (np.sqrt(2. * np.pi) * 0.5)]),
        # Peak max for two simultaneous deltas
        ([[1., 1.], [2., 1.]],
         [[1.], {'width': 1}],
         [3. / (np.sqrt(2. * np.pi))]),
        # Compare with theoretical half-maximum
        ([[0.], [1.]],
         [[np.sqrt(2 * np.log(2)) * 3],
          {'width': 3}],
         [0.5 / (np.sqrt(2 * np.pi) * 3)]),
        # And a case with multiple values, generated
        # using the ASE code (not benchmarked)
        ([[1.2, 3.4, 5], [3., 2.1, 0.]],
         [[1., 1.5, 2., 2.4], {'width': 2}],
         [0.79932418, 0.85848101, 0.88027184, 0.8695055])]

    @pytest.mark.parametrize('data, args, result',
                             sampling_data_args_results)
    def test_sampling(self, data, args, result):
        dos = RawDOSData(data[0], data[1])
        weights = dos._sample(*args[:-1], **args[-1])
        assert np.allclose(weights, result)

        with pytest.raises(ValueError):
            dos._sample([1], smearing="Gauss's spherical cousin")

    def test_sampling_error(self, sparse_dos):
        with pytest.raises(ValueError):
            sparse_dos._sample([1, 2, 3], width=0.)
        with pytest.raises(ValueError):
            sparse_dos._sample([1, 2, 3], width=-1)

    def test_sample_grid(self, sparse_dos):
        min_dos = sparse_dos.sample_grid(10, xmax=5, padding=3, width=0.1)
        assert min_dos.get_energies()[0] == pytest.approx(1.2 - 3 * 0.1)

        max_dos = sparse_dos.sample_grid(10, xmin=0, padding=2, width=0.2)
        assert max_dos.get_energies()[-1] == pytest.approx(5 + 2 * 0.2)

        default_dos = sparse_dos.sample_grid(10)
        assert np.allclose(default_dos.get_energies(),
                           np.linspace(0.9, 5.3, 10))
        dos0 = sparse_dos._sample(np.linspace(0.9, 5.3, 10))
        assert np.allclose(default_dos.get_weights(),
                           dos0)

    # Comparing plot outputs is hard, so we
    # - inspect the line values
    # - check that a line styling parameter is correctly passed through mplargs
    # - set a kwarg from self.sample() to check broadening args are recognised
    linewidths = [1, 5, None]

    @pytest.mark.usefixtures("figure")
    @pytest.mark.parametrize('linewidth', linewidths)
    def test_plot(self, sparse_dos, figure, linewidth):
        if linewidth is None:
            mplargs = None
        else:
            mplargs = {'linewidth': linewidth}

        ax = figure.add_subplot(111)
        ax_out = sparse_dos.plot(npts=5, ax=ax, mplargs=mplargs,
                                 smearing='Gauss')
        assert ax_out == ax

        line_data = ax.lines[0].get_data()
        assert np.allclose(line_data[0], np.linspace(0.9, 5.3, 5))
        assert np.allclose(line_data[1],
                           [1.32955452e-01, 1.51568133e-13,
                            9.30688167e-02, 1.06097693e-13, 3.41173568e-78])

    @pytest.mark.usefixtures("figure")
    @pytest.mark.parametrize('linewidth', linewidths)
    def test_plot_deltas(self, sparse_dos, figure, linewidth):
        if linewidth is None:
            mplargs = None
        else:
            mplargs = {'linewidth': linewidth}

        ax = figure.add_subplot(111)
        ax_out = sparse_dos.plot_deltas(ax=ax, mplargs=mplargs)
        assert ax_out == ax
        assert np.allclose(list(map(lambda x: x.vertices,
                                    ax.get_children()[0].get_paths())),
                           [[[1.2, 0.], [1.2, 3.]],
                            [[3.4, 0.], [3.4, 2.1]],
                            [[5., 0.], [5., 0.]]])


class TestGridDosData:
    """Test the grid DOS data container"""

    def test_init(self):
        # energies and weights must be equal lengths
        with pytest.raises(ValueError):
            GridDOSData(np.linspace(0, 10, 11), np.zeros(10))

        # energies must be evenly spaced
        with pytest.raises(ValueError):
            GridDOSData(np.linspace(0, 10, 11)**2, np.zeros(11))

    @pytest.fixture()
    def dense_dos(self):
        x = np.linspace(0., 10., 11)
        y = np.sin(x / 10)
        return GridDOSData(x, y, info={'symbol': 'C', 'orbital': '2s',
                                       'day': 'Tue'})

    @pytest.fixture()
    def denser_dos(self):
        x = np.linspace(0., 10., 21)
        y = np.sin(x / 10)
        return GridDOSData(x, y)

    @pytest.fixture()
    def another_dense_dos(self):
        x = np.linspace(0., 10., 11)
        y = np.sin(x / 10) * 2
        return GridDOSData(x, y, info={'symbol': 'C', 'orbital': '2p',
                                       'month': 'Feb'})

    def test_access(self, dense_dos):
        assert dense_dos.info == {'symbol': 'C', 'orbital': '2s', 'day': 'Tue'}
        assert len(dense_dos.get_energies()) == 11
        assert dense_dos.get_energies()[-2] == pytest.approx(9.)
        assert dense_dos.get_weights()[-1] == pytest.approx(np.sin(1))

    def test_copy(self, dense_dos):
        copy_dos = dense_dos.copy()
        assert copy_dos.info == dense_dos.info
        dense_dos.info['symbol'] = 'X'
        assert dense_dos.info['symbol'] == 'X'
        assert copy_dos.info['symbol'] == 'C'

    def test_addition(self, dense_dos, another_dense_dos):
        sum_dos = dense_dos + another_dense_dos
        assert np.allclose(sum_dos.get_energies(), dense_dos.get_energies())
        assert np.allclose(sum_dos.get_weights(), dense_dos.get_weights() * 3)
        assert sum_dos.info == {'symbol': 'C'}

        with pytest.raises(ValueError):
            dense_dos + GridDOSData(dense_dos.get_energies() + 1.,
                                    dense_dos.get_weights())
        with pytest.raises(ValueError):
            dense_dos + GridDOSData(dense_dos.get_energies()[1:],
                                    dense_dos.get_weights()[1:])

    def test_check_spacing(self, dense_dos):
        """Check a warning is logged when width < 2 * grid spacing"""
        # In the sample data, grid spacing is 1.0
        dense_dos._sample([1], width=2.1)

        with pytest.warns(UserWarning, match="The broadening width is small"):
            dense_dos._sample([1], width=1.9)

    def test_resampling_consistency(self, dense_dos, denser_dos):
        """Check that resampled spectra are independent of the original density

        Compare resampling of sample function on two different grids to the
        same new grid with broadening. We accept a 5% difference because the
        initial shape is slightly different; what we are checking for is a
        factor 2 difference from "double-counting" the extra data points.
        """
        sampling_params = dict(npts=500, xmin=0, xmax=10, width=4)

        from_dense = dense_dos.sample_grid(**sampling_params)
        from_denser = denser_dos.sample_grid(**sampling_params)

        assert np.allclose(from_dense.get_energies(),
                           from_denser.get_energies())
        assert np.allclose(from_dense.get_weights(),
                           from_denser.get_weights(),
                           rtol=0.05, atol=0.01)

    linewidths = [1, 5, None]

    @pytest.mark.usefixtures("figure")
    @pytest.mark.parametrize('linewidth', linewidths)
    def test_plot(self, dense_dos, figure, linewidth):
        if linewidth is None:
            mplargs = None
        else:
            mplargs = {'linewidth': linewidth}

        ax = figure.add_subplot(111)
        ax_out = dense_dos.plot(ax=ax, mplargs=mplargs,
                                smearing='Gauss')
        assert ax_out == ax

        line_data = ax.lines[0].get_data()

        # With default settings, plot data should be unchanged from input data;
        # this is a special feature of "grid" data to avoid repeated broadening
        assert np.allclose(line_data[0], np.linspace(0., 10., 11))
        assert np.allclose(line_data[1], np.sin(np.linspace(0., 1., 11)))

    @pytest.mark.usefixtures("figure")
    def test_plot_broad_dos(self, dense_dos, figure):
        # Check that setting a grid/broadening does not blow up and reproduces
        # previous results; this result has not been rigorously checked but at
        # least it should not _change_ unexpectedly
        ax = figure.add_subplot(111)
        _ = dense_dos.plot(ax=ax, npts=10, xmin=0, xmax=9,
                           width=4, smearing='Gauss')
        line_data = ax.lines[0].get_data()
        assert np.allclose(line_data[0], range(10))
        assert np.allclose(line_data[1],
                           [0.14659725, 0.19285644, 0.24345501, 0.29505574,
                            0.34335948, 0.38356488, 0.41104823, 0.42216901,
                            0.41503382, 0.39000808])

    smearing_args = [(dict(npts=0, width=None), (0, None)),
                     (dict(npts=10, width=None, default_width=5.), (10, 5.)),
                     (dict(npts=0, width=0.5, default_npts=100), (100, 0.5)),
                     (dict(npts=10, width=0.5), (10, 0.5))]

    @pytest.mark.parametrize('inputs, expected', smearing_args)
    def test_smearing_args_interpreter(self, inputs, expected):
        assert GridDOSData._interpret_smearing_args(**inputs) == expected


class TestMultiDosData:
    """Test interaction between DOS data objects"""
    @pytest.fixture()
    def sparse_dos(self):
        return RawDOSData([1.2, 3.4, 5.], [3., 2.1, 0.],
                          info={'symbol': 'H', 'number': '1', 'food': 'egg'})

    @pytest.fixture()
    def dense_dos(self):
        x = np.linspace(0., 10., 11)
        y = np.sin(x / 10)
        return GridDOSData(x, y, info={'symbol': 'C', 'orbital': '2s',
                                       'day': 'Tue'})

    def test_addition(self, sparse_dos, dense_dos):
        with pytest.raises(TypeError):
            sparse_dos + dense_dos
        with pytest.raises(TypeError):
            dense_dos + sparse_dos
