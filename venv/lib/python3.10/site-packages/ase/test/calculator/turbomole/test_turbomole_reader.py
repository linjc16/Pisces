"""unit tests for the turbomole reader module"""

import pytest

from ase.calculators.turbomole.reader import parse_data_group


def test_parse_data_group():
    """test the parse_data_group() function in the turbomole reader module"""
    assert parse_data_group('', 'empty') is None
    assert parse_data_group('$name', 'name') == ''
    assert parse_data_group('$name val', 'name') == 'val'

    dgr_dct_s = {'start': '2.000'}
    dgr_dct_l = {'start': '2.000', 'step': '0.500', 'min': '1.000'}

    dgr = '$scfdamp start=  2.000'
    assert parse_data_group(dgr, 'scfdamp') == dgr_dct_s
    dgr = '$scfdamp  start=  2.000  step =  0.500  min   = 1.000'
    assert parse_data_group(dgr, 'scfdamp') == dgr_dct_l

    dgr = '$scfdamp\n  start  2.000'
    assert parse_data_group(dgr, 'scfdamp') == dgr_dct_s
    dgr = '$scfdamp\n  start  2.000\n  step  0.500\n  min 1.000'
    assert parse_data_group(dgr, 'scfdamp') == dgr_dct_l

    dgr = '$scfdamp\n  start = 2.000'
    assert parse_data_group(dgr, 'scfdamp') == dgr_dct_s
    dgr = '$scfdamp\n  start = 2.000\n  step =0.500 \n  min =  1.000'
    assert parse_data_group(dgr, 'scfdamp') == dgr_dct_l

    dgr = '$two_flags\n flag1\n flag2'
    dgr_dct = {'flag1': True, 'flag2': True}
    assert parse_data_group(dgr, 'two_flags') == dgr_dct

    dgr = '$one_pair\n key val'
    dgr_dct = {'key': 'val'}
    assert parse_data_group(dgr, 'one_pair') == dgr_dct

    dgr_dct = {'fl_1': True, 'key_1': 'val_1', 'key_2': 'val_2', 'fl_2': True}
    dgr = '$one_line_mix fl_1 key_1 = val_1  fl_2 key_2 = val_2'
    assert parse_data_group(dgr, 'one_line_mix') == dgr_dct
    dgr = '$multi_line_eq\n fl_1\n key_1 = val_1\n  fl_2 \nkey_2 = val_2'
    assert parse_data_group(dgr, 'multi_line_eq') == dgr_dct
    dgr = '$multi_line_sp\n fl_1\n key_1  val_1\n  fl_2\n key_2  val_2'
    assert parse_data_group(dgr, 'multi_line_sp') == dgr_dct

    dgr = '$interconversion  off\n   qconv=1.d-7\n   maxiter=25'
    dgr_dct = {'off': True, 'qconv': '1.d-7', 'maxiter': '25'}
    assert parse_data_group(dgr, 'interconversion') == dgr_dct

    dgr = ('$coordinateupdate\n   dqmax=0.3\n   interpolate  on\n'
           '   statistics    5')
    dgr_dct = {'dqmax': '0.3', 'interpolate': 'on', 'statistics': '5'}
    assert parse_data_group(dgr, 'coordinateupdate') == dgr_dct

    msg = r'data group does not start with \$empty'
    with pytest.raises(ValueError, match=msg):
        parse_data_group('$other', 'empty')
