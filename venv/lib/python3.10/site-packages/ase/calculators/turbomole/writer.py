# fmt: off

"""Module containing code to manupulate control file"""
import subprocess


def add_data_group(data_group, string=None, raw=False):
    """write a turbomole data group to control file"""
    if raw:
        data = data_group
    else:
        data = '$' + data_group
        if string:
            data += ' ' + string
        data += '\n'
    with open('control', 'r+') as contr:
        lines = contr.readlines()
        contr.seek(0)
        contr.truncate()
        lines.insert(2, data)
        contr.write(''.join(lines))


def delete_data_group(data_group):
    """delete a turbomole data group from control file"""
    subprocess.run(['kdg', data_group], check=True)
