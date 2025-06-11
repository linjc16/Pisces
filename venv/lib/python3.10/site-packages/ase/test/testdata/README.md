# Purpose

In this folder, we store the reference files (input/output/data) which ASE is capable to read.
These files are used in the test suite.

# Use

In tests, the folder is accessible via a fixture `datadir`. The content of the folder is
also preserved when installing via setuptools (`setup.py`). The mask for the file
inclusion into the package allows preserving two sub folders with files below `testdata`.
