from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='fairseq',
    ext_modules=[],
    cmdclass={
        'build_ext': BuildExtension
    }
)

