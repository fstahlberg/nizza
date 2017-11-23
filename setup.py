"""Install Nizza."""

from setuptools import find_packages
from setuptools import setup

setup(
    name='nizza',
    version='0.1',
    description='Nizza',
    author='University of Cambridge',
    author_email='fs439@cam.ac.uk',
    url='https://github.com/fstahlberg/nizza',
    license='Apache 2.0',
    packages=find_packages(),
    scripts=[
        'nizza/train.py',
    ],
    install_requires=[
        'future',
        'numpy',
    ],
    extras_require={
        'tensorflow': ['tensorflow>=1.4.0'],
        'tensorflow_gpu': ['tensorflow-gpu>=1.4.0'],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='tensorflow word-alignment',)
