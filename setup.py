#!/usr/bin/env python
from setuptools import setup, find_packages


def requirements(filename='requirements.txt'):
    """Returns a list of requirements to install."""
    requires = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                # skip blank lines and comments
                continue
            requires.append(line)
    return requires

setup(
    name='finisher',
    version="1.0.0",
    url='https://github.com/slobdell/one_rep_max',
    author="Scott Lobdell",
    author_email="scott.lobdell@gmail.com",
    description=("Autocomplete package to train a model and return best results from input tokens"),
    long_description=("Autocomplete package to train a model and return best results from input tokens"),
    keywords="",
    license="",
    packages=find_packages(exclude=[]),
    include_package_data=True,
    install_requires=requirements(),
    extras_require={},
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Topic :: Other/Nonlisted Topic'],
)
