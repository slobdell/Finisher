#!/usr/bin/env python
from setuptools import setup, find_packages


setup(
    name='finisher',
    version="0.1.11",
    url='https://github.com/slobdell/Finisher',
    author="Scott Lobdell",
    author_email="scott.lobdell@gmail.com",
    description=("Autocomplete package to train a model and return best results from input tokens"),
    long_description=("Autocomplete package to train a model and return best results from input tokens"),
    keywords="python autocomplete spellcheck",
    license="MIT",
    packages=find_packages(exclude=[]),
    include_package_data=True,
    install_requires=["redis"],
    extras_require={},
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Topic :: Other/Nonlisted Topic'],
)
