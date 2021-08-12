#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import setuptools
from covate import version

requirements = [
    "cycler",
    "kiwisolver",
    "matplotlib",
    "numpy",
    "pandas",
    "patsy",
    "Pillow",
    "pyparsing",
    "python-dateutil",
    "pytz",
    "scipy",
    "six",
    "statsmodels"
]

setuptools.setup(
    name="covate",
    version=version.__version__,
    url="https://github.com/Pathogen-Genomics-Cymru/covate",

    description="Predicts time series for SARS-Cov-2 lineages",
  
    author="Anna Price",
    author_email="PriceA35@cardiff.ac.uk",

    packages=setuptools.find_packages(),
    install_requires=requirements,

    entry_points = {
        'console_scripts': [
            'covate = covate.covate:main',
        ]
    },
)
