#!python
# -*- coding:utf-8 -*-
from __future__ import print_function
from setuptools import setup, find_packages

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name="hicmsd",
    version="0.0.1",
    author="Biao Zhang",
    author_email="littlebiao@outlook.com",
    description="Code of HiCMSD",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/Codsir/HiCMSD",
    packages=find_packages(),
    install_requires=[
        "scikit-image >= 0.13.0",
        "visdom >= 0.1.8.5",
        "torch >= 0.4.1",
        "torchvision >= 0.2.1",
        "scikit-learn >= 0.20.0",
        "numpy >= 1.15.4",
        "scipy >= 1.1.0",
        ],
    classifiers=[
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3 :: Only",
        "Intended Audience :: Science/Research",
    ],
    python_requires='>=3.5',
)
