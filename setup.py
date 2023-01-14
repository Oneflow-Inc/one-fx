#!/usr/bin/env python
import os, sys
import shutil
import datetime

from setuptools import setup, find_packages
from setuptools.command.install import install

readme = open("README.md").read()

VERSION = "0.0.1"

requirements = [
    # "oneflow",
]

# import subprocess
# commit_hash = subprocess.check_output("git rev-parse HEAD", shell=True).decode('UTF-8').rstrip()
# VERSION += "_" + str(int(commit_hash, 16))[:8]
VERSION += "_" + datetime.datetime.now().strftime("%Y%m%d%H%M")[2:]

setup(
    # Metadata
    name="one-fx",
    version=VERSION,
    author="OneFlow",
    author_email="",
    url="",
    description="A toolkit for developers to simplify the transformation of nn.Module instances.",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="Apache 2.0",
    # Package info
    packages=find_packages(),
    zip_safe=True,
    install_requires=requirements,
    # Classifiers
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)