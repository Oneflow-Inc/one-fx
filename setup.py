#!/usr/bin/env python
import os
from setuptools import setup, find_packages
import subprocess

version = "0.0.2"
package_name = "onefx"
cwd = os.path.dirname(os.path.abspath(__file__))

sha = "Unknown"
try:
    sha = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
        .decode("ascii")
        .strip()
    )
except Exception:
    pass

requirements = []

readme = open("README.md").read()

def write_version_file():
    version_path = os.path.join(cwd, "onefx", "version.py")
    with open(version_path, "w") as f:
        f.write(f"__version__ = '{version}'\n")
        f.write(f"git_version = {repr(sha)}\n")

if __name__ == "__main__":
    print(f"Building wheel {package_name}-{version}")

    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

    write_version_file()

    setup(
        # Metadata
        name=package_name,
        version=version,
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