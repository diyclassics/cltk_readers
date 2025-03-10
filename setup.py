from setuptools import setup
from setuptools.command.install import install
import sys
import subprocess
from subprocess import getoutput

setup(
    name="cltk_readers",
    version="0.6.10",
    packages=["cltkreaders"],
    url="https://github.com/diyclassics/cltk_readers",
    license="MIT License",
    readme="README.md",
    author="Patrick J. Burns",
    author_email="patrick@diyclassics.org",
    description="Corpus reader extension for the Classical Language Toolkit ",
    install_requires=[
        "cltk~=1.4.0",
        "lxml==5.3.1",
        "natsort==8.4.0",
        "pyuca==1.2",
        "spacy==3.7.5",
        "la_core_web_lg@https://huggingface.co/latincy/la_core_web_lg/resolve/main/la_core_web_lg-3.7.7-py3-none-any.whl",
        "textacy==0.13.0",
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
)
