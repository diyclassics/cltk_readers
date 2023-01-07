from setuptools import setup
from setuptools.command.install import install
import sys
import subprocess
from subprocess import getoutput


# cf. https://github.com/BaderLab/saber/issues/35#issuecomment-467827175
class PostInstall(install):
    pkgs = " https://huggingface.co/diyclassics/la_dep_cltk_sm/resolve/main/la_dep_cltk_sm-0.2.0/dist/la_dep_cltk_sm-0.2.0.tar.gz"

    def run(self):
        install.run(self)
        print(getoutput("pip install" + self.pkgs))
        # https://pip.pypa.io/en/stable/user_guide/#using-pip-from-your-program
        subprocess.call([sys.executable, "-m", "pip", "install", self.pkgs])


setup(
    name="cltk_readers",
    version="0.5.2",
    packages=["cltkreaders"],
    url="https://github.com/diyclassics/cltk_readers",
    license="MIT License",
    readme="README.md",
    author="Patrick J. Burns",
    author_email="patrick@diyclassics.org",
    description="Corpus reader extension for the Classical Language Toolkit ",
    cmdclass={"install": PostInstall},
    install_requires=[
        "cltk~=1.1.5",
        "lxml==4.9.1",
        "pyuca==1.2",
        "spacy==3.4.2",
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
)
