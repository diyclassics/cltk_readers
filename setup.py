from setuptools import setup

setup(
    name='cltk_readers',
<<<<<<< HEAD
    version='0.2.1',
=======
    version='0.2.0',
>>>>>>> 3bdd00f... Update version; update readme, setup
    packages=['cltkreaders'],
    url='https://github.com/diyclassics/cltk_readers',
    license='MIT License',
    author='Patrick J. Burns',
    author_email='patrick@diyclassics.org',
    description='Corpus reader extension for the Classical Language Toolkit ',
    install_requires=['cltk~=1.0.22',
                      'pyuca==1.2',
    ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
    ],
)
