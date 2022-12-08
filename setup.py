from setuptools import setup

setup(
    name='cltk_readers',
    version='0.4.3',
    packages=['cltkreaders'],
    url='https://github.com/diyclassics/cltk_readers',
    license='MIT License',
    author='Patrick J. Burns',
    author_email='patrick@diyclassics.org',
    description='Corpus reader extension for the Classical Language Toolkit ',
    install_requires=['cltk~=1.1.5',
                      'lxml==4.9.1', 
                      'pyuca==1.2',
                      'la-core-cltk-sm @ https://github.com/diyclassics/latin-spacy-models/blob/main/la_core_cltk_sm/la_core_cltk_sm-0.1.0.tar.gz?raw=true',
                      'spacy==3.4.2',
    ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
    ],
)
