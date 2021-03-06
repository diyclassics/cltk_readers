from setuptools import setup

setup(
    name='cltk_readers',
    version='0.2.4',
    packages=['cltkreaders'],
    url='https://github.com/diyclassics/cltk_readers',
    license='MIT License',
    author='Patrick J. Burns',
    author_email='patrick@diyclassics.org',
    description='Corpus reader extension for the Classical Language Toolkit ',
    install_requires=['cltk~=1.1.1',
                      'lxml==4.7.1', 
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
