from setuptools import setup
import KSIF
setup(
    name='KSIF',
    version=str(KSIF.__version__[0]) +'.'+
            str(KSIF.__version__[1]) +'.'+
            str(KSIF.__version__[2]) +'.'+
            'dev',
    packages=['KSIF', 'KSIF.core', 'KSIF.ML', 'KSIF.test', 'KSIF.validation'],
    url='',
    license='MIT',
    author='KSIF developers',
    author_email='rambor12@business.kaist.ac.kr',
    description='KSIF Library',
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'pyprind',
        'future',
        'cython',
        'tabulate',
        'sklearn',
        'scipy',
        'decorator',
        'pandas_datareader',
        'theano'
    ],
    classifier=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers, Quants',
        'Topic :: Strategy Development :: Build Tools',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.4',

    ]
)
