"""
KSIF (KAIST Student Investement Fund) Package for Python
========================================================

This package includes modules to help investment activities such as backtest, validation.
This package is based on 'bt' and 'ffn'.
"""

from .core import base, algos, backtest, ffn, data, utils

from .core.backtest import Backtest, run
from .core.base import Strategy, Algo, AlgoStack
from .core.algos import run_always
from .core.ffn import utils, merge
from .core.data import get
from .core.utils import *

from .ML import RNN

from .test import generate

from .validation.CSCV import CSCV

core.ffn.extend_pandas()

__version__ = (0, 1, 1)
__author__ = 'Seung Hyeon Yu'
__email__ = 'rambor12@business.kaist.ac.kr'