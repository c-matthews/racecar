"""Racecar

A lightweight Python sampling package.

See https://github.com/c-matthews/racecar for information and examples.

"""
from __future__ import absolute_import

# Set explicit packages
from .sampler import Sampler as sampler
from .llh import blr, isotropic_gaussian
from ._version import version_number

__author__ = "Charles Matthews"
__license__ = "MIT"
__maintainer__ = "Charles Matthews"
__email__ = "mail@cmatthe.ws"
__version__ = version_number
