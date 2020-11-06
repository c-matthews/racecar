# racecar

A lightweight Python 3.6+ package for various sampling algorithms, particularly useful for Bayesian inference and big data.

[![Python ](https://img.shields.io/badge/python-v3.6+-blue?style=plastic&logo=python)]
[![Travis (.com)](https://img.shields.io/travis/com/c-matthews/racecar?style=plastic)](https://travis-ci.com/c-matthews/racecar)
[![PyPI - License](https://img.shields.io/pypi/l/racecar?color=yellow&style=plastic)](https://github.com/c-matthews/racecar/blob/main/LICENSE)
[![PyPI version shields.io](https://img.shields.io/pypi/v/racecar?color=blue&style=plastic)](https://pypi.org/project/racecar/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/racecar?color=9cf&style=plastic)](https://pypi.org/project/racecar/)


### Installation

You can install the package from source by cloning this repo. Otherwise it is available on pip via

    pip install racecar

### Examples

    import racecar as rc

    initial_condition = [0,0]
    learning_rate = 0.1
    llh = rc.llh.isotropic_gaussian

    # Set up the sampler
    S = rc.sampler(initial_condition, learning_rate, llh, algo='hmc')

    # Begin sampling! Try 1000 steps
    Q = S.sample(1000)
