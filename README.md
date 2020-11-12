<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/c-matthews/racecar#readme">
    <img src="https://raw.githubusercontent.com/c-matthews/racecar/main/img/logo.png" alt="Logo" width="420" height="210">
  </a>

  <p align="center">
    A Python package collecting awesome MCMC sampling algorithms.
    <br />
    <a href="https://racecar.readthedocs.io/en/latest/"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/c-matthews/racecar#examples">Examples</a>
    <span> | </span>
    <a href="https://github.com/c-matthews/racecar/issues">Report Bug</a>
    <span> | </span>
    <a href="https://github.com/c-matthews/racecar/issues">Request Feature</a>
  </p>
</p>

<div align="center">

  <a href="https://travis-ci.com/c-matthews/racecar">
    <img src="https://img.shields.io/travis/com/c-matthews/racecar?style=plastic"
      alt="Travis.com build stability" />
  </a>

  <a href="https://racecar.readthedocs.io/en/latest/?badge=latest">
    <img src="https://img.shields.io/readthedocs/racecar?style=plastic"
      alt="Docs build stability" />
  </a>

  <a href="https://github.com/c-matthews/racecar">
    <img src="https://img.shields.io/badge/python-v3.6+-blue?style=plastic&logo=python"
      alt="Python version" />
  </a>

  <a href="https://github.com/c-matthews/racecar/blob/main/LICENSE">
    <img src="https://img.shields.io/pypi/l/racecar?style=plastic"
      alt="MIT License" />
  </a>

  <a href="https://pypi.org/project/racecar/">
    <img src="https://img.shields.io/pypi/v/racecar?style=plastic"
      alt="Version" />
  </a>

  <a href="https://pypi.org/project/racecar/">
    <img src="https://img.shields.io/pypi/dm/racecar?style=plastic"
      alt="Downloads" />
  </a>

</div>

---

## Table of Contents
- [Why use Racecar?](#Why-use-Racecar)
- [Examples](#examples)
- [Installation](#installation)
- [License](#license)
- [Contact](#contact)


## Why use Racecar?

- Racecar is a lightweight Python library for sampling distributions in high dimensions using cutting-edge algorithms. It can also be used for rapid prototyping of novel methods and application to large problems.
- Pass a function evaluating the log posterior and/or its gradient, and away you go. Ideal for usage with big data applications, neural networks, regression, mixture modelling, and all sorts of Bayesian inference and sampling problems.
- Easily to use and simple to extend with new methods and use cases.
- Designed for use with stochastic gradients in mind.

<img src="https://raw.githubusercontent.com/c-matthews/racecar/main/img/example_result.png"
  alt="Results from an inference experiment" />

## Examples

##### Quickstart example

We will sample points from the one-dimensional distribution <img src="https://latex.codecogs.com/gif.latex?\pi(x)\propto%20\exp(-x^2/12-\cos(2x))" /> using the random walk Metropolis algorithm, and then plot the results.

```python
# Import racecar and numpy
import racecar as rc
import numpy as np

# Define the target log posterior function
# Note we only need it up to a constant multiple, so we do not need to
# know its normalization constant.
def log_posterior(x):
  return {
  'llh' : -( np.cos(2*x) + x**2/12 )
  }

# Create the sampler object and use Random Walk Metropolis
initial_condition = [0]
learning_rate = 0.5
S = rc.sampler(initial_condition, learning_rate, log_posterior, algo="RWMetropolis")

# Sample some points, outputting arrays of the position and the log posterior
number_of_points = 100000
Pos_traj, LLH_traj = S.sample(number_of_points, output=['pos','llh'])

# Plot the results using matplotlib
```
<img src="https://raw.githubusercontent.com/c-matthews/racecar/main/img/cos_example.png"
  alt="Results" />

##### More examples

Some more detailed examples are given in the Jupyter notebooks below

- <a href="https://github.com/c-matthews/racecar/blob/main/Examples/Gaussian_Data_Example.ipynb">Bayesian inference for the mean of Gaussian data</a>
- <a href="https://github.com/c-matthews/racecar/blob/main/Examples/Bayesian_Logistic_Regression.ipynb">Logistic regression using stochastic gradients</a>

## Installation

You can install the package from source by cloning this repo and using `setup.py`, the only dependencies are on `numpy` and `scipy`. Otherwise it is available on pip via

    pip install racecar

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Made by Charles Matthews - [www.cmatthe.ws](https://www.cmatthe.ws) - mail@cmatthe.ws

Project Link: [https://github.com/c-matthews/racecar](https://github.com/c-matthews/racecar)
