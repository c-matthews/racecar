<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/c-matthews/racecar#readme">
    <img src="https://raw.githubusercontent.com/c-matthews/racecar/main/img/logo.png" alt="Logo" width="420" height="210">
  </a>

  <p align="center">
    A Python package collecting awesome sampling algorithms.
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

  <a href="https://img.shields.io/readthedocs/racecar?style=plastic">
    <img src="https://readthedocs.org/projects/racecar/badge/?version=latest"
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
- [Features](#features)
- [Examples](#examples)
- [Installation](#installation)
- [License](#license)
- [Contact](#contact)


## Examples

##### Simple example

We will sample points from the distribution <img src="https://latex.codecogs.com/gif.latex?\pi(x)\propto%20\exp(-x^2/12-\cos(2x))" /> using the random walk Metropolis algorithm, and then plot the results.

```python
# Import racecar and numpy
import racecar as rc
import numpy as np

# Define the log likelihood function
def llh(x):
  return {
  'llh' : -( np.cos(2*x) + x**2/12 )
  }

# Create the sampler object and use Random Walk Metropolis
initial_condition = [0]
learning_rate = 0.5
S = rc.sampler(initial_condition, learning_rate, llh, algo="RWMetropolis")

# Sample some points, outputting arrays of position and log likelihood
Pos_traj, LLH_traj = S.sample(100000, output=['pos','llh'])

# Plot the results using matplotlib
```
<img src="https://raw.githubusercontent.com/c-matthews/racecar/main/img/cos_example.png"
  alt="Results" />

##### More examples

Some more detailed examples are given in detailed Jupyter notebooks below

- Example 1
- Example 2
- Example 3

## Installation

You can install the package from source by cloning this repo and using `setup.py`, the only dependencies are on `numpy` and `scipy`. Otherwise it is available on pip via

    pip install racecar

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Made by Charles Matthews - [www.cmatthe.ws](https://www.cmatthe.ws) - mail@cmatthe.ws

Project Link: [https://github.com/c-matthews/racecar](https://github.com/c-matthews/racecar)
