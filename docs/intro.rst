Introduction
============

``racecar`` is a high-level Python package which aims to provide an easy and intuitive way of sampling points from high dimensional distributions. The package allows you to compare the output between different algorithms in a simple way with low overhead.

The aim here was to create a unified framework to test novel algorithms on a set of standard problems in machine learning in a way that's easily extendable.

The current implementation has been developed in Python 3 and tested on a variety of machines, requiring only ``numpy`` as a dependency.

Motivation
**********

As an applied mathematician/statistician, I wanted to build a simple package for prototyping new algorithms and comparing the results to existing methods. The code should be scalable enough to run on HPC resources, but simple enough to be easily extendable by other users.

This Python package is intended to provide a quick, as well as (hopefully) easy to understand, way of sampling points from a distribution given the distribution function and/or its gradient.

Limitations
***********

- Algorithms that utilize multiple parallel copies of the same system are not supported out-of-the-box.

- Pre-conditioning (such as using a mass matrix) is not currently supported.

- Checkpoint-restart is not an automatic feature of the library.
