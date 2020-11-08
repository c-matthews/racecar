Examples & Quickstart
======================

After installing ``racecar`` and importing the package, we can build a new sampler object and then call the ``sample`` routine on it to sample from the target distribution.
The internal state of the integrator is kept within the sampler object, and only the required points are returned.

Below we give a quick example sampling a one-dimensional distribution.

.. code-block:: python

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

.. image:: https://raw.githubusercontent.com/c-matthews/racecar/main/img/cos_example.png
  :width: 800
  :align: center
  :alt: Results

For other examples, take a look at `examples on Github <https://github.com/c-matthews/racecar#examples>`_.
