"""
Random Walk Metropolis
======================

One of the simplest and widely used MCMC sampling algorithms, the Metropolis algorithm proposes a new point using an isotropic Gaussian distribution and accepts with a probability preserving the detailed balance condition.

The Metropolis condition ensures this method is unbiased, but can be extremely slow to converge (particularly in high dimension).

.. note::
    The acceptance rate for the scheme can be checked by calling the ``acceptance_rate`` function on the sampler. A good choice is to choose a learning rate so that acceptance is around 0.234.

Usage
^^^^^

Set ``algo="RWMetropolis"`` in :ref:`sampler`.

Requires
^^^^^^^^

The ``llh`` function needs to output the following dictionary keys:

- ``llh`` :: The value of the log posterior.

Params
^^^^^^

The behavior of the sampler can be customized by including the following arguments in the sampler's ``params`` dict.

- This scheme takes no additional parameters.
"""
from .algorithm import Algorithm


class RWMETROPOLIS(Algorithm):
    def step(self, q):

        p = self.np.random.randn(*q.shape)

        q0 = q.copy()

        V0 = self.v

        h = self.h

        q = q + h * p
        fres = self.force(q)
        V1, F = fres.get("llh"), fres.get("grad")

        dH = V0 - V1

        if self.np.log(self.np.random.rand()) < -dH:
            # Accept
            self.acc += 1
            self.v = V1
            self.f = F
            return q

        # Not accept
        return q0
