"""
Euler-Maruyama (SGLD)
=====================

The classic Euler-Maruyama scheme, also known as Stochastic Gradient Langevin Dynamics (SGLD) when a stochastic approximation is used as the gradient.

This scheme uses the gradient of the posterior to move through the configuration space, without Metropolis correction. It produces a biased trajectory (though convergent as the learning rate decreases), but is a good `first-choice` in high dimension for general distributions.

Usage
^^^^^

Set ``algo="sgld"`` in :ref:`sampler`.

Requires
^^^^^^^^

The ``llh`` function needs to output the following dictionary keys:

- ``grad`` :: The (potentially approximate) gradient of the log posterior.

Params
^^^^^^

The behavior of the sampler can be customized by including the following arguments in the sampler's ``params`` dict.

- This scheme takes no additional parameters.

References
^^^^^^^^^^

This scheme is extensively studied, it should be featured heavily in any good statistics textbook.

- `Numerical Solution of Stochastic Differential Equations by Eckhard Platen and Peter Kloeden, Springer (1992)`
- `Welling, Max, and Yee W. Teh. "Bayesian learning via stochastic gradient Langevin dynamics." Proceedings of the 28th international conference on machine learning (ICML-11). 2011` `<https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf>`__

"""
from .algorithm import Algorithm


class SGLD(Algorithm):
    def step(self, q):

        q = q + self.h * self.f + self.sq2h * self.np.random.randn(*q.shape)

        fres = self.force(q)
        self.v, self.f = fres.get("llh"), fres.get("grad")

        return q
