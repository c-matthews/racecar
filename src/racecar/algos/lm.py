"""
Leimkuhler-Matthews (LM)
========================

A biased MCMC sampling method that provides significant accuracy improvements over the Euler-Maruyama scheme, without extra gradient evaluations.
Gives `exact sampling` for Gaussian target distributions.

Usage
^^^^^

Set ``algo="LM"`` in :ref:`sampler`.

Requires
^^^^^^^^

The ``llh`` function needs to output the following dictionary keys:

- ``grad`` :: The gradient of the log posterior.

Params
^^^^^^

The behavior of the sampler can be customized by including the following arguments in the sampler's ``params`` dict.

- This scheme takes no additional parameters.

References
^^^^^^^^^^

- `Molecular Dynamics: With Deterministic and Stochastic Numerical Methods by Benedict Leimkuhler and Charles Matthews, Springer (2017)`

"""
from .algorithm import Algorithm


class LM(Algorithm):
    def __init__(self, np, ic, h, force, params):
        super().__init__(np, ic, h, force, params)

        self.R = np.random.randn(*ic.shape)

    def step(self, q):

        q = q + self.h * self.f + self.sqh2 * self.R

        self.R = self.np.random.randn(*q.shape)

        q = q + self.sqh2 * self.R

        fres = self.force(q)
        self.v, self.f = fres.get("llh"), fres.get("grad")

        return q
