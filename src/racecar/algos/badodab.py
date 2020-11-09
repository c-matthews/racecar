"""
BADODAB
========================

A scheme that adaptively learns a constant damping term required to correct the distribution when using stochastic gradients.

Usage
^^^^^

Set ``algo="badodab"`` in :ref:`sampler`.

Requires
^^^^^^^^

The ``llh`` function needs to output the following dictionary keys:

- ``grad`` :: The gradient of the log posterior.

Params
^^^^^^

The behavior of the sampler can be customized by including the following arguments in the sampler's ``params`` dict.

- ``g`` : positive float
    (Default 1.0) The damping parameter used in the Langevin dynamics.
- ``mu`` : positive float
    (Default 1.0) The precision (reciprocal variance) of the auxillary variables distribution.

References
^^^^^^^^^^

- `Leimkuhler, Benedict, and Xiaocheng Shang. "Adaptive thermostats for noisy gradient systems." SIAM Journal on Scientific Computing 38.2 (2016)`

"""
from .algorithm import Algorithm


class BADODAB(Algorithm):
    def __init__(self, np, ic, h, force, params):
        super().__init__(np, ic, h, force, params)

        self.g = params.get("g", 1)
        self.mu = params.get("mu", 1)

    def step(self, q):

        h = self.h
        h2 = self.h2

        self.p = self.p + h2 * self.f
        q = q + h2 * self.p

        self.xi = self.xi + h2 * (self.np.sum(self.p * self.p) - self.p.size) / self.mu

        c1 = self.np.exp(-self.xi * h)
        c3 = self.np.sqrt(self.np.abs(self.g * (1 - c1 * c1) / (self.xi)))
        self.p = self.p * c1 + c3 * self.np.random.randn(*self.p.shape)

        self.xi = self.xi + h2 * (self.np.sum(self.p * self.p) - self.p.size) / self.mu

        q = q + h2 * self.p

        fres = self.force(q)
        self.v, self.f = fres.get("llh"), fres.get("grad")

        self.p = self.p + h2 * self.f

        return q
