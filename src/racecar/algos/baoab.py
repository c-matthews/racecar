"""
BAOAB
========================

A biased Langevin sampling routine that incorporates momentum to expedite sampling over barriers. A damping parameter can be chosen to customize the mixing with the heat bath.
In the limit of infinite damping this method becomes the LM scheme. Similarly, the BAOAB method is exact for Gaussian distributions.


Usage
^^^^^

Set ``algo="baoab"`` in :ref:`sampler`.

Requires
^^^^^^^^

The ``llh`` function needs to output the following dictionary keys:

- ``grad`` :: The gradient of the log posterior.

Params
^^^^^^

The behavior of the sampler can be customized by including the following arguments in the sampler's ``params`` dict.

- ``g`` : positive float
    (Default 1.0) The isotropic friction constant (`gamma`) used in the Langevin dynamics.

References
^^^^^^^^^^

- `Molecular Dynamics: With Deterministic and Stochastic Numerical Methods by Benedict Leimkuhler and Charles Matthews, Springer (2017)`

"""
from .algorithm import Algorithm


class BAOAB(Algorithm):
    def __init__(self, np, ic, h, force, params):
        super().__init__(np, ic, h, force, params)

        self.g = params.get("g", 1)

    def step(self, q):

        h = self.h
        h2 = self.h2

        self.p = self.p + h2 * self.f
        q = q + h2 * self.p

        c1 = self.np.exp(-self.g * h)
        c3 = self.np.sqrt(1 - c1 * c1)
        self.p = self.p * c1 + c3 * self.np.random.randn(*self.p.shape)

        q = q + h2 * self.p

        fres = self.force(q)
        self.v, self.f = fres.get("llh"), fres.get("grad")

        self.p = self.p + h2 * self.f

        return q
