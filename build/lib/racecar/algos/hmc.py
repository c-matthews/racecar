"""
Hybrid Monte Carlo (HMC)
========================

Hybrid (or Hamiltonian) Monte Carlo takes steps of constant-energy dynamics after redrawing the kinetic energy. The symplectic scheme used for the constant-energy integration allows for very favorable behavior with dimension, in spite of the Metropolization step.

.. note::
    The acceptance rate for the scheme can be checked by calling the ``acceptance_rate`` function on the sampler. A good choice is to choose a learning rate so that acceptance is between 0.7 and 0.8.

Usage
^^^^^

Set ``algo="hmc"`` in :ref:`sampler`.

Requires
^^^^^^^^

The ``llh`` function needs to output the following dictionary keys:

- ``llh`` :: The log posterior value.
- ``grad`` :: The gradient of the log posterior.

Params
^^^^^^

The behavior of the sampler can be customized by including the following arguments in the sampler's ``params`` dict.

- ``T`` : positive int
    (Default 5) The number of constant-energy substeps to take between Metropolis tests.

References
^^^^^^^^^^

- `Hybrid Monte Carlo; Simon Duane, A.D. Kennedy, Brian J. Pendleton, Duncan Roweth (1987)` `<https://doi.org/10.1016/0370-2693(87)91197-X>`__

"""
from .algorithm import Algorithm


class HMC(Algorithm):
    def __init__(self, np, ic, h, force, params):
        super().__init__(np, ic, h, force, params)

        self.T = params.get("T", 5)

    def step(self, q):

        p = self.np.random.randn(*q.shape)

        q0 = q.copy()

        V0 = self.v
        K0 = -self.np.sum(p * p) / 2

        f = self.f.copy()

        h2 = self.h2
        h = self.h

        for m in range(self.T):
            p = p + h2 * f
            q = q + h * p
            fres = self.force(q)
            V1, f = fres.get("llh"), fres.get("grad")
            p = p + h2 * f

        K1 = -self.np.sum(p * p) / 2

        dH = (V0 - V1) + (K0 - K1)

        if self.np.log(self.np.random.rand()) < -dH:
            # Accept
            self.acc += 1
            self.v = V1
            self.f = f.copy()
            self.p = p.copy()
            return q

        # Not accept
        return q0
