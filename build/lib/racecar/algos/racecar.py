"""
Racecar
========================

A general purpose sampling algorithm, specifically designed for efficient sampling in systems with a stochastic gradient.

Usage
^^^^^

Set ``algo="racecar"`` in :ref:`sampler`.

Requires
^^^^^^^^

The ``llh`` function needs to output the following dictionary keys:

- ``grad`` :: The gradient of the log posterior.
- ``grad_data`` :: (optional) The gradient of the log likelihood with respect to the current position, itemized over the data batch. Expects a `(N,D)` array, where the position space is `N` dimensional and the batch size is `D`.

Params
^^^^^^

The behavior of the sampler can be customized by including the following arguments in the sampler's ``params`` dict.

- ``g`` : positive float
    (Default 1.0) The damping parameter used in the Langevin dynamics. Must be large enough to dominate the force noise.
- ``mu`` : positive float
    (Default 1.0) The precision (reciprocal variance) of the auxillary variables distribution.
- ``estimate_basis`` : bool
    (Default `True`) If set to true and ``grad_data`` is given, then the method will approximate the dominant directions for the gradient noise.
- ``estimate_time`` : float
    (Default 0.5) The time in seconds to spend initially estimating the basis vectors by generating new grad-data. By default it will spend 0.5s building this at setup. Set it to 0 to just use the initial `grad-data` for the guess.
- ``basis_size`` : positive int
    (Default `Ndim`) The number of basis vectors to use. A smaller number can improve efficiency in high dimensions, but will remove potential accuracy.
- ``basis`` : `NxK` numpy array
    (optional) The rank-K array of basis vectors to use for the damping of the noisy gradient.

.. note::
    The `Racecar` scheme damps the system in directions specified by basis vectors.
    If a ``basis`` is not given, and ``estimate_basis`` is True, then these vectors are estimated from the ``grad_data`` value if it is outputted from the ``llh`` function. If there is no ``grad_data``, then it will the standard (diagonal) basis instead.

"""
from .algorithm import Algorithm
import time


class RACECAR(Algorithm):
    def __init__(self, np, ic, h, force, params):

        super().__init__(np, ic, h, force, params)

        self.g = params.get("g", 1)
        self.mu = params.get("mu", 1)
        self.c1 = self.np.exp(-self.h * self.g)
        self.c3 = self.np.sqrt(1 - self.c1 * self.c1)
        ff = self.ff

        Ndim = ic.size

        self.use_basis = False
        if params.get("basis") is not None:
            self.B = params.get("basis")
            self.use_basis = True
        else:
            if (ff is not None) and (params.get("estimate_basis", True)):
                assert ff.shape[0] == Ndim
                my_grad_data = [ff]
                stime = time.time()
                while (time.time()-stime<=params.get("estimate_time",0.5)):
                    my_grad_data.append( self.force(ic)['grad_data'] )
                my_grad_data = np.hstack(my_grad_data)
                evals, self.B = np.linalg.eig((1e-6) * np.eye(Ndim) + np.cov(my_grad_data))
                evals = evals.real
                self.B = self.B.real

                sz = params.get("basis_size")
                if sz is not None:
                    self.B = self.B[:, self.np.argsort(evals)[-sz:]]
                self.use_basis = True

        if self.use_basis:
            self.xi = (1 / self.mu) * np.random.randn(self.B.shape[1], 1)
        else:
            self.xi = (1 / self.mu) * np.random.randn(Ndim, 1)

    def step_C(self):

        dh = [1.351207191959657, -1.702414383919315, 1.351207191959657]

        if self.use_basis:
            Bp = self.np.dot(self.B.T, self.p)
        else:
            Bp = self.p

        xi = self.xi

        Bp, xi = self.leapfrog_C(Bp, xi, dh[0] * self.h2)
        Bp, xi = self.leapfrog_C(Bp, xi, dh[1] * self.h2)
        Bp, xi = self.leapfrog_C(Bp, xi, dh[2] * self.h2)

        self.xi = xi

        if self.use_basis:
            self.p = self.p - self.np.linalg.multi_dot([self.B, self.B.T, self.p])
            self.p = self.p + self.np.dot(self.B, Bp)
        else:
            self.p = Bp

        return

    def leapfrog_C(self, pp, xx, h):

        h2 = h / 2

        xx = xx + (pp * pp - 1) * (h2 / self.mu)
        pp = pp * self.np.exp(-h * xx)
        xx = xx + (pp * pp - 1) * (h2 / self.mu)

        return pp, xx

    def step_E(self):

        self.p = self.c1 * self.p + self.c3 * self.np.random.randn(*self.p.shape)

        return

    def step(self, q):

        h = self.h
        h2 = self.h2

        # R
        self.p = self.p + h2 * self.f

        # A
        q = q + h2 * self.p

        # C
        self.step_C()

        # E
        self.step_E()

        # C
        self.step_C()

        # A
        q = q + h2 * self.p

        fres = self.force(q)
        self.v, self.f = fres.get("llh"), fres.get("grad")

        # R
        self.p = self.p + h2 * self.f

        return q
