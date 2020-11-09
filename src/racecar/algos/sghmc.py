"""
Stochastic Gradient HMC
========================

A biased scheme for use with stochastic gradients. Estimates the current noise term and damps the dynamics to compensate.

This implementation assumes that the gradient is a scaled partial sum over data, with the variance estimation coming from the rescaled covariance of the pieces.

Usage
^^^^^

Set ``algo="sghmc"`` in :ref:`sampler`.

Requires
^^^^^^^^

The ``llh`` function needs to output the following dictionary keys:

- ``grad`` :: The gradient of the log posterior.
- ``grad_data`` :: The gradient of the log likelihood with respect to the current position, itemized over the data batch. Expects a `(N,D)` array, where the position space is `N` dimensional and the batch size is `D`.

Params
^^^^^^

The behavior of the sampler can be customized by including the following arguments in the sampler's ``params`` dict.

- ``g`` : positive float
    (Default 1.0) The damping parameter used in the Langevin dynamics. Must be large enough to dominate the force noise.
- ``datasize`` : positive int
    The total number of data points in the dataset.
- ``batchsize`` : positive int
    The number of datapoints in a single batch.

References
^^^^^^^^^^

- `Chen, Tianqi, Emily Fox, and Carlos Guestrin. "Stochastic gradient hamiltonian monte carlo." International conference on machine learning. (2014)`

"""
from .algorithm import Algorithm


class SGHMC(Algorithm):
    def __init__(self, np, ic, h, force, params):
        super().__init__(np, ic, h, force, params)

        self.g = params.get("g", 1)
        self.datasize = params.get("datasize",1)
        self.batchsize = params.get("batchsize",1)
        self.Bfac = (self.datasize * (self.datasize - self.batchsize) / self.batchsize)

        self.C = self.np.eye( len(ic) ) * self.g

    def step(self, q):

        h = self.h
        h2 = self.h2

        q = q + h * self.p

        fres = self.force(q)
        self.v, self.f, self.ff = fres.get("llh"), fres.get("grad"), fres.get("grad_data")

        self.B = self.np.cov( self.ff ) * self.Bfac
        CminusB = self.C - self.B*h2
        sqCminusB = self.np.linalg.cholesky( CminusB )

        self.p = self.p + h * self.f - h * self.np.dot(self.C, self.p)
        self.p = self.p + self.sq2h*self.np.dot(sqCminusB , self.np.random.randn(*self.p.shape))

        return q
