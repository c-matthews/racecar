"""
Stochastic Gradient HMC
========================

A biased scheme for use with stochastic gradients. Damps the dynamics to compensate for additional noise coming from a stochastic gradient.

The covariance of the stochastic gradient term, denoted `B`, is calculated at every step using a user-supplied function that takes as input the current value of `B` and the `grad_data` terms, if supplied.
For reasons of ergodicity (i.e. to ensure proper mixing behavior in the method) a matrix `C>=B*h/2` must be given, where `h` is the learning rate or step size. In this implementation we use scale the identity matrix scaled by a parameter ``g`` as the `C` matrix.
The large damping parameter required can mean that this scheme can easily become unstable and slow to converge if the gradient noise is large.

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
    (Default 1.0) The damping parameter used in the Langevin dynamics. Must be large enough to dominate the gradient noise.
- ``auto_friction`` : bool
    (Default True) Automatically increase the friction ``g`` if the target covariance matrix is not positive-definite.
- ``B`` : numpy array
    (Default 0) The initial value to use as the damping matrix for the stochastic gradient.
- ``grad_cov`` : function
    A function estimating the covariance of the stochastic gradient, with signature `grad_cov(B, grad_data)`.
    The current covariance estimate `B` and the `grad_data` term are supplied to the function, and it outputs an `(N,N)` numpy array as the new `B` matrix.
    If a function is not specified, the default behavior is to use the last estimate of ``B`` without any change.

References
^^^^^^^^^^

- `Chen, Tianqi, Emily Fox, and Carlos Guestrin. "Stochastic gradient hamiltonian monte carlo." International conference on machine learning. (2014)`

"""
from .algorithm import Algorithm


class SGHMC(Algorithm):
    def __init__(self, np, ic, h, force, params):
        super().__init__(np, ic, h, force, params)

        self.g = params.get("g", 1)
        self.auto_friction = params.get('auto_friction', True)
        self.grad_cov = params.get('grad_cov', self.zero_cov)
        self.B = params.get('B', 0*self.np.eye( len(ic) ) )

        self.C = self.np.eye( len(ic) ) * self.g

    def step(self, q):

        h = self.h
        h2 = self.h2

        q = q + h * self.p

        fres = self.force(q)
        self.v, self.f, self.ff = fres.get("llh"), fres.get("grad"), fres.get("grad_data")

        self.B = self.grad_cov(B=self.B, grad_data=self.ff)

        if (self.auto_friction):
            success = False
            for r in range(20):
                try:
                    CminusB = self.C - self.B*h2
                    sqCminusB = self.np.linalg.cholesky( CminusB )
                    success = True
                    break
                except self.np.linalg.LinAlgError:
                    self.g *= 1.5
                    self.C *= 1.5
            if (not success):
                raise self.np.linalg.LinAlgError
        else:
            CminusB = self.C - self.B*h2
            sqCminusB = self.np.linalg.cholesky( CminusB )

        self.p = self.p + h * self.f - h * self.np.dot(self.C, self.p)
        self.p = self.p + self.sq2h*self.np.dot(sqCminusB , self.np.random.randn(*self.p.shape))

        return q

    def zero_cov(self,B,grad_data):

        return B
