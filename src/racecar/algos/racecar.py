from .algorithm import Algorithm


class RACECAR(Algorithm):
    def __init__(self, np, ic, h, force, params):

        super().__init__(np, ic, h, force, params)

        self.g = params.get("g", 1)
        self.mu = params.get("mu", 100)
        ff = self.ff

        Ndim = ic.size

        self.use_basis = False
        if params.get("basis") is not None:
            self.B = params.get("basis")
            self.use_basis = True
        else:
            if (ff is not None) and (params.get("estimate_basis", True)):
                assert ff.shape[0] == Ndim
                evals, self.B = np.linalg.eig((1e-3) * np.eye(Ndim) + np.cov(ff))
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

        xx = xx + (pp * pp - 1) * h2 / self.mu
        pp = pp * self.np.exp(-h * xx)
        xx = xx + (pp * pp - 1) * h2 / self.mu

        return pp, xx

    def step_E(self):

        c1 = self.np.exp(-self.h * self.g)
        c3 = self.np.sqrt(1 - c1 * c1)
        self.p = c1 * self.p + c3 * self.np.random.randn(*self.p.shape)

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
