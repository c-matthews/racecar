class Algorithm:
    def __init__(self, np, ic, h, force, params):

        self.np = np
        self.h = h
        self.sq2h = np.sqrt(2 * h)
        self.sqh2 = np.sqrt(h / 2)
        self.h2 = h / 2
        self.force = force
        self.acc = 0

        fres = self.force(ic)
        self.v, self.f, self.ff = (
            fres.get("llh"),
            fres.get("grad"),
            fres.get("grad_data"),
        )

        self.p = np.random.randn(*ic.shape)
        self.xi = np.random.randn(1)

    def clear(self, q):
        self.acc = 0

        fres = self.force(q)
        self.v, self.f, self.ff = (
            fres.get("llh"),
            fres.get("grad"),
            fres.get("grad_data"),
        )
        pass

    def step(self, q):
        pass
