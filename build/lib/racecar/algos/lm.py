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
