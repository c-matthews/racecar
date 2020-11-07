from .algorithm import Algorithm


class SGLD(Algorithm):
    def step(self, q):


        q = q + self.h * self.f + self.sq2h * self.np.random.randn(*q.shape)

        fres = self.force(q)
        self.v, self.f = fres.get("llh"), fres.get("grad")

        return q
