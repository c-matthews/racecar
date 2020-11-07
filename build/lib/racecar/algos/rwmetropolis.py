from .algorithm import Algorithm


class RWMETROPOLIS(Algorithm):
    def step(self, q):

        p = self.np.random.randn(*q.shape)

        q0 = q.copy()

        V0 = self.v

        h = self.h

        q = q + h * p
        fres = self.force(q)
        V1, F = fres.get("llh"), fres.get("grad")

        dH = V0 - V1

        if self.np.log(self.np.random.rand()) < -dH:
            # Accept
            self.acc += 1
            self.v = V1
            self.f = F
            return q

        # Not accept
        return q0
