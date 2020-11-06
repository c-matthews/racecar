
class HMC():

    def __init__(self,np,ic,h,force,params):

        self.h = h
        self.h2 = h/2
        self.M = params.get('M',5)
        self.force = force
        self.np = np

        fres = self.force(ic)
        self.v,self.f = fres.get('llh'), fres.get('grad')

        self.acc = 0

    def step(self,q):

        p = self.np.random.randn( *q.shape )

        q0 = q.copy()

        V0 = self.v
        K0 = -self.np.sum( p*p )/2

        f = self.f.copy()

        h2 = self.h2
        h = self.h

        for m in range( self.M ):
            p = p + h2*f
            q = q + h*p
            fres = self.force(q)
            V1,f = fres.get('llh'), fres.get('grad')
            p = p + h2*f

        K1 = -self.np.sum( p*p )/2

        dH = (V0-V1) + (K0-K1)

        if (self.np.log( self.np.random.rand() ) < -dH ):
            # Accept
            self.acc += 1
            self.v = V1
            self.f = f.copy()
            return q

        # Not accept
        return q0

    def clear(self,q):
        fres = self.force(q)
        self.v, self.f = fres.get('llh'), fres.get('grad')
        self.acc = 0
