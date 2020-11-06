
class RWMETROPOLIS():

    def __init__(self,np,ic,h,force,params):

        self.h = h
        self.force = force
        self.np = np

        fres = self.force(ic)
        self.v  = fres.get('llh')
        self.acc = 0

    def step(self,q):

        p = self.np.random.randn( *q.shape )

        q0 = q.copy()

        V0 = self.v

        h = self.h

        q = q + h*p
        fres = self.force(q)
        V1 = fres.get('llh')


        dH = (V0-V1)

        if (self.np.log( self.np.random.rand() ) < -dH ):
            # Accept
            self.acc += 1
            self.v = V1 
            return q

        # Not accept
        return q0

    def clear(self,q):
        fres = self.force(q)
        self.v  = fres.get('llh')
        self.acc = 0
