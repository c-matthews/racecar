
class BAOAB():

    def __init__(self,np,ic,h,force,params):

        self.h = h
        self.h2 = h/2
        self.np = np

        self.g = params.get('g',1)

        self.p = np.random.randn( *ic.shape )

        self.force = force

        fres = self.force(ic)
        self.v,self.f = fres.get('llh'), fres.get('grad')


    def clear(self,q):
        pass

    def step(self,q):

        h = self.h
        h2 = self.h2

        self.p = self.p + h2 * self.f
        q = q + h2 * self.p

        c1 = self.np.exp(-self.g*h)
        c3 = self.np.sqrt( 1-c1*c1 )
        self.p = self.p * c1 + c3 * self.np.random.randn( *self.p.shape )

        q = q + h2 * self.p

        fres = self.force(q)
        self.v,self.f = fres.get('llh'), fres.get('grad')

        self.p = self.p + h2 * self.f

        return q
