
class BADODAB():

    def __init__(self,np,ic,h,force,params):

        self.h = h
        self.h2 = h/2
        self.np = np

        self.g = params.get('g',1)

        self.p = np.random.randn( *ic.shape )

        self.force = force

        fres = self.force(ic)
        self.v,self.f = fres.get('llh'), fres.get('grad')

        self.xi = np.random.randn( )


    def clear(self,q):
        pass

    def step(self,q):

        h = self.h
        h2 = self.h2

        self.p = self.p + h2 * self.f
        q = q + h2 * self.p

        self.xi = self.xi + h2*( self.np.sum(self.p*self.p) - self.p.size )

        c1 = self.np.exp(-self.xi*h)
        c3 = self.np.sqrt( self.np.abs(self.g*(1-c1*c1) / (self.xi) ) )
        self.p = self.p * c1 + c3 * self.np.random.randn( *self.p.shape )

        self.xi = self.xi + h2*( self.np.sum(self.p*self.p) - self.p.size )

        q = q + h2 * self.p

        fres = self.force(q)
        self.v,self.f = fres.get('llh'), fres.get('grad')

        self.p = self.p + h2 * self.f

        return q
