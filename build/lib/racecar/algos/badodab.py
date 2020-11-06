
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


    def RK4(self):

        h = self.rk4h

        h2 = h / 2
        p = self.p
        xi = self.xi

        K1p, K1xi = self.df(p,xi)
        K2p, K2xi = self.df(p + h2*K1p,xi + h2*K1xi)
        K3p, K3xi = self.df(p + h2*K2p,xi + h2*K2xi)
        K4p, K4xi = self.df(p + h*K3p,xi + h*K3xi)

        pp = p + (K1p + 2*K2p + 2*K3p + K4p)*h/6
        xx = xi + (K1xi + 2*K2xi + 2*K3xi + K4xi)*h/6

        self.p = pp
        self.xi = xx


    def df(self,p,xi):

        dp = p.copy()
        self.np.add.at( dp , self.cii , -xi*( p.copy()[self.cjj] ) )
        dp = dp - p

        dxi = (p[self.cii]*p[self.cjj] - self.tgt.reshape(xi.shape) )

        return dp, dxi
