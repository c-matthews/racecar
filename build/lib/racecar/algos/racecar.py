
class RACECAR():

    def __init__(self,np,ic,h,force,params):

        self.h = h
        self.h2 = h/2
        self.Nrk4 = params.get('Nrk4',1)
        self.rk4h = self.h2 / self.Nrk4
        self.np = np

        g = params.get('g',1)
        self.c1 = np.exp(-g*h)
        self.c3 = np.sqrt(1-self.c1*self.c1)

        self.p = np.random.randn( *ic.shape )

        NX = len(ic) * ( 1+2*params.get('Nxi',0) )
        NX = min(NX, len(ic)**2 )

        self.xi = np.random.randn( NX,1 )

        self.force = force

        self.v,self.f, ff = force( ic )

        C = np.abs(np.cov( ff ))
        self.C = C.copy()

        C = C + np.eye(len(ic))*C.max()
        Cval = np.sort( C.flatten())[-NX:].min()
        self.Cx = self.C * (C>=Cval)
        C = (C >= Cval)
        self.cii,self.cjj = np.where(C)
        self.tgt = np.array(self.cii==self.cjj, dtype=int)


    def clear(self,q):
        pass

    def step(self,q):

        h = self.h
        h2 = self.h2

        self.p = self.p + h2 * self.f
        q = q + h2 * self.p

        for n in range( self.Nrk4 ):
            self.RK4()

        self.p = self.p * self.c1 + self.c3 * self.np.random.randn( *self.p.shape )

        for n in range( self.Nrk4 ):
            self.RK4()

        q = q + h2 * self.p
        self.v,self.f,_ = self.force(q)

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
