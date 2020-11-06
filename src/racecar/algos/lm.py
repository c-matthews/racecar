
class LM():

    def __init__(self,np,ic,h,force,params):

        self.sqh2 = np.sqrt(h/2)
        self.h = h
        self.np = np
        self.R = np.random.randn( *ic.shape )

        self.force = force

    def step(self, q ):

        fres = self.force(q)
        self.v,f = fres.get('llh'), fres.get('grad') 

        q = q + self.h*f  + self.sqh2*self.R

        self.R = self.np.random.randn( *q.shape )

        q = q + self.sqh2*self.R

        return q

    def clear(self,q):
        pass
