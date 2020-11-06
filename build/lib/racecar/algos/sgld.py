
class SGLD():

    def __init__(self,np,ic,h,force,params):

        self.sq2h = np.sqrt(2*h)
        self.h = h

        self.np = np

        self.force = force

    def step(self, q ):

        fres = self.force(q)
        self.v,f = fres.get('llh'), fres.get('grad') 

        q = q + self.h*f  + self.sq2h*self.np.random.randn( *q.shape )

        return q

    def clear(self,q):
        pass