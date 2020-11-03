from __future__ import absolute_import
import numpy as np
import time

from .algos.hmc import HMC
from .algos.lm import LM
from .algos.racecar import RACECAR
from .algos.sgld import SGLD

class Sampler():

    def __init__(self,ic, h, force, algo='racecar',params={} ):

        self.timetaken = 0

        ic = np.array(ic).reshape(-1,1)
        self.q = np.copy(ic)

        if (params.get('seed') is not None):
            np.random.seed( params.get('seed') )

        if (algo.lower()=='sgld'):
            self.ig = SGLD(np,ic,h,force,params)

        if (algo.lower()=='lm'):
            self.ig = LM(np,ic,h,force,params)

        if (algo.lower()=='hmc'):
            self.ig = HMC(np,ic,h,force,params)

        if (algo.lower()=='racecar'):
            self.ig = RACECAR(np,ic,h,force,params)


    def sample(self, Nsteps, printnum=None, thinning=1):

        Qk = 0
        NQ = 1+Nsteps//thinning
        Q = np.zeros( [NQ , len(self.q.flatten() ) ] )

        self.ig.acc = 0
        self.ig.clear(self.q)

        stime = time.time()

        if (printnum is not None):
            pf = Nsteps // printnum
        else:
            pf = Nsteps+2

        for n in range(Nsteps):

            self.q = self.ig.step( self.q )

            if (n%thinning==0) and (Qk<NQ):
                Q[Qk,:] = self.q.copy().flatten()
                Qk += 1

            if (np.isnan(self.q).any() ):
                break

            if (n+1)%pf==0:
                print('Steps:',n+1 ,'  Time:',time.time() - stime ,'  V:', self.ig.v)

        self.timetaken = time.time() - stime
        Q = Q[:Qk,:]

        return Q
