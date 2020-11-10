"""
:: racecar sampler class

A class containing the state information for the current sampler object.

"""

from __future__ import absolute_import
import time
import numpy as np

from .algos.hmc import HMC
from .algos.lm import LM
from .algos.racecar import RACECAR
from .algos.sgld import SGLD
from .algos.badodab import BADODAB
from .algos.baoab import BAOAB
from .algos.rwmetropolis import RWMETROPOLIS
from .algos.sghmc import SGHMC


class Sampler:
    """
    A class containing the state information for the current sampler object.

    Parameters
    ----------
    ic : numpy array
        initial condition for the sampler
    h : float, positive
        the step size or learning rate to use in the sampler
    llh : function
        The log posterior function to use for the sampler. The function should take the position as input, and output a dictionary.
        The information required is algorithm-dependent. The algorithms make use of keys:

        - `"llh"` (float) : The value of the posterior function at any point
        - `"grad"` (numpy array) : The gradient of the posterior
        - `"grad_data"` (numpy array) : The gradients of the posterior for each data point

    algo : string, optional
        The algorithm to use for the sampler. See the :ref:`algorithms` page  for a list of possible methods supported by the package. Uses the `racecar` algorithm as default.
    params : dict, optional
        parameters
    """

    def __init__(self, ic, h, llh, algo="racecar", params={}):
        """
        Constructs all the necessary attributes for the sampler object.
        """

        self.timetaken_value = 0

        ic = np.array(ic).reshape(-1, 1)
        self.q = np.copy(ic)

        if params.get("seed") is not None:
            np.random.seed(params.get("seed"))

        args = (np, ic, h, llh, params)

        if algo.lower() == "sgld":
            self.ig = SGLD(*args)

        if algo.lower() == "lm":
            self.ig = LM(*args)

        if algo.lower() == "hmc":
            self.ig = HMC(*args)

        if algo.lower() == "racecar":
            self.ig = RACECAR(*args)

        if algo.lower() == "badodab":
            self.ig = BADODAB(*args)

        if algo.lower() == "baoab":
            self.ig = BAOAB(*args)

        if algo.lower() == "rwmetropolis":
            self.ig = RWMETROPOLIS(*args)

        if algo.lower() == "sghmc":
            self.ig = SGHMC(*args)

    def sample(self, Nsteps, printnum=None, thin=1, output=["pos"]):
        """
        Runs the sampler and returns a trajectory (if required).

        Parameters
        ----------
        Nsteps : int
            the number of steps to produce.
        printnum : int, optional
            if given, prints a  summary of the sampling ``printnum`` times
        thin : int, optional
            thins the trajectory by only including one of each ``thin`` points
        output : list, optional
            a list of strings giving which arrays should be returned by the function.
            The output list can contain

            - "pos" : Outputs the trajectory of position
            - "grad" : Outputs the gradient at each point
            - "llh" : Output the log posterior evaluated at each point
            - "xi" : Gives the auxilliary variable's value at each step
            - 'mom' : The sampled momentum points

            The default value is ``["pos"]``.

        Returns
        -------
        Q : numpy array
            the trajectory array for sampled points
        P : numpy array
            the sampled momentum points (if applicable)
        LLH : numpy array
            the log posterior of sampled points
        XI : numpy array
            the sampled auxillary variables (if applicable)
        F : numpy array
            the gradient of the log posterior function at each sampled point (if applicable)
        """

        Qk = 0
        NQ = 1 + Nsteps // thin
        self.ig.acc = 0
        self.ig.clear(self.q)

        output_dict = {}
        for o in output:
            output_dict[o.lower()] = True

        if output_dict.get("pos"):
            Q = np.zeros([NQ, len(self.q.flatten())])
        if output_dict.get("mom"):
            P = np.zeros([NQ, len(self.ig.p.flatten())])
        if output_dict.get("llh"):
            LLH = np.zeros([NQ, 1])
        if output_dict.get("grad"):
            F = np.zeros([NQ, len(self.ig.f.flatten())])
        if output_dict.get("xi"):
            XI = np.zeros([NQ, len(self.ig.xi.flatten())])

        stime = time.time()

        if printnum is not None:
            pf = Nsteps // printnum
        else:
            pf = Nsteps + 2

        for n in range(Nsteps):

            self.q = self.ig.step(self.q)

            if (n % thin == 0) and (Qk < NQ):
                if output_dict.get("pos"):
                    Q[Qk, :] = self.q.copy().flatten()
                if output_dict.get("mom"):
                    P[Qk, :] = self.ig.p.copy().flatten()
                if output_dict.get("llh"):
                    LLH[Qk] = self.ig.v
                if output_dict.get("grad"):
                    F[Qk, :] = self.ig.f.copy().flatten()
                if output_dict.get("xi"):
                    XI[Qk, :] = self.ig.xi.copy().flatten()

                Qk += 1

            if np.isnan(self.q).any():
                break

            if (n + 1) % pf == 0:
                print(
                    "Steps:", n + 1, "  Time:", time.time() - stime, "  V:", self.ig.v
                )

        self.timetaken_value = time.time() - stime
        self.Nsteps_value = Nsteps

        output_tuple = ()
        for o in output:
            if o.lower() == "pos":
                output_tuple += (Q[:Qk, :],)
            if o.lower() == "mom":
                output_tuple += (P[:Qk, :],)
            if o.lower() == "llh":
                output_tuple += (LLH[:Qk],)
            if o.lower() == "xi":
                output_tuple += (XI[:Qk, :],)
            if o.lower() == "grad":
                output_tuple += (F[:Qk, :],)

        if len(output_tuple) == 1:
            return output_tuple[0]

        return output_tuple

    def timetaken(self):
        """
        Returns the wall time of the previously run ``sample`` call.

        Returns
        -------
        T : float
            the wall time of the last sampling run, in seconds.
        """

        return self.timetaken_value

    def acceptance_rate(self):
        """
        Returns the acceptance rate of the previously run ``sample`` call.

        Returns
        -------
        A : float
            the acceptance rate of the last sampling run, if applicable.
        """

        if (self.ig.acc==0):
            return 0

        return self.ig.acc / self.Nsteps_value
