'''
Contains examples of functions for a variety of standard likelihoods.
'''
import numpy as np



def isotropic_gaussian(q):
    '''
    An isotropic Gaussian likelihood function :math:`p(x) \propto exp(-\|x\|^2/2)`.

    Parameters
    ----------
    q : numpy array
        Position parameter


    Returns
    -------
    r : dictionary
        returns the log likelihood under the ``llh`` key, and the gradient under the ``grad`` key.
    '''

    llh = -np.sum(q * q) / 2
    grad = -q

    return {"llh": llh, "grad": grad}


def blr(q, XX, tt, idxs=None):
    '''
    Bayesian Logistic Regression with a mild Gaussian prior.

    Parameters
    ----------
    q : numpy array
        Position parameter
    XX : numpy array
        A (N,d) array, where the d datapoints have dimensionality N.
    tt : numpy array
        A (1,d) binary array of indicator values
    idxs : list or iterable
        A list of indexes to use in the BLR calculation


    Returns
    -------
    r : dictionary
        returns the log likelihood under the ``llh`` key, the gradient under the ``grad`` key and the gradients for each data point are given in ``grad_data``. 
    '''

    Ndata = XX.shape[1]

    if idxs is None:
        idxs = np.arange(Ndata)

    X = XX[:, idxs].T
    t = tt[:, idxs].T
    alpha = 100

    # Prior
    Vprior = -0.5 * np.sum(q ** 2) / alpha
    Fprior = -q / alpha

    # LLH
    tv = np.dot(X, q)
    exptv = np.exp(-tv)

    VV = tv * t - np.log(1 + exptv) - tv

    V = np.sum(tv * t) - np.sum(np.log(1 + exptv)) - np.sum(tv)
    F = X * (t - 1.0 / (1.0 + exptv))

    TotalV = -(V + Vprior)

    TotalF = (Ndata * np.mean(F, 0, keepdims=True) + Fprior.T).T

    return {"llh": TotalV, "grad": TotalF, "grad_data": F.T}
