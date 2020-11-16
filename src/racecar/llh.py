'''
Contains examples of standard likelihood functions for usage in the package.
'''
import numpy as np

def isotropic_gaussian(q):
    '''
    An isotropic Gaussian likelihood function.

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


def blr(q, data, t, idxs=None, alpha=100 ):
    '''
    Bayesian Logistic Regression with a Gaussian prior.

    Parameters
    ----------
    q : numpy array
        Position parameter
    data : numpy array
        A (N,d) array, where the d datapoints have dimensionality N.
    t : numpy array
        A (1,d) binary array of indicator values
    idxs : list or iterable, optional
        A list of indexes to use in the BLR calculation
    alpha : float, optional
        The variance of the Gaussian prior, default 100.


    Returns
    -------
    r : dictionary
        returns the log likelihood under the ``llh`` key, the gradient under the ``grad`` key and the gradients for each data point are given in ``grad_data``.
    '''

    Ndata = data.shape[1]

    if idxs is None:
        idxs = np.arange(Ndata)

    X = data[:, idxs].T
    t = t[:, idxs].T

    # Prior
    Vprior = -0.5 * np.sum(q ** 2) / alpha
    Fprior = -q / alpha

    # Posterior
    tv = np.dot(X, q)
    exptv = np.exp(-tv)

    VV = tv * t - np.log(1 + exptv) - tv

    #V = np.sum(tv * t) - np.sum(np.log(1 + exptv)) - np.sum(tv)
    F = X * (t - 1.0 / (1.0 + exptv))

    TotalV = (np.sum(VV) + Vprior)

    TotalF = (Ndata * np.mean(F, 0, keepdims=True) + Fprior.T).T

    return {"llh": TotalV, "llh_data":VV , "grad": TotalF, "grad_data": F.T}
