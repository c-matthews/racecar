import numpy as np

def blr(q, XX,tt, idxs=None):

    Ndata = XX.shape[1]

    if (idxs is None):
        idxs = np.arange(Ndata)

    X = XX[:,idxs].T
    t = tt[:,idxs].T
    alpha = 100

    # Prior
    Vprior = -0.5*np.sum(q**2) / alpha
    Fprior = -q / alpha

    # LLH
    tv = np.dot( X,q)
    exptv = np.exp(-tv)

    VV = tv*t - np.log(1+exptv) - tv

    V = np.sum( tv*t ) - np.sum( np.log( 1+exptv ) ) - np.sum(tv )
    F = X * (t - 1.0/(1.0+exptv))

    TotalV = -(V+Vprior)

    TotalF = ( Ndata*np.mean(F,0,keepdims=True)+Fprior.T ).T

    return {
    'llh' : TotalV,
    'grad' : TotalF,
    'grad_data' : F.T
    }
