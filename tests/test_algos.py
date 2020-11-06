import pytest
import racecar as rc

def test_sgld():

    S = rc.sampler([0,0,0], 0.1, rc.llh.isotropic_gaussian, algo='sgld')

    Q = S.sample(100)

    assert(Q.shape==(100,3))

def test_lm():

    S = rc.sampler([0,0,0], 0.1, rc.llh.isotropic_gaussian, algo='lm')

    Q = S.sample(100)

    assert(Q.shape==(100,3))

def test_hmc():

    S = rc.sampler([0,0,0], 0.1, rc.llh.isotropic_gaussian, algo='hmc')

    Q = S.sample(100)

    assert(Q.shape==(100,3))

def test_badodab():

    S = rc.sampler([0,0,0], 0.1, rc.llh.isotropic_gaussian, algo='badodab')

    Q = S.sample(100)

    assert(Q.shape==(100,3))

def test_racecar():

    S = rc.sampler([0,0,0], 0.1, rc.llh.isotropic_gaussian, algo='racecar')

    Q = S.sample(100)

    assert(Q.shape==(100,3))
