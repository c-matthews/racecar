import pytest
import racecar as rc

def test_sgld():

    S = rc.sampler( [0,0,0] , 0.1, rc.llh.isotropic_gaussian, algo='sgld' )

    Q = S.sample(100)

	assert(Q.shape == [100,3] )
