from numpy.testing import assert_array_almost_equal
from numpy.linalg import norm
from numpy import inf, dot
from numpy.random import random
import scipy.linalg
from nose.tools import assert_is
from expint.MatrixExponential import *

# test compute(A) and apply(v,A) function
MatExpClasses = [
        Pade_scipy,
        Pade_new,
        Pade_Cpp,
        ]

# only test apply(v,A) function
KrylovClasses = [
        Krylov_Cpp,
        Krylov_Python,
        ]

class TestMatrixExponentials(object):
    def test_compute(self):
        A = random((5,5))
        expA_exact = scipy.linalg.expm(A)
        
        for c in MatExpClasses:
            assert_array_almost_equal(c().compute(A),expA_exact)
    
    def test_apply(self):
        A = random((5,5))
        v = random((5,1))
        expAv_ex = dot(scipy.linalg.expm(A),v)
        
        #for c in KrylovClasses:
        for c in MatExpClasses + KrylovClasses:
            print "class",c
            print c().apply(v.copy(),A).shape, expAv_ex.shape
            assert_array_almost_equal( c().apply(v.copy(),A), expAv_ex )
    
    def test_compiledModules(self):
        # test if all compiled modules can be loaded correctly
        instance = Pade_Cpp()
        assert_is(instance.CppFlag,True)
    

