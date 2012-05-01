# -*- encoding: utf8 -*-

import numpy as np
from types import FunctionType

class MatrixExponential(object):
    def __init__(self):
        # check for existence of compiled modules, etc...
        pass
    
    def apply(self,v,A=None,h=1):
        """Returns exp(h*A)*v"""
        raise NotImplementedError
    
    def compute(self,A,h=1):
        """Returns the matrix exp(h*A), or an approximation to it"""
        raise NotImplementedError

class Pade_scipy(MatrixExponential):
    def compute(self,A,h=1):
        """Computes the exponential of the matrix A and stores a reference in the instance"""
        # no deep copy done!
        if not type(A) is np.ndarray:
            raise TypeError('Pade approximation only supports type(A) == np.ndarray')
        import scipy.linalg
        self.expA = scipy.linalg.expm(h*A)
        return self.expA
    
    def apply(self,v,A=None,h=1):
        if A != None:
            self.compute(A,h)
        return np.dot(self.expA,v)

class Pade_new(MatrixExponential):
    def compute(self,A,h=1):
        if not type(A) is np.ndarray:
            raise TypeError('Pade approximation only supports type(A) == np.ndarray')
        try:
            from expm import expm
        except ImportError:
            # use scipy implementation as fallback
            print "ImportError generated when trying to import function 'expm' from module 'expm'!"
            print "using scipy.linalg.expm instead"
            from scipy.linalg import expm
        self.expA = expm(h*A)
        return self.expA

    def apply(self,v,A=None,h=1):
        if A != None:
            self.compute(A,h)
        return np.dot(self.expA,v)

class Pade_Cpp(MatrixExponential):
    def __init__(self):
        self.CppFlag = True
        try:
            import expmat as e
        except ImportError:
            # don't use C++ module
            self.CppFlag = False
        
    def compute(self,A,h=1):
        if not type(A) is np.ndarray:
            raise TypeError('Pade approximation only supports type(A) == np.ndarray')
        if self.CppFlag:
            import expmat as e
            self.expA = np.zeros_like(A)
            e.expmc(h*A,self.expA)
        else:
            from scipy.linalg import expm
            self.expA = expm(A)
        return self.expA
    
    def apply(self,v,A=None,h=1):
        if A != None:
            self.compute(A,h)
        return np.dot(self.expA,v)

class KrylovMatrixExp(MatrixExponential):
    def __init__(self,k=5):
        """
        Krylov subspace approximation of the matrix exponential applied
        to a given vector.
        An Arnoldi iteration is used to determine a k-dimensional Krylov space.
        
        k: dimension of Krylov space. k << N
        
        Note: subclasses of this class do not support computation of
        the matrix exp(A), only exp(A)*v. The compute() method therefore
        raises an error.
        """
        self.k = k
    
    def compute(self,A,h=1):
        raise Exception('Arnoldi_Cpp does not support explicit computation of the exponential of a matrix. Only the application of the matrix exponential onto a vector v is supported')

# TODO: test this
class Krylov_Cpp(KrylovMatrixExp):
    """
    Krylov subspace approximation of the matrix exponential applied
    to a given vector.
    Uses C++ implementation of the Arnoldi iteration
    """
    def __init__(self,k=5):
        super(Krylov_Cpp,self).__init__(k)
        self.CppFlag = True
        try:
            from expint.lib.carnoldi import arnoldi
        except ImportError:
            # don't use C++ module
            self.CppFlag = False

    def apply(self,v,A=None,h=1):
        nrm = np.linalg.norm(v)
        v = v/nrm
        if self.CppFlag:
            from expint.lib.carnoldi import arnoldi
            V,H = arnoldi(A,v,self.k)
        else:
            from expint.util import arnoldi
            V,H = arnoldi(A,v,self.k)
        
        from scipy.linalg import expm
        result = nrm*np.dot(V,expm(h*H[:self.k,:self.k])[:,0])
        result.shape = (max(result.shape),1)
        return result

class Krylov_Python(KrylovMatrixExp):
    """
    Krylov subspace approximation of the matrix exponential applied
    to a given vector: exp(h*A)*v
    Uses Python implementation of the Arnoldi iteration
    """
    def apply(self,v,A=None,h=1):
        T = v.dtype
        nrm = np.linalg.norm(v)
        v = v/nrm
        n = np.max(v.shape)
        from expint.util import arnoldi
        from scipy.linalg import expm
        V = np.zeros((n,self.k), dtype=T)
        H = np.zeros((self.k+1,self.k), dtype=T)
        V,H = arnoldi(A,v,self.k)
        result = nrm*np.dot(V,expm(h*H[:self.k,:self.k])[:,0])
        result.shape = (max(result.shape),1)
        return result
    
class MatrixExponentialFactory(object):
    """Create a new instance of a matrix exponential
    
    Contains the createMatrixExponential method which accepts
    an argument indicating which matrix exponential is to be used.
    This function checks if any required precompiled modules are
    available and importable.
    """

    # map description string to class
    StrToClass = {'scipy'  : Pade_scipy,    # scipy implementation
                  'pade'   : Pade_new,      # Padé approx. in Python (currently better than scipy)
                  'cpade'  : Pade_Cpp,      # Padé approx. C++ (Eigen)
                  'krylov' : Krylov_Python, # Krylov subspace approximation in Python
                  'ckrylov': Krylov_Cpp,    # Krylov subspace approximation in C++
                  }

    def createMatrixExponential(self,matexp):
        """Returns a new MatrixExponential depending on the parameter
        
        If any modules are missing or not importable, the default method
        is a Padé approximation (class Pade_Python).
        """
        from types import StringType
        if type(matexp) is StringType:
            # map string to correct matrix exponential
            matexp_ret = self.StrToClass[matexp.lower()]
        elif issubclass(matexp, MatrixExponential):
            # matexp is a matrix exponential class
            matexp_ret = matexp
        elif isinstance(matexp, MatrixExponential):
            # matexp is instance of a matrix exponential class
            matexp_ret = matexp
        return matexp_ret
    

