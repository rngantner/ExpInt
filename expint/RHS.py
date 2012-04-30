import numpy as np
import types

class RHS(object):
    def __init__(self):
        raise NotImplementedError
    def normDf(self):
        raise NotImplementedError
    def ApplyDf(self,v):
        raise NotImplementedError
    def Applyg(self,v):
        raise NotImplementedError
    def Applyf(self,v):
        raise NotImplementedError

#### test this::
class fRHS(RHS):
    """
    Define a right-hand side of the form: y' = f(y,t)
    
    A numpy array can be given for f instead of a function.
    
    Examples:
    >>> DfRHS(f=lambda x,t: sin(x))
    >>> DfRHS(f=array(((1,2,0),(0,1,1),(3,4,5))))
    """
    def __init__(self,f):
        self.f = f
    def ApplyDf(self,v):
        raise Exception('fRHS subclass does not support application of A')
    def normDf(self,v):
        raise Exception('fRHS subclass does not support norm computation of A')
    def Applyg(self,v):
        raise Exception('fRHS subclass does not support application of g')
    def Applyf(self,v):
        if type(self.f) is np.ndarray:
            return np.dot(self.f,v)
        elif type(self.f) is types.FunctionType:
            return self.f(v)
        elif self.f is None:
            return v

class DfRHS(RHS):
    """
    Defines a right-hand side of the form:
    
    .. math:: y' = Df(y,t)\cdot y + g(y,t) ,
    
    where Df is the Jacobian of the right hand side :math:`f=Df\cdot y + g`
    and g(y,t) is the remaining nonlinear term.
    
    If f is given instead of g, g is defined as :math:`f-Df\cdot y`
    
    If a numpy array is given for Df instead of a function, it is
    interpreted as a constant Jacobian and the application of Df to y becomes ``dot(Df,y)``
    
    Examples::
    
    >>> DfRHS(f=lambda x,t: sin(x), Df=lambda x,t: cos(x))
    >>> DfRHS(Df=array(((1,2),(0,1))), g=lambda x,t: x**2)
    """
    def __init__(self,Df=None,g=None,f=None,normDf=None):
        self.Df = Df
        self.normDf_handle = normDf
        if not g is None:
            self.g = g
        elif not f is None: # g is none and f is given
            self.g = lambda x: f(x)-Df(x,x)
        else:
            self.g = None
    
    def getDf(self,x):
        """
        Returns Jacobian Df at point x.
        This function is not mandatory.
        """
        n = np.max(x.shape)
        if type(self.Df) is np.ndarray:
            return self.Df
        elif type(self.Df) is types.FunctionType:
            # TODO: this is a very bad idea!! (maybe warn user?)
            print "Efficiency warning: constructing Jacobian from procedural applyDf! (in DfRHS.getDf)"
            A = np.zeros((n,n))
            for i in xrange(n):
                e = np.zeros(n)
                e[i] = 1
                A[:,i] = self.Df(x,e)
            return A
        elif self.Df is None: # interpret as identity
            return np.eye(n)
    
    def normDf(self,x):
        """
        Returns norm or approximation of norm of Jacobian Df at x.
        
        This is used to test if Df is singular. If this is never the case
        (eg. Df is a constant nonsingular matrix), one can just return 1.
        """
        from numpy.linalg import norm
        if type(self.Df) is np.ndarray:
            return norm(self.Df)
        elif type(self.Df) is types.FunctionType:
            if not self.normDf_handle is None:
                return self.normDf_handle(x)
            else:
                # this is a bad idea! (not very efficient)
                print "Efficiency warning: calling getDf(x) with procedural applyDf in DfRHS.normDf"
                A = self.getDf(x)
                return norm(A)
    
    def ApplyDf(self,x,v):
        """
        Returns Jacobian Df at point x applied to v
        """
        if type(self.Df) is np.ndarray:
            return np.dot(self.Df,v)
        elif type(self.Df) is types.FunctionType:
            # Df(x) returns the Jacobian at x
            return self.Df(x,v)
        elif self.Df is None: # interpret as identity
            return v
    
    def Applyg(self,x):
        """
        Returns the nonlinear term g(x)
        """
        if type(self.g) is np.ndarray:
            return np.dot(self.g,x)
        elif type(self.g) is types.FunctionType:
            return self.g(x)
        elif self.g is None:
            return x
    
    def Applyf(self,x):
        """
        Returns the entire right-hand side f = Df*y + g
        """
        return self.ApplyDf(x,x)+self.Applyg(x)

