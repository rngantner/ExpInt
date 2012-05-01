# -*- encoding: utf8 -*-

import numpy as np
import inspect
from expint.MatrixExponential import MatrixExponentialFactory

class Method(object):
    """
    Interface for integration schemes.

    :param rhs: Right-hand side to solve.
    """
    def __init__(self,rhs=None):
        self.rhs = rhs
    
    def integrate(self, x0, t0, tend, N):
        """
        This function should implement the time integration for
        t from t0 to tend, given the initial condition x(t0) = x0.
        N is the number of timesteps.
        """
        raise NotImplementedError
    
    def setRHS(self,rhs):
        """Convenience function to set right-hand side to rhs"""
        self.rhs = rhs

class AdaptiveMethod(Method):
    """
    Marker class for adaptive methods.
    Problem class checks if method is derived from this to determine function signature.
    """
    def integrate(self, x0, t0, tend, abstol, reltol):
        """
        :param x0: initial value
        :param t0: initial time
        :param tend: end time
        :param abstol: absolute tolerance. Overwrites self.abstol
        :param reltol: relative tolerance. Overwrites self.reltol
        :returns: tuple ``(t,y)``
        """
        raise NotImplementedError

from types import StringType
class ExponentialMethod(Method):
    """
    Defines an integration method that requires a matrix exponential.
    
    :param rhs: instance of a DfRHS (fRHS not sufficient, need Jacobian)
    :param matexp: describes type of matrix exponential to use
    
    Notes:
      * Certain matrix exponential algorithms require the actual
        matrix. The Df argument of the DfRHS class constructor must
        be a numpy array; a FunctionType will not work!
      * If a method requires a specific matrix exponential, it
        should overwrite the ``__init__`` function and explicitly set
        the variable ``self.matexp`` there.
    """
    def __init__(self,rhs=None,matexp='cpade'):
        # set right-hand side
        super(ExponentialMethod,self).__init__(rhs)
        self.setMatrixExponential(matexp)
    
    def setMatrixExponential(self, matexp):
        self.matexp_class = MatrixExponentialFactory().createMatrixExponential(matexp)
    
    def matexp(self,A,v=None,h=1.):
        """
        :param A: matrix to take exponential of
        :param v: vector to multiply exp(h*A) with (if given)
        :param h: scaling factor in exp(h*A)
        :returns: exp(h*A) if v is not given.
        :returns: exp(h*A)*v is v is given.
        """
        if inspect.isclass(self.matexp_class): # create instance
            instance = self.matexp_class()
        else: # already instantiated
            instance = self.matexp_class
        if v is None:
            return instance.compute(A,h)
        else:
            return instance.apply(v,A,h)


class ExpForwardEuler(ExponentialMethod):
    """
    Implements the Exponential Forward Euler method:
    u_{n+1} = exp(h_n*A)*u_n + h_n*phi_1(h_n*A)*g(u_n)
    
    Uses the standard Padé approximation to the matrix exponential
    """
    @classmethod
    def name(self):
        return "ExpForwardEuler"
    
    def phi1(self,mat,h):
        from numpy.linalg import solve
        return solve(h*mat,(self.matexp(h*mat)-np.eye(mat.shape[0])))
    
    def integrate(self, x0, t0, tend, N=100):
        raise Exception("incorrect implementation")
        """t,y = integrate(x0, t0, tend, [N])
        N: number of timesteps"""
        h = np.double(tend-t0)/N
        t = np.zeros(N+1); t[0]=t0
        x = x0; y = [x0]
        for i in range(N):
            # build matrix for nxn problem
            if x.shape == ():
                n = 1
            else:
                n = x.shape[0]
            A = np.zeros((n,n))
            for j in range(n):
                e = np.zeros(n); e[j] = 1
                A[:,j] = self.rhs.ApplyDf(x,e)
            expAh = self.matexp(h*A)
            x = np.dot(expAh,x) + h*np.dot(self.phi1(h*A,h),self.rhs.Applyg(x)) #phi1(h*A <-- correct???
            y.append(x)
            t[i+1] = t[i]+h
        return t,np.array(y)

class ExpRosenbrockEuler(ExponentialMethod):
    """
    Implements the Exponential Rosenbrock-Euler method:
    
    .. math:: u_{n+1} = u_n + h_n \\varphi(h_n\cdot J_n)\cdot f(u_n)
    """
    @classmethod
    def name(self):
        return "ExpRosenbrockEuler"
    
    def phi1(self,mat,h):
        from numpy.linalg import solve
        return solve(h*mat,(self.matexp(h*mat)-np.eye(mat.shape[0])))
    
    def integrate(self, x0, t0, tend, N=100):
        """
        :param x0: initial value
        :param t0: initial time
        :param tend: end time
        :param N: number of timesteps
        :returns: tuple ``(t,y)``
        """
        h = np.double(tend-t0)/N
        t = np.zeros(N+1); t[0]=t0
        x = x0; y = [x0]
        for i in range(N):
            # build matrix for nxn problem
            if x.shape == ():
                n = 1
            else:
                n = x.shape[0]
            A = np.zeros((n,n))
            for j in range(n):
                e = np.zeros(n); e[j] = 1
                A[:,j] = self.rhs.ApplyDf(x,e)
            expAh = self.matexp(h*A)
            x = x + h*np.dot(self.phi1(A,h),self.rhs.Applyf(x))
            y.append(x)
            t[i+1] = t[i]+h
        return t,np.array(y)

class ode45(AdaptiveMethod):
    """
    Calls the ode45 method assumed to be in the module ode45
    """
    @classmethod
    def name(self):
        return "ode45"
    
    def integrate(self, y0, t0, tend, N=None, abstol=1e-5, reltol=1e-5):
        """t,y = integrate(y0, t0, tend, [N])
        N: number of timesteps; is ignored, as this method chooses timestep size adaptively!"""
        from ode45 import ode45
        vfun = lambda t,y: self.rhs.Applyf(y)
        vslot = (t0, tend)
        vinit = y0
        t,Y,stats = ode45(vfun,vslot,vinit,abstol=abstol,reltol=reltol,stats=True)
        self.stats = stats
        return t,Y

class ImplicitEuler(Method):
    @classmethod
    def name(self):
        return "ImplicitEuler"
    
    def __init__(self,rhs=None):
        super(ImplicitEuler,self).__init__(rhs)
        if not hasattr(rhs,'M') or not hasattr(rhs,'A') or not hasattr(rhs,'F'):
            raise Exception("This Integrator only works for RHS classes with attributes A,M and F.\n A: stiffness matrix, M: mass matrix, F: (constant) load vector")
    
    def integrate(self, y0, t0, tend, N=100, abstol=1e-5, reltol=1e-5):
        import scipy.sparse.linalg as sl
        M = self.rhs.M
        A = self.rhs.A
        F = self.rhs.F
        h = np.double(tend-t0)/N
        t = np.zeros((N+1,1)); t[0]=t0
        x = y0.copy(); y = [y0.copy()]
        for i in xrange(N):
            x = sl.linsolve.spsolve(M+h/2*A,F*h+(M-h/2*A)*x)
            y.append(x)
            t[i+1] = t[i] + h
        return t,np.array(y)
        

class ExpForwardEuler_Krylov(ExponentialMethod):
    """
    Implements the Exponential Forward Euler method:
    u_{n+1} = exp(h_n*A)*u_n + h_n*phi_1(h_n*A)*g(u_n)
    
    Uses Krylov subspace approximation of matrix exponential
    """
    def __init__(self,rhs=None):
        super(ExpForwardEuler_Krylov,self).__init__(rhs)
        self.setMatrixExponential('krylov')
    @classmethod
    def name(self):
        return "ExpForwardEuler_Krylov"
    
    def phi1(self,A,v,h):
        # TODO: need krylov subspace approximation of phi_1
        pass
    
    def integrate(self, x0, t0, tend, N=100):
        """t,y = integrate(x0, t0, tend, [N])
        N: number of timesteps"""
        h = np.double(tend-t0)/N
        t = np.zeros((N+1,1)); t[0]=t0
        x = x0.copy(); y = [x0.copy()]
        for i in xrange(N):
            g = self.rhs.Applyg(x) # evaluate vector g(x)
            A = lambda v: self.rhs.ApplyDf(x,v) # ----------- TODO: test this after implementing procedural A*x support
            x = self.matexp(A,x,h) + h*self.phi1(A,g,h)
            y.append(x)
            t[i+1] = t[i]+h
        return t,np.array(y)

class ExplicitEuler(Method):
    def __init__(self,rhs=None):
        super(self.__class__,self).__init__(rhs)
    
    @classmethod
    def name(self):
        return "ExplicitEuler"
    
    def integrate(self, x0, t0, tend, N=100):
        h = np.double(tend-t0)/N
        t = np.zeros((N+1,1)); t[0]=t0
        x = x0; y = [x0]
        for i in xrange(N):
            x = x + h*self.rhs.Applyf(x)
            y.append(x)
            t[i+1] = t[i]+h
        return t,np.array(y)
