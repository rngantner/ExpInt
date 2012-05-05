import numpy as np
from RHS import DfRHS
from expint.methods.Method import AdaptiveMethod

class Problem(object):
    """
    Describes a problem.
      * Contains a right-hand side (rhs) and a method for solving the problem.
      * Provides a wrapper method ``Problem.integrate(x0,t0,tend)`` that calls the
        corresponding function of the method class
    
    Example::
    
    >>> from Method import ExplicitEuler
    >>> from RHS import LotkaVolterra
    >>> p = Problem(ExplicitEuler(), LotkaVolterra())
    >>> t,y = p.integrate(array((1,1)),0,3)
    >>> plot(t,y); show()
    """
    def __init__(self,method=None,rhs=None):
        # store method
        import inspect
        if inspect.isclass(method): # create instance
            self.method = method(rhs=rhs)
        else: # already is instance
            self.method = method
        # store right-hand side
        self.rhs = rhs
    
    def setMethod(self,method):
        self.method = method
    
    def setRHS(self,rhs):
        self.rhs = rhs
    
    def integrate(self, x0, t0, tend, N=10, abstol=1e-5, reltol=1e-5):
        """
        Wrapper function for integrator. Takes both N and tolerance values since
        it determines the type (adaptive or not) by checking if the method
        derives from Method.AdaptiveMethod.
        
        :param x0: initial value
        :param t0: initial time
        :param tend: end time
        :param N: number of timesteps (default 10)
        :param h: initial timestep size. If not specified, (tend-t0)/100 will be used.
        :param abstol: absolute tolerance. Overwrites self.abstol
        :param reltol: relative tolerance. Overwrites self.reltol
        :returns: tuple ``(t,y)``
        """
        if not type(x0) is np.ndarray:
            x0 = np.array(x0)
        #self.method.setRHS(self.rhs)
        if not x0.shape == ():
            x0 = x0[:]
        if isinstance(self.method,AdaptiveMethod):
            return self.method.integrate(x0,t0,tend,abstol=abstol,reltol=reltol)
        else:
            return self.method.integrate(x0,t0,tend,N=N)


if __name__ == '__main__':
    # define test problem
    class TestRHS(DfRHS):
        def __init__(self):
            self.A_mat = np.array(((1,2),(0,1)))
            super(TestRHS,self).__init__(Df=self.A_mat,g=lambda x: np.zeros_like(x))
        def getA(self):
            return self.A_mat
    
    # initial conditions
    x0 = np.array((20,20))
    t0 = 0
    tend = 25
    N=400
    
    # imports
    from Method import ExpForwardEuler, ExplicitEuler, ExpForwardEuler_Krylov
    from RHS import LotkaVolterraExample
    from Exp4 import Exp4
    
    # problem
    pmethod = "ExpForwardEuler"
    p = Problem(ExpForwardEuler,LotkaVolterraExample())
    t,y = p.integrate(x0,t0,tend,N)
    
    #p2 = Problem(ExplicitEuler,LotkaVolterraExample())
    #p2 = Problem(ExplicitEuler(),TestRHS())
    p2method = "ExplicitEuler"
    p2 = Problem(ExplicitEuler,LotkaVolterraExample())
    t2,y2 = p2.integrate(x0,t0,tend,N)
    
    p3method = "Exp4"
    p3 = Problem(Exp4,LotkaVolterraExample())
    t3,y3 = p3.integrate(x0,t0,tend,N)
    
    # plot results
    import matplotlib.pyplot as plt
    plt.plot(t,y[:,0],'-b',label=pmethod+': x0')
    plt.plot(t,y[:,1],'--b',label=pmethod+': x1')
    
    plt.plot(t2,y2[:,0],'-r',label=p2method+': x0')
    plt.plot(t2,y2[:,1],'--r',label=p2method+': x1')
    
    plt.plot(t3,y3[:,0],'-g',label=p3method+': x0')
    plt.plot(t3,y3[:,1],'--g',label=p3method+': x1')
    
    plt.xlabel('Time')
    plt.ylabel('Number of Predators/Prey')
    plt.legend(loc='best')
    
    plt.show()
    
