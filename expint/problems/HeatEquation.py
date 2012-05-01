from expint.RHS import RHS
from expint.util import gaussNodes
from numpy import *
import scipy.sparse as sp
import scipy.sparse.linalg as sl
import os

#solutionFile = 'HeatEquation_solution.npz'
#solutionFile = 'HeatEquation_solution2.npz'
#solutionFile = 'HeatEquation_solution3.npz'
HEsol = array([ 0.00127801,  0.00255598,  0.00383382,  0.00511139,  0.00638846,
        0.00766478,  0.00894001,  0.01021376,  0.01148558,  0.01275499,
        0.01402141,  0.01528423,  0.0165428 ,  0.01779639,  0.01904423,
        0.02028551,  0.02151936,  0.02274488,  0.0239611 ,  0.02516703,
        0.02636163,  0.02754382,  0.0287125 ,  0.02986653,  0.03100471,
        0.03212585,  0.03322872,  0.03431206,  0.0353746 ,  0.03641503,
        0.03743204,  0.03842432,  0.03939051,  0.04032929,  0.0412393 ,
        0.04211919,  0.04296762,  0.04378323,  0.0445647 ,  0.04531068,
        0.04601988,  0.04669098,  0.04732272,  0.04791383,  0.04846309,
        0.04896929,  0.04943126,  0.04984786,  0.050218  ,  0.05054061,
        0.05081468,  0.05103923,  0.05121335,  0.05133617,  0.05140687,
        0.05142469,  0.05138894,  0.05129897,  0.05115421,  0.05095417,
        0.0506984 ,  0.05038653,  0.05001829,  0.04959344,  0.04911187,
        0.04857349,  0.04797835,  0.04732655,  0.04661827,  0.0458538 ,
        0.04503349,  0.0441578 ,  0.04322727,  0.04224253,  0.04120431,
        0.04011343,  0.03897078,  0.03777737,  0.0365343 ,  0.03524275,
        0.033904  ,  0.03251944,  0.03109053,  0.02961883,  0.02810599,
        0.02655377,  0.02496399,  0.0233386 ,  0.0216796 ,  0.0199891 ,
        0.01826929,  0.01652245,  0.01475093,  0.01295719,  0.01114374,
        0.00931318,  0.0074682 ,  0.00561153,  0.00374601,  0.00187451])

class HeatEquation(RHS):
    """
    Class representing a finite elements (spatial) discretization of the heat equation.
    A sparse LU-decomposition of the Jacobian is created in the constructor, allowing
    the action of the Jacobian on a vector u to be computed more efficiently.
    """
    def __init__(self,a=0,b=1,N=100):
        """Parameters are for spatial discretization (method of lines)"""
        self.N = N
        self.a = a
        self.b = b
        h = double(b-a)/(N+1)
        self.h = h
        # Function handles
        self.u_handle = lambda x: x*sin(pi*x)
        self.f_handle = lambda x: x*sin(pi*x)#exp(-t)*((pi**2-1)*x*sin(pi*x) - 2*pi*cos(pi*x))
        #self.f_handle = lambda x: x#(x*sin(pi*x))**2
        
        # Mass matrix
        d = h/6*ones(N+1)
        self.M = sp.spdiags([d,4*d,d],[-1,0,1],N,N).tocsc()
        # store LU factors to speed up solving
        self.M_LUsolve = sl.factorized(self.M)
        # Stiffness matrix
        d = ones(N+1)/h
        self.A = sp.spdiags([-d,2*d,-d],[-1,0,1],N,N).tocsc()
        # store domain
        self.xvals = a+h*r_[1:N+1]
        self.xvals_full = a+h*r_[0:N+2]
        # shape functions
        shap_LFE = lambda x: (x<0)*(x+1.) + (x>=0)*(1.-x)
        ## quadrature weights and nodes (overkill quadrature!):
        self.quad_x,self.quad_w = gaussNodes(1000)
        self.shap = shap_LFE(self.quad_x) # precompute shape functions
        # Load Vector
        self.F = zeros(self.N)
        for i in range(self.N):
            xi = self.a+h*(i+1)
            fvals = self.f_handle(h*self.quad_x+xi)
            self.F[i] = h*sum(self.quad_w*self.shap*fvals)
        
        # store standard initial values
        self.init1 = self.xvals*sin(pi*self.xvals)
        self.sol1 = HEsol
        # solution to this initial value for t0=0, tend=1
        #if os.path.exists(solutionFile):
        #    file = open(solutionFile)
        #    self.sol1 = load(file)['y']
        #    file.close()
        #else:
        #    print "precomputed solution could not be imported"
    def normDf(self,x):
        return 6*self.N**(2.5)*norm(x[:-1])
    def ApplyDf(self,u,v):
        return -self.M_LUsolve(self.A*v) # M_LUsolve(...) implementes M^{-1}*(...)
    def Applyf(self,u):
        return self.M_LUsolve(self.F-self.A*u)
    def Applyg(self,u):
        return self.M_LUsolve(self.F)
    def name(self):
        return r"Heat equation: $\dot{u}(x) - \Delta u(x) = f(x)$, $N=%d$"%self.N


