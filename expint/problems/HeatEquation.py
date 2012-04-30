from expint.RHS import RHS

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
        from gaussNodes import gaussNodes
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
        # solution to this initial value for t0=0, tend=1
        file = open('HeatEquation_solution.npz')
        #file = open('HeatEquation_solution2.npz')
        #file = open('HeatEquation_solution3.npz')
        self.sol1 = load(file)['y']
        file.close()
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


