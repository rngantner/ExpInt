# -*- encoding: utf-8 -*-
from expint.methods.Method import ExponentialMethod
from numpy.linalg import solve

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

