from expint.methods.Method import ExponentialMethod
import numpy as np
from numpy.linalg import solve

class ExpRosenbrockEuler(ExponentialMethod):
    """
    Implements the Exponential Rosenbrock-Euler method:
    
    .. math:: u_{n+1} = u_n + h_n \\varphi(h_n\cdot J_n)\cdot f(u_n)
    """
    @classmethod
    def name(self):
        return "ExpRosenbrockEuler"
    
    def phi1(self,mat,h):
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

