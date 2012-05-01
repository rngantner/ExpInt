from expint.RHS import RHS

class QuadraticODE(RHS):
    """
    RHS of the scalar ODE:
        y' = c * y**2
    """
    def __init__(self,c=1):
        self.c = c
    def normDf(self,x):
        return abs(2*x*self.c)
    def ApplyDf(self,x,v):
        return 2*x*v*self.c
    def Applyf(self,x):
        return self.c * x**2
    def Applyg(self,x):
        return self.Applyf(x) - self.ApplyDf(x,x)
    def solution(self,t):
        r"""Solution for initial condition $u_0 = 1$"""
        return 1./(1-t)
    def name(self):
        return r"Scalar quadratic ODE: $y'="+str(self.c)+"y^2$"

