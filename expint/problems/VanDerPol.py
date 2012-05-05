from expint.RHS import RHS
import numpy as np

class Van_der_Pol(RHS):
    """
    RHS of the Van-der-Pol Oscillator:
        y1' = y2
        y2' = -mu**2 * ((y1**2 - 1)*y2 + y1)
    """
    def __init__(self,mu):
        self.mu = mu
    def ApplyDf(self,x,v):
        Df = np.array(( (0,1), (-self.mu**2*(2*x[0]*x[1]+1), -self.mu**2*(x[0]**2-1)) ))
        return np.dot(Df,v)
    def normDf(self,x):
        from numpy.linalg import norm
        Df = np.array(( (0,1), (-self.mu**2*(2*x[0]*x[1]+1), -self.mu**2*(x[0]**2-1)) ))
        return norm(Df,2)
    def Applyf(self,x):
        return np.array(( x[1], -self.mu**2 * ((x[0]**2-1)*x[1] + x[0]) ))
    def Applyg(self,x):
        return self.Applyf(x) - self.ApplyDf(x,x)
    def name(self):
        return r"Van-der-Pol Oscillator: $\mu="+str(self.mu)+"$"

