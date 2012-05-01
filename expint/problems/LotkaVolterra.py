import numpy as np
from expint.RHS import DfRHS

class LotkaVolterra(DfRHS):
    """
    Lotka Volterra (predator-prey) model:
    Prey:       x' = x(1 - alpha*y)
    Predator:   y' = y(-1 + beta*x)
    where ' representes a derivative wrt time.
    """
    def __init__(self,alpha,beta):
        self.alpha = alpha
        self.beta = beta
        f = lambda x: np.array(( x[0]*(1 - alpha*x[1]), x[1]*(-1 + beta*x[0]) ))
        Df= lambda x,v: np.dot(np.array(( (1-alpha*x[1], -alpha*x[0]),(beta*x[1], beta*x[0]-1) )),v)
        super(LotkaVolterra,self).__init__(Df,f=f)
    def normDf(self,x):
        # Df^T*Df is hermitian 2x2 matrix. [a,b; b,c]
        a = (1-self.alpha*x[1])**2 + (self.beta*x[1])**2
        b = -(1-self.alpha*x[1])*self.alpha*x[0] + self.beta*x[1]*(self.beta*x[0]-1)
        c = (self.alpha*x[0])**2 + (self.beta*x[0]-1)**2
        l1 = 0.5*( (a+c) + np.sqrt((a+c)**2-4*(a*c-b*b)) )
        l2 = 0.5*( (a+c) - np.sqrt((a+c)**2-4*(a*c-b*b)) )
        return np.sqrt(max(l1,l2))
    def name(self):
        return r"Lotka-Volterra Model: $\alpha="+str(self.alpha)+r", \beta="+str(self.beta)+"$"

class LotkaVolterraExample(LotkaVolterra):
    """
    Lotka Volterra model with the following coefficients:
    
    alpha = 0.01, beta = 0.02
    """
    def __init__(self):
        super(LotkaVolterraExample,self).__init__(0.01,0.02)

