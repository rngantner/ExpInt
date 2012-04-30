from expint import RHS

class StiffODE(RHS):
    """
    RHS of the scalar ODE:
        y' = -c * y,  c > 0
    """
    def __init__(self,c=1):
        if not c > 0:
            raise Exception("Class StiffODE only supports c>0")
        self.c = c
    def ApplyDf(self,x,v):
        return -self.c*v
    def normDf(self,x):
        return abs(self.c)
    def Applyf(self,x):
        return np.array(-self.c * x, ndmin=1)
    def Applyg(self,x):
        return np.array(0, ndmin=1)
    def solution(self,t):
        return np.exp(-self.c*t)
    def name(self):
        return r"Stiff ODE: $y'=-"+str(self.c)+"y$"

