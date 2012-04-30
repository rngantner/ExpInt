from expint.RHS import RHS

class MathematicalPendulum(RHS):
    """
    RHS of a mathematical pendulum:
        y'' = -g/l*sin(y), y(0)=y0, y'(0)=0
    written as a 2x2 system of order-1 differential equations:
        y1' = y2
        y2' = -g/l*sin(y1)
    with initial conditions: y1(0)=0, y2(0)=y0
    """
    def __init__(self,g,l):
        self.g = g
        self.l = l
    def ApplyDf(self,x,v):
        #Df = np.array(( (0,1), (-self.g/self.l*np.cos(x[0]), 0) ))
        #return np.dot(Df,v)
        return np.array((v[1], -self.g/self.l*np.cos(x[0])*v[0]))
    def normDf(self,x):
        alpha = -self.g/self.l*np.cos(x[0])
        # analytic 2-norm: sqrt of maximal singular value
        # equivalent to sqrt of maximal eigenvalue of Df^T*Df.
        # compute this analytically, result is max(|alpha|,1)
        return max(abs(alpha),1)
    def Applyf(self,x):
        return np.array(( x[1], -self.g/self.l*np.sin(x[0]) ))
    def Applyg(self,x):
        return self.Applyf(x) - self.ApplyDf(x,x)
    def name(self):
        return r"Mathematical Pendulum: $g="+str(self.g)+", l="+str(self.l)+"$"

