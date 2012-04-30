from expint.method import Method

class ExplicitEuler(Method):
    def __init__(self,rhs=None):
        super(self.__class__,self).__init__(rhs)
    
    @classmethod
    def name(self):
        return "ExplicitEuler"
    
    def integrate(self, x0, t0, tend, N=100):
        h = np.double(tend-t0)/N
        t = np.zeros((N+1,1)); t[0]=t0
        x = x0; y = [x0]
        for i in xrange(N):
            x = x + h*self.rhs.Applyf(x)
            y.append(x)
            t[i+1] = t[i]+h
        return t,np.array(y)

