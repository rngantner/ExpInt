from expint.methods import AdaptiveMethod

class ode45(AdaptiveMethod):
    """
    Calls the ode45 method assumed to be in the module ode45
    """
    @classmethod
    def name(self):
        return "ode45"
    
    def integrate(self, y0, t0, tend, N=None, abstol=1e-5, reltol=1e-5):
        """t,y = integrate(y0, t0, tend, [N])
        N: number of timesteps; is ignored, as this method chooses timestep size adaptively!"""
        from ode45_scheme import ode45
        vfun = lambda t,y: self.rhs.Applyf(y)
        vslot = (t0, tend)
        vinit = y0
        t,Y,stats = ode45(vfun,vslot,vinit,abstol=abstol,reltol=reltol,stats=True)
        self.stats = stats
        return t,Y

