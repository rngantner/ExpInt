import numpy as np
from numpy.linalg import norm
from Method import ExponentialMethod, AdaptiveMethod
from KrylovPhi import KrylovPhi


class Exp4(ExponentialMethod):
    """
    Exp4 exponential integration algorithm as described by
    Hochbruck, Lubich and Selhofer (1998)
    
    :param rhs: instance of class implementing RHS interface 
    :param stats: True means to keep track of statistics
    
    Notes:
      * This algorithm is optimized for large, sparse systems.
        Due to the construction of Krylov subspaces, its
        performance is bad for small-dimensional systems.
    """
    def __init__(self, rhs, abstol=1e-6, reltol=1e-6, stats=False):
        # set right-hand side
        super(ExponentialMethod,self).__init__(rhs)
        self.abstol = abstol
        self.reltol = reltol
        # initialize statistics dict
        if stats:
            self.stats = {"accept":0,
                          "reject":0,
                          "RHSevals":0}
        else:
            self.stats = False
        self.y = [] # needed by scaledNorm to detect if reltol can be used
    
    @classmethod
    def name(self):
        return "Exp4"
    
    def phiv(self, h, v):
        """
        Computes phi(h*Hm)*v and two other vectors
        
        :param h: factor
        :param v: vector to apply phi(h*Hm) to
        :returns: 3-tuple of vectors ``( phi(h/3*Hm)*v, phi(2h/3*Hm)*v, phi(h*Hm)*v )``
        """
        # if Df is nearly singular, Df^-1 * (exp(Df)-1)*v is numerically ill-conditioned, but should give v
        # check norm of Jacobian:
        if self.rhs.normDf(self.y[-1]) < np.finfo(np.double).eps:
            return (v,v,v)
        
        # if v is too small, phi*v is also almost zero.
        # must check for this since arnoldi iterations needs v != 0
        if norm(v) < np.finfo(np.double).eps:
            res = np.zeros_like(v)
            return (res,res,res)
        
        # use flag argument to find correct KrylovPhi instance
        pk = KrylovPhi(self.rhs, exp4=self, tau=h)
        
        # compute Krylov subspace approximation of phi(h*A)*v
        pk.compute(v)
        p1,p2,p3 = pk.phi3()
        if self.stats:
            self.stats["RHSevals"] += pk.m-1 # dimension of Krylov space minus 1 corresponds to number of applyA calls
        
        return (p1, p2, p3)
    
    def integrate(self, x0, t0, tend, N=100):
        """
        :param x0: initial value
        :param t0: initial time
        :param tend: end time
        :param N: number of timesteps (default: 100)
        :returns: tuple ``(t,y)``
        """
        h = np.double(tend-t0)/N
        t = np.zeros((N+1,1)); t[0]=t0
        x = x0; self.y = [x0]
        for i in xrange(N):
            f_xold = self.rhs.Applyf(x)
            
            k1, k2, k3 = self.phiv(h, f_xold)
            
            w4 = -7./300*k1 + 97./150*k2 - 37./300*k3
            u4 = x+h*w4
            d4 = self.rhs.Applyf(u4) - f_xold - h*self.rhs.ApplyDf(x,w4) #dot(A,w4)
            
            k4, k5, k6 = self.phiv(h, d4)
            
            w7 = 59./300*k1 - 7./75*k2 + 269./300*k3 + 2./3*(k4+k5+k6)
            u7 = x+h*w7
            d7 = self.rhs.Applyf(u7) - f_xold - h*self.rhs.ApplyDf(x,w7)#np.dot(A,w7)
            
            k7,bla1,bla2 = self.phiv(h, d7)
            
            x = x + h*(k3 + k4 - 4./3*k5 + k6 + 1./6*k7)
            xembed1 = x + h*(k3 - 2./3*k5 + 0.5*(k6+k7-k4))
            xembed2 = x + h*(-k1 + 2*k2 - k4 + k7)
            self.y.append(x)
            t[i+1] = t[i]+h
        
        return t,np.array(self.y)
    
    def scaledNorm(self, d, abstol=None, reltol=None):
        """
        compute scaled norm: :math:`||d||_{tol} = \sqrt{ \\frac1{N} \sum_{i=1}^N (d_i/w_i)^2 }`
        
        :param abstol: absolute tolerance. can be a vector
        :param reltol: relative tolerance. can be a vector
        """
        
        # use values set in constructor if not explicitly given
        if abstol is None:
            abstol = self.abstol
        if reltol is None:
            reltol = self.reltol
        
        y = self.y
        if len(y) is 0:
            print "caution: len(y) is 0 in Exp4.scaledNorm! ignoring relative tolerance!"
            w = abstol
        elif len(y) is 1:
            w = abstol + reltol*np.abs(y[0])
        else:
            w = abstol + reltol*np.maximum(np.abs(y[-1]),np.abs(y[-2]))
        
        N = len(d)
        return np.sqrt(np.sum((d/w)**2)/N) # /w is component-wise!

class Exp4_adaptive(AdaptiveMethod,Exp4):
    """
    Exp4 exponential integration algorithm as described by
    Hochbruck, Lubich and Selhofer (1998)
    
    :param rhs: instance of class implementing RHS interface 
    :param abstol: absolute tolerance (default 1e-6)
    :param reltol: relative tolerance
    :param stats: True means to keep track of statistics
    :param krylovargs: arguments given to constructor of :ref:`krylovphi` class
    
    Notes:
      * This algorithm is optimized for large, sparse systems.
        Due to the construction of Krylov subspaces, its
        performance is bad for small-dimensional systems.
    """
    def __init__(self, rhs, abstol=1e-6, reltol=1e-6, stats=True, krylovargs=None):
        # call Exp4 constructor
        super(Exp4_adaptive,self).__init__(rhs, abstol, reltol)
        # step size control variables: negative value signifies nonexistence of previous error
        self.err_old = -1
        self.h_old = -1
        # initialize statistics dict
        if stats:
            self.stats = {"accept":0,   # number of accepted timesteps
                          "reject":0,   # number of rejected timesteps
                          "RHSevals":0, # number of RHS evaluations
                          "h_err":[],   # error timestep
                          "h_kry":[],   # Krylov timestep
                          "m_max":0,    # max. Krylov space size
                          "m":[],       # Krylov space size
                         }
                          #"h":[]} --- h is minimum(h_err,h_kry,hmax)
        else:
            self.stats == False
        # set constants
        self.normcontrol = False # use norm to control tolerance
        self.hmin = np.finfo(np.double).eps  # minimal stepsize
        self.hmax = 100 # is overwritten in integrate(..)
        self.krylov = [None,None,None]
        for i in range(3):
            self.krylov[i] = KrylovPhi(self.rhs)
            self.krylov[i].setExp4(self)
    
    @classmethod
    def name(self):
        return "Exp4_adaptive"
    
    def phiv(self, h, v, flag=-1, reuse=False):
        """
        Computes phi(h*Hm)*v and two other vectors
        
        :param h: factor
        :param v: vector to apply phi(h*Hm) to
        :param flag: if it is in (0,1,2), use corresponding instance of ``PhiKrylov``
        :param reuse: if True, attempt to reuse the old Krylov space
        :returns: 3-tuple of vectors ``( phi(h/3*Hm)*v, phi(2h/3*Hm)*v, phi(h*Hm)*v )``
        """
        # if Df is nearly singular, Df^-1 * (exp(Df)-1)*v is numerically ill-conditioned, but should give v
        # check norm of Jacobian:
        if self.rhs.normDf(self.y[-1]) < np.finfo(np.double).eps:
            return (v,v,v)
        
        # if v is too small, phi*v is also almost zero.
        # must check for this since arnoldi iterations needs v != 0
        if norm(v) < np.finfo(np.double).eps:
            res = np.zeros_like(v)
            return (res,res,res)
        
        # use flag argument to find correct KrylovPhi instance
        if not flag in [0,1,2]:
            pk = KrylovPhi(self.rhs, exp4=self, tau=h)
            reuse = False
        else:
            pk = self.krylov[flag]
            pk.tau = h
            if np.isscalar(v) or len(v) <= pk.m_max:
                reuse = False
        
        # reuse already computed krylov space
        if reuse:
            p1,p2,p3 = pk.phi3()
        else:
            # compute Krylov subspace approximation of phi(h*A)*v
            pk.compute(v)
            if self.stats:
                # dimension of Krylov space minus 1 corresponds to number of applyA calls
                self.stats["RHSevals"] += pk.m-1
        p1,p2,p3 = pk.phi3()
        
        return (p1, p2, p3)
    
    def integrate(self, x0, t0, tend, h=None, abstol=None, reltol=None):
        """
        :param x0: initial value
        :param t0: initial time
        :param tend: end time
        :param h: initial timestep size. If not specified, (tend-t0)/100 will be used.
        :param abstol: absolute tolerance. Overwrites self.abstol
        :param reltol: relative tolerance. Overwrites self.reltol
        :returns: tuple ``(t,y)``
        """
        if h is None:
            h = np.double(tend-t0)/100.
        if abstol != None:
            self.abstol = abstol
        if reltol != None:
            self.reltol = reltol
        # set initial h_kry in krylov iterations:
        for i in range(3):
            self.krylov[i].h_kry = h
        
        self.hmax = np.double(tend-t0)/2
        t = [t0]
        x = x0; self.y = [x0] # x is current iterate x(t), y is list of all iterates x_i(t)
        reuse_krylov = False # set to true if step rejected
        while t[-1] != tend:
            reuse_krylov = False
            f_xold = self.rhs.Applyf(x)
            
            k1, k2, k3 = self.phiv(h, f_xold, 0, reuse=reuse_krylov)
            
            w4 = -7./300*k1 + 97./150*k2 - 37./300*k3
            u4 = x+h*w4
            d4 = self.rhs.Applyf(u4) - f_xold - h*self.rhs.ApplyDf(x,w4)
            
            k4, k5, k6 = self.phiv(h, d4, 1, reuse=reuse_krylov)
            
            w7 = 59./300*k1 - 7./75*k2 + 269./300*k3 + 2./3*(k4+k5+k6)
            u7 = x+h*w7
            d7 = self.rhs.Applyf(u7) - f_xold - h*self.rhs.ApplyDf(x,w7)
            
            k7,bla1,bla2 = self.phiv(h, d7, 2, reuse=reuse_krylov)
            
            # compute embedded methods first (before x is changed!)
            xembed1 = x + h*(k3 - 2./3*k5 + 0.5*(k6+k7-k4))
            xembed2 = x + h*(-k1 + 2*k2 - k4 + k7)
            # update x
            x = x + h*(k3 + k4 - 4./3*k5 + k6 + 1./6*k7)
            # local error estimation: minimum of error estimation by the two embedded methods
            err_loc = np.minimum( np.abs(x-xembed1), np.abs(x-xembed2) )
            
            # compute tolerance (is function of current iterate due to reltol)
            if self.normcontrol:
                tol = max(abstol, reltol*max(norm(x,np.inf),1))
            else:
                tol = np.maximum(abstol, reltol*np.abs(x))
            
            # check if error is too large
            accept_err = True
            if np.any(err_loc > tol):
                accept_err = False
            
            # dynamic timestep adjustment
            accept_kry, h_kry = self.krylov[0].stepsize_kry(h)
            h_err = self.stepsize_err(err_loc,tol,h)
            hnew = min(h_kry, h_err, self.hmax)
            #hnew = min(h_err, self.hmax)
            ##print "h_kry:",h_kry,"\th_err",h_err,"h",h,"\n"
            
            if accept_err and accept_kry:
                # accept step
                self.y.append(x)
                t.append(t[-1]+h)
                if self.stats:
                    self.stats["accept"] += 1
                    self.stats["h_err"].append(h_err)
                    self.stats["h_kry"].append(h_kry)
                    self.stats["m_max"] = max(self.stats["m_max"],self.krylov[0].m)
                    self.stats["m"].append(self.krylov[0].m)
                # if dominated by Krylov error, approximate next Jacobian by current one
                # last cond: if last step was accepted, no longer use old Jacobian
                if h_kry > h_err and accept_kry:
                    reuse_krylov = True
                else:
                    reuse_krylov = False
            else:
                # reject step
                x = self.y[-1]
                if self.stats:
                    self.stats["reject"] += 1
                reuse_krylov = True
            
            h = hnew
            # make sure t+h <= tend
            if t[-1] + h > tend:
                h = tend - t[-1]
            
        return t,np.array(self.y)
    
    def stepsize_err(self,err,tol,h):
        """
        Compute stepsize factor needed s.t. timestep tolerances are fulfilled
        
        :param err: estimated error of current step
        :param tol: current tolerance
        :param h: current stepsize
        :returns: new stepsize
        """
        # timestep adjustment factors
        alpha = 1./3 # 1/(p+1) where p is order of lowest scheme used in error estimation (here: 2)
        safetyfac = 0.8 # safety factor
        gust_alpha = 0.7/3
        beta = 0.4/3 # for gustafsson factor
        # correct local error by replacing 0s with maximal value
        err[err==0] = np.max(err)
        # if err is all zeros, replace it by factor yielding 2h as new stepsize
        if np.all(err == 0):
            err = np.max(tol)*0.4**(1./alpha)
        
        fac_classical = safetyfac*(np.min(tol/err))**alpha # classical stepsize factor
        if np.any(self.err_old < 0): # no err_old exists yet
            factor = fac_classical
        else:
            fac_gustafsson= np.min( (tol/err)**alpha * (self.err_old/tol)**beta )
            factor = min(fac_classical, fac_gustafsson)
        self.h_old = h
        self.err_old = err
        return factor*h
    

