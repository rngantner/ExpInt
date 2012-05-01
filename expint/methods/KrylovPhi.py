#!/usr/bin/env python
import numpy as np
from numpy.linalg import norm
import types
import expint.RHS

class KrylovPhi(object):
    """
    Compute the Krylov subspace approximation of phi(t*A)v
    with phi(z) = (exp(z)-1)/z
    
    Arguments of constructor:
        A
            can be one of the following:
            - matrix represented as an nxn np.ndarray
            - function implementing the linear map :math:`v \mapsto A\cdot v`
            - instance of subclass of RHS. the ApplyDf function is used as the A*v function
        
        exp4  (optional)
            an instance of the class ``Exp4``. This enables the use of a scaled
            norm based on the current iterate.
        
        tau (default: 1)
            factor in :math:`\\varphi(\\tau\cdot A)`
        
        m
            tuple containing the following (in this order)
              * m_min: minimal Krylov space size
              * m_max: maximal Krylov space size
              * m_opt: optimal Krylov space size (m_min < m_opt < m_max)
              * m_suf: sufficient Krylov space size (*currently unused*)
        
    """
    #TODO: let constructor argument A be instance of scipy LinearOperator
    def __init__(self,A,exp4=None,tau=1.,m=None):
        """
        Krylov-space approximation of Phi: A -> A^-1*exp(A-I)
        A   :   ndarray, function or instance of rhs
        exp4:   instance of Exp4 integrator
        tau :   factor in Phi(tau*A)
        m   :   4-tuple with values (m_min, m_max, m_opt, m_suf)
                m_min: minimal Krylov space size
                m_max: maximal Krylov space size
                m_opt: optimal Krylov space size (m_min < m_opt < m_max)
                m_suf: sufficient Krylov space size
        """
        self.A = A # calls setA, since its a class property
        self.Exp4Instance = exp4
        self.tau = tau
        # global krylov step size; not the one to use in computation
        self.h_kry = tau
        # Krylov space size parameters
        if m == None:
            self.m_min = 10. # minimal Krylov space size
            self.m_max = 36. # maximal Krylov space size
            self.m_opt = 27. # optimal Krylov space size
            self.m_suf = 20. # sufficient Krylov space size
        else:
            # argument m must be a 4-tuple
            if len(m) != 4:
                raise ValueError("Argument error: m must be 4-tuple!")
            self.m_min, self.m_max, self.m_opt, self.m_suf = m
        self.m_stable_min = 2. # minimum for stability when using with very small systems
        self.exponent = 1./3 # for arnoldi. lanczos: use 1./2
        self.m_old1= self.m_max # last Krylov space size
        self.m_old2= self.m_max # before last Krylov space size
        # counter for number of steps for which m is smaller than a very small number (4)
        self.m_verysmall = 4.
        self.N_smaller_verysmall = 0
        
        # create variables V and H. should only be allocated once
        # (starting at first call of compute(v), since then n is known)
        # eg. self.V = zeros((n,self.m_max))
        self.Vm = None # N unknown here
        self.Hm = None
        self.v_m1 = None
        self.h_m1 = None
    
    def getA(self):
        return self.A_internal
    
    def setA(self,A):
        """Checks type of A and sets internal variables accordingly"""
        self.computed = False
        self.A_internal = A
        # set applyA to a function returning the matrix-vector product
        if type(self.A_internal) is np.ndarray:
            self.applyA = lambda v: np.dot(self.A_internal,v)
        elif type(self.A_internal) is types.FunctionType:
            # assume A returns a np.ndarray
            self.applyA = lambda v: self.A_internal(v)
        elif isinstance(self.A_internal, expint.RHS.RHS):
            self.applyA = lambda v: self.A_internal.ApplyDf(self.Exp4Instance.y[-1],v)
        elif self.A_internal is None:
            self.applyA = lambda v: v
    
    A = property(getA,setA) # calls setA when something is assigned to self.A
    
    def setExp4(self,exp4instance):
        """set self.Exp4Instance to given argument"""
        self.Exp4Instance = exp4instance

    def generalizedResidual(self):
        """
        :returns: generalized residual as defined in [Hochbruck et al.], Section 6.3
        
        It is a vector (in direction of :math:`v_{m+1}`) of length ``self.n = len(self.v)``.
        """
        if np.all(self.Hm < np.finfo(np.double).eps):
            phiHm = np.array((1,),ndmin=2)
        else:
            phiHm = self.phiMat(self.Hm)
        return self.normv * self.tau * self.h_m1 * phiHm[-1,0] * self.v_m1
    
    def krylovStep(self):
        """Executes one Krylov step, i.e. increases dimension of Krylov space by 1."""
        m = self.m
        
        # update Hm and Vm with info from last step
        self.Hm[m,m-1] = self.h_m1
        self.v_m1.shape = (self.v_m1.shape[0],1)
        self.Vm[:,m] = self.v_m1.copy().T #v_m1 is normalized
        
        # compute new step
        self.v_m1 = self.applyA(self.Vm[:,m])
        # Gram-Schmidt orthogonalization
        # TODO: replace with Householder ortho. -> is more stable
        for j in xrange(m+1):
            self.Hm[j, m] = np.dot(self.Vm[:,j], self.v_m1)
            self.v_m1 -= self.Hm[ j, m] * self.Vm[:, j]
        self.h_m1 = norm(self.v_m1)
        m = m+1; self.m = m
        
        # test for break-down:
        if norm(self.h_m1) < np.finfo(np.double).eps:
            self.isExact = True
            return
        
        self.v_m1 = self.v_m1 / self.h_m1
        return
    
    def compute(self,v):
        """
        Computes the Krylov subspace with starting vector v.
        sets the following member variables:
        
        :param v: Starting vector of the Arnoldi iteration.
        
        ``self.Vm``
            orthonormal basis of Krylov space
        ``self.Hm``
            Hessenberg matrix
        ``self.v_m1``
            :math:`v_{m+1}`
        ``self.h_m1``
            :math:`h_{m+1,m}`
        """
        # initialization
        self.v = np.array(v,ndmin=1)
        self.m = 1
        self.normv = norm(self.v) # needed repeatedly in generalizedResidual()
        self.isExact = False
        self.finished = False
        self.n = len(self.v)
        N = self.n
        # reduce m_max if it is larger than N
        if self.m_max > N:
            fac = np.double(N)/self.m_max # just scale everything with a linear factor
            self.m_max = np.double(int(fac*self.m_max))
            self.m_min = np.double(int(fac*self.m_min))
            self.m_opt = np.double(int(fac*self.m_opt))
            self.m_suf = np.double(int(fac*self.m_suf))
        
        # initialize V1, H1:
        if self.Vm is None:
            self.Vm = np.zeros((N,self.m_max))
            self.Hm = np.zeros((self.m_max,self.m_max)) # do this here b/c m_max may have changed
        else:
            # reset all elements of both V and H to zero
            self.Hm.fill(0)
            self.Vm.fill(0)
        
        self.Vm[:,0] = v.copy()/self.normv
        
        #print self.normv, self.Hm*self.normv, self.applyA(1.0)
        
        # new vector:
        self.v_m1 = self.applyA(self.Vm[:,0])
        self.Hm[0,0] = np.dot(self.Vm[:,0],self.v_m1)
        self.v_m1 -= self.Vm[:,0]* self.Hm[0,0]
        self.h_m1 = norm(self.v_m1)
        # check for breakdown. already done
        if self.h_m1 < np.finfo(np.double).eps:
            return
        
        self.v_m1 = self.v_m1 / self.h_m1
        
        # now:  A*Vm = Vm*Hm + h_m1*v_m1*e_1
        
        # add more krylov steps until the stopping criterion is fulfilled
        while not self.finished and self.m < self.m_max:
            self.krylovStep()
            self.finished = self.stop()
        #print "n:",len(self.v), "m:",self.m
        self.computed = True
        return
    
    def stepsize_kry(self,h,verbose=False):
        """
        Compute new Krylov stepsize
          * If the number of steps is in [m_min, m_max], preserve step size.
          * If m > m_max, reduces h_kry until error bound holds.
          * If m < m_min for at least two consecutive steps, increases h_kry.

        :returns: a tuple with boolean (accept=true) and the Krylov step size h_kry.
        """
        accept = True
        if verbose:
            print "m,min,max",self.m, self.m_min, self.m_max,"\th_kry",self.h_kry
        
        if self.m == self.n:
            if verbose:
                print "C1b m in [m_min, m_max], h_kry:",self.h_kry
            #self.h_kry = min(1.2*self.h_kry, self.Exp4Instance.hmax)
        # 1. if m in [m_min, m_max]
        elif self.m >= self.m_min and self.m <= self.m_max and self.finished:
                if verbose:
                    print "C1a m in [m_min, m_max], h_kry:",self.h_kry
                if self.m > self.m_min:
                    factor = (self.m_opt/self.m)**self.exponent
                    self.h_kry = self.h_kry*factor
                    self.h_kry = min(self.h_kry,self.Exp4Instance.hmax)
        
        # 2. if m > m_max (equality b/c iteration stops at m_max)
        # the maximal number of steps was reached: make h smaller
        elif self.m == self.m_max and not self.finished:
            self.finished = True # needed for reuse
            scaledNorm_vm1 = self.Exp4Instance.scaledNorm(self.v_m1)
            accept = False
            tau_original = self.tau
            factor = 1.
            
            res = self.tau * self.Exp4Instance.scaledNorm(self.generalizedResidual())
            res_bound = 1 # max(1,err_loc) # err_loc from Exp4 -> add argument!
            # compute phi matrices to compute error bounds
            p1,p2,p3 = self.phi3()
            #return self.normv * self.tau * self.h_m1 * phiHm[-1,0] * self.v_m1
            #phiHm = self.phiMat(self.Hm)
            #print "phiHm[-1,0]*|v|: %f\t p3[-1]: %f"%(self.normv*phiHm[-1,0]/self.tau,p3[-1])
            #res1 = abs(self.tau * self.tau * p3[-1]) * scaledNorm_vm1
            #res1 = abs(self.tau**3 * p3[-1]*self.h_m1) * scaledNorm_vm1
            #res2 = self.tau * self.Exp4Instance.scaledNorm(self.tau * p3[-1] * self.v_m1)
            #print "res1:%f\tres2:%f\tres:%f"%(res1, res2, res)
            #print "%f\t%f\t%f\t%f\t%f"%(res1-res,self.h_m1,self.tau,self.tau*phiHm[-1,0],p1[-1]/self.normv)
            #res = self.tau * self.tau * p3[-1] * scaledNorm_vm1 # h_m1 missing???
            while res > res_bound:
                # try 2/3*h and 1/3*h because those are cheap
                # use precomputed scaledNorm -> save O(n)
                if (2./3) * abs(self.tau**2 * self.h_m1 * p2[-1]) * scaledNorm_vm1 < 1:
                    factor *= 2./3
                elif (1./3) * abs(self.tau**2 * self.h_m1 * p1[-1]) * scaledNorm_vm1 < 1:
                    factor *= 1./3
                else:
                    factor *= 1./6
                self.tau = tau_original*factor # factor cumulates!
                # p1,p2 may be used in next traversal of while loop
                p1,p2,p3 = self.phi3()
                # use p3 here. divide by factor b/c already in self.tau (factor**2 in formula)
                res = self.tau * self.Exp4Instance.scaledNorm(self.tau * p3[-1] * self.v_m1)/factor
            self.tau = tau_original
            self.h_kry = factor*self.tau
            if verbose:
                print "C2 m > m_max, h_kry:",self.h_kry
        
        # 3. if m < m_min in 2 consecutive steps
        elif self.m < self.m_min:
            factor = 1.5 # in case only smaller in one step
            if self.m_old1 < self.m_min:
                factor = (self.m_opt/self.m)**self.exponent
                self.h_kry = factor*self.h_kry
                if verbose:
                    print "C3 m < m_min in 2 consecutive steps. h_kry:",self.h_kry
        
        # 4. if m < m_verysmall (=4) in j consecutive steps
        if self.m < self.m_verysmall and self.n > self.m_verysmall:
            self.N_smaller_verysmall += 1
            #self.N_smaller_verysmall = min(self.N_smaller_verysmall, 10)
            self.h_kry = 2*h
            if verbose:
                print "C4 m << m_min, h_kry:",self.h_kry
        else:
            self.N_smaller_verysmall = 0
        
        self.m_old2 = self.m_old1
        self.m_old1 = self.m
        return accept, self.h_kry
    
    def stop(self):
        """
        Stopping Criterion
        
        :returns: True if iteration is converged
        """
        if self.m < min(self.n,self.m_stable_min):
            # make sure Krylov space has some minimum size (prevents singular matrix in computation of phi)
            return False
        if self.isExact is True: # is set to True in case of a breakdown
            return True
        if self.Exp4Instance == None or not hasattr(self.Exp4Instance,'y'): # use 2-norm
            # TODO: does this case even make sense?
            print "not using weighted norm in stopping criterion!"
            return self.tau*norm(self.generalizedResidual()) < 1e-4
        else: # use scaled norm
            return self.tau*self.Exp4Instance.scaledNorm(self.generalizedResidual()) < 1
    
    def phiMat(self,z):
        """
        :param z: Matrix argument to :math:`\\varphi`
        :returns: :math:`\\varphi(z)`
        """
        from numpy.linalg import solve
        from scipy.linalg import expm
        id = np.eye(self.m)
        #if np.all(np.double(z) < np.finfo(np.double).eps):
        #    return id
        #else:
        z = z[:self.m,:self.m]
        return solve(z,( expm(z) - id ))
    
    def phi(self,f=1):
        """
        :returns: :math:`V_m\cdot\\varphi(f\cdot self.H_m)\cdot e_1 \\approx \\varphi(f\cdot A)\cdot v`
        f is a factor (eg. phi3 uses 1./3)
        
        Be sure to call compute(v) first to build the Krylov subspace!
        """
        from numpy.linalg import solve
        from scipy.linalg import expm
        id = np.eye(self.m)
        A = f*self.tau*self.Hm[:self.m,:self.m]
        if np.all(np.double(A) < np.finfo(np.double).eps):
            return self.v# exp(0)*v = v
        else:
            return np.dot(self.Vm[:,:self.m],solve(A,( expm(A) - id )[:,0]))*self.normv
    
    def phi3(self):
        """
        :returns: 3-tuple of matrices ``( phi(t*A), phi(2*t*A), phi(3*t*A) )``
        
        Be sure to call compute(v) first to build the Krylov subspace!
        """
        from numpy.linalg import solve
        from scipy.linalg import expm
        id = np.eye(self.m)
        z = (1./3)*self.tau*self.Hm[:self.m,:self.m]
        #phi1 = self.phiMat(z)
        phi1 = solve(z,( expm(z) - id ))
        zPhi1 = np.dot(z,phi1)
        phi2 = np.dot((0.5*zPhi1+id),phi1)
        phi3 = np.dot(2./3*(zPhi1+id),phi2) + 1./3*phi1
        self.phi3cache = (np.dot(self.Vm[:,:self.m],phi1[:,0])*self.normv,
                          np.dot(self.Vm[:,:self.m],phi2[:,0])*self.normv,
                          np.dot(self.Vm[:,:self.m],phi3[:,0])*self.normv )
        return self.phi3cache

def phi_2(A):
    from numpy.linalg import solve
    from scipy.linalg import expm
    n = A.shape[0]
    id = np.eye(n)
    if n == 1 and np.double(A) == 0:
        return np.array(((1.0,),))
    else:
        return solve(A,( expm(A) - id ))

def phi3_2(A,h):
    """returns 3-tuple of matrices ( exp(h/3*A), exp(2h/3*A), exp(hA) )"""
    z = (1./3)*h*A
    id = np.eye(A.shape[0])
    phi1 = phi_2(z)
    phi2 = np.dot((0.5*np.dot(z,phi1) + id), phi1)
    phi3 = np.dot(2./3*(np.dot(z,phi1)+id), phi2) + 1./3*phi1
    return (phi1, phi2, phi3)

if __name__ == '__main__':
    n = 20
    A = np.random.random((n,n))
    # create a RHS (show capability of PhiKrylov to use general A*v routines)
    from RHS import DfRHS
    rhs = DfRHS(Df=A)
    
    tau = 1
    
    from Exp4 import Exp4
    exp4 = Exp4(rhs)
    #kr = PhiKrylov(A,exp4,tau=tau)
    kr = KrylovPhi(A,exp4,tau=tau)
    
    v = np.random.random(n)
    kr.compute(v.copy())
    V = kr.Vm[:,:kr.m]
    H = kr.Hm[:kr.m,:kr.m]
    
    print "V.shape:", V.shape
    #print "diag(V^TV)", np.diag(np.dot(V.T,V))
    print "V.T*V:", np.around(np.dot(V.T, V), decimals=2)
    
    import pyarnoldi
    V2,H2 = pyarnoldi.arnoldi(tau*A,v.copy(),6)
    
    print "H.shape:", H.shape
    print "H2.shape:", H2.shape
    
    #print "H :", np.abs(H)
    #print "H2:", np.abs(H2[:-1,:])
    
    #print "norm(diff):", norm(H-H2[:-1,:])
    
    lexact = np.max(np.abs(np.linalg.eig(tau*A)[0]))
    l1 = np.max(np.abs(np.linalg.eig(H)[0]))
    l2 = np.max(np.abs(np.linalg.eig(H2[:-1,:])[0]))
    
    print "A :",lexact
    print "H :",l1, "err:", abs(lexact-l1) 
    print "H2:",l2, "err:", abs(lexact-l2)
    
    
    print "\nphi(1/3*tau*A)*v : \n"
    
    #print "Exact value:   ", np.dot(phi3_2(tau*A,1)[0],v)
    #print "Krylov approx.:", kr.phi3()[0]
    
    print "error phi1:",norm( np.dot(phi3_2(tau*A,1)[0],v) - kr.phi3()[0] )
    print "error phi2:",norm( np.dot(phi3_2(tau*A,1)[1],v) - kr.phi3()[1] )
    print "error phi3:",norm( np.dot(phi3_2(tau*A,1)[2],v) - kr.phi3()[2] )

