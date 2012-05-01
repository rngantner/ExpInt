## module gaussNodes
''' x,A = gaussNodes(m,tol=10e-9)
    Returns nodal abscissas {x} and weights {A} of
    Gauss-Legendre m-point quadrature.
'''
from math import cos,pi
from numpy import zeros, mat, hstack
from numpy.linalg import norm

def gaussNodes(m,tol=10e-9):

    def legendre(t,m):
        p0 = 1.0; p1 = t
        for k in range(1,m):
            p = ((2.0*k + 1.0)*t*p1 - k*p0)/(1.0 + k )
            p0 = p1; p1 = p
        dp = m*(p0 - t*p1)/(1.0 - t**2)
        return p,dp

    A = zeros(m)   
    x = zeros(m)   
    nRoots = (m + 1)/2          # Number of non-neg. roots
    for i in range(nRoots):
        t = cos(pi*(i + 0.75)/(m + 0.5))  # Approx. root
        for j in range(30): 
            p,dp = legendre(t,m)          # Newton-Raphson
            dt = -p/dp; t = t + dt        # method         
            if abs(dt) < tol:
                x[i] = t; x[m-i-1] = -t
                A[i] = 2.0/(1.0 - t**2)/(dp**2) # Eq.(6.25)
                A[m-i-1] = A[i]
                break
    return x,A

def arnoldi(A, v0, k):
    """
    Arnoldi algorithm (Krylov approximation of a matrix)
        input: 
            A: matrix to approximate
            v0: initial vector (should be in matrix form) 
            k: number of Krylov steps 
        output: 
            V: matrix (large, N*k) containing the orthogonal vectors
            H: matrix (small, k*k) containing the Krylov approximation of A

    Author: Vasile Gradinaru, 14.12.2007 (Rennes)
    """
    #print 'ARNOLDI METHOD'
    v0.shape = (len(v0),)
    inputtype = A.dtype.type
    V = mat( v0.copy() / norm(v0), dtype=inputtype)
    H = mat( zeros((k+1,k), dtype=inputtype) )
    V = V.T
    for m in xrange(k):
        vt = A*V[ :, m]
        for j in xrange( m+1):
            H[ j, m] = (V[ :, j].H * vt )[0,0]
            vt -= H[ j, m] * V[:, j]
        H[ m+1, m] = norm(vt);
        if m is not k-1:
            V =  hstack( (V, vt.copy() / H[ m+1, m] ) )
    return V,  H

