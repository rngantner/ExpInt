# ExpInt

ExpInt contains tools to aid in the implementaiton of exponential integrators.
A few algorithms, including the Exp4 algorithm (Hochbruck, Lubich, Selhofer) are implemented.

## Files
Description of the files in this directory:

* Examples.py
   * Contains various differential equations (as specializations of the RHS class)
   * Loops over ODEs, Methods and N/tol values to create convergence plots, result plots, etc.
    
* Exp4.py
   * Contains the implementation of the Exp4 algorithm.
   * Class Exp4: uses equidistant nodes; achieves order 4, but is often slow (no Krylov space adaptivity)
   * Class Exp4_adaptive: changes timestep size based on local error estimators and Krylov space error estimator
    
* HeatEquation.py
   * Contains a spatial FE discretization of the heat equation as a subclass of the class RHS
   * Implements all necessary functions needed by Exp4; procedural Df(x)*u, g(u), ...
   * Internally stores a sparse LU decomposition to reduce computational cost.
    
* KrylovPhi.py
   * Implements a helper class for the Exp4 algorithm that manages the Krylov space, provides error estimation and statistics.
    
* MatrixExponential.py
   * Nice class structure for describing matrix exponentials; not used in Exp4 due to its highly specialized requirements.
    
* Method.py
   * Base class "Method" from which all methods should derive.
   * A few methods are implemented here, including ExplicitEuler, ExpRosenbrockEuler, ode45 (wrapper), ...
    
* Problem.py
   * "Problem" class which provides a link between RHSs and Methods.
   * Provides some convenience functions; can differentiate between adaptive and non-adaptive methods (if they derive from correct classes).
    
* RHS.py
   * Base class for a general right-hand side.
   * One specialization, "DfRHS", describes ODE of the form u' = Df(u)*u+g(u), Df is the Jacobian of u'=f(u)
    
* Timings.py
   * executes timings of variouse algorithms and problems
     
* ode45.py
   * Implementation of the ode45 algorithm (inspired by MATLAB implementation; copied from "Numerik für Physiker" files and improved a bit.)

