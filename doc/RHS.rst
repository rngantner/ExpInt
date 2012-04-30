
===================================
Representations of Right-Hand Sides
===================================
A differential equation can be specified in many different forms, some of them more useful in the context of exponential integrators.
For example, just specifying the right-hand side :math:`f(u)` is not very useful, as the Jacobian of :math:`f` is needed in the computation.

Therefore, there exist a few different classes to represent the right-hand of the ODE :math:`u'=f(u)` in different ways.

DfRHS
=====
If the Jacobian (or a procedural version describing its application to u) is known, one can use the ``DfRHS`` class.
It is documented as follows:

.. autoclass:: RHS.DfRHS
  :members:

Examples
++++++++

The Lotka-Volterra model for modeling predator-prey relationships is implemented as an easy example in the following class. All necessary functions are already implemented in the base class ``DfRHS``.

.. literalinclude:: ../RHS.py
   :pyobject: LotkaVolterra

The standard implementation contains the necessary functions for some common cases, for example if the Jacobian can be represented as a numpy.ndarray or a procedural function.
For these cases, the ``Df, g, f`` keyword arguments can be used, as shown in the above example.

For more complicated ODEs, it may make more sense to write a new class that derives from DfRHS.
An example can be found in the class ``HeatEquation``.
It represents a high-dimensional (spatial) discretization of the heat equation which is to be solved in time.
The application of the Jacobian involves the inverse of a matrix, which is efficiently implemented by computing a sparse LU-decomposition (in the constructor) and then applying forward and backward substitution each time it is applied.

The following function (from the class ``HeatEquation``) contains the implementation of the application of the Jacobian to a vector v:

.. literalinclude:: ../HeatEquation.py
   :pyobject: HeatEquation.ApplyDf
