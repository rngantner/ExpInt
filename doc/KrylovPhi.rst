.. _krylovphi:

=========
KrylovPhi
=========


The ``KrylovPhi`` module is designed to allow efficient computation of the following function by a Krylov subspace approximation.

.. math:: \varphi(z) = z^{-1} (\exp(z)-1)

The class ``KrylovPhi`` provides a means to efficiently store a Krylov space and access the :math:`\varphi(.)` function.
This space can then be reused or a new space computed without reallocation of new memory.
Additionally, this class contains an error estimation algorithm that determines a stepsize :math:`h` for which the Krylov subspace approximation fulfills certain tolerances.
This is of central importance to the :ref:`exp4link` algorithm.

.. autoclass:: KrylovPhi.KrylovPhi
  :members:

