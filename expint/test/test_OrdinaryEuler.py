#!/usr/bin/env python2
# coding: utf-8
import time
from numpy.testing import assert_almost_equal
from expint.methods.OrdinaryEuler import *
from expint.problems import QuadraticODE
from numpy import polyfit, log

class AbstractEulerTest(object):
    def __init__(self,methodClass,order):
        self.rhs = QuadraticODE.QuadraticRHS()
        self.method,self.order = methodClass(self.rhs), order
        self.x0 = 1.0
        self.t0, self.tend = 0.,0.2
    
    def test_correctsol(self):
        meth = ExplicitEuler(self.rhs)
        t,sol_euler = self.method.integrate(self.x0,self.t0,self.tend)
        exact = self.rhs.solution(self.tend)
        print sol_euler[-1], exact
        print type(sol_euler[-1]), type(exact)
        assert_almost_equal(sol_euler[-1],exact,2)

    def test_rate(self):
        exact = self.rhs.solution(self.tend)
        err = {}
        for N in (10,20,50,100,200,500):
            t,sol = self.method.integrate(self.x0,self.t0,self.tend, N=N)
            err[N] = abs(exact - sol[-1])
        p = polyfit(log(err.keys()),log(err.values()),1)
        assert_almost_equal(-p[0],self.order,1)

class TestExplicitEuler(AbstractEulerTest):
    def __init__(self):
        # tests if solution is correct and rate is close to 1
        super(TestExplicitEuler,self).__init__(ExplicitEuler,1.)

