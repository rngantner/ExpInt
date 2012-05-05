# First, define some right-hand sides

import time, os
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
# expint includes
from expint.problems import * # example right-hand sides
from expint.methods import *
#from expint.methods import ExpForwardEuler.ExpForwardEuler, ExpRosenbrockEuler.ExpRosenbrockEuler, ode45.ode45#, Exp4, Exp4_adaptive
from expint.Problem import Problem

if __name__ == '__main__':
    # plotting symbols:
    sym = ('-o','-^','-s','-<','->')
    # right-hand sides
    he = HeatEquation()
    rhss = (
            LotkaVolterra(0.01,0.02),
            StiffODE(10),
            MathematicalPendulum(1,1),
            QuadraticODE(1),
            Van_der_Pol(1),
           )
    #rhss = ( he, )#LotkaVolterra(0.01,0.02), StiffODE(10), Van_der_Pol(1),  )
    # initial values to each right-hand side: (t0, tend, y0, y_ex(tend))
    initvals = ( # t0, tend, y0, yend (usually yend computed with ode45 with abstol=reltol=1e-16)
                 (0,25,np.array((20,20)),np.array([ 148.64271069, 45.50508679]) ), # LotkaVolterra(0.01,0.02)
                 (0.0,2,1.0,np.exp(-10*2)), # StiffODE(10)
                 (0,10,np.array((2,0)), np.array([ 0.71314818060143281, -1.5313085041358192]) ), # MathematicalPendulum(1,1)
                 (0.0,0.9,1.0,10), # QuadraticRHS(1)
                 (0, 10, np.array((2,0)), np.array([ -2.008340782579709,  0.032907065863241144]) ), #Van-der-Pol(1)
                 (0,1,he.init1,he.sol1 ), # HeatEquation
               )
    nvals_vals = ( # values for N
                  np.array(np.logspace(2.2,4,10), dtype=int),
                  np.array(np.logspace(1.2,4,10), dtype=int),
                  np.array(np.logspace(1,3,10), dtype=int),
                  np.array(np.logspace(1,4,10), dtype=int),
                  np.array(np.logspace(1.5,3.9,10), dtype=int),
                  np.array(np.logspace(1.8,3,8), dtype=int),
                  )
    
    #methods = ( ExplicitEuler, ExpRosenbrockEuler, Exp4 )
    methods = ( ExplicitEuler, ExpRosenbrockEuler, Exp4 )
    adaptivemethods = ( ode45, Exp4_adaptive )
    #adaptivemethods = (ode45,)
    
    for i,rhs in enumerate(rhss):
        # eg. odeexamples/
        directory = "odeexamples"
        if not os.path.exists(directory):
            os.mkdir(directory)
        elif not os.path.isdir(directory):
            raise Exception(directory+" is not a directory; please remove and restart script.")
        # eg. odeexamples/StiffODE/
        relpath = os.path.join(directory,rhs.__class__.__name__) # directory to save in
        if not os.path.exists(relpath):
            os.mkdir(relpath)
        elif not os.path.isdir(relpath):
            raise Exception(relpath+" is not a directory; please remove and restart script.")
        
        print "rhs:",rhs.name()
        # loop over normal methods (not adaptive)
        nvals = nvals_vals[i]
        for j,method in enumerate(methods):
            print "  method:", method.name()
            t0, tend, y0, yexact = initvals[i]
            p = Problem(method,rhs)
            
            err = []; T = []
            for N in nvals:
                time0 = time.time()
                t,y = p.integrate(y0,t0,tend,N)
                T.append(time.time()-time0)
                err.append( np.linalg.norm(y[-1] - yexact) )
#                #################
                plt.figure(2)
                if y[0].shape == ():
                    plt.plot(t,y,label='N=%d'%N)
                else:
                    plt.plot(t,y[:,0],label='N=%d'%N)
            if hasattr(rhs,'solution'):
                # x are actually t values...
                x = np.linspace(t0,tend,np.max(nvals))
                plt.plot(x,rhs.solution(x),'--',label="exact solution")
            plt.legend(loc='best')
            plt.title(rhs.name()+": "+method.name())
            lims = plt.axis()
            plt.axis([np.min(t), np.max(t), lims[2], lims[3]])
            ax = plt.axes()
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.draw()
            plt.savefig(os.path.join(relpath,method.name()+".pdf"))
            plt.close()
#            #################
            print "  time: %1.3f"%T[-1] # print last time
            # determine slope in loglog diagram
            p = np.polyfit(np.log(nvals),np.log(err),1)
            # plot in loglog with slope in label
            plt.figure(1, figsize=(8,9))
            plt.loglog(nvals,err,sym[j],label=method.name()+", slope: %1.2f"%p[0],linewidth=2,markersize=10)
        # loop over adaptive methods (different nvals)
        tolvals = ( 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11 )
        #tolvals = ( 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12)
        #tolvals = (1e-1, 1e-2, 1e-3, 1e-4)
        # following are good for HeatEquation:
        #tolvals = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
        for j,method in enumerate(adaptivemethods):
            print "  method:", method.name()
            t0, tend, y0, yexact = initvals[i]
            p = Problem(method,rhs)
            
            err = []; nvals2=[]; T = []
            for tol in tolvals:
                time0 = time.time()
                t,y = p.integrate(y0,t0,tend,abstol=tol,reltol=tol)
                T.append(time.time()-time0)
                err.append( np.linalg.norm(y[-1] - yexact) )
                nvals2.append(len(t))
#                #################
                plt.figure(2)
                if not y[0].shape == ():
                    y = y[:,0]
                plt.plot(t,y,label='tol=1e%d'%int(np.log10(tol)))
                #### plot h_err and h_kry if method == Exp4_adaptive
                if method == Exp4_adaptive and tol == tolvals[-1]:
                    plt.figure(3)
                    plt.semilogy(p.method.stats["h_err"],label=r"$h_{err}$")
                    plt.semilogy(p.method.stats["h_kry"],label=r"$h_{kry}$")
                    plt.legend()
                    plt.xlabel("Step number")
                    plt.ylabel("Step size")
                    plt.draw()
                    plt.savefig(os.path.join(relpath,"stepsize.pdf"))
                    plt.close()
                    plt.figure(2) # switch back to figure 2
                ####
            if hasattr(rhs,'solution'):
                x = np.linspace(t0,tend,np.max(nvals2))
                plt.plot(x,rhs.solution(x),'--',label="exact solution")
            lims = plt.axis()
            plt.axis([np.min(t), np.max(t), lims[2], lims[3]])
            ax = plt.axes()
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.title(rhs.name()+": "+method.name())
            plt.draw()
            plt.savefig(os.path.join(relpath,method.name()+".pdf"))
            plt.close()
#            #################
            print "  time: %1.3f"%T[-1] # print last time
            # determine slope in loglog diagram
            p = np.polyfit(np.log(nvals2[1:]),np.log(err[1:]),1)
            # plot in loglog with slope in label
            plt.figure(1, figsize=(8,9))
            plt.loglog(nvals2,err,sym[j],label=method.name()+", slope: %1.2f"%p[0],linewidth=2,markersize=10)
        
        # finalize plotting of errors
        ax = plt.axes()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0+box.height*0.2, box.width, box.height*0.8])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1))
        plt.xlabel("Number of timesteps $n$",fontsize=16)
        plt.ylabel(r"Error  $\varepsilon = ||\hat{u}(t_{end})-u_{ex}(t_{end})||$",fontsize=16)
        plt.title(rhs.name(),fontsize=16)
        ax = plt.axis()
        plt.axis((min(np.min(nvals),np.min(nvals2)), max(np.max(nvals),np.max(nvals2)), ax[2], ax[3] ))
        plt.draw()
        plt.savefig(os.path.join(relpath,rhs.__class__.__name__+".pdf"))
        #plt.show()
        plt.close()
    
