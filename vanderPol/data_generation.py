"""
PLEASE NOTE THAT THIS CODE IS ADOPTED AND REVISED FROM: http://math.colgate.edu/math329/exampleode.py
THE CODE WAS NOT ORIGINALLY WRITTEN BY HANFENG ZHAI
"""
"""
exampleode.py
=============

The example code helps demonstrate how to use one ODE solver
from the SCIPY package: scipy.integrate.odeint()
There is a more object oriented solver in SCIPY called ode
which may have some features that odeint does not.

The example set of equations is the van der Pol oscillator:
         y''-mu*(1-y^2)*y'+y=0    y(0)=2,  y'(0)=0
The first step is to rewrite as a system of equations:
         y1'=y2      y2'=mu*(1-y^2)*y'+y    y1(0)=2,  y2(0)=0
The function vanderpol() returns the RHS vector for this system.

The function run_vanderpol() sets up parameters, runs the solver,
and returns the solution. 
The function draw_figures() plots a few figures with this data.
"""
import numpy as np
from numpy import *
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def vanderpol(y,t,mu):
    """ Return the derivative vector for the van der Pol equations."""
    y1= y[0]
    y2= y[1]
    dy1=y2
    dy2=mu*(1-y1**2)*y2-y1
    return [dy1, dy2]

def run_vanderpol(yinit=[1,0], tfinal=30, mu=3): #Change hyperparameters
    """ Example for how to run odeint.

    More info found in the doc_string. In ipython type odeint?
    """
    times = linspace(0,tfinal,3000)

    rtol=1e-6
    atol=1e-10

    y = odeint(vanderpol, yinit, times, args= (mu,), rtol=rtol, atol=atol)
    return y,times


if __name__ == "__main__":
    y,t = run_vanderpol([1,0], tfinal=30, mu=3) #Change hyperparameters
    draw_figure(y,t)
    plt.show()
    np.savetxt("t.txt", np.hstack(t))
    np.savetxt("x.txt", np.hstack(y[:,0]))
    np.savetxt("xdot.txt", np.hstack(y[:,1]))
