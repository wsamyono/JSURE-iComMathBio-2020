#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 13:05:58 2020
@author: skyler
"""
import math 
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
# from scipy.interpolate import interp1d

# Import CSV data file
# Column 1 = time (t)
# Column 2 = input (u)
# Column 3 = output (yp)
data = np.loadtxt('datameagancontrol.txt',delimiter=',')
# u0 = data[0,1]
# yp0 = data[0,1]
# t = data[:,0].T - data[0,0]
t = data[:,0]
# u = data[:,1].T
yp = data[:,1]

# specify number of steps
ns = len(t)
# delta_t = t[1]-t[0]
# create linear interpolation of the u data versus time
# uf = interp1d(t,u)

# define first-order plus dead-time approximation    
# def fopdt(y,t,uf,Km,taum,thetam)
def fopdt7(y,t,x,alpha1,alpha2,alpha3,alpha4, beta, gamma):
    # arguments
    #  y      = output
    #  t      = time
    #  uf     = input linear function (for time shift)
    #  Km     = model gain
    #  taum   = model time constant
    #  thetam = model time constant
    # time-shift u]
    alpha4 = x[3]
    beta = x[4]
    gamma = x[5]
    
  #  try:
    if t <= 1:
                alpha1 = x[0]
                dydt = (alpha1 * y)*math.log1p(beta/(y + gamma)) - alpha4 * y
    else: 
        if t <= 2:
                alpha2 = x[1]
                dydt = (alpha2 * y)*math.log1p(beta/(y + gamma)) - alpha4 * y
        else:
                alpha3 = x[2]
                dydt = (alpha3 * y)*math.log1p(beta/(y + gamma)) - alpha4 * y
  #  except:
    #    print('Error with time extrapolation: ' + str(t))
    #    um = 0
    # calculate derivative
    #    dydt = Km*y                # (-(y-yp0))*Km # + Km * (um-u0))/taum
    return dydt

# simulate FOPDT model with x=[Km,taum,thetam]
def sim_model(x):
    # input arguments
    alpha1 = x[0]
    alpha2 = x[1]
    alpha3 = x[2]
    alpha4 = x[3]
    beta   = x[4]
    gamma  = x[5]
  
    # storage for model values
    ym = np.zeros(ns)  # model
    # initial condition
    ym[0] = 390000
    # loop through time steps    
    for i in range(0,ns-1):
        ts = [t[i],t[i+1]]
  #              y1 = odeint(fopdt,ym[i],ts,args=(uf,Km,taum,thetam))
        y1 = odeint(fopdt7,ym[i],ts,args=(x,alpha1,alpha2,alpha3,alpha4, beta, gamma))
        ym[i+1] = y1[-1]
    return ym

# define objective
def objective(x):
    # simulate model
    ym = sim_model(x)
    # calculate objective
    obj = 0.0
    for i in range(len(ym)):
        obj = obj + (ym[i]-yp[i])**2    
    # return result
    return obj

# initial guesses
x0 = np.zeros(6)   #       Powell
x0[0] = 4000.0 # alpha1 = 5000  
x0[1] = 6500 # alpha2 = -6800
x0[2] = 12000 # alpha3 = 1500
x0[3] = 0.5  # lpha4 = 0.5
x0[4] = 110 # beta = 110       110 
x0[5] = 0.7 # gamma = 0.7   
ym1 = sim_model(x0)
plt.plot(t,ym1,'b-',linewidth=2,label='Initial Guess')
plt.plot(t,yp,'ko-',linewidth=2,label='Experiment Data')
#plt.title('Best Fit Model with Control - Gompertz')
#plt.xlabel('Days')
#plt.ylabel('Number of Cells')
#plt.legend(loc='best')
plt.show() 

# show initial objective
print('Initial SSE Objective: ' + str(objective(x0)))
print('alpha01: ' + str(x0[0]),', alpha02: ' + str(x0[1]), ' and alpha03: ' + str(x0[2]), 'alpha04: ' + str(x0[3]))
# print('alpha04: ' + str(x0[3]))
print(' beta0: ' + str(x0[4]))
print(' gamma0: ' + str(x0[5]))
# optimize Km, taum, thetam
#solution = minimize(objective,x0,method='nelder-mead',
#               options={'xatol': 1e-8, 'maxfev': 100000, 'disp': True})
solution = minimize(objective,x0,method='powell',
               options={'xtol': 1e-8, 'maxfev': 100000, 'disp': True})

x = solution.x

# show final objective
print('Final SSE Objective: ' + str(objective(x)))

print('alpha1: ' + str(x[0]),', alpha2: ' + str(x[1]),' and alpha3: ' + str(x[2]))
print('alpha4: ' + str(x[3]))
print(' beta: ' + str(x[4]))
print(' gamma: ' + str(x0[5]))

# print('taup: ' + str(x[1]))
# print('thetap: ' + str(x[2]))

# calculate model with updated parameters
# ym1 = sim_model(x0)
ym2 = sim_model(x)
# plot results
# plt.figure(1)
# plt.subplot(2,1,1)
plt.plot(t,yp,'ko-',linewidth=2,label='Experiment Data')
plt.plot(t,ym1,'b-',linewidth=2,label='Initial Guess')
plt.plot(t,ym2,'r-',linewidth=3,label='Optimized Model')
plt.title('Best Fit Model with Control - Gompertz')
plt.xlabel('Days')
plt.ylabel('Number of Cells')
plt.legend(loc='best')
#plt.subplot(2,1,2)
#plt.plot(t,x0[1,],'bx-',linewidth=2, )
#plt.plot(t,x[1,],'r--',linewidth=3)
#plt.legend(['Measured','Interpolated'],loc='best')
#plt.ylabel('Input Data')
data = np.vstack((t,yp,ym2,)) # vertical stack
data = data.T              # transpose data
np.savetxt('outputdatamd6control.txt',data,delimiter=',')
plt.savefig('Gompertz Model With control.png', dpi=300,bbox_inches='tight')
plt.show()